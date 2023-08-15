# %%
import itertools
import time
import argparse
import numpy as np

import torch
from utils import feature_norm, process_link_dataset
import os
import dgl.data

import dgl
from models import APGE, GAEMI
from DPGNN import DPGCN
from sklearn.model_selection import train_test_split
from utils import get_train_val_test, load_link, get_attr_list, compute_loss_para, get_acc, load_credit
from measuring import get_score_attr, get_scores_edge
import scipy.sparse as sp

# Training settings
parser = argparse.ArgumentParser()

# My model args
parser.add_argument('--seed', type=int, default=3, help='Random seed.')
parser.add_argument('--hidden1', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--gpu', type=int, default=2, help='GPU number.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--model', type=str, default='APGE',
                    choices=['APGE'])
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='yale',
                    choices=['rochester', 'yale', 'credit'])
parser.add_argument('--epoch', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--use_pretrain', type=bool, default=False)

args = parser.parse_known_args()[0]
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cuda = not args.no_cuda and torch.cuda.is_available()
# cuda = False
print(args)
# %%
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
else:
    device = 'cpu'

if args.dataset == 'yale':
    n_class = 6
    attr_index = 5


def main():
    if args.dataset == 'yale' or args.dataset == 'rochester':
        adj, features, adj_train, val_edges, val_edges_false, test_edges, test_edges_false, labels = load_link(
            args.dataset)
        features_mat = features.toarray()
        attr_labels_list, dim_attr, features_rm_privacy = get_attr_list(args.dataset, labels, features_mat)
        features_rm_privacy = torch.LongTensor(features_rm_privacy)
        if args.dataset == 'yale':
            attr_index = 5
            n_class = 6
        else:
            attr_index = 1
            n_class = 2

        privacy_y = labels[:, attr_index] - 1
        filter_id = np.where(privacy_y >= 0)
        privacy_y = torch.LongTensor(privacy_y).to(device)

        g = dgl.from_scipy(adj_train).to(device)
        idx = np.arange(features.shape[0])
    else:
        if args.dataset == 'credit':   # 30000*14
            # Age? or Married  # primary: 0.7788 test: 0.8056  privacy: 0.545  test: 0.6818
            dataset = 'credit'
            sens_attr = 'Married'
            predict_attr = 'NoDefaultNextMonth'
            path = "./dataset/credit/"
            g, features, labels, privacy_y = load_credit(dataset, sens_attr, predict_attr, path=path)  # label: 0 1
            sens_attr_index = 0
            idx = np.arange(features.shape[0])
            n_class = 2

        else:
            raise Exception
        process_link_dataset(g)
        features_rm_privacy = torch.hstack([features[:, :sens_attr_index], features[:, sens_attr_index+1:]])
        filter_id = np.arange(features.shape[0])
        privacy_y = privacy_y.to(device)
        adj_train = np.load('./dataset/{}_train_edges.npy'.format(args.dataset))
        val_edges = np.load('./dataset/{}_val_edges.npy'.format(args.dataset)).transpose()
        val_edges_false = np.load('./dataset/{}_val_edges_false.npy'.format(args.dataset)).transpose()
        test_edges = np.load('./dataset/{}_test_edges.npy'.format(args.dataset)).transpose()
        test_edges_false = np.load('./dataset/{}_test_edges_false.npy'.format(args.dataset)).transpose()
        g = dgl.graph((adj_train[0], adj_train[1]), num_nodes=g.number_of_nodes())
        # g = g.add_self_loop()
        g = dgl.add_reverse_edges(g).to(device)
        features_rm_privacy = feature_norm(features_rm_privacy)

    def print_info():
        train_acc = get_acc(logits.cpu(), adj.cpu())
        val_roc, val_ap = get_scores_edge(val_edges, val_edges_false, logits.cpu())
        print(
            "Epoch:",
            "%04d" % (epoch + 1),
            "train_loss=",
            "{:.5f}".format(loss.item()),
            "train_acc=",
            "{:.5f}".format(train_acc),
            "val_roc=",
            "{:.5f}".format(val_roc),
            "val_ap=",
            "{:.5f}".format(val_ap),
        )
    adj = g.adjacency_matrix().to_dense().to(device)
    features = features_rm_privacy.to(device)
    model = APGE(in_dim=features.shape[1], hidden1_dim=args.hidden1, hidden2_dim=args.hidden2,
                 emb_dim=32, attr_dim=n_class, device=device).to(device)

    # adj = adj_train

    weight_tensor, norm = compute_loss_para(adj, device)

    optimizer_att = torch.optim.Adam(model.pri_linear.parameters(), lr=args.lr)
    optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=args.lr)
    optimizer_dis = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr)
    optimizer_o = torch.optim.Adam(itertools.chain(model.encoder.parameters(), model.extend_linear.parameters()), lr=args.lr)
    for epoch in range(args.epoch):
        model.train()
        for p_1 in range(5):
            loss_att = model.compute_privacy_loss(g, features, privacy_y, filter_id)
            optimizer_att.zero_grad()
            loss_att.backward()
            optimizer_att.step()
        loss = model.compute_reconstruction_loss(g, features, adj, weight_tensor, privacy_y, filter_id, norm)
        optimizer_o.zero_grad()
        loss.backward()
        optimizer_o.step()
        for p_2 in range(1):
            loss_dis = model.compute_dis_loss(g, features)
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()
        loss_dis = -model.compute_dis_loss(g, features)
        optimizer_encoder.zero_grad()
        loss_dis.backward()
        optimizer_encoder.step()
        emb, logits = model(g, features)

        if (epoch+1) % 10 == 0:
            print("loss attacker: {}, loss dis: {}".format(loss_att.detach(), -loss_dis.detach()))
            print_info()

    test_roc, test_ap = get_scores_edge(test_edges, test_edges_false, logits)
    np.save('./checkpoint/{}_{}_emb.npy'.format(args.dataset, args.model), emb.detach().cpu().numpy())
    p0_mlp, p0_f1, p2_mlp, p2_mlp_f1, p2_svm, p2_svm_f1 = get_score_attr(args.dataset, emb.detach().cpu().numpy(), [labels, privacy_y.cpu()], [idx, filter_id])
    print('test roc:' + str(test_roc) + '\n')
    print('test ap:' + str(test_ap) + '\n')
    print('Utility Attr MLP ACC: ' + str(p0_mlp)+'\n')
    print('Utility Attr MLP F1: ' + str(p0_f1)+'\n')
    print('Privacy MLP ACC: ' + str(p2_mlp)+'\n')
    print('Privacy MLP F1: ' + str(p2_mlp_f1)+'\n')
    print('Privacy SVM ACC: ' + str(p2_svm)+'\n')
    print('Privacy SVM F1: ' + str(p2_svm_f1)+'\n')


def main_use_pretrain():

    if args.dataset == 'yale' or args.dataset == 'rochester':
        adj, features, adj_train, val_edges, val_edges_false, test_edges, test_edges_false, labels = load_link(
            args.dataset)
        if args.dataset == 'yale':
            attr_index = 5
        else:
            attr_index = 1
        privacy_y = labels[:, attr_index] - 1
        filter_id = np.where(privacy_y >= 0)
        privacy_y = torch.LongTensor(privacy_y)

        idx = np.arange(features.shape[0])
    else:
        if args.dataset == 'credit':  # 30000*14
            # Age? or Married  # primary: 0.7788 test: 0.8056  privacy: 0.545  test: 0.6818
            dataset = 'credit'
            sens_attr = 'Married'
            predict_attr = 'NoDefaultNextMonth'
            path = "./dataset/credit/"
            g, features, labels, privacy_y = load_credit(dataset, sens_attr, predict_attr, path=path)  # label: 0 1
            idx = np.arange(features.shape[0])
        else:
            raise Exception
        filter_id = np.arange(features.shape[0])
        test_edges = np.load('./dataset/{}_test_edges.npy'.format(args.dataset)).transpose()
        test_edges_false = np.load('./dataset/{}_test_edges_false.npy'.format(args.dataset)).transpose()

    emb = np.load('./checkpoint/{}_{}_emb.npy'.format(args.dataset, args.model))

    # emb += np.random.normal(0, 0.5, emb.shape)

    logits = torch.tensor(np.matmul(emb, emb.transpose()))
    test_roc, test_ap = get_scores_edge(test_edges, test_edges_false, logits)
    p0_mlp, p0_f1, p2_mlp, p2_mlp_f1, p2_svm, p2_svm_f1 = get_score_attr(args.dataset, emb, [labels, privacy_y], [idx, filter_id])
    print('test roc:' + str(test_roc) + '\n')
    print('test ap:' + str(test_ap) + '\n')
    print('Utility Attr MLP ACC: ' + str(p0_mlp)+'\n')
    print('Utility Attr MLP F1: ' + str(p0_f1)+'\n')
    print('Privacy MLP ACC: ' + str(p2_mlp)+'\n')
    print('Privacy MLP F1: ' + str(p2_mlp_f1)+'\n')
    print('Privacy SVM ACC: ' + str(p2_svm)+'\n')
    print('Privacy SVM F1: ' + str(p2_svm_f1)+'\n')


if __name__ == '__main__':
    if args.use_pretrain:
        main_use_pretrain()
    else:
        main()
