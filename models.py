import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch.conv import GATConv, GraphConv, SAGEConv
from utils import accuracy_softmax, compute_metric, accuracy_binary
from copy import deepcopy
import dgl.function as fn
import math


class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, use_feature=True, n_node=None, n_emb=None, act=F.relu):
        super(GCN, self).__init__()
        if not use_feature:
            self.emb = nn.Embedding(n_node, n_emb)
            self.gc1 = GraphConv(n_emb, n_hid, activation=act)
        else:
            self.gc1 = GraphConv(n_feat, n_hid, activation=act)
        self.gc2 = GraphConv(n_hid, n_class)
        self.dropout = nn.Dropout(dropout)
        self.use_feature = use_feature

    def forward(self, g, x, noise=None, edge_weight=None):
        if not self.use_feature:
            x = self.emb(g.nodes())
        x = self.gc1(g, x, edge_weight=edge_weight)
        x = self.dropout(x)
        x = self.gc2(g, x, edge_weight=edge_weight)
        if noise is None:
            return x
        else:
            return x + noise

    def get_representation(self, g, x):
        return self.forward(g, x)


class GATBody(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, heads, feat_drop, att_drop, negative_slope, residual):
        super(GATBody, self).__init__()
        self.num_layers = num_layers
        self.get_layers = nn.ModuleList()
        self.activation = F.elu
        self.get_layers.append(GATConv(
            in_dim, num_hidden, heads[0], feat_drop, att_drop, negative_slope, False, self.activation))
        for i in range(1, num_layers):
            self.get_layers.append(GATConv(
                num_hidden * heads[i - 1], num_hidden, heads[i], feat_drop,
                att_drop, negative_slope, residual, self.activation))
        self.get_layers.append(GATConv(
            num_hidden * heads[-1], num_hidden, heads[-1], feat_drop, att_drop, negative_slope, residual, None
        ))

    def forward(self, g, inputs):
        h = inputs
        for i in range(self.num_layers):
            h = self.get_layers[i](g, h).flatten(1)
        logits = self.get_layers[-1](g, h).mean(1)

        return logits


class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, num_classes,
                 heads, feat_drop, att_drop, negative_slope, residual):
        super(GAT, self).__init__()
        self.body = GATBody(num_layers, in_dim, num_hidden, heads, feat_drop, att_drop, negative_slope, residual)
        # self.fc = nn.Linear(num_hidden, num_classes)

    def forward(self, g, inputs):
        logits = self.body(g, inputs)
        # logits = self.fc(logits)
        return logits


class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, device):
        super(VGAEModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        self.layer_1 = GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True)
        self.layer_2 = GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)
        self.layer_3 = GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)
        self.utility_part = nn.ModuleList([self.layer_1, self.layer_2, self.layer_3])
        self.device = device

    def encoder(self, g, features):
        h = self.layer_1(g, features)
        self.mean = self.layer_2(g, h)
        self.log_std = self.layer_3(g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(self.device)
        sampled_z = self.mean + gaussian_noise*torch.exp(self.log_std).to(self.device)
        return sampled_z

    @staticmethod
    def decoder(z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec

    def compute_loss(self, g, features, adj, norm, weight_tensor):
        z = self.encoder(g, features)
        logits = self.decoder(z)
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / logits.size(0) * (1+2*self.log_std-self.mean**2-torch.exp(self.log_std)**2).sum(1).mean()
        loss -= kl_divergence

        return loss, logits, z

    def initial_model(self, g, features):
        self.forward(g, features)


class GAEModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, device):
        super(GAEModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        self.layer_1 = GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True)
        self.layer_2 = GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)
        # self.layer_3 = GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)
        self.utility_part = nn.ModuleList([self.layer_1, self.layer_2])
        self.device = device

    def encoder(self, g, features):
        h = self.layer_1(g, features)
        z = self.layer_2(g, h)
        return z

    @staticmethod
    def decoder(z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec

    def compute_loss(self, g, features, adj, norm, weight_tensor):
        z = self.encoder(g, features)
        logits = self.decoder(z)
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)

        return loss, logits, z


class VGAEPrivacyModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, n_class, device, beta):
        super(VGAEPrivacyModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        self.layer_1 = GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True)
        self.layer_2 = GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)
        self.layer_3 = GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)
        # privacy part
        self.privacy_layer_1 = GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True)
        self.privacy_layer_2 = GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x,
                                         allow_zero_in_degree=True)
        self.privacy_layer_3 = GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x,
                                         allow_zero_in_degree=True)
        self.privacy_linear = nn.Linear(self.hidden2_dim, n_class)
        # self.privacy_part = nn.ModuleList([self.privacy_layer_1, self.privacy_layer_2, self.privacy_layer_3, self.privacy_linear])
        self.privacy_part = nn.ModuleList([self.privacy_layer_2, self.privacy_layer_3, self.privacy_linear])
        self.utility_part = nn.ModuleList([self.layer_1, self.layer_2, self.layer_3])
        self.device = device
        self.beta = beta

    def encoder(self, g, features):
        h = self.layer_1(g, features)
        self.mean = self.layer_2(g, h)
        self.log_std = self.layer_3(g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(self.device)
        sampled_z = self.mean + gaussian_noise*torch.exp(self.log_std)
        return sampled_z

    @staticmethod
    def decoder(z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec

    def compute_loss(self, g, features, adj, norm, weight_tensor):
        z = self.encoder(g, features)
        logits = self.decoder(z)
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / logits.size(0) * (
                    1 + 2 * self.log_std - self.mean ** 2 - torch.exp(self.log_std) ** 2).sum(1).mean()

        mean_new = (self.mean + self.privacy_mean.detach()) / math.sqrt(2)
        std_new = torch.sqrt((torch.exp(self.privacy_log_std.detach()) ** 2 + torch.exp(self.log_std) ** 2)/2)

        kl_privacy_divergence = 0.5 / logits.size(0) * (
            1 + torch.log(std_new) - mean_new ** 2 - std_new ** 2).sum(1).mean()
        loss -= kl_divergence
        loss -= self.beta * kl_privacy_divergence
        self.kl_privacy_divergence = kl_privacy_divergence

        return loss, logits, z

    def privacy_compute_loss(self, g, features, labels, train_id):
        # h = self.privacy_layer_1(g, features)
        h = self.layer_1(g, features)
        self.privacy_mean = self.privacy_layer_2(g, h)
        self.privacy_log_std = self.privacy_layer_2(g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(self.device)
        sampled_z = self.privacy_mean + gaussian_noise*torch.exp(self.privacy_log_std).to(self.device)
        preds = self.privacy_linear(sampled_z)
        loss_privacy = F.cross_entropy(preds[train_id], labels[train_id])
        kl_divergence = 0.5 / features.size(0) * (
                    1 + 2 * self.privacy_log_std -
                    self.privacy_mean ** 2 - torch.exp(self.privacy_log_std) ** 2).sum(1).mean()
        # loss_distance = torch.norm(self.mean.detach()-self.privacy_mean, dim=1).mean() +
        # torch.norm(self.privacy_log_std - self.log_std.detach(), dim=1).mean()
        return loss_privacy - kl_divergence

    def initial_model(self, g, features):
        h = self.layer_1(g, features)
        self.mean = self.layer_2(g, h)
        self.log_std = self.layer_3(g, h)
        h = self.privacy_layer_1(g, features)
        self.privacy_mean = self.privacy_layer_2(g, h)
        self.privacy_log_std = self.privacy_layer_2(g, h)


class APGE(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, emb_dim, attr_dim,device):
        super(APGE, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.layer_1 = GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True)
        self.layer_2 = GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)
        self.encoder = nn.ModuleList([self.layer_1, self.layer_2])
        self.extend_linear = nn.Linear(self.hidden2_dim, emb_dim)
        self.pri_linear = nn.Linear(emb_dim, attr_dim)

        self.discriminator = nn.Sequential(
            nn.Linear(hidden2_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.device = device

    def compute_privacy_loss(self, g, features, privacy_label, train_id):
        x = self.layer_1(g, features)
        emb = self.layer_2(g, x)
        emb_long = self.extend_linear(emb)
        privacy_logits = self.pri_linear(emb_long)
        pri_loss = F.cross_entropy(privacy_logits[train_id], privacy_label[train_id])
        return pri_loss

    def compute_reconstruction_loss(self, g, features, adj, weight_tensor, privacy_label, train_id, norm):
        x = self.layer_1(g, features)
        emb = self.layer_2(g, x)
        emb_long = self.extend_linear(emb)
        emb_concat = torch.hstack([privacy_label.view(-1, 1), emb_long])
        logits = torch.sigmoid(torch.matmul(emb_long, emb_long.t()))
        reconstruction_loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
        privacy_logits = self.pri_linear(emb_long)
        pri_loss = F.cross_entropy(privacy_logits[train_id], privacy_label[train_id])
        return reconstruction_loss - 0.5 * pri_loss

    def compute_dis_loss(self, g, features):
        x = self.layer_1(g, features)
        emb = self.layer_2(g, x)
        real_dis = torch.randn(emb.size(0), emb.size(1)).to(self.device)
        inputs = torch.concat([emb, real_dis], dim=0)
        preds = self.discriminator(inputs).squeeze()
        labels = torch.concat([torch.zeros(emb.size(0)), torch.ones((emb.size(0)))], dim=0).to(self.device)
        dis_loss = F.binary_cross_entropy_with_logits(preds, labels)
        return dis_loss

    def forward(self, g, features):
        x = self.layer_1(g, features)
        emb = self.layer_2(g, x)
        emb_long = self.extend_linear(emb)
        logits = torch.sigmoid(torch.matmul(emb_long, emb_long.t()))
        return emb_long, logits


class GAEMI(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, attr_dim,device):
        super(GAEMI, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.layer_1 = GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True)
        self.layer_2 = GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)
        self.encoder = nn.ModuleList([self.layer_1, self.layer_2])
        self.pri_linear = nn.Linear(hidden2_dim, attr_dim)
        self.device = device

    def compute_privacy_loss(self, g, features, privacy_label, train_id):
        x = self.layer_1(g, features)
        emb = self.layer_2(g, x)
        privacy_logits = self.pri_linear(emb)
        pri_loss = F.cross_entropy(privacy_logits[train_id], privacy_label[train_id])
        return pri_loss

    def compute_reconstruction_loss(self, g, features, adj, weight_tensor, privacy_label, train_id, norm):
        x = self.layer_1(g, features)
        emb = self.layer_2(g, x)
        logits = torch.sigmoid(torch.matmul(emb, emb.t()))
        reconstruction_loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
        privacy_logits = self.pri_linear(emb)
        pri_loss = F.cross_entropy(privacy_logits[train_id], privacy_label[train_id])
        return reconstruction_loss - 0.5 * pri_loss

    def forward(self, g, features):
        x = self.layer_1(g, features)
        emb = self.layer_2(g, x)
        logits = torch.sigmoid(torch.matmul(emb, emb.t()))
        return emb, logits


class NodeClassifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(NodeClassifier, self).__init__()
        self.n_class = n_class
        self.fc = nn.Linear(n_hid, n_class)

    def forward(self, emb):
        return self.fc(emb)


class DotLinkPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]


class MLPLinkPredictor(nn.Module):
    def __init__(self, n_feat):
        super(MLPLinkPredictor, self).__init__()
        self.W1 = nn.Linear(n_feat * 2, n_feat)
        self.W2 = nn.Linear(n_feat, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']])
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class NodeGCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, use_feature=True, n_node=None, n_emb=None, act=F.relu):
        super(NodeGCN, self).__init__()
        self.gnn = GCN(n_feat, n_hid, n_hid, dropout, use_feature, n_node, n_emb, act)
        self.fc = NodeClassifier(n_hid, n_class)

    def forward(self, g, x, noise=None, edge_weight=None):
        return self.fc(self.gnn(g, x, noise, edge_weight)).squeeze()

    def get_representation(self, g, x, noise=None, edge_weight=None):
        return self.gnn(g, x, noise, edge_weight)


class AttributeClassifier:
    def __init__(self, model, lr, weight_decay):
        super(AttributeClassifier, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def fit(self, feature, g, labels, idx_train, idx_val=None, train_iters=0, verbose=False):

        best_loss_val = np.inf
        best_acc_val = 0
        if self.model.out_dim == 1:
            criterion = F.binary_cross_entropy_with_logits
            accuracy = accuracy_binary
        else:
            criterion = F.cross_entropy
            accuracy = accuracy_softmax

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_i = 0
        for i in range(train_iters):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(g, feature).squeeze()
            loss_train = criterion(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if idx_val is None:
                continue

            self.model.eval()
            output = self.model(g, feature).squeeze()
            loss_val = criterion(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                weights = deepcopy(self.model.state_dict())
                best_i = i

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                weights = deepcopy(self.model.state_dict())
                best_i = i

            if verbose and i % 10 == 0:
                print('Classifier epoch {}, train loss: {:.4f}, train acc: '
                      '{:.4f}, val loss {:.4f}, val acc: {:.4f}'.format(i, loss_train, acc_train, loss_val, acc_val))

        if idx_val is not None:
            self.model.load_state_dict(weights)
            print('best epoch: {}'.format(best_i))


class Discriminator(nn.Module):
    def __init__(self, n_feat, n_class):
        super(Discriminator, self).__init__()
        self.embed_dim = int(n_feat)
        self.out_dim = n_class
        self.net = nn.Sequential(
            nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            # nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            # nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
            # # nn.BatchNorm1d(num_features=self.embed_dim),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            # nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )

    def forward(self, g, ents_emb):
        output = self.net(ents_emb)
        # loss = self.criterion(output.squeeze(), A_labels)
        return output
