import numpy as np
import torch
import os
import pandas as pd
import dgl
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ndcg_score, f1_score, accuracy_score, recall_score
import pickle as pkl


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id[:200]:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map = np.array(idx_map)

    return idx_map


def load_credit(dataset, sens_attr, predict_attr, path):
    print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))  # 67796*279
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove("Single")
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)  # non-sensitive
    labels = idx_features_labels[predict_attr].values
    sens = idx_features_labels[sens_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)
    g = dgl.from_scipy(adj)
    return g, features, labels, sens


def load_link(dataset):
    with open('./dataset/{}_feats.pkl'.format(dataset), 'rb') as f1:
        features = pkl.load(f1)

    with open('./dataset/{}_adj.pkl'.format(dataset), 'rb') as f2:
        adj = pkl.load(f2, encoding='latin1')

    val_edges = np.load('./dataset/{}_val_edges.npy'.format(dataset))
    val_edges_false = np.load('./dataset/{}_val_edges_false.npy'.format(dataset))
    test_edges = np.load('./dataset/{}_test_edges.npy'.format(dataset))
    test_edges_false = np.load('./dataset/{}_test_edges_false.npy'.format(dataset))

    labels = np.load('./dataset/{}_labels.npy'.format(dataset))

    with open('./dataset/{}_adj_train.pkl'.format(dataset), 'rb') as handle:
        adj_train = pkl.load(handle)

    return adj, features, adj_train, val_edges, val_edges_false, test_edges, test_edges_false, labels


def get_attr_list(dataset, labels, features_mat):
    ## Remove privacy attributes from feature matrix
    # Bulid attibute labels
    if dataset == 'yale':
        # On Yale, elements in columns 0 - 4 correspond to student/faculty status,
        # elements in columns 5,6 correspond to gender,
        # and elements in  the bottom 6 columns correspond to class year,which is privacy here.

        y = labels[:, 0]
        attr_labels = np.eye(len(np.unique(y)))[y.astype(int) - 1]
        attr_labels = attr_labels[:, 1:]

        y = labels[:, -1]
        privacy_labels = np.eye(len(np.unique(y)))[y.astype(int) - 1]
        privacy_labels = privacy_labels[:, 1:]

        attr_labels_list = [attr_labels, privacy_labels]
        dim_attr = [attr_labels.shape[1], privacy_labels.shape[1]]
        features_rm_privacy = features_mat[:, :-6]

    elif dataset == 'rochester':
        # On Rochester, elements in columns 0 - 5 correspond to student/faculty status,
        # elements in the bottom 19 columns correspond to class year,
        # and elements in  the bottom 6,7 columns correspond to gender,which is privacy here.

        y = labels[:, -1]
        attr_labels = np.eye(len(np.unique(y)))[y.astype(int) - 1]
        attr_labels = attr_labels[:, 1:]

        y = labels[:, 1]
        privacy_labels = np.eye(len(np.unique(y)))[y.astype(int) - 1]
        privacy_labels = privacy_labels[:, 1:]

        attr_labels_list = [attr_labels, privacy_labels]
        dim_attr = [attr_labels.shape[1], privacy_labels.shape[1]]
        features_rm_privacy = np.hstack((features_mat[:, :6], features_mat[:, 8:]))

    else:
        raise Exception

    return attr_labels_list, dim_attr, features_rm_privacy


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return 2 * (features - min_values).div(max_values - min_values) - 1


# def accuracy(output, labels):
#     output = output.squeeze()
#     preds = (output > 0).type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)


def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_binary(output, labels):
    output = torch.sigmoid(output)
    correct = (output > 0.5) == labels
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_train_val_test(idx, val_size, test_size, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # idx = np.arange(features.shape[0])
    train_size = 1 - val_size - test_size

    idx_train_val, idx_test = train_test_split(idx, random_state=None, test_size=test_size)

    idx_train, idx_val = train_test_split(idx_train_val, random_state=None,
                                          test_size=(val_size / (val_size + train_size)))

    return idx_train, idx_val, idx_test


def split_train(idx_train, sensitive_size=0.5):
    idx_sensitive, idx_nosensitive = train_test_split(idx_train, train_size=sensitive_size)
    return idx_sensitive, idx_nosensitive


def data_preprocessing(adj, features, labels, preprocess_adj=False, preprocess_features=False, sparse=False,
                       device=None):
    if preprocess_adj:
        adj = normalize(adj)
    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        adj = torch.FloatTensor(adj.todense())
        features = torch.FloatTensor(np.array(features.todense()))
    if preprocess_features:
        features = feature_norm(features)
    return adj.to(device), features.to(device), labels.to(device)


def process_link_dataset(g):
    g_new = g.remove_self_loop()
    adj = g_new.adj().to_dense().triu().to_sparse()
    u, v = adj.indices()
    eids = np.arange(len(u))
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    valid_size = int(len(eids) * 0.05)
    # train_size = len(eids) - test_size - valid_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    valid_pos_u, valid_pos_v = u[eids[test_size:test_size+valid_size]], u[eids[test_size:test_size+valid_size]]
    train_pos_u, train_pos_v = u[eids[test_size+valid_size:]], v[eids[test_size+valid_size:]]

    u, v = g.edges()
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense()
    adj_neg = np.triu(adj_neg)
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), len(eids))
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    valid_neg_u, valid_neg_v = neg_u[neg_eids[test_size:test_size+valid_size]], neg_v[neg_eids[test_size:test_size+valid_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size+valid_size:]], neg_v[neg_eids[test_size+valid_size:]]


    train_g = dgl.remove_edges(g, eids[:test_size])
    dataset = 'credit'
    np.save('./dataset/{}_train_edges.npy'.format(dataset), np.array([train_pos_u.numpy(), train_pos_v.numpy()]))
    np.save('./dataset/{}_train_edges_false.npy'.format(dataset), np.array([train_neg_u, train_neg_v]))
    np.save('./dataset/{}_val_edges.npy'.format(dataset), np.array([valid_pos_u.numpy(), valid_pos_v.numpy()]))
    np.save('./dataset/{}_val_edges_false.npy'.format(dataset), np.array([valid_neg_u, valid_neg_v]))
    np.save('./dataset/{}_test_edges.npy'.format(dataset), np.array([test_pos_u.numpy(), test_pos_v.numpy()]))
    np.save('./dataset/{}_test_edges_false.npy'.format(dataset), np.array([test_neg_u, test_neg_v]))

    return


def compute_metric(pos_score, neg_score):
    """Compute AUC, NDCG metric for link prediction
    """

    scores = torch.sigmoid(torch.cat([pos_score, neg_score]))  # the probability of positive label
    scores_flip = 1.0 - scores  # the probability of negative label
    y_pred = torch.transpose(torch.stack((scores, scores_flip)), 0, 1)

    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    labels_flip = 1 - labels  # to generate one-hot labels
    y_true = torch.transpose(torch.stack((labels, labels_flip)), 0, 1).int()

    # print(y_true.cpu(), y_pred.cpu())
    auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
    # ndcg = 0
    # ndcg = ndcg_score(np.expand_dims(labels.cpu(), axis=0),
    #                   np.expand_dims(scores.cpu(), axis=0))  # super slow!
    ndcg = 0
    return auc, ndcg


def unravel_index(index, array_shape):
    rows = torch.div(index, array_shape[1], rounding_mode='trunc')
    cols = index % array_shape[1]
    return rows, cols


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    # TODO: maybe using coo format would be better?
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
    """
    device = adj.device
    if sparse:
        # warnings.warn('If you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')
        # but you need to install torch_scatter
        # return normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx


def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)


def compute_alpha(n, sum_log_degrees, d_min):
    try:
        alpha = 1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha = 1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees

    return ll


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.
    """

    # Determine which degrees are to be considered, i.e. >= d_min.
    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees


def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    """ Adopted from https://github.com/danielzuegner/nettack
    """
    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]
    degree_sequence = adjacency_matrix.sum(1)
    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)
    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)
    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)
    return new_ll, new_alpha, new_n, sum_log_degrees_after


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (torch.log(torch.clamp(d_old_in_range, min=1))).sum(1) \
                            + (torch.log(torch.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min

    new_n = n_old - (old_in_range != 0).sum(1) + (new_in_range != 0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n


def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004,
                            undirected=True):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.
    """

    N = int(modified_adjacency.shape[0])
    # original_degree_sequence = get_degree_squence(original_adjacency)
    # current_degree_sequence = get_degree_squence(modified_adjacency)
    original_degree_sequence = original_adjacency.sum(0)
    current_degree_sequence = modified_adjacency.sum(0)

    concat_degree_sequence = torch.cat((current_degree_sequence, original_degree_sequence))

    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence,
                                                                                           d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(
        current_degree_sequence, d_min)

    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(concat_degree_sequence,
                                                                                           d_min)

    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.
    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               modified_adjacency,
                                                                                               d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(n_combined, new_sum_log_degrees_combined, d_min)
    new_ll_combined = compute_log_likelihood(n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold

    if allowed_edges.is_cuda:
        filtered_edges = node_pairs[allowed_edges.cpu().numpy().astype(np.bool)]
    else:
        filtered_edges = node_pairs[allowed_edges.numpy().astype(np.bool)]

    allowed_mask = torch.zeros(modified_adjacency.shape)
    allowed_mask[filtered_edges.T] = 1
    if undirected:
        allowed_mask += allowed_mask.t()
    return allowed_mask, current_ratio


def compute_loss_para(adj, device):
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def main():
    pass


if __name__ == '__main__':
    main()

