from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import svm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import average_precision_score, roc_auc_score


def get_score_attr(dataset, emb=None, labels=None, filter_id=None):
    # load attribute matrix
    # In different dataset, we set different utility attribute and private attribute.
    # Find the column index corresponding to each attribute.
    if dataset == 'yale' or dataset == 'rochester':
        attr_label = np.load('./dataset/data/{}_labels.npy'.format(dataset))
        if dataset == 'yale':
            attr_index = [0, 5]
        else:
            attr_index = [5, 1]

        utility_label = attr_label[:, attr_index[0]]
        privacy_label = labels[1]
        utility_filter = np.where(utility_label != 0)
        privacy_filter = np.where(privacy_label >= 0)
    else:
        utility_label = labels[0]
        privacy_label = labels[1]
        utility_filter = filter_id[0]
        privacy_filter = filter_id[1]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    k_fold = KFold(5)

    y = utility_label[utility_filter]
    X_emb_feat0 = emb[utility_filter]
    clf = MLPClassifier(max_iter=2000)
    #     prec = cross_val_score(clf, X_emb_feat0, y, cv=k_fold,n_jobs=-1)
    #     p0_mlp = sum(prec)/len(prec)
    predicted = cross_val_predict(clf, X_emb_feat0, y, cv=k_fold, n_jobs=-1)
    p0_mlp = accuracy_score(y, predicted)
    p0_f1 = f1_score(y, predicted, average='macro')

    y = privacy_label[privacy_filter]
    X_emb_feat2 = emb[privacy_filter]

    clf = svm.SVC()
    predicted = cross_val_predict(clf, X_emb_feat2, y, cv=k_fold, n_jobs=-1)
    p2_svm = accuracy_score(y, predicted)
    p2_svm_f1 = f1_score(y, predicted, average='macro')

    clf = MLPClassifier(max_iter=2000)
    predicted = cross_val_predict(clf, X_emb_feat2, y, cv=k_fold, n_jobs=-1)
    p2_mlp = accuracy_score(y, predicted)
    p2_mlp_f1 = f1_score(y, predicted, average='macro')
    # p2_mlp_auc = roc_auc_score(y, predicted)

    return p0_mlp, p0_f1, p2_mlp, p2_mlp_f1, p2_svm, p2_svm_f1


def get_scores_edge(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]].item())

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]].data)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

