import torch_geometric as pyg
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
import pickle
import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
np.random.seed(42)
g = pickle.load(open("GeneratedDataset_interm_graph/Graph_id_theft_2_with_anomalies.pkl", "rb"))
pyg_data = pyg.utils.from_networkx(g)

plt.figure(figsize=(7, 7))
for run in np.arange(10): 

    x = F.one_hot(pyg_data.node_attr, num_classes=3).float()  # Shape: [5000, 3]
    pyg_data.x = x
    row, col = pyg_data.edge_index  

    # Aggregate features of neighbors for each node
    x_agg = scatter_mean(pyg_data.x[col], row, dim=0, dim_size=pyg_data.num_nodes)
    node_attr_embedding = torch.cat([pyg_data.x, x_agg], dim=1)  

    #-------------- node classification --------------
    true_labels = pyg_data.anomalous.detach().numpy()

    node_indices = np.arange(len(true_labels))
    np.random.shuffle(node_indices)

    # Split once after shuffling
    train_idx = node_indices[:int(0.8 * pyg_data.num_nodes)]
    test_idx = node_indices[int(0.8 * pyg_data.num_nodes):]

    iso_forest_train = node_attr_embedding[train_idx].detach().numpy()
    iso_forest_test = node_attr_embedding[test_idx].detach().numpy()
    true_labels_test = true_labels[test_idx].tolist()

    iso_forest = IsolationForest(n_estimators=100, contamination=0.2, max_samples=256, random_state=42).fit(iso_forest_train)
    normality_score = iso_forest.decision_function(iso_forest_test) # lower and more abnormal, higher is normal
    abnormality_score = 1-normality_score
    abnormality_score_normed = MinMaxScaler().fit_transform(abnormality_score.reshape(-1, 1)).flatten()

    precision, recall, thresholds = precision_recall_curve(true_labels_test, abnormality_score_normed)
    auc_precision_recall = auc(recall, precision)
    plt.plot(recall, precision, label = f"Iso AUC = {round(auc_precision_recall, 3)}, run = {run}")

plt.hlines(true_labels_test.count(1)/len(true_labels_test), xmin=0, xmax=1, label = "Baseline curve", color='red')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Node anomaly Precision - Recall curve using Iso Forest")
plt.legend()
plt.show()

"""
tn, fp, fn, tp = confusion_matrix(true_labels_shuffled[int(0 * pyg_data.num_nodes):], result).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tn + fp + fn + tp)
print('\033[1;36m'+f"We have tn {tn}, fp {fp}, fn {fn}, tp {tp}")
print("Precision : ", precision)
print("Recall : ", recall)
print("Accuracy : ", accuracy)
print("F1 score : ", (2*tp) / (2*tp + fp + fn), "\033[0m")
"""