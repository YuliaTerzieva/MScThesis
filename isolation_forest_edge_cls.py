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
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
g = pickle.load(open("GeneratedDataset_interm_graph/Graph_edge_cls_with_anomalies.pkl", "rb"))
pyg_data = pyg.utils.from_networkx(g)

all_precision = []
n_interp_points = 500
interp_recall = np.linspace(0, 1, n_interp_points)

plt.figure(figsize=(7, 7))
for run in np.arange(10): 

    x = F.one_hot(pyg_data.node_attr, num_classes=3).float()  # Shape: [5000, 3]
    pyg_data.x = x
    row, col = pyg_data.edge_index  

    # Aggregate features of neighbors for each node
    x_agg = scatter_mean(pyg_data.x[col], row, dim=0, dim_size=pyg_data.num_nodes)
    node_attr_embedding = torch.cat([pyg_data.x, x_agg], dim=1)  

    edge_embedding = torch.zeros((pyg_data.anomalous.shape[0], 2*node_attr_embedding.shape[1]))
    for c, (i, j) in enumerate(pyg_data.edge_index.T): 
        edge_embedding[c] = torch.cat([node_attr_embedding[i], node_attr_embedding[j]])

    #-------------- iso forest --------------
    true_labels = pyg_data.anomalous.detach().numpy()

    edge_indices = np.arange(len(true_labels))
    np.random.shuffle(edge_indices)

    # Split once after shuffling
    train_idx = edge_indices[:int(0.8 * pyg_data.num_edges)]
    test_idx = edge_indices[int(0.8 * pyg_data.num_edges):]

    iso_forest_train = edge_embedding[train_idx].detach().numpy()
    iso_forest_test = edge_embedding[test_idx].detach().numpy()
    true_labels_test = true_labels[test_idx].tolist()


    iso_forest = IsolationForest(n_estimators=100, contamination=0.2, max_samples=256, random_state=42).fit(iso_forest_train)
    normality_score = iso_forest.decision_function(iso_forest_test) # lower and more abnormal, higher is normal
    abnormality_score = 1-normality_score
    abnormality_score_normed = MinMaxScaler().fit_transform(abnormality_score.reshape(-1, 1)).flatten()

    precision, recall, thresholds = precision_recall_curve(true_labels_test, abnormality_score_normed)
    auc_precision_recall = auc(recall, precision)
    # plt.plot(recall, precision, label = f"Iso AUC = {round(auc_precision_recall, 3)}, run = {run}")

    precision_interp = np.interp(interp_recall, recall[::-1], precision[::-1], left=1.0)
    
    all_precision.append(precision_interp)

# plt.hlines(true_labels_test.count(1)/len(true_labels_test), xmin=0, xmax=1, label = "Baseline curve", color='red')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title(f"Edge anomaly Precision - Recall curve using Iso Forest")
# plt.legend()
# plt.show()

# Convert to array and compute mean/std
all_precision = np.vstack(all_precision)
mean_precision = all_precision.mean(axis=0)
std_precision = all_precision.std(axis=0)

# Plot mean curve with shaded std area
plt.figure(figsize=(7, 7))
plt.plot(interp_recall, mean_precision, label=f"Mean PR curve (AUC = {auc(interp_recall, mean_precision):.3f})", color = "teal")
plt.fill_between(interp_recall, mean_precision - std_precision, mean_precision + std_precision, alpha=0.3, color = "teal")

# Baseline (random)
positive_ratio = true_labels_test.count(1) / len(true_labels_test)
plt.hlines(positive_ratio, xmin=0, xmax=1, color='red', label='Baseline')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Edge anomaly\nMean Precision-Recall Curve with Std (Isolation Forest)")
plt.legend()
plt.show()


# ---- old ----
