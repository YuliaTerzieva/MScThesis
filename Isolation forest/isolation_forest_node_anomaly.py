import torch_geometric as pyg
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter
import pickle
import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
np.random.seed(42)
training = pickle.load(open("GeneratedDataset_interm_graph/Synthetic_K15_node_train", "rb"))
training_data = pyg.utils.from_networkx(training)
training_data.x = training_data.node_attr
row, col = training_data.edge_index  

# Aggregate features of neighbors for each node
reduction_function = "mul"
x_agg = scatter(training_data.x[col], row, dim=0, dim_size=training_data.num_nodes, reduce=reduction_function)
node_attr_embedding = torch.cat([training_data.x, x_agg], dim=1) 

all_precision = []
n_interp_points = 500
interp_recall = np.linspace(0, 1, n_interp_points)

avg_precision = []
for run in np.arange(10): 

    # Training the isolation forest 
    iso_forest = IsolationForest(n_estimators=100, contamination=0.04, max_samples=256).fit(node_attr_embedding)
    
    # Preparing the testing
    testing = pickle.load(open("GeneratedDataset_interm_graph/Synthetic_K15_node_test", "rb"))
    testing_data = pyg.utils.from_networkx(testing)
    testing_data.x = testing_data.node_attr
    row_test, col_test = testing_data.edge_index  

    # Aggregate features of neighbors for each node
    testing_x_agg = scatter(testing_data.x[col_test], row_test, dim=0, dim_size=testing_data.num_nodes, reduce=reduction_function) 
    testing_node_attr_embedding = torch.cat([testing_data.x, testing_x_agg], dim=1) 
    true_labels = testing_data.anomalous.detach().numpy().tolist()


    normality_score = iso_forest.decision_function(testing_node_attr_embedding) # lower and more abnormal, higher is normal
    abnormality_score = 1-normality_score
    abnormality_score_normed = MinMaxScaler().fit_transform(abnormality_score.reshape(-1, 1)).flatten()

    precision, recall, thresholds = precision_recall_curve(true_labels, abnormality_score_normed, drop_intermediate = True)
    auc_precision_recall = auc(recall, precision)
    print(auc_precision_recall)
    plt.plot(recall, precision, ".-")#, label = f"Iso AUC = {round(auc_precision_recall, 3)}, run = {run})
    
    precision_interp = np.interp(interp_recall, recall[::-1], precision[::-1], left=1.0)
    all_precision.append(precision_interp)

    avg_precision.append(average_precision_score(true_labels, abnormality_score_normed))
    
# plt.hlines(true_labels.count(1)/len(true_labels), xmin=0, xmax=1, label = "Baseline curve", color='red')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title(f"Node anomaly Precision - Recall curve using Iso Forest")
# plt.legend()
# plt.show()


plt.figure(figsize=(5, 5))
scores = abnormality_score_normed
labels = true_labels
scores = np.array(scores)
labels = np.array(labels)
plt.hist([scores[labels==0], scores[labels==1]], bins = 50, color=["#D6A5F1", "#4A21A8"], label=['Normal', 'Anomalous'])
plt.title("Node anomaly score distribution\n Isolation Forest")
plt.xticks(np.arange(min(scores), max(scores), 0.1))
plt.xlabel("Anomaly score")
plt.ylabel("Number of nodes")
plt.yscale('log')
plt.legend()
plt.savefig(f"Plots_Isolation_Forest_NAD/{reduction_function}-NAD-ISOFOREST-score-dist")
plt.show()


# Convert to array and compute mean/std
all_precision = np.vstack(all_precision)
mean_precision = all_precision.mean(axis=0)
std_precision = all_precision.std(axis=0)

# Plot mean curve with shaded std area
plt.figure(figsize=(5, 5))
plt.plot(interp_recall, mean_precision, label=f"Mean PR curve (AP = {np.mean(avg_precision):.5f})", color = "#4A21A8")
plt.fill_between(interp_recall, mean_precision - std_precision, mean_precision + std_precision, alpha=0.5, color = "#7DC3F6")

# Baseline (random)
positive_ratio = true_labels.count(1) / len(true_labels)
plt.hlines(positive_ratio, xmin=0, xmax=1, color="#F1993A", label='Baseline')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Node anomaly on Synthetic dataset \nMean Precision-Recall Curve with Std (IsolationForest)")
plt.legend()
plt.ylim([0, 1.05])
plt.savefig(f"Plots_Isolation_Forest_NAD/{reduction_function}-NAD-ISOFOREST-AUPRC")
plt.show()


