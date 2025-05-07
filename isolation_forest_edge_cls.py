import torch_geometric as pyg
import torch
import pickle
import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

torch.manual_seed(42)
g = pickle.load(open("GeneratedDataset_interm_graph/Graph_edge_cls_with_anomalies.pkl", "rb"))
pyg_data = pyg.utils.from_networkx(g)

n2v = pyg.nn.Node2Vec(pyg_data.edge_index, embedding_dim = 1, walk_length = 15, context_size = 5, walks_per_node = 10)
node_embedding = n2v.forward(torch.arange(0, pyg_data.num_nodes))

node_attr_embedding = torch.concat((torch.nn.functional.one_hot(pyg_data.node_attr), node_embedding), dim = 1) # shape of 5000 by 131

#-------------- EDGE classification --------------

true_labels = pyg_data.anomalous.detach().numpy()

edge_embedding = torch.zeros((pyg_data.anomalous.shape[0], 2*node_attr_embedding.shape[1]))
for c, (i, j) in enumerate(pyg_data.edge_index.T): 
    edge_embedding[c] = torch.cat([node_attr_embedding[i], node_attr_embedding[j]])

iso_training = edge_embedding.detach().numpy()[:int(0.3 * pyg_data.num_edges)]
iso_testing = edge_embedding.detach().numpy()[int(0.3 * pyg_data.num_edges):]
test_true_labels = true_labels[int(0.3 * pyg_data.num_edges):]

iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42).fit(iso_training)
result = iso_forest.predict(iso_testing) # 1 for inliers, -1 for outliers.
result = np.where(result == 1, 0, result)
result = np.where(result == -1, 1, result)
result = result.tolist()

tn, fp, fn, tp = confusion_matrix(test_true_labels, result).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tn + fp + fn + tp)
print('\033[1;36m'+f"We have tn {tn}, fp {fp}, fn {fn}, tp {tp}")
print("Precision : ", precision)
print("Recall : ", recall)
print("Accuracy : ", accuracy)
print("F1 score : ", (2*tp) / (2*tp + fp + fn), "\033[0m")