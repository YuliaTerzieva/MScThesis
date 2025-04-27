import torch_geometric as pyg
import torch
import pickle
import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

torch.manual_seed(42)
g = pickle.load(open("GeneratedDataset_interm_graph/Graph_id_theft_2_with_anomalies.pkl", "rb"))
pyg_data = pyg.utils.from_networkx(g)

n2v = pyg.nn.Node2Vec(pyg_data.edge_index, embedding_dim = 1056, walk_length = 15, context_size = 10, walks_per_node = 10)

node_embedding = n2v.forward(torch.arange(0, pyg_data.num_nodes))
node_attr_embedding = torch.concat((torch.nn.functional.one_hot(pyg_data.node_attr), node_embedding), dim = 1) # shape of 5000 by 131

#-------------- EDGE classification --------------
true_labels = pyg_data.anomalous.detach().numpy()

iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
result = iso_forest.fit_predict(node_attr_embedding.detach().numpy()) # 1 for inliers, -1 for outliers.
result = np.where(result == 1, 0, result)
result = np.where(result == -1, 1, result)
result = result.tolist()

tn, fp, fn, tp = confusion_matrix(true_labels, result).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tn + fp + fn + tp)
print('\033[1;36m'+f"We have tn {tn}, fp {fp}, fn {fn}, tp {tp}")
print("Precision : ", precision)
print("Recall : ", recall)
print("Accuracy : ", accuracy, '\033[0;36m')