import torch_geometric as pyg
import torch
import pickle
import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

torch.manual_seed(42)
g = pickle.load(open("GeneratedDataset_interm_graph/Whole_Graph_with_anomalies.pkl", "rb"))
pyg_data = pyg.utils.from_networkx(g)

n2v = pyg.nn.Node2Vec(pyg_data.edge_index, embedding_dim = 128, walk_length = 15, context_size = 5, walks_per_node = 10)

node_embedding = n2v.forward(torch.arange(0, pyg_data.num_nodes))
node_attr_embedding = torch.concat((torch.nn.functional.one_hot(pyg_data.node_attr), node_embedding), dim = 1) # shape of 5000 by 131

# subgraphs = pickle.load(open("GeneratedDataset_interm_graph/ListEgoNet_with_anomalies.pkl", "rb"))
subgraphs = pickle.load(open("GeneratedDataset/RelationalDataset_with_anomaly_test", "rb"))
# breakpoint()

subgraph_embedding = torch.zeros((len(subgraphs), 16*node_attr_embedding.shape[1]))
true_labels = []
for c, subgraph in enumerate(subgraphs): 
    # subgraph_embedding[c] = torch.sum(torch.stack([node_attr_embedding[node] for node in subgraph.nodes()]), axis = 0)
    
    # embeddings = torch.stack([node_attr_embedding[node] for node in subgraph.nodes()])
    # sum_part = torch.sum(embeddings[:, :3], dim=0)
    # min_part = torch.min(embeddings[:, 3:], dim=0).values
    # subgraph_embedding[c] = torch.cat([sum_part, min_part], dim=0)
    # breakpoint()

    if  torch.cat([node_attr_embedding[node] for node in subgraph.nodes()]).shape[0] < subgraph_embedding.shape[1]:

        subgraph_embedding[c] = torch.cat( [torch.cat([node_attr_embedding[node] for node in subgraph.nodes()]), 
                                            torch.zeros(subgraph_embedding.shape[1] - torch.cat([node_attr_embedding[node] for node in subgraph.nodes()]).shape[0])]) 
    else:
        subgraph_embedding[c] = torch.cat([node_attr_embedding[node] for node in subgraph.nodes()])


    true_labels.append(int(any(v == 1 for v in nx.get_edge_attributes(subgraph, "anomalous").values())))

iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
result = iso_forest.fit_predict(subgraph_embedding.detach().numpy()) # 1 for inliers, -1 for outliers.
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