import pickle
import networkx as nx




with open("GeneratedDataset_interm_graph/Whole_Graph_with_anomalies.pkl", 'rb') as f:
    G = pickle.load(f)

# nx.write_edgelist(G, "Whole_Graph_with_anomalies_edgelist", data = False)
# nx.write_edgelist(G, "Whole_Graph_with_anomalies_edge_anomaly_label", data = True)

# breakpoint()

# with open("Whole_Graph_with_anomalies_node_class.txt", 'w') as f:
#     for node, n_class in nx.get_node_attributes(G, "node_attr").items():
#         f.write(f"{node} {n_class}\n")