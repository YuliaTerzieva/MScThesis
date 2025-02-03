import networkx as nx
import pickle as pkl
import matplotlib.pyplot as plt
import torch_geometric as pyg
import torch

print(torch.backends.mps.is_available()) 

# D = nx.DiGraph([(0, 1), (1, 2), (2, 3)]) # those are edges from 0 to 1, from 1 to 2, from 2 to 3
# din = list(d for n, d in D.in_degree())
# dout = list(d for n, d in D.out_degree())
# print(D.edges)

# print("in degrees")
# print(D.in_degree())
# print(din)

# print("out degrees")
# print(D.out_degree())
# print(dout)

# din.append(1)
# dout[0] = 2

# print(din, dout)
# We now expect an edge from node 0 to a new node, node 3 (this should be 4!).
# D = nx.directed_configuration_model([0, 2, 1], [1, 1, 1]) # in, out
# print(D.in_degree, D.out_degree)
# print(D.nodes)
# print(D.edges)

# print(D.in_degree)
# print(D.out_degree)

# nx_graphs = pkl.load(open(f'EDGE/graphs/Ego.pkl','rb')) # -> this is a list of networkx graph object 
nx_graphs = pkl.load(open(f'GeneratedDataset/Ego_Nets_conf3', 'rb'))
for n in nx_graphs:
    print(len(n.edges))
# for nx_graph in nx_graphs[:1]:
#     print("NEW ----------------------------------- > ")
#     graph = pyg.utils.from_networkx(nx_graph)
#     print(graph.num_nodes)
#     print(graph.num_edges)
#     print(graph.edge_index)
#     g = pyg.utils.to_networkx(graph, to_undirected=True)
#     nx.draw(g)
#     plt.draw()
#     plt.show()

# edge_index = torch.tensor([[0, 1, 1, 2],
#                         [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# data = pyg.data.Data(x=x, edge_index=edge_index)
# g = pyg.utils.to_networkx(data, to_undirected=True)
# nx.draw(g, with_labels = True)
# plt.draw()
# plt.show()



    


