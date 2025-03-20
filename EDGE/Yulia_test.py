import networkx as nx
import pickle as pkl
import matplotlib.pyplot as plt
import torch_geometric as pyg
import torch
import numpy as np
# from DataGeneration import *
from datasets.data_utils import preprocess

# m = torch.distributions.categorical.Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
# print(m.sample((4,)))

# print(torch.backends.mps.is_available()) 
# print(np.__version__)

# row = torch.tensor([0, 1, 0, 2, 0, 0])
# print(pyg.utils.degree(row))

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

# nx_graphs_theirs = pkl.load(open(f'EDGE/graphs/Ego.pkl','rb')) # -> this is a list of networkx graph object 
# print(len(nx_graphs_theirs))
# for n in nx_graphs_theirs[:3]:
#     print(n)
#     print(n.nodes[0])

# # Draw the graph
# pos = nx.arf_layout(nx_graphs_theirs[0])
# nx.draw(nx_graphs_theirs[0], pos, with_labels=True, node_size=500)
# plt.title("Generater Mini Graph")
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------
"""
mapping = {0: 'blue', 1: 'orange',2: 'grey'} # the reason why i have this and not one-hot-encoming is
map_color = lambda color: ([mapping[c] for c in color] if isinstance(color, list) else mapping[color])

train_graph = pkl.load(open(f'../GeneratedDataset/Small_test_no_anomaly', 'rb'))
eval_graph = train_graph[int(len(train_graph)*0.9):]
train_graph = train_graph[:int(len(train_graph)*0.9)]

print("The number of graphs in the training is ", len(train_graph))
print("The number of graphs in the eval is ", len(eval_graph))

number_of_nodes = [len(n.nodes) for n in train_graph]
plt.hist(number_of_nodes, bins=50, alpha = 0.5, label = "train")

number_of_nodes = [len(n.nodes) for n in eval_graph]
plt.hist(number_of_nodes, bins=50, alpha = 0.5, label = "eval")

test_graphs = pkl.load(open(f'../GeneratedDataset/Small_test_no_anomaly_test_graphs', 'rb'))
print("The number of graphs in the testing is ", len(test_graphs))

number_of_nodes = [len(n.nodes) for n in test_graphs]
plt.hist(number_of_nodes, bins=50, alpha = 0.5, label = "test")
plt.legend()
plt.xlabel("Number of nodes")
plt.ylabel("Number of graphs")
plt.show()
"""
# ------------------------------------------------------------------------------------------------------------------------

run = "./wandb/Small_test_no_anomaly/multinomial_diffusion/multistep/2025-03-19_19-46-30"
training_loss = pkl.load(open(run+"/metrics_train.pickle", "rb"))
eval_loss = pkl.load(open(run+"/metrics_eval.pickle", "rb"))

plt.plot(np.arange(1, 261), training_loss['bpd'], label = "training BPD")
plt.plot(np.arange(1, 261, 5), eval_loss['bpd'], label = "evaluation BPD")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print(np.argmin(training_loss['bpd']))
print(np.argmin(eval_loss['bpd'])*5)
# breakpoint()
# for n in train_graph[:5]:
#     node_colors = map_color([n.nodes[node]['node_attr'] for node in n.nodes()])
#     pos = nx.arf_layout(n)
#     nx.draw(n, pos, with_labels=True, node_color=node_colors)
#     plt.title("Training Graph")
#     plt.show()

# for n in test_graphs[:5]:
#     node_colors = map_color([n.nodes[node]['node_attr'] for node in n.nodes()])
#     pos = nx.arf_layout(n)
#     nx.draw(n, pos, with_labels=True, node_color=node_colors)
#     plt.title("Testing Graph")
#     plt.show()

# num_nodes = []
# for n in nx_graphs:
#     num_nodes.append(len(n.nodes()))

# print(num_nodes)

    # print(n.nodes[:3])
    # print(n.edges[:3])
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



"""Here i am starting the alligator

test_graphs, central_nodes = Ego_Net_Generation(None, 1, ego_file, node_file, edge_file, False)

for graph, target in zip(test_graphs, central_nodes):
    # Step 1 is remove all the edges from the graph:
    empty_graph = graph.copy()
    empty_graph.remove_edges_from(empty_graph.edges())

    # step 2 is reconstruct the graph edges
    reconstructed = model(empty_graph)

    # do a similarity measure / check how different the graphs are
    anomaly = graph - reconstructed

    # check if you were correct
    if anomaly > 0.5:
        if target.anomaly == 1 : 
            TP +=1 
        else:
            FP +=1
    else:
        if target.anomaly == 1 : 
            FN +=1 
        else:
            TN +=1

"""

# summ = np.sum([ 1.,  2.,  1.,  2.,  1.,  1., 22.,  1.,  1.,  2.,  2.,  2.,  2.,  1.,
#          2.,  1.,  1.,  1.,  1.,  2.,  1.,  1.,  1.,  1.,  1.,  3.,  8.,  1.,
#          2.,  3.,  1.,  2.,  1.,  2.,  3.,  2.,  1.,  1.,  1.,  1.,  1.,  2.,
#          2.,  1.,  3.,  2.,  1.,  2.,  1.,  1., 29.,  1.,  2.,  1.,  1.,  1.,
#          1.,  2.,  2.])
# print(summ)

