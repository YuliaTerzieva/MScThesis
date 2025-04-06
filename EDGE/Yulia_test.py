import networkx as nx
import pickle as pkl
import matplotlib.pyplot as plt
import torch_geometric as pyg
import torch
import numpy as np
from datasets.data_utils import preprocess

# for the bash colors check : https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux

def plot_graph_freq_wrt_node_edge(dataset_name) -> None:
    mapping = {0: 'blue', 1: 'orange',2: 'grey'} # the reason why i have this and not one-hot-encoming is
    map_color = lambda color: ([mapping[c] for c in color] if isinstance(color, list) else mapping[color])

    train_graph = pkl.load(open(f'../GeneratedDataset/{dataset_name}', 'rb'))
    eval_graph = train_graph[int(len(train_graph)*0.9):]
    train_graph = train_graph[:int(len(train_graph)*0.9)]

    print('\033[95m' + "The number of graphs in the training is ", len(train_graph))
    print("The number of graphs in the eval is ", len(eval_graph))

    train_number_of_nodes = [len(n.nodes) for n in train_graph]
    plt.hist(train_number_of_nodes, alpha = 0.5, label = "train")

    eval_number_of_nodes = [len(n.nodes) for n in eval_graph]
    plt.hist(eval_number_of_nodes, alpha = 0.5, label = "eval")

    test_graphs = pkl.load(open(f'../GeneratedDataset/{dataset_name}_test_graphs', 'rb'))
    print("The number of graphs in the testing is ", len(test_graphs))
    print('\033[1;30m')

    test_number_of_nodes = [len(n.nodes) for n in test_graphs]
    plt.hist(test_number_of_nodes, alpha = 0.5, label = "test")
    plt.legend()
    plt.xlabel("Number of nodes")
    plt.ylabel("Number of graphs")
    plt.title(f"Dataset {dataset_name}")
    plt.show()

    # ---- EDGEs now

    train_number_of_edges = [len(n.edges) for n in train_graph]
    plt.hist(train_number_of_edges, alpha = 0.5, label = "train")

    eval_number_of_edges = [len(n.edges) for n in eval_graph]
    plt.hist(eval_number_of_edges, alpha = 0.5, label = "eval")

    test_graphs = pkl.load(open(f'../GeneratedDataset/{dataset_name}_test_graphs', 'rb'))

    test_number_of_edges = [len(n.edges) for n in test_graphs]
    plt.hist(test_number_of_edges, alpha = 0.5, label = "test")
    plt.legend()
    plt.xlabel("Number of edges")
    plt.ylabel("Number of graphs")
    plt.title(f"Dataset {dataset_name}")
    plt.show()

def plot_training_loss(dataset) -> None:
    
    f, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i, run in enumerate(dataset) :

        _ax = ax[int(i//2), int(i%2)]

        training_loss = pkl.load(open(run+"/metrics_train.pickle", "rb"))
        eval_loss = pkl.load(open(run+"/metrics_eval.pickle", "rb"))

        min_train_idx = np.argmin(training_loss['bpd'])
        min_train_bpd = min(training_loss['bpd'])

        min_eval_idx = np.argmin(eval_loss['bpd'])
        min_eval_epoch = min_eval_idx * 10
        min_eval_bpd = eval_loss['bpd'][min_eval_idx]

        
        print('\033[1;36m' + "For Dataset ", run[8:-52], '\033[0;36m')
        print("The epoch with the lowest training bpd is ", min_train_idx)
        print("The epoch with the lowest evaluation bpd is", min_eval_idx*10)
        print("Lowest BPD of the evaluation is: ", round(min_eval_bpd, 4), "\n" + '\033[0;30m')

        _ax.plot(np.arange(1, len(training_loss['bpd'])+1), training_loss['bpd'], label = "training BPD")
        _ax.plot(np.arange(1, len(training_loss['bpd'])+1,10), eval_loss['bpd'], label = "evaluation BPD")
        _ax.set_xlabel("Epoch")

        _ax.scatter(min_train_idx + 1, min_train_bpd, color='blue', label=f'Min Training BPD = {round(min_train_bpd, 3)}', zorder=5)
        _ax.scatter(min_eval_epoch + 1, min_eval_bpd, color='orange', label=f'Min Eval BPD = {round(min_eval_bpd, 3)}', zorder=5)

        _ax.set_title(run[8:-52])
        _ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__': 

    """ # this is test to see what is gimble noise

    uniform = torch.rand_like(torch.zeros(1000))
    print(uniform)
    print(-torch.log(uniform + 1e-30))
    print(-torch.log(-torch.log(uniform + 1e-30) + 1e-30))


    plt.hist(uniform, label = "1", alpha = 0.5, density=True)
    plt.hist(-torch.log(uniform + 1e-30), label = "2", alpha = 0.5,  density=True)
    plt.hist(-torch.log(-torch.log(uniform + 1e-30) + 1e-30), label = "3", alpha = 0.5,  density=True)
    plt.legend()
    plt.show()

    """

    dataset_name_list = ["Basic_test_no_anomaly", "Basic_test_with_anomaly", # 0, 1
                         "Small_test_no_anomaly", "Small_test_with_anomaly", # 2, 3
                         "Mid_test_no_anomaly", "Mid_test_with_anomaly",     # 4, 5
                         "relation_based_test", "RelationalDataset_no_anomaly"]# 6, 7
    dataset_number = 7
    dataset_name = dataset_name_list[dataset_number]
    plot_graph_freq_wrt_node_edge(dataset_name)
    # plot_graph_freq_wrt_node_edge("Ego.pkl")


    # Friday the 21st
    Small_test_no_anomaly = "./wandb/Small_test_no_anomaly/multinomial_diffusion/multistep/2025-03-21_10-10-38"
    Mid_test_no_anomaly = "./wandb/Mid_test_no_anomaly/multinomial_diffusion/multistep/2025-03-21_10-12-13"

    # Friday the 21st with Anomalies
    Small_test_with_anomaly = "./wandb/Small_test_with_anomaly/multinomial_diffusion/multistep/2025-03-21_22-24-50"
    Mid_test_with_anomaly = "./wandb/Mid_test_with_anomaly/multinomial_diffusion/multistep/2025-03-21_22-25-56"

    # Wednesday 26th type 3 no anomalies
    relational_dataset = "./wandb/relation_based_test/multinomial_diffusion/multistep/2025-03-26_18-36-15"

    # Wednesday 2th of April no anomalies
    dataset_list = [Small_test_no_anomaly, Mid_test_no_anomaly, Small_test_with_anomaly, Mid_test_with_anomaly, relational_dataset]

    first_run = "./wandb/RelationalDataset_no_anomaly/multinomial_diffusion/multistep/2025-04-02_13-18-55"
    second_run = "./wandb/RelationalDataset_no_anomaly/multinomial_diffusion/multistep/2025-04-03_18-07-30"


    plot_training_loss([first_run, second_run])


# ------------------------------------------------------------------------------------------------------------------------
# THE FOLLOWING IS SOME TEST CODE I HAVE TO DELETE 
"""
m = torch.distributions.categorical.Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
print(m.sample((4,)))

print(torch.backends.mps.is_available()) 
print(np.__version__)

row = torch.tensor([0, 1, 0, 2, 0, 0])
print(pyg.utils.degree(row))

D = nx.DiGraph([(0, 1), (1, 2), (2, 3)]) # those are edges from 0 to 1, from 1 to 2, from 2 to 3
din = list(d for n, d in D.in_degree())
dout = list(d for n, d in D.out_degree())
print(D.edges)

print("in degrees")
print(D.in_degree())
print(din)

print("out degrees")
print(D.out_degree())
print(dout)

din.append(1)
dout[0] = 2

print(din, dout)
We now expect an edge from node 0 to a new node, node 3 (this should be 4!).
D = nx.directed_configuration_model([0, 2, 1], [1, 1, 1]) # in, out
print(D.in_degree, D.out_degree)
print(D.nodes)
print(D.edges)

print(D.in_degree)
print(D.out_degree)

nx_graphs_theirs = pkl.load(open(f'EDGE/graphs/Ego.pkl','rb')) # -> this is a list of networkx graph object 
print(len(nx_graphs_theirs))
for n in nx_graphs_theirs[:3]:
    print(n)
    print(n.nodes[0])

# Draw the graph
pos = nx.arf_layout(nx_graphs_theirs[0])
nx.draw(nx_graphs_theirs[0], pos, with_labels=True, node_size=500)
plt.title("Generater Mini Graph")
plt.show()

breakpoint()
for n in train_graph[:5]:
    node_colors = map_color([n.nodes[node]['node_attr'] for node in n.nodes()])
    pos = nx.arf_layout(n)
    nx.draw(n, pos, with_labels=True, node_color=node_colors)
    plt.title("Training Graph")
    plt.show()

for n in test_graphs[:5]:
    node_colors = map_color([n.nodes[node]['node_attr'] for node in n.nodes()])
    pos = nx.arf_layout(n)
    nx.draw(n, pos, with_labels=True, node_color=node_colors)
    plt.title("Testing Graph")
    plt.show()

num_nodes = []
for n in nx_graphs:
    num_nodes.append(len(n.nodes()))

print(num_nodes)

    print(n.nodes[:3])
    print(n.edges[:3])
for nx_graph in nx_graphs[:1]:
    print("NEW ----------------------------------- > ")
    graph = pyg.utils.from_networkx(nx_graph)
    print(graph.num_nodes)
    print(graph.num_edges)
    print(graph.edge_index)
    g = pyg.utils.to_networkx(graph, to_undirected=True)
    nx.draw(g)
    plt.draw()
    plt.show()

edge_index = torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = pyg.data.Data(x=x, edge_index=edge_index)
g = pyg.utils.to_networkx(data, to_undirected=True)
nx.draw(g, with_labels = True)
plt.draw()
plt.show()
"""



