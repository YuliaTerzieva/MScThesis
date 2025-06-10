import networkx as nx
import pickle as pkl
import matplotlib.pyplot as plt
import torch_geometric as pyg
import torch
import numpy as np
from datasets.data_utils import preprocess
import pickle

# for the bash colors check : https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux

def plot_graph_freq_wrt_node_edge(dataset_name) -> None:
    mapping = {0: 'blue', 1: 'orange',2: 'grey'} # the reason why i have this and not one-hot-encoming is
    map_color = lambda color: ([mapping[c] for c in color] if isinstance(color, list) else mapping[color])

    train_graph = pkl.load(open(f'../GeneratedDataset/{dataset_name}_train', 'rb'))
    eval_graph = pkl.load(open(f'../GeneratedDataset/{dataset_name}_eval', 'rb'))
    test_graph = pkl.load(open(f'../GeneratedDataset/{dataset_name}_test', 'rb'))

    print('\033[95m' + "The number of graphs in the training is ", len(train_graph))
    print("The number of graphs in the eval is ", len(eval_graph))
    print("The number of graphs in the testing is ", len(test_graph))
    print('\033[1;30m')

    train_number_of_nodes = [len(n.nodes) for n in train_graph]
    plt.hist(train_number_of_nodes, alpha = 0.5, label = "train")
    print("The training number of nodes have mean and std : ", np.mean(train_number_of_nodes), np.std(train_number_of_nodes))


    eval_number_of_nodes = [len(n.nodes) for n in eval_graph]
    plt.hist(eval_number_of_nodes, alpha = 0.5, label = "eval")

    test_number_of_nodes = [len(n.nodes) for n in test_graph]
    plt.hist(test_number_of_nodes, alpha = 0.5, label = "test")
    plt.legend()
    plt.xlabel("Number of nodes")
    plt.ylabel("Number of graphs")
    plt.title(f"Dataset {dataset_name}")
    plt.show()

    # ---- EDGEs now

    train_number_of_edges = [len(n.edges) for n in train_graph]
    plt.hist(train_number_of_edges, alpha = 0.5, label = "train")
    print("The training number of eges have mean and std : ", np.mean(train_number_of_edges), np.std(train_number_of_edges))

    eval_number_of_edges = [len(n.edges) for n in eval_graph]
    plt.hist(eval_number_of_edges, alpha = 0.5, label = "eval")

    test_number_of_edges = [len(n.edges) for n in test_graph]
    plt.hist(test_number_of_edges, alpha = 0.5, label = "test")
    plt.legend()
    plt.xlabel("Number of edges")
    plt.ylabel("Number of graphs")
    plt.title(f"Dataset {dataset_name}")
    plt.show()

def plot_training_loss(dataset, eval_every_int) -> None:
    
    f, ax = plt.subplots(2, 2, figsize=(10, 8))
    for i, run in enumerate(dataset) :

        _ax = ax[int(i//2), int(i%2)]

        training_loss = pkl.load(open(run+"/metrics_train.pickle", "rb"))
        eval_loss = pkl.load(open(run+"/metrics_eval.pickle", "rb"))

        min_train_idx = np.argmin(training_loss['bpd'])
        min_train_bpd = min(training_loss['bpd'])

        min_eval_idx = np.argmin(eval_loss['bpd'])
        min_eval_epoch = min_eval_idx * eval_every_int
        min_eval_bpd = eval_loss['bpd'][min_eval_idx]

        
        print('\033[1;36m' + "For Dataset ", run[8:-52], '\033[0;36m')
        print("The epoch with the lowest training bpd is ", min_train_idx)
        print("The epoch with the lowest evaluation bpd is", min_eval_idx*eval_every_int)
        print("Lowest BPD of the evaluation is: ", round(min_eval_bpd, 4), "\n" + '\033[0;30m')

        _ax.plot(np.arange(1, len(training_loss['bpd'])+1), training_loss['bpd'], label = "training BPD")
        _ax.plot(np.arange(1, len(training_loss['bpd'])+1,eval_every_int)[:len(eval_loss['bpd'])], eval_loss['bpd'], label = "evaluation BPD")
        _ax.set_xlabel("Epoch")

        _ax.scatter(min_train_idx + 1, min_train_bpd, color='blue', label=f'Min Training BPD = {round(min_train_bpd, 3)}', zorder=5)
        _ax.scatter(min_eval_epoch + 1, min_eval_bpd, color='orange', label=f'Min Eval BPD = {round(min_eval_bpd, 3)}', zorder=5)

        _ax.set_title(run[8:-52])
        _ax.legend()

    plt.tight_layout()
    plt.show()

def plot_single_training_loss(dataset, eval_every_int) -> None :
    training_loss = pkl.load(open(dataset+"/metrics_train.pickle", "rb"))
    eval_loss = pkl.load(open(dataset+"/metrics_eval.pickle", "rb"))

    min_train_idx = np.argmin(training_loss['bpd'])
    min_train_bpd = min(training_loss['bpd'])

    min_eval_idx = np.argmin(eval_loss['bpd'])
    min_eval_epoch = min_eval_idx * eval_every_int
    min_eval_bpd = eval_loss['bpd'][min_eval_idx]

    
    print('\033[1;36m' + "For Dataset ", dataset[8:-52], '\033[0;36m')
    print("The epoch with the lowest training bpd is ", min_train_idx)
    print("The epoch with the lowest validation bpd is", min_eval_idx*eval_every_int)
    print("Lowest BPD of the validation is: ", round(min_eval_bpd, 4), "\n" + '\033[0;30m')

    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(1, len(training_loss['bpd'])+1), training_loss['bpd'], label = "training BPD", color = "#4A21A8")
    plt.plot(np.arange(1, len(training_loss['bpd'])+1,eval_every_int)[:len(eval_loss['bpd'])], eval_loss['bpd'], label = "validation BPD", color = "#F1993A")
    plt.xlabel("Epoch")
    plt.ylabel("Bits per dimentions")

    plt.scatter(min_train_idx + 1, min_train_bpd, color="#4A21A8", label=f'Min Training BPD = {round(min_train_bpd, 3)}', zorder=5)
    plt.scatter(min_eval_epoch + 1, min_eval_bpd, color = "#F1993A", label=f'Min Validation BPD = {round(min_eval_bpd, 3)}', zorder=5)

    plt.title("Loss")
    plt.legend()
    plt.savefig("Loss-NAD")
    plt.show()
    
if __name__ == '__main__': 

    NAD_Synthetic_DS4A1 = "EDGE/wandb/Synthetic_K15_node/multinomial_diffusion/multistep/DS4A1"
    NAD_Synthetic_DS4A31 = "EDGE/wandb/Synthetic_K15_node/multinomial_diffusion/multistep/DS4A31"
    NAD_Synthetic_DS4A331 = "EDGE/wandb/Synthetic_K15_node/multinomial_diffusion/multistep/DS4A331"

    NAD_Synthetic_DS3A31 = "EDGE/wandb/Synthetic_K15_node/multinomial_diffusion/multistep/DS3A31"
    NAD_Synthetic_DS5A31 = "EDGE/wandb/Synthetic_K15_node/multinomial_diffusion/multistep/DS5A31"

    EAD_Synthetic_DS5A1 = "EDGE/wandb/Synthetic_K7_edge/multinomial_diffusion/multistep/DS5A1"
    EAD_Synthetic_DS5A31 = "EDGE/wandb/Synthetic_K7_edge/multinomial_diffusion/multistep/DS5A31"
    EAD_Synthetic_DS5A331 = "EDGE/wandb/Synthetic_K7_edge/multinomial_diffusion/multistep/DS5A331"

    Cora = "EDGE/wandb/Cora_K15_node/multinomial_diffusion/multistep/DD32A71"
    Cora_771 = "EDGE/wandb/Cora_K15_node/multinomial_diffusion/multistep/DD64A771"

    Cora_full_A1 = "EDGE/wandb/Cora_full__K15_node/multinomial_diffusion/multistep/DD64A1"
    Cora_full_A71 = "EDGE/wandb/Cora_full__K15_node/multinomial_diffusion/multistep/DD64A71"
    Cora_full_A1001 = "EDGE/wandb/Cora_full__K15_node/multinomial_diffusion/multistep/DD64A1001"

    plot_single_training_loss(NAD_Synthetic_DS4A31, 3)

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



