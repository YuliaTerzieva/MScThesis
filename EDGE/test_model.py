import pickle
import torch
from datasets.data import get_data
import torch_geometric as pyg
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from model import get_model
from typing import Tuple 

def get_graph_probability(graph, MC_edge_probabilities, num_MC_sim) -> Tuple[float, float]:
    """
    The edges are independent, thus we can calculate the probabability of this graph by summing
    the individual probaility of the edges, we also need to take the number of edges into account. 
    
    Parameters
    ----------
    graph : networkx graph
            This is the original graph which probability we are trying to calculate
    MC_edge_probabilities : dictionary
            a dictionaly with key edhes (node tuples) and values the probability of that edge
            (the MC probability of being generated)
    num_MC_sim : int
                the number of Monte carlo simulations

    Returns
    -------
    lowest_probability : float
    probability : float
    """
    probability = 0
    lowest_probability = (float('inf'), None)
    for edge in graph.edges:
        probability += MC_edge_probabilities[edge] / num_MC_sim
        if MC_edge_probabilities[edge] / num_MC_sim < lowest_probability[0] : 
            lowest_probability = (MC_edge_probabilities[edge] / num_MC_sim, edge)
    probability /= graph.number_of_edges()


    return lowest_probability, probability

def plot_mean_probability_per_nodes_per_edges(graph_statistic) -> None:
    """
    Parameters
    ----------
    graph_statistic : np.array(np.array(int, int, float))
                      a list of tuples (number_of_nodes, number_of_edges, graph_probability, lowest_edge_probability)
    """
    
    nodes = graph_statistic[:, 0]
    edges = graph_statistic[:, 1]
    probabilities = graph_statistic[:, 2]
    lowest_edge_probability = graph_statistic[:, 3]


    unique_nodes, mean_prob_per_node = np.unique(nodes), np.array([probabilities[nodes == n].mean() for n in np.unique(nodes)])
    unique_edges, mean_prob_per_edge = np.unique(edges), np.array([probabilities[edges == e].mean() for e in np.unique(edges)])

    f, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(unique_nodes, mean_prob_per_node, "o-")
    axes[0].set_ylabel("Mean probability")  
    axes[0].set_xlabel("Number of nodes")
    axes[0].set_title("Mean (MC calculated) probability of graphs \n Per their nodes")


    axes[1].plot(unique_edges, mean_prob_per_edge, "o-")
    axes[1].set_ylabel("Mean probability")
    axes[1].set_xlabel("Number of eges")
    axes[1].set_title("Mean (MC calculated) probability of graphs \n Per their edges")

    plt.show()

    #---------------

    f, axes = plt.subplots(2, 1, figsize=(9, 9))

    nodes_prob_data = [probabilities[nodes == n] for n in unique_nodes]
    axes[0].boxplot(nodes_prob_data, tick_labels=unique_nodes.astype(int))
    axes[0].set_xlabel("Number of nodes")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Probability distribution per nodes")

    edges_prob_data = [probabilities[edges == e] for e in unique_edges]
    axes[1].boxplot(edges_prob_data, tick_labels=unique_edges.astype(int))
    axes[1].set_xlabel("Number of edges")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Probability distribution per edges")

    plt.tight_layout()
    plt.show()

    #---------------
    
    scatter = plt.scatter(nodes, edges, s=probabilities*200, alpha=0.6)
    plt.xlabel("Number of nodes")
    plt.ylabel("Number of edges")
    plt.title("Bubble plot: Nodes vs Edges (size = probability)")

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, func=lambda x:x/200)
    plt.legend(handles, labels, loc="lower right", title="Sizes")
    plt.show()

    #---------------
    f, axes = plt.subplots(2, 1, figsize=(9, 9))

    lowest_edge_probability_wrt_nodes = [lowest_edge_probability[nodes == n] for n in unique_nodes]
    axes[0].boxplot(lowest_edge_probability_wrt_nodes, tick_labels=unique_nodes.astype(int))
    axes[0].set_xlabel("Number of nodes")
    axes[0].set_ylabel("Lowest edge probability")
    axes[0].set_title("Distribution of lowest edge probability per nodes")

    lowest_edge_probability_wet_edges = [lowest_edge_probability[edges == e] for e in unique_edges]
    axes[1].boxplot(lowest_edge_probability_wet_edges, tick_labels=unique_edges.astype(int))
    axes[1].set_xlabel("Number of edges")
    axes[1].set_ylabel("Lowest edge probability")
    axes[1].set_title("Distribution of lowest edge probability per edges")

    plt.tight_layout()
    plt.show()

    return 

def pretty_print_matrix(matrix):
    """
    Pretty prints the between_class_edge_occurence matrix with color labels.

    Parameters:
    matrix (np.array): A 3x3 matrix containing edge occurrences between node groups.

    Returns:
    None
    """
    color_labels = ["Blue", "Orange", "Gray"]  # Corresponding to indices 0, 1, 2

    print("\nBetween-Class Edge Occurrences\n")
    
    # Print column headers
    print(f"{' ':<10} {' | '.join(f'{color:<6}' for color in color_labels)}")
    print("-" * 40)

    # Print upper triangular matrix with row labels
    for i in range(len(matrix)):
        row_str = []
        for j in range(len(matrix)):
            if j >= i:  
                row_str.append(f"{round(matrix[i, j], 4):<6}")
            else:
                row_str.append("   -  ") 

        print(f"{color_labels[i]:<10} {' | '.join(row_str)}")

def between_node_class_stats(graphs, edge_list_counter, num_MC_sim) -> int:
    """
    Parameters
    ----------
    graphs : torch geometric graph

    edge_list_counter : List(Dict(Tuple(), int)) 
                        A list of dictionaries with key : edges (node tuples) and values :the number of occurences of this edge across all (num_MC_sim) graph generations
                        Each dictionary is for a graph in graphs
    num_MC_sim : int 
                the number of Monte carlo simulations
    """

    between_class_edge_occurence = np.zeros((3, 3), dtype=int) # order is blue, orange, gray just like the mapping

    for g_index, graph in enumerate(graphs):
        blue_index = (graph.node_attr == 0).nonzero().squeeze().tolist()
        orange_index = (graph.node_attr == 1).nonzero().squeeze().tolist()
        gray_index = (graph.node_attr == 2).nonzero().squeeze().tolist()

        if isinstance(blue_index, int):
            blue_index = [blue_index]
        if isinstance(orange_index, int):
            orange_index = [orange_index]
        if isinstance(gray_index, int):
            gray_index = [gray_index]

        node_to_group = {node: 0 for node in blue_index}
        node_to_group.update({node: 1 for node in orange_index})
        node_to_group.update({node: 2 for node in gray_index})

        for (u, v), occurence in edge_list_counter[g_index].items():
            # if u in node_to_group.keys() and v in node_to_group.keys():
            between_class_edge_occurence[node_to_group[u], node_to_group[v]] += occurence
            if u != v:
                between_class_edge_occurence[node_to_group[v], node_to_group[u]] += occurence
        # between_class_edge_occurence = between_class_edge_occurence / num_MC_sim
    # between_class_edge_occurence = between_class_edge_occurence / len(graphs)
    pretty_print_matrix(between_class_edge_occurence)
    return between_class_edge_occurence


which_run = "./wandb/Basic_test_no_anomaly/multinomial_diffusion/multistep/2025-03-15_20-14-25"

path = which_run+"/check/checkpoint_49.pt"
num_samples = 3
Monte_Carlo = 10

path_args = which_run+ "/args.pickle"
with open(path_args, 'rb') as f:
    args = pickle.load(f)

# print(args)
args.device = 'cpu'
_, _, _, _, _, _, _, _, initial_graph_sampler, _, _, _ = get_data(args)

model = get_model(args, initial_graph_sampler=initial_graph_sampler)
checkpoint = torch.load(path, map_location=args.device, weights_only=False)
model.load_state_dict(checkpoint['model'])

if torch.cuda.is_available():
    model = model.to(args.device)

model.eval()

# print(model) # BinomialDiffusionActive _denoise_fn TGNN_degree_guided

# sample 
original_graphs, sampled_pygraph, per_graph_edge_list_counter = model.sample_and_MC(num_samples, w = 0, MC = Monte_Carlo) 

# print(per_graph_edge_list_counter)

mapping = {0: 'blue', 1: 'orange', 2: 'grey'}
# fig, axes = plt.subplots(num_samples, 2)
# if num_samples == 1:
#     axes = np.array(axes).reshape(1, 2)

graph_statistic = []
for count, (OG_graph, generated) in enumerate(zip(original_graphs, sampled_pygraph.to_data_list())):

    OG_node_colors = [mapping[node_class.item()] for node_class in OG_graph.node_attr]
    og_gen = pyg.utils.to_networkx(OG_graph, to_undirected=True)
    
    generated_node_colors = [mapping[node_class.item()] for node_class in generated.node_attr]
    g_gen = pyg.utils.to_networkx(generated, to_undirected=True)

    g_gen.clear_edges()
    g_gen.remove_edges_from(list(g_gen.edges))
    edges_with_weights = [(u, v, {'probability': count / Monte_Carlo}) for (u, v), count in per_graph_edge_list_counter[count].items()]
    g_gen.add_edges_from(edges_with_weights)
    edge_labels = {(u, v): f"{data['probability']:.2f}" for u, v, data in g_gen.edges(data=True)}

    
    lowest_edge_probability, graph_probabilitiy = get_graph_probability(og_gen, per_graph_edge_list_counter[count], Monte_Carlo)

    # if lowest_edge_probability[0] < 0.01 : 
    fig, axes = plt.subplots(1, 2)
    pos = nx.circular_layout(og_gen)
    nx.draw(og_gen, pos, ax=axes[0],with_labels=True, node_color=OG_node_colors)
    pos = nx.circular_layout(g_gen)
    nx.draw(g_gen, pos, ax=axes[1], with_labels=True, node_color=generated_node_colors)
    nx.draw_networkx_edge_labels(g_gen, pos, ax=axes[1], edge_labels=edge_labels, font_color='gray')
    
    axes[0].set_title(f"Original graph with probability {graph_probabilitiy :.3f}\n edge with lowest porbability {lowest_edge_probability[1]}")
    axes[1].set_title(f"Edge probability over {Monte_Carlo} generated graphs")
    print(per_graph_edge_list_counter[count])
    print(lowest_edge_probability[1])
    plt.tight_layout()
    plt.show()    
        
    graph_statistic.append(np.array([og_gen.number_of_nodes(), og_gen.number_of_edges(), graph_probabilitiy, lowest_edge_probability[0]]))

print()
graph_statistic = np.array(graph_statistic)
# print(graph_statistic)
# plot_mean_probability_per_nodes_per_edges(graph_statistic)
between_node_class_stats(original_graphs, per_graph_edge_list_counter, Monte_Carlo)

