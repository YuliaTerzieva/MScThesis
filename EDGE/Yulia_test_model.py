import pickle
import torch
from datasets.data import get_data
import torch_geometric as pyg
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from model import get_model
from typing import Tuple 
from collections import defaultdict
import matplotlib.colors as mcolors

def get_graph_probability(graph, MC_edge_probabilities, num_MC_sim) -> Tuple[float, float]:
    """
    The edges are independent, thus we can calculate the probabability of this graph by summing
    the individual probaility of the edges, we also need to take the number of edges into account. 
    
    Parameters
    ----------
    graph : networkx graph
            This is the original graph which probability we are trying to calculate
    MC_edge_probabilities : dictionary
            a dictionaly with key edges (node tuples) and values the probability of that edge
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

def get_graph_probability_2(graph, MC_edge_probabilities, num_MC_sim) -> Tuple[float, float]:
    """
    The edges are independent, thus we can calculate the probabability of this graph by 
    the formula P(G_c | G) = Prod{e in G_c} P(e) * Prod_{e in G / G_c} 1 - P(e)
    
    Parameters
    ----------
    graph : networkx graph
            This is the original graph which probability we are trying to calculate
    MC_edge_probabilities : dictionary
            a dictionaly with key edges (node tuples) and values the probability of that edge
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
    for edge in MC_edge_probabilities.keys():
        if edge in graph.edges():
            probability += MC_edge_probabilities[edge] / num_MC_sim
        else :
            probability += 1 - (MC_edge_probabilities[edge] / num_MC_sim)

        if MC_edge_probabilities[edge] / num_MC_sim < lowest_probability[0] : 
            lowest_probability = (MC_edge_probabilities[edge] / num_MC_sim, edge)

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

def pretty_print_matrix(matrix, matrix_title):
    """
    Pretty prints the between_class_edge_occurence matrix with color labels.

    Parameters:
    matrix (np.array): A 3x3 matrix containing edge occurrences between node groups.

    Returns:
    None
    """
    color_labels = ["Blue", "Orange", "Gray"]  # Corresponding to indices 0, 1, 2

    print(f"\n{matrix_title}\n")
    
    # Print column headers
    print('\033[1;35m', f"{' ':<10} {' | '.join(f'{color:<6}' for color in color_labels)}")
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
    print('\033[0m')

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
    pretty_print_matrix(between_class_edge_occurence, "Between-Class Edge Occurrences")
    return between_class_edge_occurence

def edge_type_probability_distribution(graphs, per_graph_edge_list_counter, num_MC_sim):
    """
    Builds a nested dictionary of edge-type probability distributions per (num_nodes, num_edges).

    Returns
    -------
    edge_prob_stats: dict
        Format: {
            num_nodes: {
                (color1, color2): [prob1, prob2, ...]
            }
        }
    """

    edge_prob_stats = defaultdict(lambda: defaultdict(list))

    # for each graph in the original list
    for g_idx, graph in enumerate(graphs):
        num_nodes = graph.num_nodes
        edge_counts = per_graph_edge_list_counter[g_idx]
        for (u, v), count in edge_counts.items():
            # for each generated edge for that graph save its probability 
            color_u = int(graph.node_attr[u])
            color_v = int(graph.node_attr[v])
            edge_type = tuple(sorted([color_u, color_v]))
            probability = count / num_MC_sim

            edge_prob_stats[num_nodes][edge_type].append(probability)

    return edge_prob_stats

def print_mean_edge_probs(edge_prob_stats):
    """
    Computes the mean probability for each edge type across all num_nodes.

    Parameters
    ----------
    edge_prob_stats : dict
        Nested dict with structure: {
            num_nodes: {
                (color1, color2): [prob1, prob2, ...]
            }
        }

    Returns
    -------
    mean_edge_probs : dict
        Format: {
            (color1, color2): mean_prob
        }
    """
    aggregated = defaultdict(list)

    for node_dict in edge_prob_stats.values():
        for edge_type, probs in node_dict.items():
            aggregated[edge_type].extend(probs)

    matrix = np.zeros((3, 3), dtype=float)
    for edge_type, probs in aggregated.items():
        matrix[edge_type[0], edge_type[1]] = round(np.mean(probs), 4)

    pretty_print_matrix(matrix, "Mean probability")

def plot_edge_distribution_violin_boxplots(edge_prob_stats, number_nodes = None) -> None:
    """
    Plots a box plot and a violin plot for the distribution of edge type probabilities
    for all edge types: blue-blue, blue-orange, blue-grey, orange-orange, orange-grey, and grey-grey.
    Each plot is colored based on the node colors. For mixed edge types, the colors are blended.
    
    Parameters:
    -----------
    edge_prob_stats : dict
        Nested dictionary of edge-type probability distributions per number of nodes.
        Format: {
            num_nodes: {
                (color1, color2): [prob1, prob2, ...]
            }
        }
    number_nodes : int
        This is a parameter to filter and use only statistics from graphs nodes less than "number_nodes" 
    """

    all_edge_type_probs = defaultdict(list)

    if number_nodes == None :
        for node_size_dict in edge_prob_stats.values():
            for edge_type, probs in node_size_dict.items():
                all_edge_type_probs[edge_type].extend(probs)
    else :
        for num_nodes, node_size_dict in edge_prob_stats.items():
            if num_nodes <= number_nodes:
                for edge_type, probs in node_size_dict.items():
                    all_edge_type_probs[edge_type].extend(probs)
    
    edge_type_order = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    node_color = {0: 'blue', 1: 'orange', 2: 'grey'}
    labels = [f"{node_color[a]}-{node_color[b]}" for a, b in edge_type_order]
    
    # Create an edge color mapping. For mixed edges, blend the two colors.
    edge_color_map = {}
    for etype in edge_type_order:
        if etype[0] == etype[1]:
            edge_color_map[etype] = node_color[etype[0]]
        else:
            rgb1 = np.array(mcolors.to_rgb(node_color[etype[0]]))
            rgb2 = np.array(mcolors.to_rgb(node_color[etype[1]]))
            blended = (rgb1 + rgb2) / 2
            edge_color_map[etype] = mcolors.to_hex(blended)
    
    # for each possible relation, get the probability distribution or an empty array
    data = [all_edge_type_probs.get(etype, []) for etype in edge_type_order]
    
    for relation_number, relation_probabilities in enumerate(data):
        if len(relation_probabilities) > 0:
            sorted_relation_probabilities = sorted(relation_probabilities)
            plt.plot(*np.unique(sorted_relation_probabilities, return_counts=True))
            plt.title(f"Relation {edge_type_order[relation_number]}")
            plt.show()
        else:
            print(f" Relation {edge_type_order[relation_number]} doesn't exist")


    # Create a figure with two subplots: one for the box plot and one for the violin plot.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # -- Box Plot --
    bp = axes[0].boxplot(data, patch_artist=True, tick_labels=labels)
    # Color each box with the corresponding edge type color.
    for patch, etype in zip(bp['boxes'], edge_type_order):
        patch.set_facecolor(edge_color_map[etype])
        patch.set_alpha(0.7)
    axes[0].set_title("Box Plot of Edge Type Probabilities")
    axes[0].set_xlabel("Edge Types")
    axes[0].set_ylabel("Probability")
    
    # -- Violin Plot --
    vp = axes[1].violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    # Color each violin body.
    for i, body in enumerate(vp['bodies']):
        etype = edge_type_order[i]
        body.set_facecolor(edge_color_map[etype])
        body.set_edgecolor('black')
        body.set_alpha(0.7)
    axes[1].set_title("Violin Plot of Edge Type Probabilities")
    axes[1].set_xlabel("Edge Types")
    axes[1].set_ylabel("Probability")
    axes[1].set_xticks(np.arange(1, len(labels) + 1))
    axes[1].set_xticklabels(labels)
    
    plt.tight_layout()
    plt.show()

# which_run = "./wandb/Small_test_no_anomaly/multinomial_diffusion/multistep/2025-03-19_19-46-30"
# which_run = "./wandb/Small_test_no_anomaly/multinomial_diffusion/multistep/2025-03-20_19-13-14"

# Friday 21st
# which_run = "./wandb/Mid_test_no_anomaly/multinomial_diffusion/multistep/2025-03-21_10-12-13" # here checkpoint 254
# which_run = "./wandb/Small_test_no_anomaly/multinomial_diffusion/multistep/2025-03-21_10-10-38" # here checkpoint 469

# Wednesday 26th 
# which_run = "./wandb/relation_based_test/multinomial_diffusion/multistep/2025-03-26_18-36-15" # here checkpoint 189

# Wednesday 2th 
which_run = "./wandb/RelationalDataset_no_anomaly/multinomial_diffusion/multistep/2025-04-02_13-18-55" # here checkpoint 929

path = which_run+"/check/checkpoint_929.pt"
num_samples = 10
Monte_Carlo = 1

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


original_graphs, sampled_pygraph, per_graph_edge_list_counter = model.sample_and_MC(num_samples, lambda_guidance = 0.5, MC = Monte_Carlo) # 0 only conditioned, >0 subtracks the unconditioned, actively reducing the probability of generating samples that ignore the conditioning

# print(per_graph_edge_list_counter)

mapping = {0: 'blue', 1: 'orange', 2: 'grey'}
# fig, axes = plt.subplots(num_samples, 2)
# if num_samples == 1:
#     axes = np.array(axes).reshape(1, 2)

graph_statistic = []
for sample_nb, (OG_graph, generated) in enumerate(zip(original_graphs, sampled_pygraph.to_data_list())):

    OG_node_colors = [mapping[node_class.item()] for node_class in OG_graph.node_attr]
    og_gen = pyg.utils.to_networkx(OG_graph, to_undirected=True)
    
    generated_node_colors = [mapping[node_class.item()] for node_class in generated.node_attr]
    g_gen = pyg.utils.to_networkx(generated, to_undirected=True)

    g_gen.clear_edges()
    g_gen.remove_edges_from(list(g_gen.edges))
    edges_with_weights = [(u, v, {'probability': count / Monte_Carlo}) for (u, v), count in per_graph_edge_list_counter[sample_nb].items()]
    g_gen.add_edges_from(edges_with_weights)
    edge_labels = {(u, v): f"{data['probability']:.2f}" for u, v, data in g_gen.edges(data=True)}

    
    lowest_edge_probability, graph_probabilitiy = get_graph_probability(og_gen, per_graph_edge_list_counter[sample_nb], Monte_Carlo)

    # if lowest_edge_probability[0] < 0.01 : 
    if sample_nb < 10:
        fig, axes = plt.subplots(1, 2)
        pos = nx.circular_layout(og_gen)
        nx.draw(og_gen, pos, ax=axes[0],with_labels=True, node_color=OG_node_colors)
        pos = nx.circular_layout(g_gen)
        nx.draw(g_gen, pos, ax=axes[1], with_labels=True, node_color=generated_node_colors)
        nx.draw_networkx_edge_labels(g_gen, pos, ax=axes[1], edge_labels=edge_labels, font_color='gray')
        
        axes[0].set_title(f"Original graph with probability {graph_probabilitiy :.3f}\n edge with lowest porbability {lowest_edge_probability[1]}")
        axes[1].set_title(f"Edge probability over {Monte_Carlo} generated graphs")
        plt.tight_layout()
        plt.show()    
        
    graph_statistic.append(np.array([og_gen.number_of_nodes(), og_gen.number_of_edges(), graph_probabilitiy, lowest_edge_probability[0]]))

print()
# graph_statistic = np.array(graph_statistic)
# plot_mean_probability_per_nodes_per_edges(graph_statistic)
# between_node_class_stats(original_graphs, per_graph_edge_list_counter, Monte_Carlo)
edge_prob_stats = edge_type_probability_distribution(original_graphs, per_graph_edge_list_counter, Monte_Carlo)
print_mean_edge_probs(edge_prob_stats)
# plot_edge_distribution_violin_boxplots(edge_prob_stats)
#------------------------------------------------------------------------------------

