import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def _random_subset(seq, m, rng): # I took this from networkx library
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = random.choice(seq)
        targets.add(x)
    return targets

# -------------- The following is for within class --------------
def uniform_graph(n_nodes, m, seed = 42) -> nx.Graph:
    """
    The function creates a graph with nodes that are all connected to 
    at least m other nodes 
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    for node in G.nodes():
        connections = _random_subset(G.nodes().remove(node), m, seed)
        G.add_edges_from(zip([node] * m, connections))
    
    return G

def uniform_graph_2(n_nodes, m, seed = 42) -> nx.Graph:
    """
    The function creates a graph with nodes that are all connected to 
    exactly m other nodes 
    """
    G = nx.configuration_model([m] * n_nodes, nx.Graph, seed=seed)
    
    return G

# -------------- The following is for between class --------------
def bipartite_barabasi_albert(n_nodes_lhs, n_nodes_rhs, m, seed = 42) -> nx.Graph:
    """
    In this case there two classes of nodes, for each node in the first class 
    the algorithm choose m nodes to connect to from the second class, this is repeated
    and the targets of the each iterations are added to a preferential attachment list
    """

    if n_nodes_rhs < m:
        raise nx.NetworkXError(
            f"There must be at least {m} nodes in from the second class, check your node class cardinality"
        )

    G = nx.Graph()
    
    lhs_node_list = [*range(n_nodes_lhs)]
    rhs_node_list = [*range(n_nodes_lhs, n_nodes_lhs + n_nodes_rhs)]

    G.add_nodes_from(lhs_node_list)
    G.add_nodes_from(rhs_node_list)

    for node in lhs_node_list :
        targets = _random_subset(rhs_node_list, m, seed)
        G.add_edges_from(zip([node] * m, targets))
        rhs_node_list.extend(targets)

    return  G

def bipartite_barabasi_albert_2(n_nodes_lhs, n_nodes_rhs, m, seed = 42) -> nx.Graph:
    """
    Here I have two groups of nodes; For every edge to be added m, 
    I roll a dice if it is from class 1 to class 2 or vice versa
    """
    edges_to_add = 10

    G = nx.Graph()
    
    lhs_node_list = [*range(n_nodes_lhs)]
    rhs_node_list = [*range(n_nodes_lhs, n_nodes_lhs + n_nodes_rhs)]

    G.add_nodes_from(lhs_node_list)
    G.add_nodes_from(rhs_node_list)

    for _ in range(edges_to_add):
        if random.random() < 0.5 : 
            node = random.choice(lhs_node_list)
            targets = _random_subset(rhs_node_list, m, seed)
            G.add_edges_from(zip([node] * m, targets))
            rhs_node_list.extend(targets)
            lhs_node_list.extend([node] * m) 
        else : 
            node = random.choice(rhs_node_list)
            targets = _random_subset(lhs_node_list, m, seed)
            G.add_edges_from(zip([node] * m, targets))
            lhs_node_list.extend(targets)
            rhs_node_list.extend([node] * m) 

    return  G

def bipartite_uniform_graph(n_nodes_lhs, n_nodes_rhs, m, seed = 42) -> nx.Graph: 
    """
    This function creates a graph with two types of nodes, such that the first n_nodes_lhs nodes are in a class
    and the following n_nodes_rhs are in onother class. Each node from the first class is connected to m
    nodes from the second class, the nodes are sampled uniformly. 
    NOTE that here it is imporant which is the left hand side node class and which is the right hand side node! 
    """

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes_lhs))
    G.add_nodes_from(range(n_nodes_lhs, n_nodes_lhs + n_nodes_rhs))

    rhs_node_ids = [*range(n_nodes_lhs, n_nodes_lhs + n_nodes_rhs)]
    for node_id in range(n_nodes_lhs):
        connections = _random_subset(rhs_node_ids, m, seed)
        G.add_edges_from(zip([node_id] * m, connections))

    return G
# ----------------------------------------------------------------------------
"""
The parameters are : 

N : number of node classes, 
NC : node class cardinality
R : relations matrix (N x N) with tuples of relations type and algorithm parameters e.g.
[[("BA", 2), None, ("R", 0.0001)], 
 [None, None, ("BA", 2)], 
 [None, ("Uni", 2), None]]

BA is barabasi albert
R is random
Uni is uniform

Output is a big graph
"""

N = 3 # B, O, G
NC = [4_0, 3_5, 2_5] # 40%, 35%, 25% -> 10 000 nodes in total -> 100 anomalies -> 50 anomalous relations
R = [("BA", 2), None, ("R", 0.001), # BB, BO, BG
     None, None, ("BA", 2), # OB, OO, OG
     None, ("Uni", 2), None] # GB, GO, GG

reproducibility_seed = 42
random.seed(reproducibility_seed)
np.random.seed(reproducibility_seed)

assert len(NC) == N
assert len(R) == N*N 
assert sum(NC) == 10_0 # TODO this can be removed or added as a separate variable

mapping_abbr_2_alg_within_class = {"BA" : nx.barabasi_albert_graph, 
                                   "R" : nx.erdos_renyi_graph, 
                                   "Uni" : uniform_graph}
mapping_abbr_2_alg_between_class = {"BA" : bipartite_barabasi_albert, 
                                    "R" : nx.bipartite.random_graph, 
                                    "Uni" : bipartite_uniform_graph}

graphs_to_overlay = []
for r_count, relation in enumerate(R):
    if relation is not None :
        lhs_class = r_count // N
        rhs_class = r_count % N
        algorithm_parameter = relation[1] # this is either m in the case of BA and Uni; or p in the case of Random

        # breakpoint()
        if lhs_class == rhs_class :
            algorithm = mapping_abbr_2_alg_within_class[relation[0]]
            generated_graph = algorithm(NC[lhs_class], algorithm_parameter, reproducibility_seed)
            nx.set_node_attributes(generated_graph, lhs_class, "node_attr")
            
        else :
            algorithm = mapping_abbr_2_alg_between_class[relation[0]]
            generated_graph = algorithm(NC[lhs_class], NC[rhs_class], algorithm_parameter, reproducibility_seed)
            class_labels = dict(zip(range(NC[lhs_class]), [lhs_class]*NC[lhs_class]))
            class_labels.update(dict(zip(range(NC[lhs_class], NC[lhs_class] + NC[rhs_class]), [rhs_class] * NC[rhs_class])))
            nx.set_node_attributes(generated_graph, class_labels, "node_attr")
        
        graphs_to_overlay.append(generated_graph)

    
"""
Plotting
"""
print(graphs_to_overlay)

for print_graph in graphs_to_overlay:
    mapping_to_color = {0:'blue', 1: 'orange', 2: 'grey'}
    map_to_color = lambda color: ([mapping_to_color[c] for c in color] if isinstance(color, list) else mapping_to_color[color])
    node_colors = [print_graph.nodes[node]['node_attr'] for node in print_graph.nodes()]

    nx.draw(print_graph, with_labels=True, node_color=map_to_color(node_colors))
    plt.show()
