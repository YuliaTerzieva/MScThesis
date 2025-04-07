import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import pickle
from itertools import product

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
def plot_degree_distribution(g, l_cls = None, r_cld= None, relation=(None, None)):

    algorithm, alg_param = relation
    degree_sequence = sorted((d for n, d in g.degree()), reverse=True)

    plt.bar(*np.unique(degree_sequence, return_counts=True))
    if l_cls is not None and r_cld is not None and relation is not None :
        plt.title(f"Algorithm {algorithm}, param {alg_param}, node class {l_cls} -> {r_cld}")
        if relation[0] == 'BA':
            plt.yscale("log")
    else:
        plt.title(f"Complete network")
        plt.yscale("log")

    plt.xlabel("Degree")
    plt.ylabel("# of Nodes")
    plt.show()

def generate_whole_graph(N, NC, R, reproducibility_seed) -> nx.Graph:
    """
    
    """
    assert len(NC) == N
    assert len(R) == N*N 

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
            
            graphs_to_overlay.append((generated_graph, lhs_class, rhs_class))
            # plot_degree_distribution(generated_graph, lhs_class, rhs_class, relation)

    Final_Graph = nx.Graph()
    node_id_to_class = [(i, {'node_attr': cls})
        for cls, _ in enumerate(NC)
        for i in range(sum(NC[:cls]), sum(NC[:cls+1]))]
    Final_Graph.add_nodes_from(node_id_to_class)

    # Precompute class start indices in the global graph
    class_start = [sum(NC[:i]) for i in range(len(NC))]

    # Overlay all graphs into Final_Graph
    for H, lhs_class, rhs_class in graphs_to_overlay:
        lhs_offset = class_start[lhs_class]
        rhs_offset = class_start[rhs_class]
        lhs_count = NC[lhs_class]

        def map_node(n):
            return lhs_offset + n if n < lhs_count else rhs_offset + (n - lhs_count)

        Final_Graph.add_edges_from(
            (map_node(u), map_node(v)) for u, v in H.edges()
        )

    # plot_degree_distribution(Final_Graph)
    return Final_Graph

def add_anomalous_relations(N, nb_anomalous_relations, graph) -> nx.Graph:

    assert len(nb_anomalous_relations) == N*N

    for r_count, relation in enumerate(nb_anomalous_relations):
        if relation > 0:
            lhs_class = r_count // N
            rhs_class = r_count % N

            lhs_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_attr') == lhs_class]
            rhs_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_attr') == rhs_class]

            if lhs_class == rhs_nodes:
                # Avoid self-loops for same-class pairing
                possible_pairs = [(u, v) for u, v in product(lhs_nodes, repeat=2) if u < v and not graph.has_edge(u, v)]
            else:
                possible_pairs = [(u, v) for u, v in product(lhs_nodes, rhs_nodes) if u != v and not graph.has_edge(u, v)]

            sampled_pairs = random.sample(possible_pairs, relation)
            for u, v in sampled_pairs:
                graph.add_edge(u, v, anomalous = True)
                graph.nodes[u]['anomalous'] = True
                graph.nodes[v]['anomalous'] = True

    return graph

def generate_ego_graphs(big_graph, K = 10) -> list[nx.Graph]:
    
    ego_graphs = []
    is_ego_graph_center_anomaly = np.zeros(len(big_graph.nodes))

    for node in big_graph.nodes:
        ppr = nx.pagerank(Final_Graph, personalization={node: 1.0})

        top_k_nodes = sorted((n for n in ppr if n != node), key=ppr.get, reverse=True)[:K]

        ego_nodes = [node] + top_k_nodes
        ego_subgraph = Final_Graph.subgraph(ego_nodes).copy()
        ego_subgraph.remove_nodes_from(list(nx.isolates(ego_subgraph)))
        ego_graphs.append(ego_subgraph)
        if "anomalous" in big_graph.nodes[node].keys():
            is_ego_graph_center_anomaly[node] == 1

    return ego_graphs, is_ego_graph_center_anomaly

def plot_graph(graph) -> None:
    mapping_to_color = {0:'blue', 1: 'orange', 2: 'grey'}
    map_to_color = lambda color: ([mapping_to_color[c] for c in color] if isinstance(color, list) else mapping_to_color[color])
    node_colors = [graph.nodes[node]['node_attr'] for node in graph.nodes()]
    nx.draw(graph, with_labels=True, node_color=map_to_color(node_colors))
    plt.show()

if __name__ == '__main__':
    
    N = 3 # B, O, G
    N_total_nodes = 5000
    NC_perc = np.array([0.25, 0.35, 0.4]) # 25%, 35%, 40% -> 5 000 nodes in total -> 50 anomalies -> 25 anomalous relations
    NC = (NC_perc * N_total_nodes).astype(int).tolist()
    print(f"The node class cardinality is {NC}")
    R = [("BA", 2), None, ("R", 0.0005), # BB, BO, BG 
         None, None, ("BA", 1), # OB, OO, OG
         None, ("Uni", 2), None] # GB, GO, GG
    K = 15

    # the goal is that 1% of the nodes are anomalous, this means that we have halv the relations being anomalous
    # because both noed of anomalous edge are set as anomalies
    N_anomalous_relations = int(0.01 * N_total_nodes / 2) 
    anomalous_R_perc = np.array([0, 0.2, 0, 
                                 0.2, 0.2, 0, 
                                 0.2, 0, 0.2])
    nb_anomalous_relations = anomalous_R_perc * N_anomalous_relations
    print(nb_anomalous_relations)
    assert sum(nb_anomalous_relations) == N_anomalous_relations

    reproducibility_seed = 42
    random.seed(reproducibility_seed)
    np.random.seed(reproducibility_seed)

    # ------------------------------------------------------------
    # """
    start_time = time.time()

    Final_Graph = generate_whole_graph(N, NC, R, reproducibility_seed)

    end_time = time.time()
    print(f"Time it took to generate {sum(NC)} nodes is { end_time - start_time } seconds ") # Time that took to generate 10000 nodes is 0.04338574409484863 seconds 

    with open("GeneratedDataset_new/Whole_Graph.pkl", 'wb') as f:
        pickle.dump(Final_Graph, f)
    # """
    # ------------------------------------------------------------
    # """
    start_time = time.time()

    Final_Graph_witn_anomalies = add_anomalous_relations(N, (nb_anomalous_relations).astype(int).tolist(), Final_Graph)

    end_time = time.time()
    print(f"Time it took to generate the anomalies is { end_time - start_time } seconds ") # Time that took to generate 10000 nodes is 0.04338574409484863 seconds 

    with open("GeneratedDataset_interm_graph/Whole_Graph_with_anomalies.pkl", 'wb') as f:
        pickle.dump(Final_Graph_witn_anomalies, f)
    # """
    # ------------------------------------------------------------
    # """
    start_time = time.time()
    ego_net_list, is_central_node_anomalous = generate_ego_graphs(Final_Graph_witn_anomalies, K = K)
    end_time = time.time()
    print(f"Time it took to get the ego networks of {sum(NC)} nodes is { end_time - start_time } seconds ") # Time it took to get the ego networks of 10000 nodes is 245.056871175766 (~ 4 minutes) seconds with 15 K

    random.shuffle(ego_net_list)
    with open("GeneratedDataset_interm_graph/ListEgoNet_with_anomalies.pkl", "wb") as f:
        pickle.dump(ego_net_list, f)
    with open("GeneratedDataset_interm_graph/is_central_node_anomalous.pkl", "wb") as f:
        pickle.dump(is_central_node_anomalous, f)
    # """
    # ------------------------------------------------------------
    
    # with open("GeneratedDataset_interm_graph/ListEgoNet.pkl", "rb") as f:
    #     ego_net_list = pickle.load(f)
    for graph in ego_net_list[:20]:
        plot_graph(graph)
    
    with open("GeneratedDataset/RelationalDataset_with_anomaly", "wb") as f:
        pickle.dump(ego_net_list[:int(0.8*len(ego_net_list))], f)
    with open("GeneratedDataset/RelationalDataset_with_anomaly_test_graphs", "wb") as f:
        pickle.dump(ego_net_list[int(0.8*len(ego_net_list)):], f)
    

    