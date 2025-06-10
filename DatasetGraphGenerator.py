import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import pickle
from itertools import product
from collections import defaultdict

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
    print("The mean node degree in the whole network is : ", np.mean(degree_sequence), np.std(degree_sequence))

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

def plot_degree_distribution_by_node_class(g, l_cls = None, r_cls= None, relation=(None, None), class_labels = None):

    algorithm, alg_param = relation
    if l_cls != r_cls:
        degree_sequence_l_cls = sorted((d for n_id, d in g.degree() if class_labels[n_id] == l_cls), reverse=True)
        degree_sequence_r_cls = sorted((d for n_id, d in g.degree() if class_labels[n_id] == r_cls), reverse=True)

        x, y = np.unique(degree_sequence_l_cls, return_counts=True)
        print(x, y)
        plt.plot(x, y, label=f"Degree of node type {l_cls}")
        plt.xlabel("Degree")
        plt.ylabel("# of Nodes")
        # plt.title(f"Distribution of the node degrees of class {l_cls} ert \nalgorithm {algorithm}, param {alg_param}, node class {l_cls} -> {r_cls}")
        # plt.show()
        x, y = np.unique(degree_sequence_r_cls, return_counts=True)
        print(x, y)
        plt.plot(x, y, label=f"Degree of node type {r_cls}")
        plt.xlabel("Degree")
        plt.ylabel("# of Nodes")
        plt.title(f"Distribution of the node degrees \nalgorithm {algorithm}, param {alg_param}, node class {l_cls} -> {r_cls}")
        plt.legend()
        plt.show()
        
    else:
        degree_sequence = sorted((d for n, d in g.degree()), reverse=True)
        plt.bar(*np.unique(degree_sequence, return_counts=True))

        if l_cls is not None and r_cls is not None and relation is not None :
            plt.title(f"Algorithm {algorithm}, param {alg_param}, node class {l_cls} -> {r_cls}")
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
                plot_degree_distribution_by_node_class(generated_graph, lhs_class, rhs_class, relation)
                
            else :
                algorithm = mapping_abbr_2_alg_between_class[relation[0]]
                generated_graph = algorithm(NC[lhs_class], NC[rhs_class], algorithm_parameter, reproducibility_seed)
                class_labels = dict(zip(range(NC[lhs_class]), [lhs_class]*NC[lhs_class]))
                class_labels.update(dict(zip(range(NC[lhs_class], NC[lhs_class] + NC[rhs_class]), [rhs_class] * NC[rhs_class])))
                nx.set_node_attributes(generated_graph, class_labels, "node_attr")
                plot_degree_distribution_by_node_class(generated_graph, lhs_class, rhs_class, relation, class_labels)
            
            graphs_to_overlay.append((generated_graph, lhs_class, rhs_class))
            

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

    plot_degree_distribution(Final_Graph)
    return Final_Graph

def add_anomalous_relations(N :int , nb_anomalous_relations : list[int], graph: nx.Graph) -> nx.Graph:

    assert len(nb_anomalous_relations) == N*N

    nx.set_edge_attributes(graph, 0, 'anomalous')

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
                graph.add_edge(u, v, anomalous = 1)

    return graph

def generate_edge_ego_graphs(big_graph, K) -> list[nx.Graph]:
    
    edge_ego_graphs = []

    ego_nodes = defaultdict(list)
    for node in big_graph.nodes:
        ppr = nx.pagerank(big_graph, personalization={node: 1.0})
        top_k_nodes = sorted((n for n in ppr if n != node), key=ppr.get, reverse=True)[:K]
        ego_nodes[node] = [node] + top_k_nodes

    for edge in big_graph.edges(data=True):
        ego_subgraph = big_graph.subgraph(ego_nodes[edge[0]] + ego_nodes[edge[1]]).copy()
        ego_subgraph.remove_nodes_from(list(nx.isolates(ego_subgraph)))
        nx.set_edge_attributes(ego_subgraph, 0, 'central_edge')
        ego_subgraph[edge[0]][edge[1]]['central_edge'] = 1
        edge_ego_graphs.append(ego_subgraph)

    return edge_ego_graphs

def plot_graph(graph) -> None:
    mapping_to_color = {0:'blue', 1: 'orange', 2: 'grey'}
    map_to_color = lambda color: ([mapping_to_color[c] for c in color] if isinstance(color, list) else mapping_to_color[color])
    node_colors = [graph.nodes[node]['node_attr'] for node in graph.nodes()]
    nx.draw(graph, with_labels=True, node_color=map_to_color(node_colors))
    plt.show()

def add_anomalous_nodes(N_nodes, G) -> nx.Graph:
    # 80 is from O to G 
    # M is from G to O this should be bigger because this is a bigger anomaly! 
    print(f"Injecting {N_nodes} anomalous nodes")
    M = int(0.8 * N_nodes)
    N = N_nodes - M

    nx.set_node_attributes(G, 0, 'anomalous')

    nodes = np.array(G.nodes)
    attrs = np.array([G.nodes[n]['node_attr'] for n in nodes])
    idx_1 = np.where(attrs == 1)[0] # Orange Nodes
    idx_2 = np.where(attrs == 2)[0] # Grey nodes

    assert len(G.nodes) == len(idx_1) + len(idx_2) + len(np.where(attrs == 0)[0])
    
    selected_1_Orange = np.random.choice(idx_1, N, replace=False)
    selected_2_Gray = np.random.choice(idx_2, M, replace=False)

    for i in selected_1_Orange: # this is from Orange to Grey
        G.nodes[nodes[i]]['node_attr'] = 2
        G.nodes[nodes[i]]['anomalous'] = 1

    for i in selected_2_Gray: # this is from Grey to Orange
        G.nodes[nodes[i]]['node_attr'] = 1
        G.nodes[nodes[i]]['anomalous'] = 1

    return G
    
def generate_ego_graph(G, K) -> list[nx.Graph]:
    
    ego_graphs = []
    count = 0

    for node in G.nodes:
        if len([edge for edge in G.edges() if edge[0] == node or edge[1]==node]) == 0:
            count+=1
            continue
        ppr = nx.pagerank(G, personalization={node: 1.0})
        top_k_nodes = sorted((n for n in ppr if n != node), key=ppr.get, reverse=True)[:K]
        ego_subgraph = G.subgraph([node] + top_k_nodes).copy()
        ego_subgraph.remove_nodes_from(list(nx.isolates(ego_subgraph)))

        if len(ego_subgraph.edges) <=5 : 
            count+=1
            continue

        nx.set_node_attributes(ego_subgraph, 0, 'central_node')
        ego_subgraph.nodes[node]['central_node'] = 1
        ego_graphs.append(ego_subgraph)

    print(count, "those are the isolated nodes out of ", len(G.nodes))
    return ego_graphs

# ----------------------------------------------------------------------------

def inductive_edge_split_and_save_node_anomaly(
    graph: nx.Graph,
    K: int,
    split_ratios=(0.25, 0.25, 0.25, 0.25),
    seed=42,
    dataset_name="Synthetic"
):
    """
    Performs inductive 4-way edge split on a graph, generates ego-graphs, and saves them to disk.
    
    Parameters:
        graph (nx.Graph): Input undirected graph with attributes.
        K (int): Hop size for ego-graph generation.
        split_ratios (tuple): Four-way split
        seed (int): Random seed for reproducibility.
        dataset_name (str): Prefix for output filenames.
    """

    edges = list(graph.edges)
    random.Random(seed).shuffle(edges)
    n_total = len(edges)

    split_1 = round(split_ratios[0] * n_total)
    split_2 = round((split_ratios[0] + split_ratios[1]) * n_total)
    split_3 = round((split_ratios[0] + split_ratios[1] + split_ratios[2]) * n_total)

    edge_splits = {
        'Training': edges[:split_1],
        'Validation 1': edges[split_1:split_2],
        'Validation 2': edges[split_2:split_3],
        'Testing': edges[split_3:]
    }

    for name, edge_list in edge_splits.items():
        subgraph = nx.Graph()
        subgraph.add_edges_from(edge_list)

        for node in subgraph.nodes():
            if node in graph.nodes:
                subgraph.nodes[node].update(graph.nodes[node])
            else :
                print("This is impossible so there is a mistake")
                breakpoint()

        print(f"In {name} number of isolated nodes is :", list(nx.isolates(subgraph)))
        subgraph.remove_nodes_from(list(nx.isolates(subgraph)))

        N_anomalous_nodes = int(0.04 * len(subgraph.nodes)) 
        add_anomalous_nodes(N_anomalous_nodes, subgraph)

        # """ Plotting
        attrs = [0, 1, 2]
        degrees = {a: [d for n, d in subgraph.degree() if subgraph.nodes[n].get('node_attr') == a] for a in attrs}

        # Determine the full range of degrees
        all_degrees = np.arange(0, 21)
        counts = {}

        # Build a consistent count array for each attribute
        for a in attrs:
            unique, count = np.unique(degrees[a], return_counts=True)
            count_dict = dict(zip(unique, count))
            counts[a] = np.array([count_dict.get(deg, 0) for deg in all_degrees])

        colors = ["blue", "orange", "gray"]
        bottom = np.zeros_like(all_degrees)
        plt.figure()
        for a in attrs:
            plt.bar(all_degrees, counts[a], bottom=bottom, label=f'Node type {colors[a]}', color=colors[a], alpha=0.4)
            bottom += counts[a]
            # plt.bar(*np.unique(degrees[a], return_counts=True), alpha=0.4, label=f'Node type {colors[a]}', color=colors[a])

        # Styling
        plt.legend()
        plt.xticks(np.arange(0, 21, 1))
        plt.xlim([0, 20])
        plt.ylim([0, 210])
        plt.xlabel('Node Degree')
        plt.ylabel('Number of nodes')
        plt.title(f'Node degree distribution of {name} subgraph')
        plt.savefig(f'Node degree distribution of {name} subgraph')
        plt.show()

        # """
        continue

        nodes = subgraph.nodes
        attrs = np.array([nodes[n]['node_attr'] for n in nodes])
        one_hot = np.eye(attrs.max() + 1)[attrs]
        nx.set_node_attributes(subgraph, {n: v for n, v in zip(nodes, one_hot)}, 'node_attr')

        # with open(f"GeneratedDataset_interm_graph/{dataset_name}_K{K}_node_{name}", "wb") as f:
        #     pickle.dump(subgraph, f)

        # continue
        ego_net_list = generate_ego_graph(subgraph, K)

        with open(f"GeneratedDataset/{dataset_name}_K{K}_node_{name}", "wb") as f:
            pickle.dump(ego_net_list, f)

def inductive_edge_split_and_save_edge_anomaly(
        graph: nx.Graph, 
        K :int, 
        split_ratios = (0.25, 0.25, 0.25, 0.25),
        seed = 42, 
        dataset_name = "Synthetic"
):
    """
    Performs inductive 4-way edge split on a graph, generates ego-graphs, and saves them to disk.
    
    Parameters:
        graph (nx.Graph): Input undirected graph with attributes.
        K (int): Hop size for ego-graph generation.
        split_ratios (tuple): Four-way split
        seed (int): Random seed for reproducibility.
        dataset_name (str): Prefix for output filenames.
    """

    edges = list(graph.edges)
    random.Random(seed).shuffle(edges)
    n_total = len(edges)

    split_1 = round(split_ratios[0] * n_total)
    split_2 = round((split_ratios[0] + split_ratios[1]) * n_total)
    split_3 = round((split_ratios[0] + split_ratios[1] + split_ratios[2]) * n_total)

    edge_splits = {
        'Training': edges[:split_1],
        'Validation 1': edges[split_1:split_2],
        'Validation 2': edges[split_2:split_3],
        'Testing': edges[split_3:]
    }

    for name, edge_list in edge_splits.items():
        subgraph = nx.Graph()
        subgraph.add_edges_from(edge_list)

        for node in subgraph.nodes():
            if node in graph.nodes:
                subgraph.nodes[node].update(graph.nodes[node])
            else :
                print("This is impossible so there is a mistake")
                breakpoint()

        print(f"In {name} number of isolated nodes is :", list(nx.isolates(subgraph)))
        subgraph.remove_nodes_from(list(nx.isolates(subgraph)))

        print(f"Total number on edges in the {name} graph before anomalies : {subgraph.number_of_edges()}")
        N_anomalous_relations = round(0.04 * subgraph.number_of_edges()) 
        anomalous_R_perc = np.array([0, 0.33, 0, 
                                    0, 0.33, 0, 
                                    0, 0, 0.34])
        nb_anomalous_relations = anomalous_R_perc * N_anomalous_relations
        # print((nb_anomalous_relations).astype(int).tolist())
        assert sum(nb_anomalous_relations) == N_anomalous_relations

        print(f"Adding the following number of edges per relation type {nb_anomalous_relations}")

        add_anomalous_relations(3, (nb_anomalous_relations).astype(int).tolist(), subgraph)

        # """ Plotting
        attrs = [0, 1, 2]
        degrees = {a: [d for n, d in subgraph.degree() if subgraph.nodes[n].get('node_attr') == a] for a in attrs}

        # Determine the full range of degrees
        all_degrees = np.arange(0, 21)
        counts = {}

        # Build a consistent count array for each attribute
        for a in attrs:
            unique, count = np.unique(degrees[a], return_counts=True)
            count_dict = dict(zip(unique, count))
            counts[a] = np.array([count_dict.get(deg, 0) for deg in all_degrees])

        colors = ["blue", "orange", "gray"]
        bottom = np.zeros_like(all_degrees)
        plt.figure()
        for a in attrs:
            plt.bar(all_degrees, counts[a], bottom=bottom, label=f'Node type {colors[a]}', color=colors[a], alpha=0.4)
            bottom += counts[a]

        # Styling
        plt.legend()
        plt.xticks(np.arange(0, 21, 1))
        plt.xlim([0, 20])
        plt.ylim([0, 210])
        plt.xlabel('Node Degree')
        plt.ylabel('Number of nodes')
        plt.title(f'Node degree distribution of {name} subgraph')
        plt.savefig(f'Node degree distribution of {name} subgraph -> edge anomaly detection')
        plt.show()

        continue

        # """
    
        nodes = subgraph.nodes
        attrs = np.array([nodes[n]['node_attr'] for n in nodes])
        one_hot = np.eye(attrs.max() + 1)[attrs]
        nx.set_node_attributes(subgraph, {n: v for n, v in zip(nodes, one_hot)}, 'node_attr')

        with open(f"GeneratedDataset_interm_graph/{dataset_name}_K{K}_edge_{name}", "wb") as f:
            pickle.dump(subgraph, f)

        ego_net_list = generate_edge_ego_graphs(subgraph, K = K)
    
        with open(f"GeneratedDataset/{dataset_name}_K{K}_edge_{name}", "wb") as f:
            pickle.dump(ego_net_list, f)

def generate_edge_anomaly_dataset() -> None :

    with open("GeneratedDataset_interm_graph/Synthetic_node.pkl", 'rb') as f:
        Final_Graph = pickle.load(f)
    
    K = 7
    inductive_edge_split_and_save_edge_anomaly(Final_Graph, K)

def generate_node_anomaly_dataset() -> None:
    N = 3 # B, O, G
    N_total_nodes = 1000
    NC_perc = np.array([0.25, 0.35, 0.4]) 
    NC = (NC_perc * N_total_nodes).astype(int).tolist()
    print(f"The node class cardinality is {NC}")
    R = [("BA", 3), None, ("R", 0.01), # BB, BO, BG
         None, None, ("Uni", 7), # OB, OO, OG
         None, ("BA", 8), None] # GB, GO, GG
    K = 15

    reproducibility_seed = 42
    random.seed(reproducibility_seed)
    np.random.seed(reproducibility_seed)


    Final_Graph = generate_whole_graph(N, NC, R, reproducibility_seed)
    
    # node_color_mapping = {0: 'blue', 1: 'orange', 2: 'grey'}
    # original_node_colors = [node_color_mapping[n[1]['node_attr']] for n in Final_Graph.nodes(data = True)]
    # nx.draw(Final_Graph,  node_color=original_node_colors, node_size = 50, alpha = 0.7) 
    # plt.show()
    
    with open("GeneratedDataset_interm_graph/Synthetic_node.pkl", 'wb') as f:
        pickle.dump(Final_Graph, f)
    
    # """
    # ------------------------------------------------------------
    # """
    
    inductive_edge_split_and_save_node_anomaly(Final_Graph, K)
# ----------------------------------------------------------------------------

def plot_graph_freq_wrt_node_edge(dataset_name) -> None:
    # mapping = {0: 'blue', 1: 'orange',2: 'grey'} # the reason why i have this and not one-hot-encoming is
    # map_color = lambda color: ([mapping[c] for c in color] if isinstance(color, list) else mapping[color])

    train_graph = pickle.load(open(f'GeneratedDataset/{dataset_name}_train', 'rb'))
    eval_graph = pickle.load(open(f'GeneratedDataset/{dataset_name}_eval', 'rb'))
    # tune_graph = pickle.load(open(f'GeneratedDataset/{dataset_name}_tune', 'rb'))
    test_graph = pickle.load(open(f'GeneratedDataset/{dataset_name}_test', 'rb'))

    print('\033[95m' + "The number of graphs in the training is ", len(train_graph))
    print("The number of graphs in the eval is ", len(eval_graph))
    # print("The number of graphs in the tune is ", len(tune_graph))
    print("The number of graphs in the testing is ", len(test_graph))
    print('\033[1;30m')

    plt.figure()
    train_number_of_nodes = [len(n.nodes) for n in train_graph]
    plt.hist(train_number_of_nodes, alpha = 0.5, label = "train")
    print("The training number of nodes have mean and std : ", np.mean(train_number_of_nodes), np.std(train_number_of_nodes))

    eval_number_of_nodes = [len(n.nodes) for n in eval_graph]
    plt.hist(eval_number_of_nodes, alpha = 0.5, label = "eval")

    # tune_number_of_nodes = [len(n.nodes) for n in tune_graph]
    # plt.hist(tune_number_of_nodes, alpha = 0.5, label = "tune")

    test_number_of_nodes = [len(n.nodes) for n in test_graph]
    plt.hist(test_number_of_nodes, alpha = 0.5, label = "test")
    plt.legend()
    plt.xlabel("Number of nodes")
    plt.ylabel("Number of graphs")
    plt.title(f"Ego-induced subgraphs from {dataset_name[:-8]} dataset")
    plt.show()

    # ---- EDGEs now

    train_number_of_edges = [len(n.edges) for n in train_graph]
    plt.hist(train_number_of_edges, alpha = 0.5, label = "train")
    print("The training number of eges have mean and std : ", np.mean(train_number_of_edges), np.std(train_number_of_edges))

    eval_number_of_edges = [len(n.edges) for n in eval_graph]
    plt.hist(eval_number_of_edges, alpha = 0.5, label = "eval")

    # tune_number_of_edges = [len(n.edges) for n in tune_graph]
    # plt.hist(tune_number_of_edges, alpha = 0.5, label = "tune")

    test_number_of_edges = [len(n.edges) for n in test_graph]
    plt.hist(test_number_of_edges, alpha = 0.5, label = "test")
    plt.legend()
    plt.xlabel("Number of edges")
    plt.ylabel("Number of graphs")
    plt.title(f"Ego-induced subgraphs from {dataset_name[:-8]} dataset")
    plt.show()

def plot_example_graphs_node(dataset_name):

    graphs = pickle.load(open(f"GeneratedDataset/{dataset_name}", "rb"))
    node_color_mapping = {0: 'blue', 1: 'orange', 2: 'grey'}
    np.random.shuffle(graphs)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten() 

    anomalous_gs = [g for g in graphs if nx.get_node_attributes(g, "anomalous")[next(n for n, d in g.nodes(data=True) if d.get('central_node') == 1)] == 1]
    nomal_gs = [g for g in graphs if g not in anomalous_gs]

    for i, g in enumerate(nomal_gs[:3]):
        ax = axes[i]

        node_color = [node_color_mapping[int(np.argmax(data['node_attr']))] for _, data in g.nodes(data=True)]

        central_node_id = next(n for n, d in g.nodes(data=True) if d.get('central_node') == 1)

        anomalous = "anomalous" if g.nodes[central_node_id].get('anomalous', 0) == 1 else "not anomalous"

        nx.draw(g, with_labels=True, node_color=node_color, alpha = 0.7, ax=ax, node_size=300, font_size=8)
        ax.set_title(f"Central: {central_node_id}\n({anomalous})", fontsize=10)

    for i, g in enumerate(anomalous_gs[:3]):
        ax = axes[i+3]

        node_color = [node_color_mapping[int(np.argmax(data['node_attr']))] for _, data in g.nodes(data=True)]

        central_node_id = next(n for n, d in g.nodes(data=True) if d.get('central_node') == 1)

        anomalous = "anomalous" if g.nodes[central_node_id].get('anomalous', 0) == 1 else "not anomalous"

        nx.draw(g, with_labels=True, node_color=node_color, alpha = 0.7, ax=ax, node_size=300, font_size=8)
        ax.set_title(f"Central: {central_node_id}\n({anomalous})", fontsize=10)

    # fig.suptitle("Example of subgraphs for node anomaly detection\n(input to node-guided topology reconstructor)")
    plt.tight_layout()
    plt.show()

def plot_example_graphs_edge(dataset_name):

    graphs = pickle.load(open(f"GeneratedDataset/{dataset_name}", "rb"))
    node_color_mapping = {0: 'blue', 1: 'orange', 2: 'grey'}
    np.random.shuffle(graphs)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten() 

    anomalous_gs = [G for G in graphs if any(d.get('central_edge') == 1 and d.get('anomalous') == 1 for _, _, d in G.edges(data=True))]
    nomal_gs = [g for g in graphs if g not in anomalous_gs]

    for i, g in enumerate(nomal_gs[:3]):
        ax = axes[i]

        node_color = [node_color_mapping[int(np.argmax(data['node_attr']))] for _, data in g.nodes(data=True)]

        central_edge = next((n1, n2) for n1, n2, d in g.edges(data=True) if d.get('central_edge') == 1)

        nx.draw(g, with_labels=True, node_color=node_color, alpha = 0.7, ax=ax, node_size=300, font_size=8)
        ax.set_title(f"Central: {central_edge}\n(not anomalous)", fontsize=10)

    for i, g in enumerate(anomalous_gs[:3]):
        ax = axes[i+3]

        node_color = [node_color_mapping[int(np.argmax(data['node_attr']))] for _, data in g.nodes(data=True)]

        central_edge = next((n1, n2) for n1, n2, d in g.edges(data=True) if d.get('central_edge') == 1)

        nx.draw(g, with_labels=True, node_color=node_color, alpha = 0.7, ax=ax, node_size=300, font_size=8)
        ax.set_title(f"Central: {central_edge}\n(anomalous)", fontsize=10)

    # fig.suptitle("Example of subgraphs for node anomaly detection\n(input to node-guided topology reconstructor)")
    plt.tight_layout()
    plt.show()

def plot_example_subgraphs_cora(dataset_name):

    graphs = pickle.load(open(f"GeneratedDataset/{dataset_name}", "rb"))
    np.random.shuffle(graphs)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten() 

    struc_anomalous_gs = [g for g in graphs if nx.get_node_attributes(g, "struc_anomaly")[next(n for n, d in g.nodes(data=True) if d.get('central_node') == 1)] == 1]
    contextual_anomalous_gs = [g for g in graphs if nx.get_node_attributes(g, "contextual_anomaly")[next(n for n, d in g.nodes(data=True) if d.get('central_node') == 1)] == 1]
    nomal_gs = [g for g in graphs if (g not in struc_anomalous_gs) and (g not in contextual_anomalous_gs)]

    for i, g in enumerate(nomal_gs[:3]):
        ax = axes[i]
        central_node_id = next(n for n, d in g.nodes(data=True) if d.get('central_node') == 1)
        nx.draw(g, with_labels=True, alpha = 0.7, ax=ax, node_size=300, font_size=8)
        ax.set_title(f"Central: {central_node_id}\n(normal)", fontsize=10)

    for i, g in enumerate(struc_anomalous_gs[:3]):
        ax = axes[i+3]
        central_node_id = next(n for n, d in g.nodes(data=True) if d.get('central_node') == 1)
        nx.draw(g, with_labels=True, alpha = 0.7, ax=ax, node_size=300, font_size=8)
        ax.set_title(f"Central: {central_node_id}\n(structural anomaly)", fontsize=10)

    for i, g in enumerate(contextual_anomalous_gs[:3]):
        ax = axes[i+6]
        central_node_id = next(n for n, d in g.nodes(data=True) if d.get('central_node') == 1)
        nx.draw(g, with_labels=True, alpha = 0.7, ax=ax, node_size=300, font_size=8)
        ax.set_title(f"Central: {central_node_id}\n(Contextual anomaly)", fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_max_degree(dataset_names):
    bar_width = 0.2
    all_x_vals = set()
    datasets_names = ["Training", "Validation 1", "Validation 2", "Testing"]

    colors = ["#F1993A", "#4A21A8" , "#7DC3F6" , "#D6A5F1"] 

    degree_counts_list = []
    for dataset_name in dataset_names:
        graphs = pickle.load(open(f"GeneratedDataset/{dataset_name}", "rb"))
        max_node_degrees = [np.max([d for n, d in G.degree()]) for G in graphs]
        unique, counts = np.unique(max_node_degrees, return_counts=True)
        degree_counts_list.append((unique, counts))
        all_x_vals.update(unique)

    all_x_vals = sorted(list(all_x_vals))
    x_indices = {x: i for i, x in enumerate(all_x_vals)}  # map degree -> index

    # Plot
    plt.figure()
    for i, (unique, counts) in enumerate(degree_counts_list):
        y_vals = np.zeros(len(all_x_vals))
        for u, c in zip(unique, counts):
            y_vals[x_indices[u]] = c
        x = np.arange(len(all_x_vals)) + i * bar_width
        label = f"{datasets_names[i]}"
        plt.bar(x, y_vals, width=bar_width, alpha=1, label=label, color=colors[i % len(colors)])

    plt.xticks(np.arange(len(all_x_vals)) + bar_width * (len(dataset_names) - 1) / 2, all_x_vals)
    plt.title("Distribution of highest node degree \nof subgraphs for edge anomaly detection\n(input to node-guided topology reconstructor)")
    plt.xlabel("Highest node degree")
    plt.ylabel("Number of subgraphs")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    generate_node_anomaly_dataset()
    generate_edge_anomaly_dataset()
    


    K = 15
    dataset_name = f"Synthetic_K{K}_node" 
    plot_graph_freq_wrt_node_edge(dataset_name)

    plot_max_degree([f"{dataset_name}_train", f"{dataset_name}_eval", f"{dataset_name}_tune", f"{dataset_name}_test"])
    plot_example_graphs_node(f"{dataset_name}_train")
    plot_example_graphs_node(f"{dataset_name}_eval")
    plot_example_graphs_node(f"{dataset_name}_tune")
    plot_example_graphs_node(f"{dataset_name}_test")

    