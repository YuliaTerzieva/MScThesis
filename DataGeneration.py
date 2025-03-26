import argparse
import pandas as pd
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric as pyg
import pickle
import json
import random


def GeneratePattern1(node_file_name, edge_file_name,  number_of_times=1):
    """Generates pattern 1 : 
        in a circle connected : blue, orange, grey, orange, blue
        Where the first and the last blue are connected to each other
        The last blue is connected to the first blue of the next patters
        the last blue of the last generated pattern is connected to the first blue of the first pattern

        Note that this is undirectional pattern, however, it is saved as directional 
        and transformed to undirectional when generating the ego networks

    Parameters
    ----------
    node_file_name : str
        The file name of the csv file with nodes
    edge_file_name : str
        The file name of the csv file with the edges
    number_of_times : int, optional
        Number of times the pattern should be generated (default is 1)
    
    Returns
    -------
    None
    
    """

    
    if not os.path.isfile(node_file_name):
        pd.DataFrame(columns=['node_id', 'color', 'anomaly']).to_csv(node_file_name, index=False)

    if not os.path.isfile(edge_file_name):
        pd.DataFrame(columns=['source', 'target']).to_csv(edge_file_name, index=False)

    with open(node_file_name, 'r') as file_reader : 
        number_of_existing_nodes = len(file_reader.readlines())-1
    
    node_writer = open(node_file_name, "a")
    edge_writer = open(edge_file_name, "a")
    
    for i in range(number_of_times):
        node_id_0 = number_of_existing_nodes
        node_writer.write(f"{node_id_0},blue,0\n")
        node_writer.write(f"{node_id_0+1},orange,0\n")
        node_writer.write(f"{node_id_0+2},grey,0\n")
        node_writer.write(f"{node_id_0+3},orange,0\n")
        node_writer.write(f"{node_id_0+4},blue,0\n")
        edge_writer.write(f"{node_id_0},{node_id_0+1}\n") # blue to orange
        edge_writer.write(f"{node_id_0+1},{node_id_0+2}\n") # orange to grey
        edge_writer.write(f"{node_id_0+2},{node_id_0+3}\n") # grey to orange
        edge_writer.write(f"{node_id_0+3},{node_id_0+4}\n") # orange to blue
        edge_writer.write(f"{node_id_0},{node_id_0+4}\n") # blue 0 to blue 4
        number_of_existing_nodes +=5

        if i > 0 :
            edge_writer.write(f"{node_id_0-1},{node_id_0}\n") # last patter last blue to current pattern first blue

        if i == number_of_times-1:
            edge_writer.write(f"{node_id_0+4},{0}\n") # blue 0 to blue 4

    node_writer.close()
    edge_writer.close()
    return 

def GeneratePattern2(node_file_name, edge_file_name,  number_of_times=1):
    """Generates pattern 2 : in-burst with 5 blue to 1 orange in combination with convey orange to orange

    Parameters
    ----------
    node_file_name : str
        The file name of the csv file with nodes
    edge_file_name : str
        The file name of the csv file with the edges
    number_of_times : int, optional
        Number of times the pattern should be generated (default is 1)
    
    Returns
    -------
    None
    
    """

    
    if not os.path.isfile(node_file_name):
        pd.DataFrame(columns=['node_id', 'color', 'anomaly']).to_csv(node_file_name, index=False)

    if not os.path.isfile(edge_file_name):
        pd.DataFrame(columns=['source', 'target']).to_csv(edge_file_name, index=False)

    with open(node_file_name, 'r') as file_reader : 
        number_of_existing_nodes = len(file_reader.readlines())-1
    
    node_writer = open(node_file_name, "a")
    edge_writer = open(edge_file_name, "a")
    
    for i in range(number_of_times):
        node_id_0 = number_of_existing_nodes
        node_writer.write(f"{node_id_0},blue,0\n")
        node_writer.write(f"{node_id_0+1},blue,0\n")
        node_writer.write(f"{node_id_0+2},blue,0\n")
        node_writer.write(f"{node_id_0+3},blue,0\n")
        node_writer.write(f"{node_id_0+4},orange,0\n")
        node_writer.write(f"{node_id_0+5},orange,0\n")
        edge_writer.write(f"{node_id_0},{node_id_0+4}\n")
        edge_writer.write(f"{node_id_0+1},{node_id_0+4}\n")
        edge_writer.write(f"{node_id_0+2},{node_id_0+4}\n")
        edge_writer.write(f"{node_id_0+3},{node_id_0+4}\n")
        edge_writer.write(f"{node_id_0+4},{node_id_0+5}\n")
        number_of_existing_nodes +=6

    node_writer.close()
    edge_writer.close()
    return 

def ConnectPatterns(node_file_name, edge_file_name, new_connection = 10):
    """Connects patterns by injecting new connections between existing nodes. 
    Those connections are anomalous and thus both nodes are then flagged as anomalies

    Parameters
    ----------
    node_file_name : str
        The file name of the csv file with nodes
    edge_file_name : str
        The file name of the csv file with the edges
    new_connection : int, optional
        Number of new connections to be added between existing nodes (default is 10)
    
    Returns
    -------
    None
    
    """
    nodes = pd.read_csv(node_file_name)
    edges = pd.read_csv(edge_file_name)

    node_id = nodes['node_id'].values
    edges_set = set(map(tuple, edges[['source', 'target']].values))
    total_nodes = len(node_id)
    new_edges = set()
    attempts = 0

    while len(new_edges) < new_connection and attempts < new_connection * 2:
        sampled_indices = np.random.choice(total_nodes, (new_connection * 2, 2), replace=True)
        valid_pairs = [(u, v) for u, v in sampled_indices if u != v and (u, v) not in edges_set]

        
        for u, v in valid_pairs:
            if len(new_edges) >= new_connection:
                break
            edges_set.add((u, v))
            new_edges.add((u, v))
            # Directly update anomaly status
            nodes.at[u, 'anomaly'] = 1
            nodes.at[v, 'anomaly'] = 1
        attempts += 1
    
    new_edges_df = pd.DataFrame(list(new_edges), columns=['source', 'target'])
    new_edges_df.to_csv(edge_file_name, mode='a', header=False, index=False)
    nodes.to_csv(node_file_name, index=False)

    return 

def InjectRandomNodes(node_file_name, edge_file_name, number_random = 10, mu = 2.5, std = 1):
    """Adding random "grey" nodes to a network and connecting them to existing nodes / or themselvs
    by drawing their degree from a gaussian distribution (mean and std from parameters)

    Parameters
    ----------
    node_file_name : str
        The file name of the csv file with nodes
    edge_file_name : str
        The file name of the csv file with the edges
    number_random : int, optional
        Number of new random gray nodes to be added (default is 10)
    mu : int, optional
        The degree of the new nodes are drawn from Gaussian distribution, this is the mean (default is 2.5)
    std : int, optional
        The degree of the new nodes are drawn from Gaussian distribution, this is the standard deviation (default is 1)
    
    """
    nodes_df = pd.read_csv(node_file_name)
    max_node_id = nodes_df['node_id'].max() if not nodes_df.empty else 0
    new_node_ids = range(max_node_id + 1, max_node_id + number_random + 1)

    new_random_nodes_df = pd.DataFrame({
        'node_id': new_node_ids,
        'color': ['grey'] * number_random,
        'anomaly': [0] * number_random
    })

    nodes_df = pd.concat([nodes_df, new_random_nodes_df], ignore_index=True)
    updated_nodes_ids = nodes_df['node_id'].values # <- all node ids including the new gray nodes
    
    new_edges = set()
    new_node_degrees = np.clip(np.random.normal(loc=mu, scale=std, size=number_random).astype(int), 1, len(updated_nodes_ids)-1)

    # print(new_node_ids, new_node_degrees)
    
    for node, degree in zip(new_node_ids, new_node_degrees):
        connections = np.random.choice(updated_nodes_ids[updated_nodes_ids != node], degree, replace=False) # <- no self-loops! 
        directions = np.random.choice([0, 1], size = degree)

        new_edges.update((node, target) if direction == 0 else (target, node)
                         for target, direction in zip(connections, directions))
        # print(f"after node {node} we have the following edges: ")
        # print(new_edges)
        
    new_edges_df = pd.DataFrame(list(new_edges), columns=['source', 'target'])
    new_edges_df.to_csv(edge_file_name, mode='a', header=False, index=False)
    new_random_nodes_df.to_csv(node_file_name, mode='a', header=False, index=False)

def Visualizegraph(node_file_name, edge_file_name):
    """Visualizing the graph from the csv files, retaining the node color and adding red outline for anomaloud nodes
    Returning the networkx object created in the process

    Parameters
    ----------
    node_file_name : str
        The file name of the csv file with nodes
    edge_file_name : str
        The file name of the csv file with the edges

    Returns
    -------
    G : networkx.DiGraph instance
        The graph created from the csv files used for the visualisation
    """
    G = nx.DiGraph()
    
    nodes = pd.read_csv(node_file_name)
    edges = pd.read_csv(edge_file_name)
    edges = edges[['source', 'target']].apply(tuple, axis=1).tolist()

    # Add nodes with attributes
    for _, row in nodes.iterrows():
        G.add_node(row['node_id'], color=row['color'], anomaly=row['anomaly'])
    
    # Add edges
    G.add_edges_from(edges)
    
    # Extract node attributes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_border_colors = ['red' if G.nodes[node]['anomaly'] == 1 else 'black' for node in G.nodes()]
    
    # Draw the graph
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edgecolors=node_border_colors, node_size=500)
    plt.title("Generater Mini Graph")
    plt.show()

    return G

def Ego_Net_Generation(graph, k_hop, ego_net_file_name, test_ego_net_file_name, node_file_name, edge_file_name, print_subgraphs=False):
    """Creating k-hop ego networks and saving them as a pickled list of networkx objects in the file ego_net_file_name. 
    The subgraph includes 
        - the central node, 
        - all the k_hop neighbours,
        - the connections between the central node the other nodes
        - the connections between the peripheral nodes as well  !!! 

    Parameters
    ----------
    k_hop : int
        how big thge ego_net to be hop-wise
    ego_net_file_name : str
        The directory name where the ego networks files would be stores with name based on the central node_id 
    """

    mapping = {'blue': 0, 'orange': 1, 'grey': 2} # the reason why i have this and not one-hot-encoming is
    # because in the code, when calling the training funtions, they make the node attribute one-hot encoded :
    # -> diffusion/diffusion_bimonial_active.py _train_loss
    # -> diffusion/diffusion_binomial_vanilla.py _q_sample_and_set_xtmin1_xt_given_x0
    # -> diffusion/diffusion_base.py index_to_log_onehot
    map_color = lambda color: ([mapping[c] for c in color] if isinstance(color, list) else mapping[color])

    mapping_to_color = {0:'blue', 1: 'orange', 2: 'grey'}
    map_to_color = lambda color: ([mapping_to_color[c] for c in color] if isinstance(color, list) else mapping[color])

    print_patience = 10
    if not isinstance(graph, nx.Graph):
        graph = nx.Graph()
        nodes = pd.read_csv(node_file_name)
        edges = pd.read_csv(edge_file_name)
        edges = edges[['source', 'target']].apply(tuple, axis=1).tolist()

        # Add nodes with attributes
        for _, row in nodes.iterrows():
            graph.add_node(row['node_id'], node_attr=map_color(row['color']), anomaly=row['anomaly'])
        
        # Add edges
        graph.add_edges_from(edges)
    
    if not os.path.exists(ego_net_file_name):
        with open(ego_net_file_name, 'x') as file:
            print(f"File created: {ego_net_file_name}")
    if not os.path.exists(test_ego_net_file_name):
        with open(test_ego_net_file_name, 'x') as file:
            print(f"File created: {test_ego_net_file_name}")

    list_of_ego_nets = []

    for node in graph.nodes.items(): # example of node is : (0, {'color': 'blue', 'anomaly': 0})
        sub_graph = nx.ego_graph(graph, node[0], radius=k_hop, center=True, undirected=True)
        if sub_graph.number_of_nodes() > 2:
            list_of_ego_nets.append(sub_graph)

            if print_subgraphs and print_patience >0: # if you want to test make this True, otherwise False
                node_colors = [sub_graph.nodes[node]['node_attr'] for node in sub_graph.nodes()]
                pos = nx.arf_layout(sub_graph)
                nx.draw(sub_graph, pos, with_labels=True, node_color=map_to_color(node_colors))
                plt.show()
                print_patience -=1


    random.shuffle(list_of_ego_nets)
    dbfile = open(ego_net_file_name, 'ab')
    pickle.dump(list_of_ego_nets[:int(0.8*len(list_of_ego_nets))], dbfile)
    dbfile.close()

    dbfile = open(test_ego_net_file_name, 'ab')
    pickle.dump(list_of_ego_nets[int(0.8*len(list_of_ego_nets)):], dbfile)
    dbfile.close()

    return list_of_ego_nets, graph.nodes.items()

def Generate_Dataset_Type_1(arguments):
    """
    Dataset Tupe 1 is circular type dataset, check PDF with exmaplation and examples. 
    
    """
    pattern_number = arguments["pattern_number"] # the number of times the pattern would be repeated
    new_connections = arguments["new_connections"] # number of new connections to be added (this would be the anomaly)
    node_file = arguments["node_file"]
    edge_file = arguments["edge_file"]
    ego_file = arguments["ego_file"]
    hop = arguments["hop"]

    GeneratePattern1(node_file, edge_file, pattern_number)
    ConnectPatterns(node_file, edge_file, new_connections)
    # Visualizegraph(node_file, edge_file)
    Ego_Net_Generation(None, hop, ego_file, ego_file+"_test_graphs", node_file, edge_file, False)

    return 

def Generate_Dataset_Type_2(arguments):
    pattern_number = arguments["pattern_number"] # the number of times the pattern would be repeated
    new_connections = arguments["new_connections"] # number of new connections to be added (this would be the anomaly)
    number_random = arguments["number_random"] # number of nodes that are random/gray/fillers and have random connections
    node_file = arguments["node_file"]
    edge_file = arguments["edge_file"]
    ego_file = arguments["ego_file"]
    gray_degree_mu = arguments["mu"]
    gray_degree_std = arguments["std"]
    hop = arguments["hop"]

    """ Note that connecting patters, which the the function that intorduces anomalies, needs to executed before the InjectRandomNodes. 
    Otherwise it would intorduce anomalies between gray nodes, which would be wrong. """

    GeneratePattern2(node_file, edge_file, pattern_number)
    ConnectPatterns(node_file, edge_file, new_connections)
    InjectRandomNodes(node_file, edge_file, number_random, gray_degree_mu, gray_degree_std)
    # Visualizegraph(node_file, edge_file)
    Ego_Net_Generation(None, hop, ego_file, ego_file+"_test_graphs", node_file, edge_file, False)

    return

#---------

def Generate_nodes(N_nodes, n_blue, n_orange, n_grey, node_file_name):
    if not os.path.isfile(node_file_name):
        pd.DataFrame(columns=['node_id', 'color', 'anomaly']).to_csv(node_file_name, index=False)

    with open(node_file_name, 'r') as file_reader : 
        number_of_existing_nodes = len(file_reader.readlines())-1
    
    node_writer = open(node_file_name, "a")
    
    for i in range(N_nodes):
        if i < int(N_nodes * (n_blue/100)):
            node_writer.write(f"{i},blue,0\n")
        elif i < int(N_nodes * (n_blue/100 + n_orange/100)):
            node_writer.write(f"{i},orange,0\n")  
        else:
            node_writer.write(f"{i},grey,0\n")  

    node_writer.close()
    return 

def Generate_edges(N_edges, edge_prob_list, edge_file_name, node_file_name):
    """
    N_edges: int 
    edge_prob_list: list[int]
        The lngth of which has to be (N_node_classes**2+N_node_classes)/2. 
        The list holds the respective percentage for the given type of relation.
        The order is given by the order of node classes with within class relations coming first, 
        For example given for nodes clases B, O, G, the list is:
        [BB, BO, BG, OO, OG, GG]; note : because we work with undirected graphs BO = OB
    """
    
    types = { 'blue': 0, 'orange': 1, 'grey': 2 }
    nodes_by_type = {0: [], 1: [], 2: []}
    with open(node_file_name) as f:
        next(f)
        for line in f:
            i, color, _ = line.strip().split(',')
            nodes_by_type[types[color]].append(int(i))

    type_pairs = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)] # [BB, BO, BG, OO, OG, GG]
    counts = [int(p / 100 * N_edges) for p in edge_prob_list]

    edges = set()
    for (t1, t2), count in zip(type_pairs, counts):
        a, b = nodes_by_type[t1], nodes_by_type[t2]
        if not a or not b: continue
        while count > 0:
            x, y = random.choice(a), random.choice(b)
            if x != y:
                edge = tuple(sorted((x, y)))
                if edge not in edges:
                    edges.add(edge)
                    count -= 1

    with open(edge_file_name, 'w') as f:
        f.write('source,target\n')
        for x, y in edges:
            f.write(f'{x},{y}\n')
                
def Generate_Dataset_Type_3(arguments):
    """
    "relation_based_test":{
        "dataset_type" : 3, 
        "total_number_nodes" : 500, 
        "blue_perc" : 60, 
        "orange_perc" : 15, 
        "grey_perc": 25, 
        "total_number_relations" : 250, 
        "relation_perc" : [25, 0, 15, 0, 60, 0],
        "node_file": "GeneratedDataset/Mid_test_with_anomaly_nodes.csv",
        "edge_file": "GeneratedDataset/Mid_test_with_anomaly_edges.csv",
        "ego_file": "GeneratedDataset/Mid_test_with_anomaly",
        "hop":2
    }
    """
    total_number_nodes = arguments["total_number_nodes"]
    blue_perc, orange_perc, grey_perc = arguments["blue_perc"], arguments["orange_perc"], arguments["grey_perc"]
    total_number_relations = arguments["total_number_relations"]
    relation_perc = arguments["relation_perc"]
    node_file = arguments["node_file"]
    edge_file = arguments["edge_file"]
    ego_file = arguments["ego_file"]
    hop = arguments["hop"]

    assert blue_perc + orange_perc + grey_perc == 100
    assert sum(relation_perc) == 100

    Generate_nodes(total_number_nodes, blue_perc, orange_perc, grey_perc, node_file)
    Generate_edges(total_number_relations, relation_perc, edge_file, node_file)
    Visualizegraph(node_file, edge_file)
    Ego_Net_Generation(None, hop, ego_file, ego_file+"_test_graphs", node_file, edge_file, True)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Load configuration settings.")
    parser.add_argument("--config", type=str, default="configurations.json", help="Path to the configuration file")
    parser.add_argument("--setting", type=str, required=True, help="Name of the setting to load")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = json.load(file)[args.setting]

    if config["dataset_type"] == 1: 
        Generate_Dataset_Type_1(config)
    if config["dataset_type"] == 2: 
        Generate_Dataset_Type_2(config)
    if config["dataset_type"] == 3: 
        Generate_Dataset_Type_3(config)
    
    
