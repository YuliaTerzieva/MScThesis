import pandas as pd
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt


def GeneratePattern1(node_file_name, edge_file_name,  number_of_times=1):
    """Generates pattern 1 : in-burst with 5 blue to 1 orange in combination with convey orange to orange

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

    Parameters
    ----------
    node_file_name : str
        The file name of the csv file with nodes
    edge_file_name : str
        The file name of the csv file with the edges

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
    pos = nx.arf_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edgecolors=node_border_colors, node_size=500)
    plt.title("Generater Mini Graph")
    plt.show()


if __name__ == '__main__':
    pattern_number = 3 # the number of times the pattern would be repeated
    new_connections = 5 # number of new connections to be added (this would be the anomaly)
    number_random = 3 # number of nodes that are random/gray/fillers and have random connections
    node_file = "GeneratedDataset/testnodes.csv"
    edge_file = "GeneratedDataset/testedges.csv"

    """ Note that connecting patters, which the the function that intorduces anomalies, needs to executed before the InjectRandomNodes. 
    Otherwise it would intorduce anomalies between gray nodes, which would be wrong. """

    # GeneratePattern1(node_file, edge_file, pattern_number)
    # ConnectPatterns(node_file, edge_file, new_connections)
    # InjectRandomNodes(node_file, edge_file, number_random, 3, 2)
    Visualizegraph(node_file, edge_file)
