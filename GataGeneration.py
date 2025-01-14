import numpy as np
import networkx as nx

pattern_number = 100 # the number of times the pattern would be repeated
percent_reconnected = 10 # percent of OG connections to be reconnected (this would be the anomaly)
number_random = 100 # number of nodes that are random/gray/fillers and have random connections


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
    node_list
        List with node tuples (node_id, node_color, anomaly_bool)
    edge_list
        List with edges represented by node tuples (node_id, node_id)
    
    """

    with open(node_file_name, 'r') as file_reader : 
        number_of_existing_nodes = len(file_reader.readlines())
    
    node_writer = open(node_file_name, "a")
    edge_writer = open(edge_file_name, "a")
    
    for i in range(number_of_times):
        node_id_0 = number_of_existing_nodes
        node_writer.write(f"{node_id_0}, blue, 0\n")
        node_writer.write(f"{node_id_0+1}, blue, 0\n")
        node_writer.write(f"{node_id_0+2}, blue, 0\n")
        node_writer.write(f"{node_id_0+3}, blue, 0\n")
        node_writer.write(f"{node_id_0+4}, orange, 0\n")
        node_writer.write(f"{node_id_0+5}, orange, 0\n")
        edge_writer.write(f"{node_id_0}, {node_id_0+4}\n")
        edge_writer.write(f"{node_id_0+1}, {node_id_0+4}\n")
        edge_writer.write(f"{node_id_0+2}, {node_id_0+4}\n")
        edge_writer.write(f"{node_id_0+3}, {node_id_0+4}\n")
        edge_writer.write(f"{node_id_0+4}, {node_id_0+5}\n")
        number_of_existing_nodes +=6

    node_writer.close()
    edge_writer.close()
    return 


GeneratePattern1("GeneratedDataset/nodes.txt", "GeneratedDataset/edges.txt", 2)


"""
nodes = [] # id: (type, in_doing_degree, out_going_degree)
edges = [] # (node, node)


# half of the nodes have low connectivity
low_connect_node_degree = np.random.normal(mu_low, sigma_low, number_of_nodes//2).astype(int)
high_connect_node_degree = np.random.normal(mu_high, sigma_high, number_of_nodes//2).astype(int)

nid = 0

for count, node_degree in enumerate(high_connect_node_degree):

    if count%3 == 0:
        ingoing = node_degree//2
        nodes.append([nid, 'A', ingoing, node_degree - ingoing]) # balanced
    
    if count%3 == 1:
        outgoing = node_degree//3
        nodes.append([nid, "B", node_degree - outgoing, outgoing]) # majority ingoing (sink)
   
    if count%3 == 2:
        ingoing = node_degree//3
        nodes.append([nid, "C", ingoing, node_degree-ingoing]) # majority outgoing (source)

    nid+=1

for count, node_degree in enumerate(low_connect_node_degree):

    if count%3 == 0:
        ingoing = node_degree//2
        nodes.append([nid, "D", ingoing, node_degree - ingoing]) # balanced
    
    if count%3 == 1:
        outgoing = node_degree//3
        nodes.append([nid, "E", node_degree - outgoing, outgoing]) # majority ingoing (sink)
   
    if count%3 == 2:
        ingoing = node_degree//3
        nodes.append([nid, "F", ingoing, node_degree-ingoing]) # majority outgoing (source)
    nid+=1

print(nodes)
in_degrees_all_nodes = np.array([n[2] for n in nodes])
out_degrees_all_nodes = np.array([n[3] for n in nodes])
print(in_degrees_all_nodes.sum(), out_degrees_all_nodes.sum())

D = nx.directed_configuration_model(in_degrees_all_nodes, out_degrees_all_nodes)
"""