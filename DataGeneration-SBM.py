import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# num_nodes = 10_00
# num_types = 12
# node_type_distribution = np.array([0.15, 0.01, 0.015, 0.23, 0.05, 0.001, 0.075, 0.075, 0.030, 0.014, 0.1, 0.25])
# node_group_sizes = (node_type_distribution * num_nodes).astype(int)
# print(node_group_sizes) 

num_nodes = 10
num_types = 3
node_type_distribution = np.array([0.6, 0.3, 0.2])
node_group_sizes = (node_type_distribution * num_nodes).astype(int)
print(node_group_sizes) 


distribution_generators = [
    lambda size: np.random.beta(1, 3, size),                        # Beta
    lambda size: np.clip(np.random.normal(0.5, 0.3, size), 0, 1),   # Normal
    lambda size: np.random.uniform(0, 1, size),                     # Uniform
    lambda size: np.random.beta(0.5, 0.5, size),                    # Beta
    lambda size: np.random.beta(2, 5, size),                        # Beta
    lambda size: np.random.triangular(0, 0.5, 1, size)              # Triangular
]
distribution_generators_names = ["Beta(0.5, 0.5)", "Normal", "Uniform", "Beta(1, 3)", "Beta(2, 5)", "Triangular"]


# Step 4: Plot distributions for visualization
def plot_distributions(distributions, samples_per_dist=100000):
    plt.figure(figsize=(10, 6))
    for i, dist_func in enumerate(distributions):
        samples = dist_func(samples_per_dist)
        plt.hist(samples, bins= 100, alpha=0.5, density=True, label=f" {distribution_generators_names[i]}")
    plt.legend()
    plt.title("Visualizing Distributions")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

# plot_distributions(distribution_generators)

# print("TEST")
# print([i%len(distribution_generators) for i in range(0, num_types)])


matrix = np.array([distribution_generators[i%len(distribution_generators)](num_types) for i in range(0, num_types)])
np.fill_diagonal(matrix, 0)
normalized_matrix = matrix / matrix.sum(axis=1, keepdims=True)
# print(normalized_matrix.sum(axis = 1))

# print(normalized_matrix)
plot = plt.imshow(normalized_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(plot) 
plt.show()

# Verify row sums
# row_sums = normalized_matrix.sum(axis=1)
# print("\nRow sums (should all be 1):", row_sums)

G = nx.stochastic_block_model(node_group_sizes, normalized_matrix, directed=True)

# Assign node types and random features
for node, data in G.nodes(data=True):
    data['type'] = data['block']

# Add anomalous edges
num_anomalous_edges = 10
for _ in range(num_anomalous_edges):
    u = np.random.randint(0, num_nodes)
    v = np.random.randint(0, num_nodes)
    while v == u:
        v = np.random.randint(0, num_nodes)
    if not G.has_edge(u, v):
        G.add_edge(u, v, anomaly=True)


# Step 6: Visualize the graph
def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    node_colors = [data['type'] for _, data in graph.nodes(data=True)]
    edge_colors = ['red' if data.get('anomaly') else 'black' for _, _, data in graph.edges(data=True)]

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color=node_colors, cmap=plt.cm.Set1)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, alpha=0.7)
    
    # Add labels
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color='white')

    plt.title("Graph with Anomalous Edges Highlighted")
    plt.show()


# Call the visualization function
visualize_graph(G)

adj_matrix = nx.adjacency_matrix(G).todense()
print("Adjacency Matrix:")
print(adj_matrix)



