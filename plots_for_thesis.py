import matplotlib.pyplot as plt
import pickle
import networkx as nx
import numpy as np

with open("GeneratedDataset_interm_graph/Core_node_big_graph", "rb") as f:
    G = pickle.load(f)

degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
dmax = max(degree_sequence)
dmean = np.mean(degree_sequence)
breakpoint()

fig = plt.figure("Degree of Cora Graph", figsize=(10, 4))
axgrid = fig.add_gridspec(1, 4)

ax0 = fig.add_subplot(axgrid[0, 0:2])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos = nx.spring_layout(Gcc, seed=10396953)
# mapping_to_color = {0:'blue', 1: 'orange', 2: 'grey'}
# map_to_color = lambda color: ([mapping_to_color[c] for c in color] if isinstance(color, list) else mapping_to_color[color])
# node_colors = [Gcc.nodes[node]['node_attr'] for node in Gcc.nodes()]
nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20, alpha=0.3) #node_color=map_to_color(node_colors)
nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
ax0.set_title("Graph")
ax0.set_axis_off()

ax2 = fig.add_subplot(axgrid[0, 2:])
ax2.bar(*np.unique(degree_sequence, return_counts=True))
ax2.set_title("Degree histogram")
ax2.set_xlabel("Degree")
ax2.set_ylabel("# of Nodes")
fig.tight_layout()
plt.show()