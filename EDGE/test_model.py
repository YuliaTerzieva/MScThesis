import pickle
import torch
from datasets.data import get_data
import torch_geometric as pyg
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Model
from model import get_model

which_run = "./wandb/Small_test_no_anomaly/multinomial_diffusion/multistep/2025-03-10_17-56-42"

path = which_run+"/check/checkpoint_44.pt"
num_samples = 1
Monte_Carlo = 10000

path_args = which_run+ "/args.pickle"
with open(path_args, 'rb') as f:
    args = pickle.load(f)

# print(args)
args.device = 'cpu'
train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics = get_data(args)

model = get_model(args, initial_graph_sampler=initial_graph_sampler)
checkpoint = torch.load(path, map_location=args.device, weights_only=False)
model.load_state_dict(checkpoint['model'])

if torch.cuda.is_available():
    model = model.to(args.device)

model.eval()

# print(model) # BinomialDiffusionActive _denoise_fn TGNN_degree_guided

# sample 
original_graphs, sampled_pygraph, per_graph_edge_list_counter = model.sample_and_MC(num_samples, w = 0.4, MC = Monte_Carlo) 

# print(per_graph_edge_list_counter)

mapping = {0: 'blue', 1: 'orange', 2: 'grey'}
fig, axes = plt.subplots(num_samples, 2)
if num_samples == 1:
    axes = np.array(axes).reshape(1, 2)
for count, (OG_graph, generated) in enumerate(zip(original_graphs, sampled_pygraph.to_data_list())):

    OG_node_colors = [mapping[node_class.item()] for node_class in OG_graph.node_attr]
    og_gen = pyg.utils.to_networkx(OG_graph, to_undirected=True)
    pos = nx.arf_layout(og_gen)
    nx.draw(og_gen, pos, ax=axes[count][0],with_labels=True, node_color=OG_node_colors)
    axes[count][0].set_title("Original graph")
    
    generated_node_colors = [mapping[node_class.item()] for node_class in generated.node_attr]
    g_gen = pyg.utils.to_networkx(generated, to_undirected=True)

    # new code possible wrong
    g_gen.clear_edges()
    g_gen.remove_edges_from(list(g_gen.edges))
    edges_with_weights = [(u, v, {'probability': count / Monte_Carlo}) for (u, v), count in per_graph_edge_list_counter[count].items()]
    g_gen.add_edges_from(edges_with_weights)
    edge_labels = {(u, v): f"{data['probability']:.2f}" for u, v, data in g_gen.edges(data=True)}


    pos = nx.arf_layout(g_gen)
    nx.draw(g_gen, pos, ax=axes[count][1], with_labels=True, node_color=generated_node_colors)
    nx.draw_networkx_edge_labels(g_gen, pos, ax=axes[count][1], edge_labels=edge_labels, font_color='gray')
    axes[count][1].set_title("Generated graph")

plt.tight_layout()
plt.show()
