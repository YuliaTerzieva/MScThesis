import pickle
import torch
from datasets.data import get_data
import torch_geometric as pyg
import networkx as nx
import matplotlib.pyplot as plt

# Model
from model import get_model

#"./wandb/Big_Ego_Nets_non_dir/multinomial_diffusion/multistep/2025-02-18_11-24-25/check/checkpoint_49.pt"
path ="./wandb/140_nodes_graph/multinomial_diffusion/multistep/2025-03-03_13-16-32/check/checkpoint_49.pt"
num_samples = 3

path_args = "./wandb/140_nodes_graph/multinomial_diffusion/multistep/2025-03-03_13-16-32/args.pickle"
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

print(model) # BinomialDiffusionActive _denoise_fn TGNN_degree_guided

# sample 
# breakpoint()
sampled_pygraph = model.sample(num_samples) 
# print(type(sampled_pygraph)) # <class 'abc.DataBatch'>
pyg_datas = sampled_pygraph.to_data_list()
generated_nxgraphs = []

for pyg_data in pyg_datas:
    g_gen = pyg.utils.to_networkx(pyg_data, to_undirected=True)

    nx.draw(g_gen)
    plt.show()
