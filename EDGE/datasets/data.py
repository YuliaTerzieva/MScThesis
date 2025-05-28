import math
import torch
import os 
import networkx as nx
import numpy as np

import pickle as pkl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.data import data
import torch_geometric as pyg
import random
from functools import partial
from torch_geometric.datasets import QM9
from datasets.data_utils import EmpiricalEmptyGraphGenerator, NeuralEmptyGraphGenerator, EmptyGraphGeneratorWithNodeAttributes, preprocess, collate_fn, FEATURE_EXTRACTOR
# from datasets.evaluator import NetworkEvaluator, GenericGraphEvaluator


class NetworkDataset(Dataset):
    def __init__(self, pyg_graph, num_iter, transform=None):
        super().__init__()
        self.pyg_data = pyg_graph
        self.transform = transform
        self.num_iter = num_iter

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.pyg_graph)
        return self.pyg_data

    def __len__(self):
        return self.num_iter


class GraphDataset(Dataset):
    def __init__(self, pyg_datas):
        super().__init__()
        self.pyg_datas = pyg_datas

    def __getitem__(self, index):
        return self.pyg_datas[index]#, self.denses[index]

    def __len__(self):
        return len(self.pyg_datas)


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dim_node_attr', type = int)
    # Train params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_iter', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=eval, default=True)

    parser.add_argument('--empty_graph_sampler', type=str, default='empirical', help='empirical | neural') 
    parser.add_argument('--degree', action='store_true') # Yulia : this means that if the --degree flag is provided, the argument will be set to True; otherwise False
    parser.add_argument('--augmented_features', type=str, nargs="*", default=[])
    parser.add_argument('--p_uncon', type=float,  default=0.5, help="the probability of dropping the node features during training") # Added by Yulia

def get_data_id(args):
    return '{}'.format(args.dataset)

def get_data(args):
    num_node_classes = args.dim_node_attr # here I have 3 (blue, orange and gray) or 30 for cora
    num_edge_classes = 2 # no edge / edge
    num_node_feat = 1 
    
    train_nx_graphs = pkl.load(open(f"../GeneratedDataset/{args.dataset}_train", 'rb'))
    eval_nx_graphs = pkl.load(open(f"../GeneratedDataset/{args.dataset}_eval", 'rb'))

    max_degree = max([max([d for n, d in train_nx_graph.degree()]) for train_nx_graph in train_nx_graphs])

    train_pygraphs = []
    eval_pygraphs = []

    for nx_graph in train_nx_graphs:
        # Yulia: This preprocessing is to turn the data from networkx object to a pyg object
        pyg_data = preprocess(nx_graph, degree=args.degree, p_uncon = args.p_uncon)
        train_pygraphs.append(pyg_data)

    for nx_graph in eval_nx_graphs:
        pyg_data = preprocess(nx_graph, degree=args.degree)
        eval_pygraphs.append(pyg_data)
        
    train_set = GraphDataset(train_pygraphs)
    eval_set = GraphDataset(eval_pygraphs)
    
    if args.empty_graph_sampler == 'file':
        initial_graph_sampler = EmptyGraphGeneratorWithNodeAttributes(file_path = args.dataset + "_test") # TODO : Yulia : add parameter empty_graphs_file_name to get from the arguments and pass here!
    else : 
        raise NotImplementedError

    monitoring_statistics = ['bpd']

    augmented_feature_dict = {}

    # Data Loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=partial(collate_fn))
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)

    return train_loader, eval_loader, None, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, None, None, monitoring_statistics
 
