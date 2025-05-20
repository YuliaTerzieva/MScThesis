import torch
import random
import networkx as nx
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import to_dense_adj
from torch_geometric import transforms as T
from torch.nn import functional as F
from layers.layers import BitModel, NodeModel
import pickle as pkl

def preprocess(g, degree=True, p_uncon = None):
    if isinstance(g, nx.Graph):
        pyg_data = pyg.utils.from_networkx(g)
        adj = torch.from_numpy(nx.to_numpy_array(g).astype(np.int64)).long() # Yulia : changed "int" to "int64"
    elif isinstance(g, pyg.data.Data):
        pyg_data = g
        adj = to_dense_adj(g.edge_index)[0].long()
    else:
        raise NotImplementedError()
    
    if hasattr(pyg_data, 'anomalous'):
        del pyg_data.anomalous

    if hasattr(pyg_data, 'edge_anomalous'):
        del pyg_data.edge_anomalous
        
    if hasattr(pyg_data, 'ano_label'):
        del pyg_data.ano_label
        del pyg_data.str_ano_label
        del pyg_data.attr_ano_label
        del pyg_data.weight

    row, col = torch.triu_indices(pyg_data.num_nodes, pyg_data.num_nodes,1)
    # Yulia : full_edge_index has shape (2 by N) it is all the possible undirected edges that can be added to a graph with N  = (number of nodes ^ 2 - number of nodes)/2
    # note we are wokring with undirectional graph with no self loops!
    pyg_data.full_edge_index = torch.stack([row, col])

    # Yulia : full_edge_attr has shape (1 by N), it is based on the possible edges from the full_edge_index, 
    # there are 1s for the edges that actually occur in the graph and 0s for the rest of the edges
    pyg_data.full_edge_attr = adj[pyg_data.full_edge_index[0], pyg_data.full_edge_index[1]]

    if not hasattr(pyg_data, 'node_attr') or (p_uncon is not None and torch.rand(1) < p_uncon): # This part is changed by Yulia, based on Algorithm 1 in the thesis 
        # TODO check if the algorithm number is still correct
        # pyg_data.node_attr = torch.full([pyg_data.num_nodes], -1, dtype=torch.int) # this is for my dataset 
        # breakpoint()
        pyg_data.node_attr = torch.full(pyg_data.node_attr.shape, 0, dtype=torch.int) # this is for Cora dataset

    if degree:
        # Yulia : these are the degrees for each node in the graph
        pyg_data.degree = pyg.utils.degree(pyg_data.edge_index[0]).long() # make sure edge_index is bi-directional
    
    
    return pyg_data

def collate_fn(pyg_datas, repeat=1):
    """
    Yulia : this function cloned the original pyg_datas repeat times
    then it batched it and adds two featues to the batch nodes_per_graph and edges_per_graph, 
    where edges_per_graph is the total possible edges a graph with this many nodes can have

    """
    # breakpoint()
    pyg_datas = sum([[pyg_data.clone() for _ in range(repeat)]for pyg_data in pyg_datas],[])
    batched_data = pyg.data.Batch.from_data_list(pyg_datas)
    batched_data.nodes_per_graph = torch.tensor([pyg_data.num_nodes for pyg_data in pyg_datas])
    batched_data.edges_per_graph = torch.tensor([pyg_data.num_nodes * (pyg_data.num_nodes-1)//2 for pyg_data in pyg_datas])

    return batched_data 

class EmptyGraphGeneratorWithNodeAttributes:

    def __init__(self, file_path):

        self.testing_graphs_nx = pkl.load(open(f"../GeneratedDataset/{file_path}", 'rb'))
        # self.testing_graphs = [preprocess(graph, degree=True) for graph in testing_graphs_nx]

    def _fill_needed_features(self, graphs):
        """
        This function takes a list of graphs and removes the edge_index attribute of all of them. 
        """
        return_data_list = []

        for graph in graphs:
            graph.edge_index=None
            return_data_list.append(graph)

        batched_data = collate_fn(return_data_list)
        
        return batched_data

    def sample(self, num_samples):
        """
        This function samples "num_samples" from a list of graphs.
        The list is created from graphs in the test section of the dataset
        The graphs already have the following features : 
            - full_edge_index
            - full_edge_attr
            - node_attr
            - degree
        Then, other features are added to the pyg.data.Data() object 
        that are required for the algorithm.
        The features that are added/removed are : 
            - edge_index is removed 
            - nodes_per_graph is added
            - edges_per_graph is added
        """
        if num_samples == len(self.testing_graphs_nx):
            sampled_graphs = self.testing_graphs_nx
        elif type(num_samples) == int:
            sampled_graphs = random.choices(self.testing_graphs_nx, k=num_samples)
        else : # if case the num_samples are a list of indices as in sample_and_MC
            sampled_graphs = [self.testing_graphs_nx[i] for i in num_samples]
        
        # I first sample, then I clone
        # the original sampled graphs don't need to be preprocessed, only truened into pygeometric
        # the one that owul be used for inference need to be preprocessed

        sampled_graphs_clone = [graph.copy() for graph in sampled_graphs]

        sampled_graphs_clone = [preprocess(graph, degree=True) for graph in sampled_graphs_clone]
        sampled_graphs = [pyg.utils.from_networkx(graph) for graph in sampled_graphs]

        empty_pyg_datas = self._fill_needed_features(sampled_graphs_clone)
        return sampled_graphs, empty_pyg_datas

#--------------------------------------------------------------------------------------
# THIS CODE IS FOR THE ORIGINAL EDGE ALGORITHM

FEATURE_EXTRACTOR = {
}

def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    
def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def unpack_deg_matrix(degs):
    res = []
    for deg in degs:
        deg = deg.long().tolist()
        r = []
        for d in deg:
            if (sum(r)==0) or (d > 0):
                r.append(d)
        res.append(r)
    return res

def deg_hist_to_deg_seq(deg_hist):
    ret = torch.zeros(sum(deg_hist))
    cum = 0
    for d, num_nodes in enumerate(deg_hist):
        ret[cum:cum+num_nodes] = d+1
        cum = cum+num_nodes
    return ret


class EmpiricalEmptyGraphGenerator:
    def __init__(self, train_pyg_datas, degree=False, augment_features=[]):
        # pmf of graph size
        num_nodes = torch.tensor([pyg_data.num_nodes for pyg_data in train_pyg_datas])

        self.min_node = num_nodes.min().long().item()
        self.max_node = num_nodes.max().long().item()

        unnorm_p = torch.histc(num_nodes.float(), bins=self.max_node-self.min_node+1)

        self.empirical_graph_size_dist = unnorm_p/unnorm_p.sum()

        # empty graph table
        self.empty_graphs = {}

        # degree table
        self.degree = degree
        self.augment_features = augment_features

        self.empirical_node_feat_dist = {}

        for pyg_data in train_pyg_datas:
            if pyg_data.num_nodes not in self.empirical_node_feat_dist:
                self.empirical_node_feat_dist[pyg_data.num_nodes] = []
            feats = {}
            if self.degree:
                feats['degree'] = pyg.utils.degree(pyg_data.edge_index[0],num_nodes=pyg_data.num_nodes)
            for feat_name in self.augment_features:
                feats[feat_name] = getattr(pyg_data, feat_name)# FEATURE_EXTRACTOR[feat_name]['func'](pyg_data)
            # feats['x'] = pyg_data.x
            self.empirical_node_feat_dist[pyg_data.num_nodes].append(feats)


    def _sample_graph_size_and_features(self, num_samples):
        ret = self.empirical_graph_size_dist.multinomial(num_samples=num_samples, replacement=True) + self.min_node
        ret = ret.tolist()
        xT_feats = [] 
        for n_node in ret:
            xT_feats.append(random.choice(self.empirical_node_feat_dist[n_node]))
        # xT_feats will be a list of dicts
        return ret, xT_feats

    def _generate_empty_data(self, num_node_per_graphs, xT_feats):
        """
        this funcition receives a list of integers num_node_per_graphs and 
        a list of dictionary, with degree key showing the degree of each node
        """
        # breakpoint()
        return_data_list = []

        for num_node, xT_feat in zip(num_node_per_graphs, xT_feats):
            if num_node not in self.empty_graphs:
                pyg_data = pyg.data.Data()
                row, col = torch.triu_indices(num_node, num_node,1)
                pyg_data.full_edge_index = torch.stack([row, col])

                pyg_data.full_edge_attr = torch.zeros((pyg_data.full_edge_index[0].shape[0],), dtype=torch.long)
                pyg_data.node_attr = torch.zeros((num_node,), dtype=torch.long)

                pyg_data.num_nodes = num_node
                self.empty_graphs[num_node] = pyg_data

            pyg_data = self.empty_graphs[num_node].clone()
            for feat_name in xT_feat:
                setattr(pyg_data, feat_name, xT_feat[feat_name])
            
            return_data_list.append(pyg_data)

        batched_data = collate_fn(return_data_list)
        return batched_data

    def sample(self, num_samples): # return type is -> class 'abc.DataBatch'
        num_node_per_graphs, xT_feats = self._sample_graph_size_and_features(num_samples)
        # print(f"I made a graph with the following numbers of nodes {num_node_per_graphs} with features {xT_feats}")
        empty_pyg_datas = self._generate_empty_data(num_node_per_graphs, xT_feats)

        return empty_pyg_datas

class NeuralEmptyGraphGenerator:
    def __init__(self, train_pyg_datas, neural_attr_sampler, degree=False, device='cuda:0'):
        # now only support degree features, other features are left to future.

        num_nodes = torch.tensor([pyg_data.num_nodes for pyg_data in train_pyg_datas])

        self.min_node = num_nodes.min().long().item()
        self.max_node = num_nodes.max().long().item()

        unnorm_p = torch.histc(num_nodes.float(), bins=self.max_node-self.min_node+1)
        # empty graph table
        self.empty_graphs = {}
        self.empirical_graph_size_dist = unnorm_p/unnorm_p.sum()
        self.degree = degree
        self.neural_attr_sampler = neural_attr_sampler
        self.device = device
        
        self.node_model = NodeModel(num_bits=neural_attr_sampler['NUM_BITS'], max_num_nodes=neural_attr_sampler['MAX_NUM_NODES'], seq_lens=neural_attr_sampler['SEQ_LENS'])
        self.bit_model = BitModel(num_bits=neural_attr_sampler['NUM_BITS'], max_num_nodes=neural_attr_sampler['MAX_NUM_NODES'])
        
        self.node_model.to(self.device)
        self.bit_model.to(self.device)

        self.node_model.load_state_dict(neural_attr_sampler['modelNode'])
        self.bit_model.load_state_dict(neural_attr_sampler['modelBit'])
 
    def _sample_graph_size_and_features(self, num_samples):
        ret = self.empirical_graph_size_dist.multinomial(num_samples=num_samples, replacement=True) + self.min_node
        if self.degree:
            x = torch.zeros(num_samples, 1, self.neural_attr_sampler['NUM_BITS'])
            g = r = ret[:, None]
            x = x.to(self.device)
            g = g.to(self.device)
            r = r.to(self.device)
            self.node_model.eval()
            self.bit_model.eval()
            with torch.no_grad():
                for i in range(self.neural_attr_sampler['SEQ_LENS']):      
                    node_hidden = self.node_model(x, g, r)[:,-1,:]
                    y = (torch.ones(num_samples, 1).long().to(self.device)*2).long()

                    for j in range(self.neural_attr_sampler['NUM_BITS']):
                        prediction = self.bit_model(y.view(-1, j+1), node_hidden.view(-1, node_hidden.shape[-1]), r[:,-1][:,None])[:,-1,:]
                        prediction = F.sigmoid(prediction)
                        index = prediction.bernoulli().long()
                        y = torch.cat([y, index],dim=-1)
                    y = y[:, 1:]
                    n_j = bin2dec(y, self.neural_attr_sampler['NUM_BITS'])-1
                    r = torch.cat([r, (r[:, -1]-n_j)[:,None]],dim=-1)
                    x = torch.cat([x, y[:,None,:]], dim=1)
                
                x = (bin2dec(x, self.neural_attr_sampler['NUM_BITS'])-1).clamp(0)[:, 1:]
            xT_feats = unpack_deg_matrix(x)
            ret = [sum(xT_feat) for xT_feat in xT_feats]
            xT_feats = [{'degree':deg_hist_to_deg_seq(xT_feat)} for xT_feat in xT_feats]
        else:
            xT_feats = [{} for _ in ret]
            ret = ret.tolist()
        return ret, xT_feats

    def _generate_empty_data(self, num_node_per_graphs, xT_feats):
        return_data_list = []

        for num_node, xT_feat in zip(num_node_per_graphs, xT_feats):
            if num_node not in self.empty_graphs:
                pyg_data = pyg.data.Data()
                row, col = torch.triu_indices(num_node, num_node,1)
                pyg_data.full_edge_index = torch.stack([row, col])

                pyg_data.full_edge_attr = torch.zeros((pyg_data.full_edge_index[0].shape[0],), dtype=torch.long)
                pyg_data.node_attr = torch.zeros((num_node,), dtype=torch.long)

                pyg_data.num_nodes = num_node
                self.empty_graphs[num_node] = pyg_data

            pyg_data = self.empty_graphs[num_node].clone()
            for feat_name in xT_feat:
                setattr(pyg_data, feat_name, xT_feat[feat_name])
            
            return_data_list.append(pyg_data)

        batched_data = collate_fn(return_data_list)
        return batched_data

    def sample(self, num_samples):
        num_node_per_graphs, xT_feats = self._sample_graph_size_and_features(num_samples)
        empty_pyg_datas = self._generate_empty_data(num_node_per_graphs, xT_feats)
        return empty_pyg_datas
