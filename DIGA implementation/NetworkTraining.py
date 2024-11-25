
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
import numpy as np
import torch
import torch.nn as nn
import random

def ego_network_ppr(data, center_nodes, k_hop=2, top_k=4):
    """
    Extract the k-hop ego network for multiple center nodes and compute the PPR to return the top-k nodes subgraph for each.

    Args:
        data (torch_geometric.data.Data): Input graph as a PyTorch Geometric Data object.
        center_nodes (torch.Tensor): A tensor of indices of the center nodes.
        k_hop (int): Number of hops for the ego network (default: 2).
        top_k (int): Number of top nodes to retain based on PPR values (default: 4).

    Returns:
        list[Data]: A list of PyTorch Geometric Data objects, one for each center node's induced subgraph.
    """
    subgraphs = []

    for center_node in center_nodes:
        # Step 1: Extract the k-hop ego network
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            center_node.item(), k_hop, data.edge_index, relabel_nodes=True
        )

        # Extract the node features and edge attributes for the ego network
        if data.x is not None:
            ego_x = data.x[subset]
        else:
            ego_x = None

        if data.edge_attr is not None:
            ego_edge_attr = data.edge_attr[edge_mask]
        else:
            ego_edge_attr = None

        # Create a new Data object for the ego network
        ego_data = Data(
            x=ego_x,
            edge_index=edge_index,
            edge_attr=ego_edge_attr,
            num_nodes=len(subset)
        )

        # Step 2: Compute PPR on the ego network
        num_nodes = ego_data.num_nodes
        edge_index = ego_data.edge_index

        # Create adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes))
        adj[edge_index[0], edge_index[1]] = 1

        # Initialize PPR scores
        ppr = torch.zeros(num_nodes)
        ppr[mapping] = 1.0  # Start with all the mass at the center node

        # Define damping factor and number of iterations
        alpha = 0.85
        max_iter = 100
        tol = 1e-6

        # Power iteration to compute PPR
        for _ in range(max_iter):
            prev_ppr = ppr.clone()
            ppr = alpha * torch.matmul(adj.T, ppr) + (1 - alpha) * torch.zeros_like(ppr)
            ppr[mapping] += (1 - alpha)  # Add back teleportation to the center node
            if torch.norm(ppr - prev_ppr, p=1) < tol:
                break

        # Step 3: Select top-k nodes based on PPR
        top_k_indices = torch.topk(ppr, top_k).indices

        # Extract the subgraph induced by top-k nodes
        induced_edge_index, induced_edge_attr = subgraph(
            top_k_indices,
            ego_data.edge_index,
            edge_attr=ego_data.edge_attr,
            num_nodes=num_nodes,
            relabel_nodes=True
        )

        if ego_data.x is not None:
            induced_node_features = ego_data.x[top_k_indices]
        else:
            induced_node_features = None

        # Create the final Data object
        induced_subgraph = Data(
            x=induced_node_features,
            edge_index=induced_edge_index,
            edge_attr=induced_edge_attr,
            num_nodes=top_k
        )

        subgraphs.append(induced_subgraph)

    return subgraphs



class DIGA_trainer:
    def __init__(self, denoise_network, guiding_classifier, dataset, optimizer, training_step, batch_size, epochs, device, max_norm, pt_model_file_name, tau = 0.8, T_diffusion = 20):
        self.denoise_network = denoise_network
        self.guiding_classifier = guiding_classifier
        self.dataset = dataset
        self.optimizer = optimizer
        self.training_step = training_step
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.max_norm = max_norm
        self.pt_model_file_name = pt_model_file_name
        self.tau = tau
        self.T_diffusion = T_diffusion

        self.patience = 5
        self.min_delta = 1e-2
        self.best_epoch_loss = float('inf')
        self.plauteu_counter = 0

        self.beta = torch.linspace(10**-4, 0.02, self.T_diffusion)
        self.alpha = 1-self.beta
        self.alpha_bar = torch.cumprod(self.alpha)

        self.diffusion_loss = []
        self.guidence_loss = []

    
    def train_model(self):


        for epoch in range(self.epochs):

            self.denoise_network.train()
            self.guiding_classifier.train()

            total_epoch_loss = 0
            number_of_batches = 0

            for i in self.training_step:
                data = self.dataset.get_data(time_step = i).to(self.device)
                transform = ToUndirected()
                data_undirected = transform(data)


                train_loader = NeighborLoader(
                    data = data_undirected, 
                    num_neighbors = [-1]*2, 
                    replace = False, 
                    shuffle = False, 
                    batch_size = self.batch_size
                )

                for batch in train_loader:
                    batch = batch.to(self.device)
                    target_nodes = batch.n_id[:batch.batch_size]

                    t = torch.randint(1, self.T_diffusion + 1, (self.batch_size, 1))
                    
                    noise = torch.randn((self.batch_size, 1))

                    Z_c_t = torch.sqrt(self.alpha_bar[t]) * batch.x + torch.sqrt(1 - self.alpha_bar[t]) * noise

                    if np.random.rand(1)[0] > 0.8:

                        bce_loss = nn.BCEloss()
                        self.optimizer.zero_grad()

                        G_c = ego_network_ppr(batch.x, target_nodes)

                        predicted_labels = self.guiding_classifier(G_c, Z_c_t, t)

                        true_labels = [int(data.y[data.n_id == target_node]) for target_node in target_nodes]

                        loss = bce_loss(predicted_labels, true_labels)
                        loss.backwards()

                        nn.utils.clip_grad_norm_(self.guiding_classifier.parameters(), max_norm=self.max_norm)
                        self.optimizer.step()

                        total_epoch_loss += loss.item()
                        self.guidence_loss.append(loss.item())
                        number_of_batches += 1

                    else : 

                        mse_loss = nn.MSELoss()
                        self.optimizer.zero_grad()

                        G_c = ego_network_ppr(batch.x, target_nodes)

                        predicted_noise = self.denoise_network(G_c, Z_c_t, t)

                        loss = mse_loss(predicted_noise, noise)
                        loss.backwards()

                        nn.utils.clip_grad_norm_(self.guiding_classifier.parameters(), max_norm=self.max_norm)
                        self.optimizer.step()

                        total_epoch_loss += loss.item()
                        self.guidence_loss.append(loss.item())
                        number_of_batches += 1


            # Early stopping based on loss platau 
            
            avg_epoch_loss = total_epoch_loss / number_of_batches
            print("...")

            if self.best_epoch_loss - avg_epoch_loss > self.min_delta:
                None
                

                        













