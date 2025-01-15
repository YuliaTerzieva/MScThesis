import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GATv2Conv
import numpy as np


class GAT(nn.Module):
    def __init__(self, in_channel, hidden_channels, output_channel):

        """
            In this case the channels are the dimention of the node features? 
        """

        self.conv1 = GATv2Conv(in_channel+1, hidden_channels, dropout=0.1) # the +1 is because we have the time : bottom right of page 4408
        self.conv2 = GATv2Conv(hidden_channels, output_channel, dropout=0.1)                                   
                                   
    def forward(self, x, edge_index):

        h1 = self.conv1(x, edge_index)
        a1 = f.leaky_relu(h1, 0.1) # number of nodes x  hidden_dimention
        h2 = self.conv2(a1, edge_index) # number of nodes x  output_dimention
        a2 = f.leaky_relu(h2, 0.1)

        return a1, a2 

#################################################################################

class Denoising_network(nn.Module):

    def __init__(self, gnn, output_channel):
        super(GAT, self)

        self.gnn = gnn
        self.mlp = nn.Sequential(
            nn.Linear(output_channel, 32), 
            nn.LeakyReLU(0.1), 
            nn.Linear(32, 16), 
            nn.LeakyReLU(0.1), 
            nn.Linear(16, 1, bias=False), 
            nn.Sigmoid() # We need to see what is the range of the noise
       )

                                             
    def forward(self, G_c, Z_c_t, t):

        x = torch.cat(Z_c_t, torch.fill(size = Z_c_t.shape[0], value=t), dim = 1)

        _, output = self.gnn(x, G_c)
        output = self.mlp(output)

        return output
    
#################################################################################
    
class Guiding_classifier(nn.Module):
    def __init__(self, gnn, in_channel, hidden_channels, output_channel, l): # l is lambda
        super(GAT, self)

        self.gnn = gnn
        self.mlp_h1 = nn.Sequential(
            nn.Linear(hidden_channels, 1), 
            nn.Sigmoid()
            )
        self.mlp_h2 = nn.Sequential(
            nn.Linear(output_channel, 1), 
            nn.Sigmoid()
            )
        self.mlp_z = nn.Sequential(
            nn.Linear(in_channel, 1), 
            nn.Sigmoid()
            )
        self.mlp_readout = nn.Sequential(
            nn.Linear(2*output_channel, 1), 
            nn.Sigmoid()
            )

        self.l = l

    
    def forward(self, G_c, Z_c_t, t):

        x = torch.cat(Z_c_t, torch.fill(size = Z_c_t.shape[0], value=t), dim = 1)

        a1, a2 = self.gnn(x, G_c)

        # This is layer 1

        s1 = self.l * self.mlp_h1(a1) + (1-self.l) * self.mlp_z(Z_c_t) # self.mlp_h1(a1) output is number_of_nodes x hidden_channels; so then s1 should be : number of nodes x 1

        _, top2node_indeces = torch.topk(s1, k=2)

        top2node = a1[top2node_indeces]

        readout1 = torch.cat(torch.mean(top2node, dim = 0), torch.max(top2node, dim = 0))

        # this is layer 2

        s2 = self.l * self.mlp_h2(a2) + (1-self.l) * self.mlp_z(Z_c_t) 

        _, top1node_indeces = torch.topk(s2, k=1)

        top1node = a2[top1node_indeces]

        readout2 = torch.cat(torch.mean(top1node, dim = 0), torch.max(top1node, dim = 0))
        test = torch.cat(top1node, top1node)

        assert readout2 == test

        readout_sum = readout1 + readout2

        return self.mlp_readout(readout_sum)
    

    """
    Equation 9 : s h and the output of the MLP are all a single number, no an array. 
    We need to calculate s for each node and then select the top N ones. 
    We can see from Eq. 7 that h there is also a single number (in that equation)
    The implementation however needs to be for an array. The right side of eq 9 is the same for all nodes. 
    so we need to calculate s as given and then select top N nodes. Is that indeed 2 and 1 node for us? That seems rediculously low!
    Because we need to do the read out only on the nodes left after the pooling (so we do a readout on 2 nodes and then on 1?). 
    The last readout is the same number :D

    How is the sum done? How do we sum the readouts?
    """

    """
    
        We came to the comclusion that all the channels in the GNN must be the same (i.e. the input, hidden and output features should be the same dimention).
        The reason for that is because we need to sum all the readouts, which would only be possible if all the layers have the same dim. 

    """