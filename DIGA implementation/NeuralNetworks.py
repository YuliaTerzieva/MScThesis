import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GATv2Conv


class GAT(nn.Module):
    def __init__(self, in_channel, hidden_channels, output_channel):

        self.conv1 = GATv2Conv(in_channel+1, hidden_channels, dropout=0.1) # the +1 is because we have the time : bottom right of page 4408
        self.conv2 = GATv2Conv(hidden_channels, output_channel, dropout=0.1)                                   
                                   
    def forward(self, x, edge_index):

        h1 = self.conv1(x, edge_index)
        a1 = f.leaky_relu(h1, 0.1)
        h2 = self.conv2(a1, edge_index)
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
            nn.Sigmoid()
       )

                                             
    def forward(self, G_c, Z_c_t, t):

        x = torch.cat(Z_c_t, torch.fill(size = Z_c_t.shape[0], value=t), dim = 1)

        _, output = self.gnn(x, G_c)
        output = self.mlp(output)

        return output
    
#################################################################################
    
class Guiding_classifier(nn.Module):
    def __init__(self, gnn, in_channel, hidden_channels, output_channel, l): 
        super(GAT, self)

        self.gnn = gnn
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channel, hidden_channels), 
            nn.Sigmoid()
            )
        self.mlp2 = nn.Sequential(
            nn.Linear(in_channel, output_channel), 
            nn.Sigmoid()
            )
        self.l = l

        

                                             
    def forward(self, G_c, Z_c_t, t):

        x = torch.cat(Z_c_t, torch.fill(size = Z_c_t.shape[0], value=t), dim = 1)

        a1, a2 = self.gnn(x, G_c)

        s1 = self.l * a1 + (1-self.l) * self.mlp1(Z_c_t)
        readout1 = torch.cat()

        s2 = self.l * a2 + (1-self.l) * self.mlp2(Z_c_t)
        readout2 = 

        return 