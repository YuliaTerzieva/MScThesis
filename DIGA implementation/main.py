from NeuralNetworks import GAT, Denoising_network, Guiding_classifier
from NetworkTraining import DIGA_trainer, ego_network_ppr
import torch
import numpy as np




# Example usage
if __name__ == "__main__":

    # Initialisation -------------
    in_channel = hidden_channels = output_channel = 10 # random guess check elliptic ++
    guiding_lambda = 0.6 # double check page 4410 section 5.1.4

    base_GNN = GAT(in_channel, hidden_channels, output_channel)
    denoising_network = Denoising_network(base_GNN, output_channel)
    guiding_network = Guiding_classifier(base_GNN, in_channel, hidden_channels, output_channel, guiding_lambda)

    dataset = None
    optimizer = None
    training_step = None
    batch_size = 64
    epochs = None # many
    device = 'mps'
    max_norm = None
    pt_model_file_name = "DIGA"

    s = 3 # this is the gradient scale 𝑠. based on Figure 3 right most plot i selected 3. Double check! 

    # Training -------------

    model_trainer = DIGA_trainer(denoising_network, guiding_network, dataset, optimizer, training_step, batch_size, epochs, device, max_norm, pt_model_file_name)
    model_trainer.train_model()


    # Inference -------------

    denoising_network.eval()
    guiding_network.eval()

    T = 20

    eval_data = None
    eval_nodes = None

    G_c = ego_network_ppr(eval_data.x, eval_nodes)
    Z_c_T = torch.sqrt(model_trainer.alpha_bar[T]) * eval_data.x +  torch.sqrt(1 - model_trainer.alpha_bar[T]) * torch.randn((eval_data.x.shape[0], 1))
    
    for t in np.linspace(20, 1, 20):
        gradient = torch.autograd.grad()
        pred_noise = denoising_network(G_c, Z_c_T, t) - s * torch.sqrt(1 - model_trainer.alpha_bar[t]) * gradient



