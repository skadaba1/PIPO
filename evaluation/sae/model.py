# 1. Define the Sparse Autoencoder
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sparsity_weight=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.activation = nn.ReLU()
        self.sparsity_weight = sparsity_weight
        self.device = device
        self.to(device)

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)

        # Encode
        latent = self.encoder(x)
        latent_activated = self.activation(latent)

        # Decode
        reconstruction = self.decoder(latent_activated)

        return reconstruction, latent_activated

    def l1_loss(self, latent_activations):
        return self.sparsity_weight * torch.mean(torch.abs(latent_activations))