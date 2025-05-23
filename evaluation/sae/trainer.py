from train_utils import get_cosine_schedule_with_warmup
import torch.nn as nn
import torch
import torch.optim as optim

# 4. Function to train the SAE
def train_sae(sae, activations, batch_size=256, num_epochs=50, lr=5e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_epochs*len(activations)//batch_size)
    # Ensure activations are on the correct device
    activations = activations.to(device)

    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            # Get batch of activations (already on device)
            act_batch = batch[0]

            # Forward pass
            reconstructions, latents = sae(act_batch)

            # Calculate losses
            reconstruction_loss = mse_loss(reconstructions, act_batch)
            sparsity_loss = sae.l1_loss(latents)
            loss = reconstruction_loss + sparsity_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.6f}")

    return sae