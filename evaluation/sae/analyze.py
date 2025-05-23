import torch 
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import SparseAutoencoder
from trainer import train_sae

# 2. Create a hook to extract activations from a specific layer
def get_activation_extractor(model, target_layer_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
    activations = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            # For attention layers that return (attention_output, attention_weights)
            # Usually we want the first element
            activations['value'] = output[0].detach().to(device)
        else:
            # For layers that return a single tensor
            activations['value'] = output.detach().to(device)

    # Attach hook to the target layer
    for name, module in model.named_modules():
        if name == target_layer_name:
            module.register_forward_hook(hook_fn)

    return activations

# 3. Function to collect activations from a dataset
def collect_activations(model, dataset_loader, activation_extractor, max_samples=10000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    activations_list = []
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataset_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass through the model to trigger hooks
            model(**batch)
            # Get the activations (already on device from the hook)
            batch_activations = activation_extractor['value']


            # Reshape to (batch_size * seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = batch_activations.shape
            reshaped_activations = batch_activations.view(-1, hidden_dim)

            activations_list.append(reshaped_activations)

            sample_count += reshaped_activations.shape[0]
            if sample_count >= max_samples:
                break

    # Concatenate all activations (all on device already)
    all_activations = torch.cat(activations_list, dim=0)

    # Subsample if we have too many
    if all_activations.shape[0] > max_samples:
        indices = torch.randperm(all_activations.shape[0], device=device)[:max_samples]
        all_activations = all_activations[indices]

    return all_activations

from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and pi * num_cycles after a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# 5. Main function to extract and analyze features
def analyze_with_sae(model, tokenizer, dataset, target_layer, latent_dim=256, num_epochs=50, lr=5e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Move model to device
    model = model.to(device)

    # 5.1 Set up activation extraction
    activation_extractor = get_activation_extractor(model, target_layer, device)

    # 5.2 Create DataLoader
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 5.3 Collect activations
    activations = collect_activations(model, data_loader, activation_extractor, device=device)

    # 5.4 Create and train SAE
    input_dim = activations.shape[1]  # Hidden dimension of the layer
    sae = SparseAutoencoder(input_dim, latent_dim, device=device)
    trained_sae = train_sae(sae, activations, device=device, num_epochs=num_epochs, lr=lr)

    # 5.5 Analyze feature patterns
    # You can now use trained_sae to analyze patterns in new inputs

    return trained_sae, activations

# Analyze which features activate for specific protein motifs
# 7. Analyze which features activate for specific protein motifs
def analyze_feature_activations(sae, model, tokenizer, protein_sequence, target_layer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Tokenize the sequence
    inputs = tokenizer(protein_sequence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model activations
    activation_extractor = get_activation_extractor(model, target_layer, device)
    model.eval()
    with torch.no_grad():
        model(**inputs)

    # Get activations and reshape
    activations = activation_extractor['value']
    reshaped_activations = activations.view(-1, activations.shape[-1])

    # Get SAE feature activations
    _, feature_activations = sae(reshaped_activations)

    # Reshape back to (seq_len, num_features)
    feature_map = feature_activations.reshape(activations.shape[1], -1)

    return feature_map