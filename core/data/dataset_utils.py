import torch
import random

def collate_fn(batch, device):
    # Unpack the batch
    prefered_ids, disprefered_ids, prefered_mask, disprefered_mask, strengths = zip(*batch)

    # Stack the tensors
    prefered_ids = torch.stack(prefered_ids).to(device)
    disprefered_ids = torch.stack(disprefered_ids).to(device)
    prefered_mask = torch.stack(prefered_mask).to(device)
    disprefered_mask = torch.stack(disprefered_mask).to(device)
    strengths = torch.tensor(strengths).to(device)

    return {'prefered_ids': prefered_ids,
            'disprefered_ids': disprefered_ids,
            'prefered_mask': prefered_mask,
            'disprefered_mask': disprefered_mask,
            'strengths': strengths}

# scales column to range specified by low, high
def scale(column, low, high):
    # Step 1: Apply sigmoid scaling based on the mean
    # mean_val = column.mean()
    # sigmoid_scaled = 1 / (1 + np.exp(-10*(column - mean_val)))

    # Step 2: Scale sigmoid values to the specified low and high range
    min_val = column.min()
    max_val = column.max()
    scaled_column = (column - min_val) / (max_val - min_val) * (high - low) + low + 1e-6

    return scaled_column

def sample_combinations(n, k):
    if k > (n * (n - 1)) // 2:
        raise ValueError("k exceeds the number of possible unique pairs")

    # Convert sampled indices to pairs
    pairs = []
    for idx in range(k):
        random_pair = tuple(random.sample(range(n), 2))
        pairs.append(random_pair)

    return pairs