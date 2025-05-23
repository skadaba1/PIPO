import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ..analyze.py import analyze_feature_activations

def compute_top_correlated_features(sae, model, tokenizer, sequences, scores, target_layer,
                                   k=10, correlation_type='pearson',
                                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Compute the top-k features most correlated with sequence scores.

    Args:
        sae: Trained Sparse Autoencoder model
        model: ESM protein language model
        tokenizer: ESM tokenizer
        sequences: List of protein sequences
        scores: List of corresponding scores for each sequence
        target_layer: Layer in ESM model to extract activations from
        k: Number of top correlated features to return
        correlation_type: 'pearson' or 'spearman'
        device: Computing device

    Returns:
        Dictionary with top-k features, their correlation values, and p-values
    """
    # Ensure inputs are valid
    assert len(sequences) == len(scores), "Sequences and scores must have the same length"
    assert correlation_type in ['pearson', 'spearman'], "Correlation type must be 'pearson' or 'spearman'"

    # Collect feature activations for all sequences
    print("Extracting features for all sequences...")
    all_feature_activations = []

    for sequence in tqdm(sequences):
        # Use the existing function to get feature map
        feature_map = analyze_feature_activations(sae, model, tokenizer, sequence, target_layer, device)

        # Pool features across sequence (max pooling)
        if torch.is_tensor(feature_map):
            pooled_features = torch.max(feature_map, dim=0)[0].cpu().detach().numpy()
        else:
            pooled_features = np.max(feature_map, axis=0)

        all_feature_activations.append(pooled_features)

    # Convert to numpy array for correlation analysis
    feature_matrix = np.vstack(all_feature_activations)
    scores_array = np.array(scores)

    # Compute correlation between each feature and the scores
    n_features = feature_matrix.shape[1]
    correlations = np.zeros(n_features)
    p_values = np.zeros(n_features)

    print("Computing correlations...")
    for i in range(n_features):
        if correlation_type == 'pearson':
            corr, p_val = pearsonr(feature_matrix[:, i], scores_array)
        else:  # spearman
            corr, p_val = spearmanr(feature_matrix[:, i], scores_array)

        correlations[i] = corr
        p_values[i] = p_val

    # Get top-k features by absolute correlation
    abs_correlations = np.abs(correlations)
    top_indices = np.argsort(-abs_correlations)[:k]

    # Create result dictionary
    results = {
        'feature_indices': top_indices,
        'correlation_values': correlations[top_indices],
        'p_values': p_values[top_indices],
        'absolute_correlations': abs_correlations[top_indices]
    }

    # Visualize results
    plt.figure()
    plt.bar(range(k), correlations[top_indices])
    plt.xlabel('Feature Index')
    plt.ylabel(f'{correlation_type.capitalize()} Correlation')
    plt.title(f'Top {k} Features Correlated with Scores')
    plt.xticks(range(k), top_indices)

    # Add correlation values as text
    # for i in range(k):
    #     plt.text(i, correlations[top_indices[i]],
    #             f'{correlations[top_indices[i]]:.3f}\np={p_values[top_indices[i]]:.3e}',
    #             ha='center', va='bottom' if correlations[top_indices[i]] > 0 else 'top')

    plt.tight_layout()
    # plt.savefig("top_correlated_features.png", dpi=300)
    plt.show()

    # Additional visualization: correlation heatmap for top features
    plt.figure()
    top_features_matrix = feature_matrix[:, top_indices]
    correlation_data = np.column_stack((top_features_matrix, scores_array.reshape(-1, 1)))

    # Fix the heatmap visualization part
    plt.figure()
    top_features_matrix = feature_matrix[:, top_indices]

    # Create a new correlation matrix manually to match the first calculation
    heatmap_corr_matrix = np.zeros((k+1, k+1))
    # Fill in feature-score correlations (last row and last column)
    for i in range(k):
        heatmap_corr_matrix[i, k] = correlations[top_indices[i]]  # Last column
        heatmap_corr_matrix[k, i] = correlations[top_indices[i]]  # Last row

    # Fill in feature-feature correlations using pearsonr or spearmanr
    for i in range(k):
        heatmap_corr_matrix[i, i] = 1.0  # Diagonal is always 1
        for j in range(i+1, k):
            if correlation_type == 'pearson':
                corr, _ = pearsonr(top_features_matrix[:, i], top_features_matrix[:, j])
            else:  # spearman
                corr, _ = spearmanr(top_features_matrix[:, i], top_features_matrix[:, j])
            heatmap_corr_matrix[i, j] = corr
            heatmap_corr_matrix[j, i] = corr  # Matrix is symmetric

    # Set score-score correlation to 1
    heatmap_corr_matrix[k, k] = 1.0

    # Create mask for upper triangle
    mask = np.zeros_like(heatmap_corr_matrix)
    mask[np.triu_indices_from(mask)] = True

    feature_labels = [f'Feature {idx}' for idx in top_indices]
    feature_labels.append('Score')

    sns.heatmap(heatmap_corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
                center=0, annot=True, fmt='.2f', linewidths=0.5,
                xticklabels=feature_labels, yticklabels=feature_labels)
    plt.title(f'Correlation Between Top {k} Features and Score')
    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/SMPO/Figures/Appendices/feature_correlations_SMPO_noMLM.png", dpi=300)
    plt.show()

    return results