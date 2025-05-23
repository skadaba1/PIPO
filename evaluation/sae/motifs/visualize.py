import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd

def visualize_feature_map(feature_map, sequence=None, title="Feature Map Visualization", device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Convert to numpy for visualization if on GPU
    if torch.is_tensor(feature_map):
        feature_map = feature_map.detach().cpu().numpy()

    # 1. Heatmap visualization
    plt.figure()
    ax = sns.heatmap(feature_map, cmap="viridis")
    plt.title(f"{title} - Heatmap")
    plt.xlabel("Feature Index")
    plt.ylabel("Sequence Position")
    plt.tight_layout()
    # plt.savefig("feature_map_heatmap.png", dpi=300)
    plt.show()

    # 2. Top features per position
    if sequence:
        seq_len = min(len(sequence), feature_map.shape[0])
        top_n_features = 3

        # Find top features for each position
        top_features_idx = np.argsort(-feature_map[:seq_len], axis=1)[:, :top_n_features]
        top_features_val = np.take_along_axis(feature_map[:seq_len], top_features_idx, axis=1)

        # Create a dataframe for visualization
        position_data = []
        for pos in range(seq_len):
            amino_acid = sequence[pos]
            for i in range(top_n_features):
                feature_idx = top_features_idx[pos, i]
                activation = top_features_val[pos, i]
                if activation > 0.1:  # Only show significant activations
                    position_data.append({
                        'Position': pos + 1,
                        'Amino Acid': amino_acid,
                        'Feature': f'Feature {feature_idx}',
                        'Activation': activation
                    })

        if position_data:
            df = pd.DataFrame(position_data)
            plt.figure()
            sns.scatterplot(data=df, x='Position', y='Activation', hue='Feature',
                            size='Activation', sizes=(20, 200), alpha=0.7)

            # Annotate with amino acids for top activations
            for i, row in df.iterrows():
                if row['Activation'] > 0.5:  # Only annotate strong activations
                    plt.annotate(row['Amino Acid'],
                                (row['Position'], row['Activation']),
                                xytext=(0, 5),
                                textcoords='offset points',
                                ha='center')

            plt.title(f"{title} - Top Features by Position")
            plt.xlabel("Sequence Position")
            plt.ylabel("Feature Activation")
            plt.tight_layout()
            # plt.savefig("top_features_by_position.png", dpi=300)
            plt.show()

    # 3. Feature activation distribution
    plt.figure()
    # Calculate sparsity (percentage of neurons that are active)
    sparsity = (feature_map > 0.1).mean(axis=0) * 100
    plt.bar(range(len(sparsity)), sparsity)
    plt.title(f"{title} - Feature Activation Percentage")
    plt.xlabel("Feature Index")
    plt.ylabel("% of Positions Where Feature is Active")
    plt.tight_layout()
    # plt.savefig("feature_activation_distribution.png", dpi=300)
    plt.show()

# Visualizing specific features across the sequence
def visualize_specific_feature(feature_idx, feature_map, sequence, title=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if torch.is_tensor(feature_map):
        feature_map = feature_map.detach().cpu().numpy()

    # Extract the feature activations
    feature_activations = feature_map[:, feature_idx]

    plt.figure()
    plt.bar(range(len(feature_activations)), feature_activations)

    # Add amino acid labels on x-axis
    if sequence:
        seq_len = min(len(sequence), len(feature_activations))
        plt.xticks(range(0, seq_len, 5),
                   [sequence[i:i+5] for i in range(0, seq_len, 5)],
                   rotation=45)

    # Highlight top activations
    threshold = np.percentile(feature_activations, 95)
    top_positions = np.where(feature_activations > threshold)[0]

    # Annotate top activations with amino acids
    for pos in top_positions:
        if pos < len(sequence):
            plt.annotate(sequence[pos],
                        (pos, feature_activations[pos]),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontweight='bold')

    plt.title(title or f"Feature {feature_idx} Activation Pattern")
    plt.xlabel("Sequence Position")
    plt.ylabel(f"Feature {feature_idx} Activation")
    plt.tight_layout()
    # plt.savefig(f"feature_{feature_idx}_pattern.png", dpi=300)
    plt.show()

# Creating a motif visualization
def visualize_feature_motifs(sae, top_k_features=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Get the decoder weights
    decoder_weights = sae.decoder.weight.detach().cpu().numpy()

    # Compute feature importance (L2 norm of each feature's decoder weights)
    feature_importance = np.linalg.norm(decoder_weights, axis=0)

    # Get top features
    top_features = np.argsort(-feature_importance)[:top_k_features]

    # Visualize each top feature's decoder weights
    plt.figure()
    for i, feature_idx in enumerate(top_features):
        plt.subplot(top_k_features, 1, i+1)
        weights = decoder_weights[:, feature_idx]
        plt.bar(range(len(weights)), weights)
        plt.title(f"Feature {feature_idx} Decoder Weights")
        plt.xlabel("Dimension in Original Space")
        plt.ylabel("Weight")

    plt.tight_layout()
    # plt.savefig("top_feature_motifs.png", dpi=300)
    plt.show()

# Calculate feature similarity
def visualize_feature_similarity(feature_map, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if torch.is_tensor(feature_map):
        feature_map = feature_map.detach().cpu().numpy()

    # Calculate correlation matrix between features
    n_features = feature_map.shape[1]
    correlation_matrix = np.corrcoef(feature_map.T)

    # Visualize correlation matrix
    plt.figure()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, cmap="coolwarm", vmin=-1, vmax=1,
                center=0, square=True, linewidths=.5, annot=False)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()