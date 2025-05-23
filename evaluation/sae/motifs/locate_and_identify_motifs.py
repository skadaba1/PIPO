import numpy as np
from scipy.interpolate import interp1d

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

def interpolate_sequence(seq, target_len=120):
    """
    seq: np.array of shape (N, 120)
    target_len: desired length after resampling
    Returns: np.array of shape (target_len, 120)
    """
    N, F = seq.shape
    x_old = np.linspace(0, 1, N)
    x_new = np.linspace(0, 1, target_len)

    # Use vectorized interpolation across all features at once
    interpolated = interp1d(x_old, seq, axis=0, kind='linear', fill_value="extrapolate")(x_new)
    return interpolated

def find_feature_triggers(feature_idx, target_layer, sae, model, tokenizer, sequences,
                          top_k=5, motif_length=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Find amino acid motifs that trigger a specific feature.

    Args:
        feature_idx: Index of the feature to analyze
        target_layer: Layer in ESM model to extract activations from
        sae: Trained Sparse Autoencoder model
        model: ESM protein language model
        tokenizer: ESM tokenizer
        sequences: List of protein sequences to analyze
        top_k: Number of top motifs to return
        motif_length: Length of continuous motifs to find (default=1 for single amino acids)
        device: Computing device

    Returns:
        Lists of top motifs and their activation values
    """
    all_motifs = []
    all_activations = []

    # Ensure motif_length is valid
    motif_length = max(1, int(motif_length))

    # Add logging info
    start_time = time.time()
    total_sequences = len(sequences)
    print(f"Processing {total_sequences} sequences for feature {feature_idx} with motif length {motif_length}...")

    # Use tqdm for progress tracking
    for i, sequence in enumerate(tqdm(sequences, desc=f"Feature {feature_idx}, Motif length {motif_length}")):
        # Get feature activations for this sequence
        feature_map = analyze_feature_activations(sae, model, tokenizer, sequence, target_layer, device)

        # Convert to numpy if needed
        if torch.is_tensor(feature_map):
            feature_activations = feature_map[:, feature_idx].cpu().detach().numpy()
        else:
            feature_activations = feature_map[:, feature_idx]

        # Get all possible motifs of the specified length
        seq_len = min(len(sequence), len(feature_activations))

        for j in range(seq_len - motif_length + 1):
            # Extract the motif
            motif = sequence[j:j+motif_length]

            # For motifs longer than 1, use the average activation across the motif positions
            if motif_length == 1:
                activation = feature_activations[j]
            else:
                # We could use different aggregation methods here:
                # - Mean: average activation across the motif
                # - Max: strongest activation within the motif
                # - Sum: total activation across the motif
                activation = np.mean(feature_activations[j:j+motif_length])

            all_motifs.append(motif)
            all_activations.append(activation)

        # Add intermediate logging every 20% of sequences
        if (i+1) % max(1, total_sequences // 5) == 0:
            elapsed = time.time() - start_time
            avg_time_per_seq = elapsed / (i+1)
            remaining = avg_time_per_seq * (total_sequences - (i+1))
            print(f"  Processed {i+1}/{total_sequences} sequences ({(i+1)/total_sequences*100:.1f}%)")
            print(f"  Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
            print(f"  Current motif count: {len(all_motifs)}")

    elapsed = time.time() - start_time
    print(f"Finished processing {total_sequences} sequences in {elapsed:.1f}s")
    print(f"Total motifs found: {len(all_motifs)}")

    # Convert to numpy array for sorting
    activations = np.array(all_activations)

    # Find top activating motifs
    if len(activations) > 0:
        print("Finding top motifs...")
        top_indices = np.argsort(-activations)[:top_k]
        top_motifs = [all_motifs[i] for i in top_indices]
        top_activation_values = [activations[i] for i in top_indices]
        bottom_indices = np.argsort(activations)[:top_k]
        bottom_motifs = [all_motifs[i] for i in bottom_indices]
        bottom_activation_values = [activations[i] for i in bottom_indices]

        print(f"Top motifs for feature {feature_idx}, length {motif_length}:")
        for i, (motif, activation) in enumerate(zip(top_motifs, top_activation_values)):
            print(f"  {i+1}. {motif}: {activation:.4f}")

        print(f"Bottom motifs for feature {feature_idx}, length {motif_length}:")
        for i, (motif, activation) in enumerate(zip(bottom_motifs, bottom_activation_values)):
            print(f"  {i+1}. {motif}: {activation:.4f}")
    else:
        top_motifs = []
        top_activation_values = []
        print("No motifs found.")

    # Visualize the results
    if len(top_motifs) > 0:
        print("Creating visualization...")
        plt.figure()
        y_pos = range(len(top_motifs))
        plt.barh(y_pos, top_activation_values)
        plt.yticks(y_pos, top_motifs)
        plt.xlabel('Activation Strength')
        plt.title(f'Top {top_k} Motifs (Length {motif_length}) for Feature {feature_idx}')
        plt.tight_layout()
        # plt.savefig(f"feature_{feature_idx}_motifs_length_{motif_length}.png", dpi=300)
        plt.show()

    return top_motifs, top_activation_values

# Function to find consensus patterns across top motifs
def find_consensus_pattern(motifs):
    """
    Find the consensus pattern in a list of motifs of the same length.
    Returns a consensus string with most common amino acid at each position.

    Args:
        motifs: List of motifs of the same length

    Returns:
        Consensus pattern and position-specific frequency information
    """
    if not motifs or len(motifs) == 0:
        return "", {}

    motif_length = len(motifs[0])
    position_frequencies = []

    # Count amino acid frequencies at each position
    for pos in range(motif_length):
        aa_counts = {}
        for motif in motifs:
            if pos < len(motif):
                aa = motif[pos]
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
        position_frequencies.append(aa_counts)

    # Find the most common amino acid at each position
    consensus = ""
    for pos_counts in position_frequencies:
        if pos_counts:
            most_common_aa = max(pos_counts.items(), key=lambda x: x[1])[0]
            consensus += most_common_aa
        else:
            consensus += "X"  # Use X for positions with no clear consensus

    return consensus, position_frequencies

# Enhanced function to analyze multiple motif lengths
def analyze_feature_motifs(feature_idx, target_layer, sae, model, tokenizer, sequences,
                         max_motif_length=5, top_k=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Analyze a feature across multiple motif lengths and find patterns.

    Args:
        feature_idx: Index of the feature to analyze
        target_layer: Layer in ESM model to extract activations from
        sae: Trained Sparse Autoencoder model
        model: ESM protein language model
        tokenizer: ESM tokenizer
        sequences: List of protein sequences to analyze
        max_motif_length: Maximum motif length to analyze
        top_k: Number of top motifs to return for each length
        device: Computing device

    Returns:
        Dictionary of results for each motif length
    """
    results = {}
    total_motif_lengths = max_motif_length

    print(f"Starting analysis of feature {feature_idx} across {total_motif_lengths} motif lengths")
    start_time = time.time()

    for motif_length in tqdm(range(max_motif_length, max_motif_length + 1), desc="Analyzing motif lengths"):
        motif_start_time = time.time()
        print(f"\nAnalyzing motifs of length {motif_length}/{max_motif_length}...")

        top_motifs, top_activations = find_feature_triggers(
            feature_idx, target_layer, sae, model, tokenizer, sequences,
            top_k, motif_length, device
        )

        # Skip if no motifs found
        if not top_motifs:
            print(f"No motifs found for length {motif_length}, skipping.")
            continue

        # Find consensus pattern for motifs
        print("Finding consensus pattern...")
        consensus, position_frequencies = find_consensus_pattern(top_motifs)

        # Visualize position-specific amino acid frequencies
        if motif_length > 1:
            print("Creating position frequency visualization...")
            fig, axes = plt.subplots(motif_length, 1, sharex=False)

            # Add overall title to the figure instead of individual subplot titles
            fig.suptitle('Amino Acid Frequencies by Position', fontsize=14)

            # Add common y-label
            fig.text(0.04, 0.5, 'Frequency (%)', va='center', rotation='vertical')

            aggregate_df = []  # List to store each position's DataFrame

            for pos in range(motif_length):
                ax = axes[pos] if motif_length > 1 else axes

                pos_counts = position_frequencies[pos]
                aa_list = sorted(pos_counts.keys())
                frequencies = [pos_counts[aa] / len(top_motifs) * 100 for aa in aa_list]

                # Build position-specific DataFrame
                df = pd.DataFrame({
                    'position': pos,         # Add position as a column
                    'aa': aa_list,
                    'frequency': frequencies
                })
                aggregate_df.append(df)

                ax.bar(range(len(aa_list)), frequencies)
                ax.set_xticks(range(len(aa_list)))
                ax.set_xticklabels(aa_list)

                # Only add position indicator on the right side
                ax.text(1.02, 0.5, f'Pos {pos+1}', transform=ax.transAxes,
                        verticalalignment='center')

                # Remove individual y-labels and titles
                ax.set_ylabel('')

                # Only show y-ticks for leftmost subplot
                if pos > 0:
                    ax.set_yticklabels([])
            final_df = pd.concat(aggregate_df, ignore_index=True)
            final_df.to_csv(f'/content/drive/MyDrive/SMPO/Figures/Appendices/{feature_idx}_pssm_SMPO_noMLM_.csv')
            # Adjust spacing between subplots to prevent overlap
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, left=0.15, right=0.85)

            # plt.savefig(f"feature_{feature_idx}_position_frequencies_length_{motif_length}.png", dpi=300)
            plt.show()

        # Store results
        results[motif_length] = {
            'motifs': top_motifs,
            'activations': top_activations,
            'consensus': consensus,
            'position_frequencies': position_frequencies
        }

        motif_elapsed = time.time() - motif_start_time
        print(f"Consensus pattern for length {motif_length}: {consensus}")
        print(f"Completed motif length {motif_length} in {motif_elapsed:.1f}s")

    total_elapsed = time.time() - start_time
    print(f"\nAnalysis complete for feature {feature_idx}")
    print(f"Total time: {total_elapsed:.1f}s")

    # Summary of best motifs by activation strength
    if results:
        print("\nSummary of best motifs by length:")
        for length, result in sorted(results.items()):
            if result['motifs']:
                best_motif = result['motifs'][0]
                best_activation = result['activations'][0]
                print(f"Length {length}: {best_motif} (activation: {best_activation:.4f})")

        # Find best overall motif
        best_length = max(results.keys(), key=lambda k: max(results[k]['activations']))
        best_motif = results[best_length]['motifs'][0]
        best_activation = results[best_length]['activations'][0]

        print(f"\nBest overall motif: {best_motif} (length {best_length}, activation: {best_activation:.4f})")
        print(f"Consensus pattern: {results[best_length]['consensus']}")

    return results

# Function to scan sequences for motifs
def scan_sequences_for_motif(motif, sequences, feature_idx, target_layer, sae, model, tokenizer,
                            threshold=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Scan sequences for occurrences of a motif and report activation levels.

    Args:
        motif: Motif to scan for (exact string or consensus pattern with 'X' wildcards)
        sequences: List of protein sequences to scan
        feature_idx: Feature index to check activations for
        target_layer: Target layer for feature extraction
        sae: Trained SAE model
        model: ESM model
        tokenizer: ESM tokenizer
        threshold: Optional activation threshold (default: None = report all occurrences)
        device: Computing device

    Returns:
        Dictionary with occurrences and their activation levels
    """
    results = {}
    motif_length = len(motif)
    is_consensus = 'X' in motif

    print(f"Scanning {len(sequences)} sequences for motif: {motif}")

    for seq_idx, sequence in enumerate(tqdm(sequences, desc="Scanning sequences")):
        # Get feature activations
        feature_map = analyze_feature_activations(sae, model, tokenizer, sequence, target_layer, device)

        if torch.is_tensor(feature_map):
            feature_activations = feature_map[:, feature_idx].cpu().detach().numpy()
        else:
            feature_activations = feature_map[:, feature_idx]

        # Find motif occurrences
        occurrences = []

        for pos in range(len(sequence) - motif_length + 1):
            current_segment = sequence[pos:pos+motif_length]

            # Check for match (exact or consensus with wildcards)
            match = True
            if is_consensus:
                for i, aa in enumerate(motif):
                    if aa != 'X' and aa != current_segment[i]:
                        match = False
                        break
            else:
                match = (current_segment == motif)

            if match:
                # Calculate average activation for this occurrence
                activation = np.mean(feature_activations[pos:pos+motif_length])

                # Only include if above threshold (if specified)
                if threshold is None or activation >= threshold:
                    occurrences.append({
                        'position': pos,
                        'activation': activation,
                        'segment': current_segment
                    })

        # Sort occurrences by activation
        occurrences.sort(key=lambda x: x['activation'], reverse=True)

        # Store results if any occurrences found
        if occurrences:
            results[seq_idx] = occurrences

    # Summary statistics
    total_occurrences = sum(len(occs) for occs in results.values())
    num_sequences_with_motif = len(results)

    print(f"\nScan complete:")
    print(f"- Motif found in {num_sequences_with_motif}/{len(sequences)} sequences ({num_sequences_with_motif/len(sequences)*100:.1f}%)")
    print(f"- Total occurrences: {total_occurrences}")
    print(f"- Average occurrences per sequence (where present): {total_occurrences/max(1, num_sequences_with_motif):.2f}")

    if total_occurrences > 0:
        # Collect all activations for histogram
        all_activations = [occ['activation'] for seq_occs in results.values() for occ in seq_occs]

        plt.figure()
        plt.hist(all_activations, bins=30)
        plt.xlabel('Activation Level')
        plt.ylabel('Frequency')
        plt.title(f'Activation Distribution for Motif "{motif}" (Feature {feature_idx})')
        plt.axvline(np.mean(all_activations), color='r', linestyle='dashed', label=f'Mean: {np.mean(all_activations):.4f}')
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f"feature_{feature_idx}_motif_{motif.replace('X', 'x')}_activation_histogram.png", dpi=300)
        plt.show()

    return results