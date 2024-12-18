import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import stats
import pickle
import torch

from core.losses.mlm import get_sequence_probs

def extract_membrane_indices(transmembrane_string):
    indices = []
    splits = transmembrane_string.split('TRANSMEM')
    for split in splits:
        indices_instance = split.strip()
        indices_instance = indices_instance.split('..')
        try:
            indices_instance[1] = indices_instance[1].split(';')[0]
            indices.append(tuple(indices_instance))
        except:
            continue
    return indices

def extract_seqs(uniprot_df):
    sequences = []
    for _, row in uniprot_df.iterrows():
        for tm_index in row['tm_indices']:
            try:
                sequences.append(row['Sequence'][int(tm_index[0]):int(tm_index[1])])
            except:
                pass
    return sequences

def find_motifs(sequences, motif):
    motif_regex = motif.replace('X', '.')  # '.' matches any character
    pattern = re.compile(motif_regex)

    result = []
    for seq in sequences:
        matches = [(m.start(), m.group()) for m in pattern.finditer(seq)]
        if matches:
            result.append((seq, matches))
    return result

def main():
    parser = argparse.ArgumentParser(description="Analyze sequence motifs and mutations.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.")
    parser.add_argument("--sklearn_model_path", type=str, required=True, help="Path to the fitted sklearn model.")
    parser.add_argument("--uniprot_path", type=str, required=True, help="Path to the Uniprot dataset.")
    parser.add_argument("--tm_library_path", type=str, required=True, help="Path to the transmembrane library dataset.")
    parser.add_argument("--motifs", type=str, required=True, help="Comma-separated list of motifs to analyze.")
    args = parser.parse_args()
    # Define motifs of interest
    motifs_of_interest = args.motifs.split(",")
    # Load models and data
    model = torch.load(args.model_path)
    tokenizer = pickle.load(open(args.tokenizer_path, 'rb'))
    fitted_model = pickle.load(open(args.sklearn_model_path, 'rb'))

    uniprot_df = pd.read_csv(args.uniprot_path, sep='\t', on_bad_lines='skip')
    uniprot_df['tm_indices'] = uniprot_df['Transmembrane'].apply(lambda x: extract_membrane_indices(x))
    
    tm_df = pd.read_csv(args.tm_library_path)
    sequences = set(extract_seqs(uniprot_df) + tm_df['aa_seq'].tolist())

    hits = []
    for motif_of_interest in motifs_of_interest:
        hits += find_motifs(sequences, motif_of_interest)

    hits_to_ala_scan = {}
    hits_to_random_scan = {}

    for hit in hits:
        roi = 'G'
        num_to_replace = 1
        motif = hit[1][0][1]
        len_motif = len(motif)

        mutagenesis = motif.replace(roi, 'A', num_to_replace)
        insert_pos = hit[1][0][0]
        new_seq = hit[0][:insert_pos] + mutagenesis + hit[0][insert_pos+len_motif:]

        num_replacements = motif.count(roi) if num_to_replace is None else num_to_replace
        random_start_pos = np.random.randint(0, len(hit[0]) - num_replacements)
        random_seq = hit[0][:random_start_pos] + 'A' * num_replacements + hit[0][random_start_pos+num_replacements:]

        hits_to_ala_scan[hit[0]] = new_seq
        hits_to_random_scan[hit[0]] = random_seq

    original_seqs = list(hits_to_ala_scan.keys())
    new_seqs = list(hits_to_ala_scan.values())
    random_seqs = list(hits_to_random_scan.values())

    max_length = 40
    probs_wt = fitted_model.predict(get_sequence_probs(original_seqs, model, tokenizer, max_length).tolist())
    probs_ala_mut = fitted_model.predict(get_sequence_probs(new_seqs, model, tokenizer, max_length).tolist())
    probs_random_mut = fitted_model.predict(get_sequence_probs(random_seqs, model, tokenizer, max_length).tolist())

    probs_ala_delta = probs_ala_mut - probs_wt
    probs_random_delta = probs_random_mut - probs_wt

    # Plot distributions
    plt.hist(probs_ala_delta, bins=20, label='ala-mut', alpha=0.7, density=True)
    plt.hist(probs_random_delta, bins=20, label='rand-mut', alpha=0.7, density=True)
    plt.xlim(-1, 1)
    plt.legend()
    plt.show()

    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(probs_ala_delta, probs_random_delta)

    print(f"K-S test statistic: {ks_statistic}")
    print(f"P-value: {p_value}")

    print(f"Mean of probs_ala_delta: {np.mean(probs_ala_delta)}, median: {np.median(probs_ala_delta)}")
    print(f"Mean of probs_random_delta: {np.mean(probs_random_delta)}, median: {np.median(probs_random_delta)}")

if __name__ == "__main__":
    main()
