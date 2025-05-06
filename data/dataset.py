from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from itertools import combinations
import torch
class Library(Dataset):
    """
    DPO preference-pair dataset with optional strength-threshold filtering.
    Each item is a dict with keys:
      - 'prefered_ids', 'disprefered_ids', 'prefered_mask', 'disprefered_mask', 'strengths'
    """
    def __init__(self, file_path, tokenizer, max_length, split, strength_threshold=None):
        # Read raw data
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.threshold = strength_threshold

        # Generate all possible index pairs
        index_pairs = list(combinations(range(len(self.data)), 2))
        # Filter pairs based on split and threshold
        self.pairs, self.abs_diffs, self.num_samples = self.filter_combination_indexes(
            index_pairs, split
        )
        # Iterator over filtered pairs
        self._pair_iter = iter(self.pairs)

        # Pre-tokenize all sequences for efficiency
        sequences = self.data['aa_seq'].tolist()
        tokenized = tokenizer(
            sequences,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        self.input_ids = tokenized['input_ids']
        self.attention_mask = tokenized['attention_mask']

    def filter_combination_indexes(self, index_pairs, split):
        # preare dataframe
        index1, index2 = zip(*index_pairs)
        scores1 = self.data['avg_induction_score'].values[np.array(index1)]
        scores2 = self.data['avg_induction_score'].values[np.array(index2)]
        sequence1 = self.data['aa_seq'].values[np.array(index1)]
        sequence2 = self.data['aa_seq'].values[np.array(index2)]

        abs_diff = np.array(np.abs(scores1 - scores2))
        min_diff, max_diff = np.min(abs_diff), np.max(abs_diff)
        # print(f"Min abs_diff: {np.min(abs_diff)}, Max abs_diff: {max_diff}")
        if max_diff > min_diff:
            norm_diff = (abs_diff - min_diff) / (max_diff - min_diff)
        else:
            norm_diff = np.zeros_like(abs_diff)

        # Build DataFrame for sorting/shuffling
        df = pd.DataFrame({
            'index1': index1,
            'index2': index2,
            'abs_diff': abs_diff,
            'normalized_abs_diff': norm_diff
        })

        # Sort or shuffle
        if split == 'none':
            df = df.sort_values(by='abs_diff', ascending=False)
        else:
            df = df.sample(frac=1).reset_index(drop=True)

        # Apply threshold filtering
        if self.threshold is not None:
            df = df[df['normalized_abs_diff'] >= self.threshold].reset_index(drop=True)
            print(f"Number of pairs after thresholding: {len(df)}")
        pairs = list(zip(df['index1'], df['index2']))
        abs_diffs = df['normalized_abs_diff'].tolist()
        num_samples = len(pairs)
        print(f"Number of pairs: {num_samples}")
        return pairs, abs_diffs, num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Fetch next pair, cycling if exhausted
        try:
            i, j = next(self._pair_iter)
        except StopIteration:
            self._pair_iter = iter(self.pairs)
            i, j = next(self._pair_iter)

        # Determine which sequence is preferred
        score_i = self.data.iloc[i]['avg_induction_score']
        score_j = self.data.iloc[j]['avg_induction_score']
        if score_i > score_j:
            pref_idx, disp_idx = i, j
        else:
            pref_idx, disp_idx = j, i

        # Gather tokenized tensors
        preferred_ids   = self.input_ids[pref_idx]
        preferred_mask  = self.attention_mask[pref_idx]
        disprefered_ids = self.input_ids[disp_idx]
        disprefered_mask= self.attention_mask[disp_idx]

        # Compute DPO strength
        strength = 1 / (2 * abs(np.log(score_i+1e-10) - np.log(score_j+1e-10)))

        return preferred_ids, disprefered_ids, preferred_mask, disprefered_mask, strength


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