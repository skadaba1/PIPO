from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict
import math
import numpy as np
from itertools import combinations
from tqdm import tqdm
from dataset_utils import sample_combinations, scale
import display 

class TMLibrary(Dataset):
    def __init__(self, file_path, tokenizer, max_length, max_visitations, split, threshold, n_samples):

        data = pd.read_csv(file_path)
        self.seq_key = 'Sequence'
        self.score_key = 'Pred_affinity'
        self.data, self.num_seqs, combination_indexes = self.__process_df__(data, n_samples)
        self.threshold = threshold

        # compute number of combinations for num_seqs
        self.combination_indexes, self.num_samples = self.filter_combination_indexes(combination_indexes, split)

        self.observed_seqs = defaultdict(int)
        self.tokenizer = tokenizer
        self.tokenized = self.tokenizer(self.data[self.seq_key].tolist(), padding=True, return_tensors="pt", max_length=max_length, truncation=True )
        self.max_visitations = max_visitations


    def __process_df__(self, df, n_samples):
        df.dropna(subset=[self.seq_key, self.score_key])
        df = df[df[self.score_key].apply(np.isfinite)]

        self.modified_score_key = 'transformed_' + self.score_key
        df[self.modified_score_key] = scale(df[self.score_key], 1.0, math.e**(10))

        combination_indexes = sample_combinations(len(df), n_samples)#list(combinations(range(self.num_seqs), 2))
        return df, len(df), combination_indexes

    def filter_combination_indexes(self, index_pairs, split):

        # preare dataframe
        index1, index2 = zip(*index_pairs)
        scores1 = self.data[self.score_key].values[np.array(index1)]
        scores2 = self.data[self.score_key].values[np.array(index2)]
        sequence1 = self.data[self.seq_key].values[np.array(index1)]
        sequence2 = self.data[self.seq_key].values[np.array(index2)]

        abs_diff = np.abs(scores1 - scores2)
        relative_abs_diff = abs_diff / np.minimum(scores1, scores2)
        normalized_abs_diff = (abs_diff - np.min(abs_diff)) / (np.max(abs_diff) - np.min(abs_diff))

        df = pd.DataFrame(
            {
              'index1': index1,
              'index2': index2,
              'score1': scores1,
              'score2': scores2,
              'abs_diff': abs_diff,
              'relative_abs_diff': relative_abs_diff,
              'normalized_abs_diff': normalized_abs_diff,
              'sequence1':sequence1,
              'sequence2':sequence2,
            })

        threshold = self.threshold
        df = df[((df['score1'] >= threshold) | (df['score2'] >= threshold))]

        def custom_sort(row):
            return row['relative_abs_diff']

        if(split == 'none'):
          df_sorted = df.iloc[df.apply(custom_sort, axis=1).argsort()]
          # reverse order
          df_sorted = df_sorted.iloc[::-1]
        else:
          df_sorted = df.sample(frac=1).reset_index(drop=True)

        display(df_sorted)
        selected_pairs = list(zip(df_sorted['index1'], df_sorted['index2']))

        filtered_abs_diff = df_sorted['normalized_abs_diff'].tolist()
        num_samples = len(selected_pairs)
        print(f'num_samples: {num_samples}')

        return selected_pairs, num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pair = self.combination_indexes[idx]
        self.observed_seqs[pair[0]] += 1
        self.observed_seqs[pair[1]] += 1

        # Calculate the average number of times a pair has been seen
        average_visitations = np.average(list(self.observed_seqs.values()))
        if average_visitations > self.max_visitations:
            raise ValueError(f"Error: Average number of times pairs have been seen ({average_visitations}) exceeds {self.max_visitations}.")

        if(len(self.observed_seqs.keys()) == self.num_seqs):
          print(f"{1} pass through sequences completed! Average number of visitations: {average_visitations}")
          self.observed_seqs = defaultdict(int)

        # take sigmoid of score here
        score_1 = self.data.iloc[pair[0]][self.modified_score_key]
        score_2 = self.data.iloc[pair[1]][self.modified_score_key]
        high = 0.0
        low = 0.0

        if(score_1 > score_2):
          preferred_ids = self.tokenized['input_ids'][pair[0]]
          preferred_mask = self.tokenized['attention_mask'][pair[0]]
          disprefered_ids = self.tokenized['input_ids'][pair[1]]
          disprefered_mask = self.tokenized['attention_mask'][pair[1]]
          high = score_1
          low = score_2

        else:
          preferred_ids = self.tokenized['input_ids'][pair[1]]
          preferred_mask = self.tokenized['attention_mask'][pair[1]]
          disprefered_ids = self.tokenized['input_ids'][pair[0]]
          disprefered_mask = self.tokenized['attention_mask'][pair[0]]
          high = score_2
          low = score_1

        # check if denominator will be zero, then print
        strengths = 1/(2*np.log(high/low))
        # print(strengths)
        #1/2*np.exp(abs(score_1-score_2))
        #1/2*(np.log(abs(score_1-score_2)+EPS))
        return preferred_ids, disprefered_ids, preferred_mask, disprefered_mask, strengths
