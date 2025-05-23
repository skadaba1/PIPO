import pandas as pd
JXT = 8
def extract_membrane_indices(transmembrane_string):
    # Initialize an empty list to store the indices
    indices = []

    # Iterate through the string
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
  for indx, row in uniprot_df.iterrows():
    for tm_index in row['tm_indices']:
      try:
        sequences.append(row['Sequence'][int(tm_index[0]):int(tm_index[1]) + JXT])
      except:
        pass
  return sequences

uniprot_df = pd.read_csv('/content/idmapping_2024_10_07.tsv', sep='\t', on_bad_lines='skip')
uniprot_df['tm_indices'] = uniprot_df['Transmembrane'].apply(lambda x: extract_membrane_indices(x))
tm_df = pd.read_csv('/content/tm_library_annotated.tsv', sep='\t', on_bad_lines='skip')
tm_df = tm_df[tm_df['jxt'] == True]
sequences = set((extract_seqs(uniprot_df)) + (tm_df['aa_seq'].values.tolist()))
