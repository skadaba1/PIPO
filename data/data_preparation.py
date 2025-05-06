import os
import pandas as pd
import numpy as np
from graph_part import train_test_validation_split
import torch
from torch.utils.data import DataLoader
import itertools
from functools import partial
from .dataset import Library, collate_fn

def read_fasta(path: str) -> (list, list):
    """
    Read sequences from a FASTA file and return a tuple of (headers, sequences).
    Headers do NOT include the leading '>' and are kept as-is to match TXT name column.
    Sequences are uppercase strings.
    """
    headers = []
    seqs = []
    buf = []
    current_header = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    seqs.append("".join(buf).upper())
                    buf = []
                current_header = line[1:].strip()
                headers.append(current_header)
            else:
                buf.append(line)
        if current_header is not None and buf:
            seqs.append("".join(buf).upper())
    if len(headers) != len(seqs):
        raise ValueError(f"FASTA parse error: {len(headers)} headers but {len(seqs)} sequences")
    return headers, seqs

def load_dataset(
    csv_file: str = None,
    fasta_file: str = None,
    txt_file: str = None,
    txt_sep: str = None
) -> pd.DataFrame:
    """
    Load dataset either from a CSV or from a FASTA + TXT pair.

    - CSV must contain 'aa_seq' and 'avg_induction_score' columns.
    - FASTA + TXT: TXT must have a 'name' column matching FASTA headers.
      Sequences are injected as 'aa_seq'.
      If TXT has 'mean_rate', it's renamed to 'avg_induction_score'.

    Returns a DataFrame with 'aa_seq', 'avg_induction_score' first, followed by other columns.
    """
    if csv_file:
        df = pd.read_csv(csv_file)
        required = {'aa_seq', 'avg_induction_score'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"CSV input is missing required columns: {missing}")
    elif fasta_file and txt_file:
        headers, seqs = read_fasta(fasta_file)
        df_txt = pd.read_csv(txt_file, sep=txt_sep) if txt_sep else pd.read_csv(txt_file)
        if 'name' not in df_txt.columns:
            raise ValueError("TXT input must contain 'name' column matching FASTA headers")
        df_fasta = pd.DataFrame({'name': headers, 'aa_seq': seqs})
        df = df_txt.merge(df_fasta, on='name', how='inner')
        if len(df) != len(df_txt):
            raise ValueError("Some FASTA headers did not match TXT 'name' entries")
        # unify score column
        if 'avg_induction_score' not in df.columns:
            if 'mean_rate' in df.columns:
                df.rename(columns={'mean_rate': 'avg_induction_score'}, inplace=True)
            else:
                raise ValueError(
                    "TXT input must contain either 'avg_induction_score' or 'mean_rate'"
                )
        cols = ['aa_seq', 'avg_induction_score'] + [c for c in df.columns
                                                   if c not in ('aa_seq', 'avg_induction_score')]
        df = df[cols]
    else:
        raise ValueError("Must provide either csv_file or (fasta_file and txt_file)")

    # --- new NaN filtering step ---
    # drop any rows missing sequence or score
    df = df.dropna(subset=['aa_seq', 'avg_induction_score']).reset_index(drop=True)

    return df

def random_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    valid_size: float = 0.05,
    random_state: int = None
) -> tuple:
    """
    Randomly split DataFrame indices into train, test, and validation sets.
    Returns (train_idx, test_idx, valid_idx) as lists of integers.
    """
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * valid_size)
    if n_test + n_val >= n:
        raise ValueError("Test + validation size must be smaller than total samples")
    indices = np.arange(n)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    return train_idx.tolist(), test_idx.tolist(), val_idx.tolist()

def split_dataset(
    df: pd.DataFrame,
    method: str = 'alignment',
    **kwargs
) -> tuple:
    """
    Split dataset indices using either 'alignment' via graph_part or 'random'.
    """
    if method == 'alignment':
        seqs = df['aa_seq'].tolist()
        return train_test_validation_split(
            seqs,
            alignment_mode=kwargs.get('alignment_mode', 'needle'),
            threads=kwargs.get('threads', 1),
            threshold=kwargs.get('threshold', 0.5),
            test_size=kwargs.get('test_size', 0.15),
            valid_size=kwargs.get('valid_size', 0.05),
        )
    elif method == 'random':
        return random_split(
            df,
            test_size=kwargs.get('test_size', 0.15),
            valid_size=kwargs.get('valid_size', 0.05),
            random_state=kwargs.get('random_state', None),
        )
    else:
        raise ValueError("Unknown split method: choose 'alignment' or 'random'")

def save_splits(
    df: pd.DataFrame,
    train_idx: list,
    test_idx: list,
    valid_idx: list,
    output_dir: str = 'data/outputs'
) -> None:
    """
    Save train/test/validation splits into CSV files in 'data/outputs/'.
    """
    os.makedirs(output_dir, exist_ok=True)
    df.iloc[train_idx].to_csv(os.path.join(output_dir, 'train_set.csv'), index=False)
    df.iloc[test_idx].to_csv(os.path.join(output_dir, 'test_set.csv'), index=False)
    df.iloc[valid_idx].to_csv(os.path.join(output_dir, 'val_set.csv'), index=False)

def prepare_dataloaders(
    tokenizer,
    max_length: int,
    batch_size: int,
    device,
    strength_threshold: float = 0.5,
    # dataset source args
    csv_file: str = None,
    fasta_file: str = None,
    txt_file: str = None,
    txt_sep: str = None,
    # split args
    split_method: str = 'alignment',
    split_kwargs: dict = None,
    # always use data/outputs
    output_dir: str = 'data/outputs'
) -> tuple:
    """
    Load or generate train/val/test splits in 'data/outputs/', then return DataLoaders.
    """
    split_kwargs = split_kwargs or {}
    os.makedirs(output_dir, exist_ok=True)
    train_fp = os.path.join(output_dir, 'train_set.csv')
    val_fp   = os.path.join(output_dir, 'val_set.csv')
    test_fp  = os.path.join(output_dir, 'test_set.csv')

    # regenerate splits if missing
    if not (os.path.isfile(train_fp) and os.path.isfile(val_fp) and os.path.isfile(test_fp)):
        df = load_dataset(csv_file=csv_file,
                          fasta_file=fasta_file,
                          txt_file=txt_file,
                          txt_sep=txt_sep)
        train_idx, test_idx, val_idx = split_dataset(df,
                                                     method=split_method,
                                                     **split_kwargs)
        save_splits(df, train_idx, test_idx, val_idx, output_dir=output_dir)

    # build datasets
    print(f"train datasets (filtered with threshold {strength_threshold}):")
    train_ds = Library(file_path=train_fp,
                       tokenizer=tokenizer,
                       max_length=max_length,
                       split='train',
                       strength_threshold=strength_threshold)
    print("val datasets:")
    val_ds = Library(file_path=val_fp,
                     tokenizer=tokenizer,
                     max_length=max_length,
                     split='val')
    print("test datasets:")
    test_ds = Library(file_path=test_fp,
                      tokenizer=tokenizer,
                      max_length=max_length,
                      split='test')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, device=device)
    )
    val_loader = itertools.cycle(DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, device=device)
    ))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, device=device)
    )

    return train_loader, val_loader, test_loader
