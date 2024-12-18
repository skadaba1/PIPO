from graph_part import train_test_validation_split, stratified_k_fold
from sys import exit
import condacolab
import pandas as pd
import numpy as np

def homology_partition(file_path, sequences):
    train_idx, test_idx, valid_idx = train_test_validation_split(sequences,
                                                                alignment_mode='needle',
                                                                threads = 8,
                                                                threshold = 0.5,
                                                                test_size = 0.15,
                                                                valid_size = 0.05,
                                                                )
    import pandas as pd
    df = pd.read_csv(file_path)
    df_train = df.iloc[train_idx]
    df_train.to_csv("tf_library_train.csv")
    df_val = df.iloc[valid_idx]
    df_val.to_csv("tf_library_val.csv")
    df_test = df.iloc[test_idx]
    df_test.to_csv("tf_library_test.csv")
    print(len(train_idx), len(valid_idx), len(test_idx))

def filter_unbounded(df, column_name):
    return df[np.isfinite(df[column_name])]

def split_and_save_dataframe(df, train_ratio=0.8, val_ratio=0.05, test_ratio=0.15, train_file='train.csv', val_file='val.csv', test_file='test.csv'):
    assert train_ratio + val_ratio + test_ratio == 1.0, "The ratios must sum to 1.0"

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # df = filter_unbounded(df, 'Pred_affinity')
    # df = filter_unbounded(df, 'avg_retention_score')
    # df['ret_induc_prod'] = df['avg_retention_score'] * df['avg_induction_score']

    train_size = int(train_ratio * len(df))
    val_size = int(val_ratio * len(df))
    test_size = len(df) - train_size - val_size

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    return train_df, val_df, test_df