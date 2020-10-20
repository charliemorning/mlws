import numpy as np


def train_test_split_from_data_frame(df, test_size=0.3, reset_index=True):

    msk = np.random.rand(len(df)) > test_size
    train_df = df[msk]
    test_df = df[~msk]

    if reset_index is True:
        return train_df.reset_index(), test_df.reset_index()
    else:
        return train_df, test_df

