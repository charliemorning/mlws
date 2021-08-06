import pandas as pd

from train.dataset import TextDataset
from preprocess.text.tokenize import NGramTokenizer
from util.label import encode_labels


def load_labelled_data_as_dataset(path):

    df = pd.read_csv(path, index_col="id")
    df['sequences'] = df["text"].apply(str.split)

    dataset = TextDataset(df['sequences'].tolist(), df["label"].tolist(), tokenizer=NGramTokenizer(), seq_length=128, label_encoder=encode_labels)
    return dataset


def load_test_data_as_dataset(path):
    df = pd.read_csv(path, index_col="id")
    df['sequences'] = df["text"].apply(str.split)

    dataset = TextDataset(df['sequences'].tolist(), df["label"].tolist(), tokenizer=NGramTokenizer(), seq_length=128,
                          label_encoder=encode_labels)
    return dataset