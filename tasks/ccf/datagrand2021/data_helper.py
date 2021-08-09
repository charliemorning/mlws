import json

import pandas as pd

from train.dataset import TextDataset
from preprocess.text.tokenize import NGramTokenizer
from util.label import encode_labels


def load_labelled_data_as_dataset(path):

    df = pd.read_csv(path, index_col="id")
    df['sequences'] = df["text"].apply(str.split)

    dataset = TextDataset(df['sequences'].tolist(), df["label"].tolist(), tokenizer=NGramTokenizer(2), seq_length=128, vocab=None, label_encoder=None)
    return dataset


def load_test_data_as_dataset(path, vocab, label_encoder):
    df = pd.read_csv(path, index_col="id")
    df['sequences'] = df["text"].apply(str.split)

    dataset = TextDataset(df['sequences'].tolist(), None, tokenizer=NGramTokenizer(2), seq_length=128, vocab=vocab,
                          label_encoder=label_encoder)
    return dataset


def make_submissions(path, data):
    id = [i for i in range(len(data))]
    df = pd.DataFrame({"id": id, "label": data})
    df.to_csv(path, index=False)


def prepare_glove_corpus(src_path, dst_path):
    with open(dst_path) as w_f:
        with open(src_path) as r_f:
            for l in r_f:
                obj = json.loads(l)
                title = obj["title"]
                content = obj["content"]
                text = title + " " + content + "\n"
                w_f.write(text)
            r_f.close()
        w_f.close()

