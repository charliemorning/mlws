from collections import Counter

import numpy as np
import pandas as pd


UNKNOWN = "<unk>"


def build_word_index_and_counter(sequences: iter) -> (dict, Counter):

    word_counter = Counter()
    word_index = {}
    word_index[UNKNOWN] = 0
    for seq in sequences:
        for word in seq:
            word_counter[word] += 1
            if word not in word_index:
                word_index[word] = len(word_index)
    return word_index, word_counter


def build_embedding_matrix(embedding_index, word_index, embedding_dim) -> np.array:

    embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))

    for word, i in word_index.items():
        if word in embedding_index:
            embedding_matrix[i] += embedding_index[word]
        else:
            embedding_matrix[i] += np.random.rand(embedding_dim)

    return embedding_matrix


def transform_token_seqs_to_word_index_seqs(sequences, seq_length: int, word_index=None) -> np.array:

    if word_index is None:
        word_index, _ = build_word_index_and_counter(sequences)

    one_hots = np.zeros((len(sequences), seq_length), dtype="int32")
    i = 0
    for seq in sequences:
        j = seq_length - 1
        for word in reversed(seq):
            if j < 0:
                break
            if word not in word_index:
                continue
            one_hots[i][j] = word_index[word] if word in word_index else 1
            j -= 1
        i += 1
    return one_hots

