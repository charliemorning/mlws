from collections import Counter

import numpy as np


UNKNOWN = "<unk>"


def build_label_index(labels: iter,
                      start_from: int = 0
                      ) -> (dict, Counter):
    label_counter = Counter()
    label_index = {}
    for label in labels:
        label_counter[label] += 1
        if label not in label_index:
            label_index[label] = len(label_index) + start_from

    return label_index, label_counter


def build_word_index_and_counter(sequences: iter,
                                 with_unknown: bool = True,
                                 unknown_str: str = UNKNOWN,
                                 start_from: int = 1
                                 ) -> (dict, Counter):

    word_counter = Counter()
    word_index = {}
    if with_unknown:
        word_index[unknown_str] = start_from

    for seq in sequences:
        for word in seq:
            word_counter[word] += 1
            if word not in word_index:
                word_index[word] = len(word_index) + start_from
    return word_index, word_counter


def build_embedding_matrix(embedding_index, word_index, embedding_dim) -> np.array:

    embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))

    for word, i in word_index.items():
        if word in embedding_index:
            embedding_matrix[i] += embedding_index[word]
        else:
            embedding_matrix[i] += np.random.rand(embedding_dim)

    return embedding_matrix


def transoform_seq_to_one_hot(sequences,
                              seq_length: int,
                              word_index=None) -> np.array:

    if word_index is None:
        word_index, _ = build_word_index_and_counter(sequences, with_unknown=False, start_from=0)

    onehot = np.zeros((len(sequences), seq_length, len(word_index)), dtype="int32")

    for i, seq in enumerate(sequences):
        for j, word in enumerate(seq):
            if j > seq_length - 1:
                break
            onehot[i][j][word_index[word]] = 1

    return onehot


def transform_token_seqs_to_word_index_seqs(sequences,
                                            seq_length: int,
                                            from_right_to_left=False,
                                            word_index=None) -> np.array:

    if word_index is None:
        word_index, _ = build_word_index_and_counter(sequences)

    index_seq = np.zeros((len(sequences), seq_length), dtype="int32")
    for i, seq in enumerate(sequences):

        if from_right_to_left:
            j = seq_length - 1
            for word in reversed(seq):
                if j < 0:
                    break

                index_seq[i][j] = word_index[word] if word in word_index else word_index[UNKNOWN]
                j -= 1
        else:
            for k, word in enumerate(seq):
                if k > seq_length - 1:
                    break

                index_seq[i][k] = word_index[word] if word in word_index else word_index[UNKNOWN]

    return index_seq

