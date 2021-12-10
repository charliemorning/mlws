from typing import Union, List, Iterable
from collections import Counter

import numpy as np

UNKNOWN_ENTRY = object()

UNKNOWN_STR = "<unk>"


def build_index_and_counter_from_sequence(sequence: Union[List, Iterable, np.array],
                                          start_from: int = 0):
    """Given a sequence, to build its index and count of each entry in sequence.

    TODO: This is a common method probably, move it to global package.

    >>> sequence = ['a', 'b', 'c', 'b', 'c', 'd']
    >>> index, counter = build_index_and_counter_from_sequence(sequence)
    >>> print(index)
    ... {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    >>> print(counter)
    ... {'a': 1, 'b': 2, 'c': 2, 'd': 1}
    """
    assert sequence is not None, "sequences must be not None."
    assert start_from > 0, "start_from should be great than 0."

    entry_counter = Counter()
    entry_index = {}

    for entry in sequence:
        entry_counter[entry] += 1
        if entry not in entry_index:
            entry_index[entry] = len(entry_index) + start_from
    return entry_index, entry_counter


def build_index_and_counter_from_sequences(sequences: Union[List, iter],
                                           with_unknown: bool = True,
                                           unknown=UNKNOWN_ENTRY,
                                           start_from: int = 1
                                           ) -> (dict, Counter):
    """Given a list of sequences, build its index and count of each entry in sequence of sequences.

    with_unknown is default to be True, which means an object represents unknown value is add to the index table.
    counter returned by method does not include the unknown object.

    TODO: This is a common method probably, move it to global package.

    >>> sequences = [['a', 'b', 'c'], ['b', 'c', 'd']]
    >>> index, counter = build_index_and_counter_from_sequences(sequences)
    >>> print(index)
    ... {'<unk>': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5}
    >>> print(counter)
    ... {'a': 1, 'b': 2, 'c': 2, 'd': 1}
    """
    assert sequences is not None, "sequences must be not None."
    assert unknown is not None, "unknown must be not None."
    assert start_from > 0, "start_from should be great than 0."

    entry_counter = Counter()
    entry_index = {}

    if with_unknown:
        entry_index[unknown] = start_from

    for sequence in sequences:
        for entry in sequence:
            entry_counter[entry] += 1
            if entry not in entry_index:
                entry_index[entry] = len(entry_index) + start_from

    return entry_index, entry_counter


def build_label_index(labels: Union[List, Iterable, np.array],
                      start_from: int = 0
                      ) -> (dict, Counter):
    """To build label index from label sequence.
    """
    label_index, _ = build_index_and_counter_from_sequence(labels, start_from=start_from)
    return label_index


def build_word_index_and_counter(sequences: Union[List, Iterable, np.array],
                                 with_unknown: bool = True,
                                 unknown_str: str = UNKNOWN_STR,
                                 start_from: int = 1
                                 ) -> (dict, Counter):
    """To build word index and word counter from text sequence
    """
    assert type(unknown_str) is str, "unknown_str must be string."
    return build_index_and_counter_from_sequences(sequences, with_unknown, unknown_str, start_from)


def transform_sequence_to_one_hot_matrix(sequence: Union[List, Iterable, np.array],
                                          seq_length: int,
                                          entry_index: dict = None) -> np.array:
    """
    """
    if entry_index is None:
        entry_index, _ = build_index_and_counter_from_sequence(sequence, start_from=0)

    # 2-D array
    one_hot_matrix = np.zeros((seq_length, len(entry_index)), dtype="int32")

    for i, entry in enumerate(sequence):
        if i > seq_length - 1:
            break
        one_hot_matrix[i][entry_index[entry]] = 1

    return one_hot_matrix


def transform_sequences_to_one_hot_matrix(sequences: Union[List, Iterable, np.array],
                                          seq_length: int,
                                          entry_index: dict = None) -> np.array:
    if entry_index is None:
        entry_index, _ = build_index_and_counter_from_sequences(sequences, with_unknown=False, start_from=0)

    # 3-D array
    one_hot_matrix = np.zeros((len(sequences), seq_length, len(entry_index)), dtype="int32")

    for i, sequence in enumerate(sequences):
        for j, entry in enumerate(sequence):
            if j > seq_length - 1:
                break
            one_hot_matrix[i][j][entry_index[entry]] = 1

    return one_hot_matrix


def transform_sequences_to_index_sequences(sequences: Union[List, Iterable, np.array],
                                           seq_length: int,
                                           from_right_to_left=False,
                                           entry_index=None) -> np.array:
    if entry_index is None:
        entry_index, _ = build_word_index_and_counter(sequences)

    index_sequences = np.zeros((len(sequences), seq_length), dtype="int32")
    for i, seq in enumerate(sequences):

        if from_right_to_left:
            j = seq_length - 1
            for word in reversed(seq):
                if j < 0:
                    break
                index_sequences[i][j] = entry_index[word] if word in entry_index else entry_index[UNKNOWN_STR]
                j -= 1
        else:
            for k, word in enumerate(seq):
                if k > seq_length - 1:
                    break
                index_sequences[i][k] = entry_index[word] if word in entry_index else entry_index[UNKNOWN_STR]

    return index_sequences


def build_embedding_matrix(embedding_index, word_index, embedding_dim) -> np.array:
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        if word in embedding_index:
            embedding_matrix[i] += embedding_index[word]
        else:
            embedding_matrix[i] += np.random.rand(embedding_dim)

    return embedding_matrix