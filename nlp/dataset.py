from typing import Any, List, Iterable

from collections import Counter

import numpy as np

from nlp.preprocess.text.tokenize import Tokenizer
from nlp.preprocess.feature.transform import build_index_and_counter_from_sequence, build_index_and_counter_from_sequences, build_label_index, \
    build_word_index_and_counter, transform_sequences_to_index_sequences, transform_sequences_to_one_hot_matrix
from util.eda import SequenceStatistics


class Vocabulary(object):

    def __init__(self, sequences):
        word_index, counter = build_word_index_and_counter(sequences)
        self.w2i = word_index
        self.i2w = {i: w for w, i in word_index.items()}
        self.wc = counter

    def word_index(self):
        return self.w2i

    def index_word(self):
        return self.i2w

    def check_coverage(self, vocab):

        oov = Counter()
        iv = Counter()

        for w in vocab.w2i:
            if w in self.w2i:
                iv[w] += vocab.wc[w]
            else:
                oov[w] += vocab.wc[w]

        ivc, oovc = sum(iv.values()), sum(oov.values())

        return iv, ivc, oov, oovc

    def __len__(self):
        return len(self.w2i)

    def __contains__(self, token):
        return token in self.w2i.keys()


class BaseEncoder(object):
    def __init__(self,
                 entries: Any[List, Iterable, np.array],
                 index_counter_builder_fn):
        entry_index, _ = index_counter_builder_fn(entries)
        self.entry_to_index = entry_index
        self.index_to_entry = {index: entry for entry, index in entry_index.items()}

    def entry_index(self) -> dict:
        return self.entry_to_index

    def index_entry(self) -> dict:
        return self.index_to_entry

    def index_size(self) -> int:
        return len(self.entry_to_index)

    def encode(self, entries: List) -> np.array:
        assert sum([1 for entry in entries if entry in self.entry_to_index.keys()]) == len(entries)
        return np.asarray([self.entry_to_index[entry] for entry in entries])

    def decode(self, indexes) -> np.array:
        assert sum([1 for index in indexes if index in self.index_to_entry.keys()]) == len(indexes)
        return np.asarray([self.index_to_entry[index] for index in indexes])


class EntryEncoder(BaseEncoder):

    def __init__(self, entries: Any[List, Iterable, np.array]):
        super().__init__(entries, build_index_and_counter_from_sequence)


class SequenceEncoder(BaseEncoder):

    def __init__(self, entries: Any[List, Iterable, np.array]):
        super().__init__(entries, build_index_and_counter_from_sequences)


class LabelEncoder(EntryEncoder):

    def __init__(self, labels: Iterable):
        super().__init__(labels)

    def label_index(self):
        return super().entry_index()

    def index_label(self):
        return self.i2l

    def label_size(self):
        return len(self.l2i)

    def encode(self, labels):
        return super().entry_index(labels)

    def decode(self, labels):
        return super().decode(labels)


class OneHotEncoder:

    def __init__(self,
                 sequences: Any[List, Iterable, np.array],
                 seq_length: int,
                 entry_index: dict = None):
        transform_sequences_to_one_hot_matrix(sequences, seq_length=seq_length, entry_index=entry_index)
        self.seq_length = seq_length
        self.entry_index = entry_index

    def encode(self, sequences: Any[List, Iterable, np.array]):
        return transform_sequences_to_one_hot_matrix(sequences, seq_length=self.seq_length, entry_index=self.entry_index)

    def decode(self):
        pass


class SequenceDataset(object):

    def __init__(self,
                 sequences: list,
                 labels: list,
                 seq_length: int,
                 sequence_encoder: SequenceEncoder,
                 label_encoder: LabelEncoder
                 ):
        """

        """
        self.sequences = sequences

        if labels is not None:
            if label_encoder is None:
                self.label_encoder = LabelEncoder(labels)
            else:
                self.label_encoder = label_encoder
            self.labels = self.label_encoder.encode(labels)

        if sequence_encoder is None:
            self.sequence_encoder = SequenceEncoder(sequences)
        else:
            self.sequence_encoder = sequence_encoder

        self.index_sequences = transform_sequences_to_index_sequences(
            self.sequences,
            entry_index=self.sequence_encoder.entry_index(),
            seq_length=seq_length
        )

    def get_sequence_lengths(self):
        return np.asarray([len(seq) for seq in self.sequences])

    def entry_set_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.texts)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    def get_label_size(self):
        return self.label_encoder.label_size()

    def get_sequences(self):
        return self.index_sequences

    def get_vocab(self):
        return self.vocab

    def get_label_encoder(self):
        return self.label_encoder

    def get_sequences_by_index(self, index):
        return self.index_sequences[index]

    def get_labels_by_index(self, index):
        return self.labels[index]

    def stats(self) -> str:
        stats = SequenceStatistics(self.sequences)
        return stats.report()

    def vocab_coverage(self) -> float:

        hit_count = 0
        miss_count = 0
        for sequence in self.sequences:
            for token in sequence:
                if token in self.vocab:
                    hit_count += 1
                else:
                    miss_count += 1

        if hit_count + miss_count == 0:
            return 0.0

        return hit_count / float(hit_count + miss_count)

    def get_oov(self) -> Counter:
        oov = Counter()
        for sequence in self.sequences:
            for token in sequence:
                if token not in self.vocab:
                    oov[token] += 1

        return oov

    def label_dist(self):
        labels_count = Counter()

        for label in self.labels:
            labels_count[label] += 1

        return labels_count


class TextDataset(object):

    def __init__(
            self,
            texts: list,
            labels: list,
            tokenizer: Tokenizer,
            seq_length: int,
            vocab: Vocabulary,
            label_encoder: LabelEncoder
    ):
        self.texts = texts

        self.label_encoder = label_encoder
        if labels is not None:
            if label_encoder is None:
                self.label_encoder = LabelEncoder(labels)
            self.labels = self.label_encoder.encode(labels)

        self.tokenizer = tokenizer

        self.sequences = tokenizer.tokenize(texts)
        if vocab is None:
            self.vocab = Vocabulary(self.sequences)
        else:
            self.vocab = vocab

        self.index_sequences = transform_sequences_to_index_sequences(
            self.sequences,
            entry_index=self.vocab.w2i,
            seq_length=seq_length
        )

    def get_sequence_lengths(self):
        return np.asarray([len(seq) for seq in self.sequences])

    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.texts)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    def get_label_size(self):
        return self.label_encoder.label_size()

    def get_sequences(self):
        return self.index_sequences

    def get_vocab(self):
        return self.vocab

    def get_label_encoder(self):
        return self.label_encoder

    def get_sequences_by_index(self, index):
        return self.index_sequences[index]

    def get_labels_by_index(self, index):
        return self.labels[index]

    def stats(self) -> str:
        stats = SequenceStatistics(self.sequences)
        return stats.report()

    def vocab_coverage(self) -> float:

        hit_count = 0
        miss_count = 0
        for sequence in self.sequences:
            for token in sequence:
                if token in self.vocab:
                    hit_count += 1
                else:
                    miss_count += 1

        if hit_count + miss_count == 0:
            return 0.0

        return hit_count / float(hit_count + miss_count)

    def get_oov(self) -> Counter:
        oov = Counter()
        for sequence in self.sequences:
            for token in sequence:
                if token not in self.vocab:
                    oov[token] += 1

        return oov

    def label_dist(self):
        labels_count = Counter()

        for label in self.labels:
            labels_count[label] += 1

        return labels_count
