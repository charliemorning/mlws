from collections import Counter

import numpy as np

from nlp.preprocess import Tokenizer
from nlp.preprocess.feature import build_label_index, build_word_index_and_counter, transform_token_seqs_to_word_index_seqs
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


class LabelEncoder(object):

    def __init__(self, labels):
        label_index, label_counter = build_label_index(labels)
        self.l2i = label_index
        self.i2l = {i: w for w, i in label_index.items()}
        self.lc = label_counter

    def label_index(self):
        return self.l2i

    def index_label(self):
        return self.i2l

    def label_size(self):
        return len(self.l2i)

    def encode(self, labels):
        assert sum([1 for l in labels if l in self.l2i.keys()]) == len(labels)
        return np.asarray([self.l2i[l] for l in labels])

    def decode(self, labels):
        assert sum([1 for l in labels if l in self.i2l.keys()]) == len(labels)
        return np.asarray([self.i2l[i] for i in labels])


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

        self.index_sequences = transform_token_seqs_to_word_index_seqs(
            self.sequences,
            word_index=self.vocab.w2i,
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
