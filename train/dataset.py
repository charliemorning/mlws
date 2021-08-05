import numpy as np

from preprocess.text.tokenize import Tokenizer
from preprocess.feature.transform import build_word_index_and_counter, transform_token_seqs_to_word_index_seqs
from util.label import encode_onehot_labels


class Vocabulary(object):

    def __init__(self, sequences):
        word_index, counter = build_word_index_and_counter(sequences)
        self.w2i = word_index
        self.i2w = {i: w for w, i in word_index.items()}

    def word_index(self):
        return self.w2i

    def index_word(self):
        return self.i2w

    def __len__(self):
        return len(self.w2i)


class TextDataset(object):

    def get_vocab(self):
        return self.vocab

    def __init__(
            self,
            texts: list,
            labels: list,
            tokenizer: Tokenizer,
            seq_length: int,
            label_encoder=None
    ):
        self.texts = texts
        if label_encoder is None:
            self.labels = np.asarray(labels)
        else:
            self.label_index, self.labels = label_encoder(labels)
            self.onehot_labels = encode_onehot_labels(labels, self.label_index, self.labels)
        self.tokenizer = tokenizer

        self.sequences = tokenizer.tokenize(texts)
        self.vocab = Vocabulary(self.sequences)

        self.index_sequences = transform_token_seqs_to_word_index_seqs(
            self.sequences,
            word_index=self.vocab.w2i,
            seq_length=seq_length
        )

    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.texts)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    def get_sequences_by_index(self, index):
        return self.index_sequences[index]

    def get_labels_by_index(self, index):
        if self.onehot_labels is not None:
            return self.onehot_labels[index]
        else:
            return self.labels[index]