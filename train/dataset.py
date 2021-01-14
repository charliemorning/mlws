import numpy as np

from preprocess.text.tokenize import Tokenizer
from preprocess.feature.transform import build_word_index_and_counter, transform_token_seqs_to_word_index_seqs


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

    def __init__(
            self,
            texts: list,
            labels: list,
            tokenizer: Tokenizer,
            seq_length: int
    ):
        self.texts = texts
        self.labels = np.asarray(labels)
        self.tokenizer = tokenizer

        self.sequences = tokenizer.tokenize(texts)
        self.vocab = Vocabulary(self.sequences)

        self.index_sequences = transform_token_seqs_to_word_index_seqs(
            self.sequences,
            word_index=self.vocab.w2i,
            seq_length=seq_length
        )

    def get_vocab(self):
        return self.vocab

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
        return self.labels[index]