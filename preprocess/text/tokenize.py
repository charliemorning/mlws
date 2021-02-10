import jieba
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from nltk.tokenize import word_tokenize


def jieba_tokenize(text: str) -> list:
    return [w for w in jieba.cut(text)]


def keras_tokenizer(num_words=5000,
                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                    lower=True
                    ):
    return KerasTokenizer(num_words=num_words,
                     filters=filters,
                     lower=lower,
                     split=' ',
                     char_level=False,
                     oov_token=None)


class Tokenizer(object):

    def __init__(self,
                 vocab_size=5000,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True
                 ):
        pass

    def __len__(self):
        pass

    def word_index(self):
        pass

    def tokenize(self, texts):
        pass


class NLTKTokenizer(Tokenizer):

    def tokenize(self, texts):
        return [[token for token in word_tokenize(text)] for text in texts]