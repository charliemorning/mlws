import jieba
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer


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


class Tokenizer:

    def __init__(self,
                 num_words=5000,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True
                 ):
        pass

    def word_index(self):
        pass