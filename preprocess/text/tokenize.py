import jieba
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from nltk.tokenize import word_tokenize


START_TOKEN = "<s>"
END_TOKEN = "<e>"


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


class NGramTokenizer(Tokenizer):

    def __init__(self, n=1):
        self.n = n

    def tokenize(self, texts):

        ngrams_list = []

        for tokens in texts:
            expand_tokens = [START_TOKEN] * (self.n - 1) + tokens + [END_TOKEN] * (self.n - 1)
            ngrams = []
            for i in range(len(tokens) + 1):
                ngram = tuple(expand_tokens[i:i+self.n])
                ngrams.append(ngram)
            ngrams_list.append(ngrams)

        return ngrams_list


if __name__ == '__main__':
    texts = [
        ["a", "b", "c"],
        ["b", "c", "d"],
        ["c", "d", "e"]
    ]

    unigram_tokenizer = NGramTokenizer(1)
    bigram_tokenizer = NGramTokenizer(2)
    trigram_tokenizer = NGramTokenizer(3)

    print(unigram_tokenizer.tokenize(texts))
    print(bigram_tokenizer.tokenize(texts))
    print(trigram_tokenizer.tokenize(texts))
