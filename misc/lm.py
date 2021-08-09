from gensim.test.utils import common_texts
from gensim.models import Word2Vec, FastText
import json


START_TOKEN = "<s>"
END_TOKEN = "<e>"


def tokenize(text, n):

    expand_tokens = [START_TOKEN] + text + [END_TOKEN]
    ngrams = []
    for i in range(len(text) + 1):
        ngram = tuple(expand_tokens[i:i + n])
        ngrams.append(str(ngram))

    return ngrams


# datagrand_2021_unlabeled_data.json
class MySentences(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        for line in open(self.filepath):
            obj = json.loads(line)
            title = obj["title"].split()
            content = obj["content"].split()
            text = title + content
            # print(text)
            yield tokenize(text, 2)


sentences = MySentences("L:/developer/datagrand_2021_unlabeled_data.json")

model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=5, workers=8)


model.save("L:/developer/word2vec.skipgram.bigram.300d.model")