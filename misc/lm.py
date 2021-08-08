from gensim.test.utils import common_texts
from gensim.models import Word2Vec, FastText
import json

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
            yield text


sentences = MySentences("L:/developer/datagrand_2021_unlabeled_data.json")

model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=1, workers=4)


model.save("L:/developer/word2vec.skipgram.unigram.300d.model")