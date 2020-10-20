from collections import Counter
import pprint

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class TextStatistics:
    """
    provide char-level statistics.
    """
    def __init__(self, texts):
        self.max_len = max(texts.str.len())
        self.min_len = min(texts.str.len())
        self.avg_len = texts.str.len().mean()
        self.std_len = texts.str.len().std()

    def report(self):
        stat_str = "maximum length: {max_len}\n" \
                   + "minimum length: {min_len}\n" \
                   + "average length: {avg_len}\n" \
                   + "standard length: {std_len}\n"
        return stat_str.format(max_len=self.max_len,
                               min_len=self.min_len,
                               avg_len=self.avg_len,
                               std_len=self.std_len)

    def __str__(self):
        return self.report()


class SequenceStatistics:

    def __init__(self, sequences, top_num=100):

        self.top_num = top_num
        self.word_counter = SequenceStatistics.build_word_counter(sequences)
        self.num_words = sum(self.word_counter.values())

        self.num_docs = len(sequences)

        self.seq_len_list = [len(sequence) for sequence in sequences]
        self.max_len = max(self.seq_len_list)
        self.min_len = min(self.seq_len_list)
        self.avg_len = np.mean(self.seq_len_list)
        self.std_len = np.std(self.seq_len_list)
        self.seq_len_counter = SequenceStatistics.build_seq_len_counter(self.seq_len_list)

        self.numeric_token_counter = SequenceStatistics.build_numeric_token_counter(sequences)

    @staticmethod
    def build_word_counter(sequences):
        word_counter = Counter()
        for sequence in sequences:
            for word in sequence:
                word_counter[word] += 1
        return word_counter

    @staticmethod
    def build_seq_len_counter(seq_len_list):
        len_counter = Counter()
        for seq_len in seq_len_list:
            len_counter[seq_len] += 1
        return len_counter

    @staticmethod
    def build_numeric_token_counter(sequences):
        num_token_counter = Counter()
        for sequence in sequences:
            for word in sequence:
                if any((type(word) is str and word.isdigit(), type(word) is int)):
                    num_token_counter[word] += 1
        return num_token_counter

    # @staticmethod
    def plot_len_hist(self):
        import matplotlib.pyplot as plt
        plt.hist(self.seq_len_counter.values())

    def report(self):
        stat_str = "=" * 45 + " Sequence Statistics " + "=" * 45 + "\n" \
        + "= number of documents: {num_docs:10}\n" \
        + "= number of words: {num_words:10}\n" \
        + "= most common words: {words}\n" \
        + "= most common len: {lens}\n" \
        + "= maximum length: {max_len:10}\n" \
        + "= minimum length: {min_len:10}\n" \
        + "= average length: {avg_len:10.2f}\n" \
        + "= standard length: {std_len:10.2f}\n" \
        + "= most common numeric tokens: {num_tokens}\n" \
        + "=" * 100

        return stat_str.format(num_docs=self.num_docs,
                               num_words=self.num_words,
                               words=self.word_counter.most_common(self.top_num),
                               lens=self.seq_len_counter.most_common(self.top_num),
                               max_len=self.max_len,
                               min_len=self.min_len,
                               avg_len=self.avg_len,
                               std_len=self.std_len,
                               num_tokens=self.numeric_token_counter.most_common(self.top_num)
                               )

    def __str__(self):
        return self.report()


def feature_selection(sequences, labels):
    strs = [" ".join(_) for _ in sequences]
    vectorizer = TfidfVectorizer(lowercase=False)
    tf_idf_vecs = vectorizer.fit_transform(strs).toarray()
    feature_names = vectorizer.get_feature_names()
    model = SelectKBest(chi2, k=20)
    model.fit(tf_idf_vecs, labels)
    features = {feature: model.scores_[i] for i, feature in enumerate(feature_names)}
    pprint.pprint(sorted(features.items(), key=lambda x: x[1], reverse=True)[:1000])