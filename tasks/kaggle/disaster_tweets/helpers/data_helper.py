import os
import re
import itertools

import emoji
from tqdm import tqdm
import pandas as pd

from nlp.preprocess.text.tokenize import NLTKTokenizer
from nlp.dataset import TextDataset
from util.model import load_embedding_index, rebuild_embedding_index, build_embedding_matrix

tqdm.pandas()

# nlp = spacy.load("en_core_web_sm")

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "thx": "thanks"
}


def remove_contractions(text):
    expaneded = []
    for w in text.split(" "):
        expaneded.append(contractions[w] if w in contractions.keys() else text)
    return " ".join(expaneded)


CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'


def remove_special_char(text):
    return "".join(c for c in text if c not in CHARS_TO_REMOVE)


def clean_dataset(text):
    # To lowercase
    text = text.lower()
    # Remove hashtag while keeping hashtag text
    text = re.sub(r'#', '', text)
    # Remove HTML special entities (e.g. &amp;)
    text = re.sub(r'\&\w*;', '', text)
    # Remove tickers
    text = re.sub(r'\$\w*', '', text)
    # Remove hyperlinks
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)
    # Remove whitespace (including new line characters)
    text = re.sub(r'\s\s+', '', text)
    text = re.sub(r'[ ]{2, }', ' ', text)
    # Remove URL, RT, mention(@)
    text = re.sub(r'http(\S)+', '', text)
    text = re.sub(r'http ...', '', text)
    text = re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+', '', text)
    text = re.sub(r'RT[ ]?@', '', text)
    text = re.sub(r'@[\S]+', '', text)
    # Remove words with 2 or fewer letters
    text = re.sub(r'\b\w{1,2}\b', '', text)
    # &, < and >
    text = re.sub(r'&amp;?', 'and', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    # Insert space between words and punctuation marks
    text = re.sub(r'([\w\d]+)([^\w\d ]+)', '\1 \2', text)
    text = re.sub(r'([^\w\d ]+)([\w\d]+)', '\1 \2', text)
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    text = ''.join(c for c in text if c <= '\uFFFF')
    text = text.strip()
    # Remove misspelling words
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    # Remove punctuation
    text = ' '.join(re.sub("[\.\,\!\?\:\;\-\=\/\|\'\(\']", " ", text).split())
    # Remove emoji
    text = emoji.demojize(text)
    text = text.replace(":", " ")
    text = ' '.join(text.split())
    text = re.sub("([^\x00-\x7F])+", " ", text)
    # Remove Mojibake (also extra spaces)
    text = ' '.join(re.sub("[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
    return text


def prepare_data(data_home):
    df = pd.read_csv(os.path.join(data_home, "train.csv"))
    df["target"] = df["target"].progress_apply(lambda x: int(x))
    df["text"] = df["text"].progress_apply(lambda s: s.lower())
    df['text'] = df['text'].progress_apply(remove_contractions)
    # df['text'] = df['text'].progress_apply(clean_dataset)
    return df


def prepare_data_for_cnn_and_rnn(data_home, model_path):
    df = pd.read_csv(os.path.join(data_home, "train.csv"))
    df["target"] = df["target"].progress_apply(lambda x: int(x))
    df["text"] = df["text"].progress_apply(lambda s: s.lower())
    df['text'] = df['text'].progress_apply(remove_contractions)
    # df['text'] = df['text'].progress_apply(clean_dataset)

    dataset = TextDataset(df['text'].tolist(), df["target"].tolist(), tokenizer=NLTKTokenizer(), seq_length=128)

    embedding_index = load_embedding_index(model_path)
    embedding_index = rebuild_embedding_index(embedding_index, dataset.get_vocab().word_index())
    embedding_matrix = build_embedding_matrix(embedding_index, dataset.get_vocab().word_index())

    return dataset, embedding_matrix
