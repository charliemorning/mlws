import numpy as np
import pickle
import os
import logging

import torch

from train.torch.nlp.dataset import TextMCCDataset
from preprocess.feature.transform import UNKNOWN

def load_embedding_index(path: str, ext="pickle"):

    pickle_path = path + "." + ext

    if os.path.exists(pickle_path):
        logging.info("load embedding from %s..." % pickle_path)
        with open(pickle_path, "rb") as f:
            embedding_index = pickle.load(f)
    else:
        logging.info("load embedding from %s..." % path)
        embedding_index = {}
        with open(path, "rb") as f:
            for line in f:
                values = line.split()
                word = values[0].decode("utf-8")
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs

        num_words = len(embedding_index)
        logging.info("load %d words." % (num_words))
        with open(pickle_path, "wb") as f:
            pickle.dump(embedding_index, f)

    return embedding_index


def rebuild_embedding_index(embedding_index, word_index):

    new_embedding_index = {}
    for word in word_index:
        if word in embedding_index:
            new_embedding_index[word] = embedding_index[word]

    return new_embedding_index


def build_embedding_matrix(embedding_index, word_index, oov_strategy=None):

    # get embedding size
    for embedding in embedding_index.values():
        embedding_size = len(embedding)
        break

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_size), dtype=np.float)

    for word, i in word_index.items():
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        else:
            embedding_matrix[i] = np.random.randn(embedding_size)

    return embedding_matrix


def predict(model, device, xs_test, batch_size):
    model.eval()

    test_dataset = TextMCCDataset(xs_test, None)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    preds = torch.tensor([], dtype=torch.int)

    with torch.no_grad():
        for data in test_dataloader:
            # data = data.to(device)
            output = model(data)

            batch_preds = torch.argmax(torch.softmax(output, dim=1), dim=1).flatten()
            preds = torch.cat((preds, batch_preds))

    return preds

