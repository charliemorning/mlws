import numpy as np
import pickle
import os
import logging


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
