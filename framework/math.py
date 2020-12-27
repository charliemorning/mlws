import numpy as np


def kullback_leibler_divergence(p, q):
    return np.sum(p * np.log(p / q))
