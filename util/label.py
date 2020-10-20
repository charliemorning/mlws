import numpy as np

from sklearn.preprocessing import LabelEncoder


def encode_labels(ys):
    le = LabelEncoder()
    encoded_labels = le.fit_transform(ys)
    return np.unique(encoded_labels).astype("str")