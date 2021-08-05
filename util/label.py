import numpy as np

from sklearn.preprocessing import LabelEncoder


def encode_labels(ys):
    le = LabelEncoder()
    encoded_labels = le.fit_transform(ys)
    return np.unique(encoded_labels).astype("str"), encoded_labels


def encode_onehot_labels(ys, labels_index, encoded_labels):
    if labels_index is None or encoded_labels is None:
        labels_index, encoded_labels = encode_labels(ys)
    onehot_labels = np.zeros((len(ys), len(labels_index)))
    for i, col_i in enumerate(encoded_labels):
        onehot_labels[i][col_i] = 1
    return onehot_labels


