import os
from preprocess.feature.transform import build_label_index, build_word_index_and_counter, transform_token_seqs_to_word_index_seqs, transoform_seq_to_one_hot

DATA_HOME = r"L:\developer\msra"


def get_data(file_path):

    xs, ys = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        x, y = [], []
        for line in f:
            splits = line.rstrip().split()

            if len(splits) != 2:
                xs.append(x)
                ys.append(y)
                x, y = [], []
                continue

            x.append(splits[0])
            y.append(splits[1])

    return xs, ys


def prepare_for_data():

    xs_train, ys_train = get_data(os.path.join(DATA_HOME, "msra_train_bio.txt"))
    xs_test, ys_test = get_data(os.path.join(DATA_HOME, "msra_test_bio.txt"))

    seqs = xs_train + xs_test
    labels = ys_train + ys_test

    word_index, _ = build_word_index_and_counter(seqs)

    label_index, _ = build_word_index_and_counter(labels, with_unknown=False, start_from=0)

    xs_seq_train = transform_token_seqs_to_word_index_seqs(xs_train, 128, word_index=word_index)
    xs_seq_test = transform_token_seqs_to_word_index_seqs(xs_test, 128, word_index=word_index)

    ys_index_train = transoform_seq_to_one_hot(ys_train, 128, word_index=label_index)
    ys_index_test = transoform_seq_to_one_hot(ys_test, 128, word_index=label_index)

    return xs_seq_train, ys_index_train, xs_seq_test, ys_index_test, word_index, label_index
