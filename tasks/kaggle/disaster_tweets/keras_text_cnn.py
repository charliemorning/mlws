import os
import logging
import argparse

import pandas as pd


from nlp.preprocess import keras_tokenizer
from nlp.preprocess.feature import transform_token_seqs_to_word_index_seqs

from util.eda import *
from util.corpus import train_test_split_from_data_frame
from util.metric import precision, recall, f1


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='train cls train on text convolutional.')

    parser.add_argument('data_home', type=str)

    args = parser.parse_args()

    data_home = args.data_home

    df = pd.read_csv(os.path.join(data_home, "train.csv"))
    df["target"] = df["target"].apply(lambda x: str(x))

    tokenizer = keras_tokenizer()
    tokenizer.fit_on_texts(df["text"])
    df["seq"] = tokenizer.texts_to_sequences(df["text"])

    seq_stat = SequenceStatistics(df["seq"])
    logging.info(seq_stat.report())

    train_df, dev_df = train_test_split_from_data_frame(df)
    xs_train = transform_token_seqs_to_word_index_seqs(train_df["seq"], 128)
    ys_train = train_df["target"].apply(lambda x: float(x))

    xs_dev = transform_token_seqs_to_word_index_seqs(dev_df["seq"], 128)
    ys_dev = dev_df["target"].apply(lambda x: float(x))

    # test_pd = pd.read_csv(os.path.join(data_home, "test.csv"))
    # xs_test, ys_test = dev_df["text"], dev_df["target"].apply(lambda x: str(x))

    config = SupervisedNNModelTrainConfig(epoch=10, train_batch_size=128)

    cnn_model_config = TextCNNModelConfig(
        max_features=8000,
        max_seq_length=128,
        embedding_size=300,
        filters=250,
        kernel_size=3,
        dim_out=1
    )

    text_cnn = TextCNN(
        train_config=config,
        cnn_model_config=cnn_model_config,
        metric_callbacks=['accuracy', precision, recall, f1]
    )

    text_cnn.fit(xs_train, ys_train, validation_data=(xs_dev, ys_dev))
    print("done")