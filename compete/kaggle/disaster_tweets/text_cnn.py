import os
import logging
import argparse

import pandas as pd


from preprocess.text.tokenize import keras_tokenizer
from preprocess.feature.transform import transform_sequences_to_one_hot
from algorithm.nlp.framework import SupervisedNNModelTrainConfig
from algorithm.nlp.text_cnn.tf_keras_ver import TextCNN
from util.eda import *
from util.corpus import train_test_split_from_data_frame
from util.metric import precision, recall, f1


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='train cls model on text cnn.')

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
    xs_train = transform_sequences_to_one_hot(train_df["seq"], 128)
    ys_train = train_df["target"].apply(lambda x: float(x))

    # test_pd = pd.read_csv(os.path.join(data_home, "test.csv"))
    # xs_test, ys_test = dev_df["text"], dev_df["target"].apply(lambda x: str(x))

    config = SupervisedNNModelTrainConfig(epoch=50, batch_size=128)

    text_cnn = TextCNN(
        config,
        5000,
        128,
        300,
        250,
        3,
        1,
        metric_callbacks=['accuracy', precision, recall, f1]
    )

    text_cnn.fit(xs_train, ys_train)
    print("done")