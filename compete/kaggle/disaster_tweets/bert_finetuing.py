import os
import argparse

import pandas as pd


from algorithm.nlp.tf.v1.bert.bert_models import BertTrainer
from util.data.split import train_test_split_from_data_frame
from util.data.label import encode_labels



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='train cls model on bert.')

    parser.add_argument('bert_home', type=str)

    parser.add_argument('data_home', type=str)

    parser.add_argument('--batch-size', type=int, default=4, metavar='N', help='input batch size for training (default: 4)')

    args = parser.parse_args()

    bert_home = args.bert_home
    data_home = args.data_home

    df = pd.read_csv(os.path.join(data_home, "train.csv"))[:100]
    df["target"] = df["target"].apply(lambda x: str(x))
    train_df, dev_df = train_test_split_from_data_frame(df)

    # test_pd = pd.read_csv(os.path.join(data_home, "test.csv"))
    # xs_test, ys_test = dev_df["text"], dev_df["target"].apply(lambda x: str(x))

    bert = BertTrainer(
        os.path.join(bert_home, 'bert_config.json'),
        os.path.join(bert_home, 'vocab.txt'),
        os.path.join(bert_home, 'bert_model.ckpt'),
        data_home,
        os.path.join(data_home, 'output'),
        BertTrainer.DataFrameInput(
            BertTrainer.DataFrameSingleTextInputProcessor("id", "text", "target"), encode_labels(df["target"]), train_df, dev_df),
        train_batch_size=8, num_train_epochs=1, max_seq_length=16
    )

    bert.train()
    print("done")