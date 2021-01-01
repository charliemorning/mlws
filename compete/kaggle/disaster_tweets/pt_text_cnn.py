import os
import string
import pandas as pd

from models.nlp.framework import SupervisedNNModelTrainConfig
from models.nlp.text_cnn.pt_ver import TextCNNTrainer, TextCNNModelConfig, PoolingType
from preprocess.feature.transform import build_word_index_and_counter, transform_token_seqs_to_word_index_seqs
from util.corpus import train_test_split_from_data_frame

DATA_HOME = r"C:\Users\Charlie\Corpus\kaggle\nlp-getting-started"
MODEL_HOME = r"L:\developer\model\bert-base-uncased"


def main():
    df = pd.read_csv(os.path.join(DATA_HOME, "train.csv"))
    df["target"] = df["target"].apply(lambda x: int(x))
    df["sequence"] = df["text"].apply(lambda s: [t for t in s.split() if t not in string.punctuation])

    # build whole word index
    word_index, _ = build_word_index_and_counter(df["sequence"])

    train_df, dev_df = train_test_split_from_data_frame(df)

    xs_train = train_df["sequence"].tolist()
    ys_train = train_df["target"].tolist()

    xs_dev = dev_df["sequence"].tolist()
    ys_dev = dev_df["target"].tolist()

    xs_word_index_train = transform_token_seqs_to_word_index_seqs(xs_train, word_index=word_index, seq_length=128)
    xs_word_index_dev = transform_token_seqs_to_word_index_seqs(xs_dev, word_index=word_index, seq_length=128)


    train_config = SupervisedNNModelTrainConfig(
        learning_rate=0.005,
        epoch=80,
        train_batch_size=64,
        eval_batch_size=64
    )

    cnn_model_config = TextCNNModelConfig(
        max_features=len(word_index),
        max_seq_length=128,
        embedding_size=300,
        filters=[100, 100, 100],
        kernel_size=[3, 4, 5],
        padding=[0, 0, 0],
        dilation=[1, 1, 1],
        stride=[1, 1, 1],
        # filters=[100],
        # kernel_size=[3],
        # padding=[0],
        # dilation=[1],
        # stride=[1],
        pooling_type=PoolingType.AVG_POOLING,
        global_pooling=False,
        dim_out=1
    )

    trainer = TextCNNTrainer(train_config=train_config, cnn_model_config=cnn_model_config)

    trainer.fit(train_data=(xs_word_index_train, ys_train), eval_data=(xs_word_index_dev, ys_dev))


if __name__ == "__main__":
    main()