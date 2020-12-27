import os
import pandas as pd

from models.nlp.framework import SupervisedNNModelTrainConfig
from models.nlp.bert.pt.text_cls_ft_trainer import BertModelConfig, BertTrainDataset, QuickBertTrainer, BertTrainer
from util.corpus import train_test_split_from_data_frame


DATA_HOME = r"C:\Users\Charlie\Corpus\kaggle\nlp-getting-started"
MODEL_HOME = r"L:\developer\model\bert-base-uncased"


def main():
    df = pd.read_csv(os.path.join(DATA_HOME, "train.csv"))
    df["target"] = df["target"].apply(lambda x: int(x))
    train_df, dev_df = train_test_split_from_data_frame(df)

    xs_train = train_df["text"].tolist()
    ys_train = train_df["target"].tolist()

    xs_dev = dev_df["text"].tolist()
    ys_dev = dev_df["target"].tolist()

    # label_list = list(set(train_ys))
    # label2id = {label: i for i, label in enumerate(label_list)}

    train_config = SupervisedNNModelTrainConfig(
        epoch=7,
        train_batch_size=64,
        eval_batch_size=64,
        logging_dir="L:/tmp/logging",
        output_dir="L:/tmp/output",
        warmup_steps=500
    )

    bert_model_config = BertModelConfig(
        config_path=os.path.join(MODEL_HOME, "config.json"),
        tokenizer_path=os.path.join(MODEL_HOME, "vocab.txt"),
        model_path=os.path.join(MODEL_HOME, "pytorch_model.bin"),
        cache_dir="L/tmp/cache",
        freeze_pretrained_model_while_training=True
    )

    trainer = BertTrainer(train_config=train_config, bert_model_config=bert_model_config)
    trainer.fit(train_data=(xs_train, ys_train), valid_data=(xs_dev, ys_dev))


if __name__ == "__main__":
    main()