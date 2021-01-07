import os

from models.torch.trainer import SupervisedNNModelTrainConfig
from models.torch.nlp.network.bert.text_cls_ft import BertModelConfig, TextClsCustomedBertTrainer
from util.corpus import train_test_split_from_data_frame
from tasks.kaggle.disaster_tweets import prepare_data

MODEL_HOME = r"L:\developer\model\bert-base-uncased"


def main():
    df = prepare_data()
    train_df, dev_df = train_test_split_from_data_frame(df)

    xs_train = train_df["text"].tolist()
    ys_train = train_df["target"].tolist()

    xs_dev = dev_df["text"].tolist()
    ys_dev = dev_df["target"].tolist()

    # label_list = list(set(train_ys))
    # label2id = {label: i for i, label in enumerate(label_list)}

    train_config = SupervisedNNModelTrainConfig(
        epoch=20,
        train_batch_size=64,
        eval_batch_size=64,
        logging_dir="L:/tmp/logging",
        output_dir="L:/tmp/output",
        warmup_steps=500,
        dim_out=1
    )

    bert_model_config = BertModelConfig(
        config_path=os.path.join(MODEL_HOME, "config.json"),
        tokenizer_path=os.path.join(MODEL_HOME, "vocab.txt"),
        model_path=os.path.join(MODEL_HOME, "pytorch_model.bin"),
        cache_dir="L/tmp/cache",
        freeze_pretrained_model_while_training=False
    )

    trainer = TextClsCustomedBertTrainer(train_config=train_config, bert_model_config=bert_model_config)
    trainer.fit(train_data=(xs_train, ys_train), eval_data=(xs_dev, ys_dev))


if __name__ == "__main__":
    main()