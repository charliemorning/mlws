import random
import numpy as np
import torch

from transformers import BertConfig, BertTokenizer, TrainingArguments

from models.nlp.framework import SupervisedNNModelTrainConfig, TrainFramework
from models.nlp.bert import BertModelConfig


def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


set_seed()


class BertTrainDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            xs: list,
            ys: list,
            tokenizer: BertTokenizer,
            max_length: int
    ):
        self.tokenized = tokenizer(
            xs,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=True
        )

        # self.tokenized.pop("token_type_ids")

        self.labels = [label for label in map(lambda x: torch.tensor(x), ys)]
        self.tokenized["label"] = self.labels

    def __len__(self):
        return len(self.tokenized["input_ids"])

    def __getitem__(self, index):
        return {
            "attention_mask": self.tokenized["attention_mask"][index],
            "input_ids": self.tokenized["input_ids"][index],
            "label": self.tokenized["label"][index]
        }


class BertTrainerBase(TrainFramework):
    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig,
            bert_model_config: BertModelConfig
    ):
        super(BertTrainerBase, self).__init__(
            train_config=train_config
        )

        self.bert_model_config = bert_model_config

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_config.tokenizer_path)

        self.bert_config = BertConfig.from_pretrained(
            pretrained_model_name_or_path=bert_model_config.config_path,
            # num_labels=len(label2id),
            # label2id=label2id,
            # id2label={id: label for label, id in label2id.items()},
            finetuning_task="text-classification",
            cache_dir=bert_model_config.cache_dir
        )

        self.training_args = TrainingArguments(
            output_dir=train_config.output_dir,
            num_train_epochs=train_config.epoch,
            per_device_train_batch_size=train_config.train_batch_size,
            per_device_eval_batch_size=train_config.eval_batch_size,
            warmup_steps=train_config.warmup_steps,
            weight_decay=train_config.weight_decay,
            logging_dir=train_config.logging_dir
        )

    def fit(self, train_data, eval_data=None):
        super().fit(train_data, eval_data)

        if type(train_data) is tuple\
                and len(train_data) == 2:
            xs_train, ys_train = train_data
        else:
            raise TypeError()

        # if eval_data is not None\
        #         and type(eval_data) is tuple\
        #         and len(eval_data) == 2:
        #     xs_eval, ys_eval = eval_data
        #     self.eval_dataset = BertTrainDataset(xs_eval, ys_eval, self.tokenizer, max_length=self.train_config.max_input_length)
        # elif eval_data is None:
        #     self.eval_dataset = None
        # else:
        #     raise TypeError()

        self.train_dataset = BertTrainDataset(xs_train, ys_train, self.tokenizer, max_length=self.train_config.max_input_length)

    def evaluate(self, eval_data):
        super().evaluate(eval_data)
        if eval_data is not None\
                and type(eval_data) is tuple\
                and len(eval_data) == 2:
            xs_eval, ys_eval = eval_data
            self.eval_dataset = BertTrainDataset(xs_eval, ys_eval, self.tokenizer, max_length=self.train_config.max_input_length)
        else:
            raise TypeError()

    def predict(self, test_dataset):
        super().predict(test_dataset)