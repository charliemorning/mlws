import time

import numpy as np
from tqdm import tqdm
import torch

from transformers import BertModel, BertForSequenceClassification, Trainer
from transformers import AdamW, get_linear_schedule_with_warmup

from models.trainer import SupervisedNNModelTrainConfig
from models.torch.nlp.network.bert import BertModelConfig, BertTrainerBase
from util.metric import transformers_aprf_metrics, precision_recall_f1_score


class TextClsBertTrainer(BertTrainerBase):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig,
            bert_model_config: BertModelConfig
    ):
        super(TextClsBertTrainer, self).__init__(
            train_config=train_config,
            bert_model_config=bert_model_config)

        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=bert_model_config.model_path,
            config=self.bert_config,
            cache_dir=bert_model_config.cache_dir
        )

        if self.bert_model_config.freeze_pretrained_model_while_training:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def fit(self, train_data, eval_data=None):

        super().fit(train_data, eval_data)

        self.trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=self.training_args,  # training arguments, defined above
            train_dataset=self.train_dataset,  # training dataset
            eval_dataset=self.eval_dataset if eval_data is not None else None,  # evaluation dataset
            compute_metrics=transformers_aprf_metrics,
        )
        self.trainer.train()
        self.trainer.evaluate()

    def evaluate(self, eval_dataset):
        self.trainer.evaluate(eval_dataset)

    def predict(self, test_dataset):
        self.trainer.predict(test_dataset)


class TextClsCustomedBertTrainer(BertTrainerBase, torch.nn.Module):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig,
            bert_model_config: BertModelConfig
    ):
        super(TextClsCustomedBertTrainer, self).__init__(
            train_config=train_config,
            bert_model_config=bert_model_config
        )

        dim_in, h, dim_out = 768, 50, 1
        self.bert = BertModel(config=self.bert_config)
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Conv1d(768, 128, kernel_size=3, padding=1),
        #     GlobalAvgPool1D(),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.Linear(128, dim_out),
        # )
        # self.conv = torch.nn.Conv1d(768, 128, kernel_size=3, padding=1)
        # self.pool = GlobalAvgPool1D()
        # self.dropout = torch.nn.Dropout(0.5)
        # self.fc = torch.nn.Linear(128, dim_out)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(768 * 2, dim_out)
        )

        self.loss_fn = torch.nn.BCELoss()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # the last hidden state of the token `[CLS]` for classification task
        cls_last_hidden_state = outputs[0][:, 0, :]

        cls_non_masked_state = outputs[1]

        # index of '[SEP]' token
        sep_index = torch.count_nonzero(attention_mask, dim=1) - 1

        # the last hidden state of the token '[SEP]'
        sep_last_hidden_state = torch.gather(outputs[0], dim=1,
                                             index=sep_index.unsqueeze(1).repeat(1, 768).unsqueeze(1)).squeeze(1)

        # sum all last hidden state of tokens except for '[CLS]' and '[SEP]'
        sum_masked_state = torch.sum(outputs[0], dim=1) - cls_last_hidden_state - sep_last_hidden_state

        # exclude [CLS] and [SEP]
        mask_seq_len = torch.count_nonzero(attention_mask, dim=1) - 2

        avg_masked_state = sum_masked_state / mask_seq_len.unsqueeze(1).repeat(1, 768).float()

        # token_state = outputs[0]

        # token_state = token_state.permute(0, 2, 1)

        # x = self.conv(token_state)
        # x = self.pool(x).squeeze(2)
        # x = self.dropout(x)
        # logits = self.fc(x)

        # Feed input to classifier to compute logits
        logits = self.classifier(torch.cat((cls_last_hidden_state, avg_masked_state), dim=1))
        # logits = self.classifier(avg_masked_state)

        return torch.sigmoid(logits)

    def fit(self, train_data, eval_data=None):
        super().fit(train_data, eval_data)

        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler,
                                                       batch_size=self.train_config.train_batch_size)

        optimizer = AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,  # Default learning rate
            eps=1e-8  # Default epsilon value
        )

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.train_config.warmup_steps,  # Default value
                                                    num_training_steps=len(train_dataloader) * self.train_config.epoch)

        # scaler = torch.cuda.amp.GradScaler()
        for epoch_i in range(self.train_config.epoch):
            t0_epoch = time.time()
            total_loss = 0
            self.train()
            self.to(self.device)

            y_preds = []
            y_true = []

            # For each batch of training data...
            t = tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit='example')
            t.set_description("epoch {epoch_i} training".format(epoch_i=epoch_i))
            for step, batch in t:
                b_input_ids, b_attn_mask, b_labels = (batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device))
                self.zero_grad()

                # with torch.cuda.amp.autocast():
                logits = self(b_input_ids, b_attn_mask)
                logits = logits.squeeze(1)

                y_preds.extend([i for i in torch.round(logits).int().cpu()])
                y_true.extend([i for i in b_labels.int().cpu()])

                loss = self.loss_fn(logits, b_labels.float())
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                # scaler.scale(loss).backward()
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                # Update parameters and the learning rate
                # scaler.step(optimizer)
                optimizer.step()
                scheduler.step()
                # scaler.step(scheduler)

                # scaler.update()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)
            t.update()

            train_loss = loss / len(train_dataloader)
            train_acc = (np.asarray(y_preds) == np.asarray(y_true)).mean()
            train_prec, train_recall, train_f1 = precision_recall_f1_score(y_true, y_preds)

            print("[train_loss:{train_loss:.3f}; train_acc:{train_acc:.3f}; "
                  "train_prec:{train_prec:.3f}; train_recall:{train_recall:.3f}; "
                  "train_f1:{train_f1:.3f};]".format(
                     train_loss=train_loss,
                     train_acc=train_acc,
                     train_prec=train_prec,
                     train_recall=train_recall,
                     train_f1=train_f1
                 )
            )

            if eval_data is not None:
                eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1score = self.evaluate(eval_data)

                time_elapsed = time.time() - t0_epoch

                print("[eval_loss:{eval_loss:.3f}; eval_acc:{eval_acc:.3f}; "
                      "eval_prec:{eval_prec:.3f}; eval_recall:{eval_recall:.3f}; "
                      "eval_f1:{eval_f1:.3f}; time:{time_elapsed:.2f}s]".format(
                        loss=avg_train_loss,
                        eval_loss=eval_loss,
                        eval_acc=eval_accuracy,
                        eval_prec=eval_precision,
                        eval_recall=eval_recall,
                        eval_f1=eval_f1score,
                        time_elapsed=time_elapsed
                    )
                )

        print("Training complete!")

    def evaluate(self, eval_data):
        super().evaluate(eval_data)

        eval_sampler = torch.utils.data.RandomSampler(self.eval_dataset)
        eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, sampler=eval_sampler,
                                                       batch_size=self.train_config.eval_batch_size)
        self.eval()
        loss = 0.0
        y_preds = []
        y_true = []

        # For each batch in our validation set...
        for batch in eval_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = (
            batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device))

            # Compute logits
            with torch.no_grad():
                logits = self(b_input_ids, b_attn_mask)
                logits = logits.squeeze(1)

            # Compute loss
            batch_loss = self.loss_fn(logits, b_labels.float())
            loss += batch_loss.item()

            # Get the predictions
            batch_preds = torch.round(logits)

            y_preds.extend([i for i in batch_preds.int().cpu()])
            y_true.extend([i for i in b_labels.int().cpu()])

        # Compute the average accuracy and loss over the validation set.
        loss = loss / len(eval_dataloader)
        accuracy = (np.asarray(y_preds) == np.asarray(y_true)).mean()
        precision, recall, f1 = precision_recall_f1_score(y_true, y_preds)

        return loss, accuracy, precision, recall, f1