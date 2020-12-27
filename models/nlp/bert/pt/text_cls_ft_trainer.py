import time

import numpy as np
import torch
import transformers

from transformers import BertConfig, BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup

from models.nlp.bert import BertModelConfig
from models.nlp.bert.pt import BertTrainDataset, BertTrainerBase
from models.nlp.framework import SupervisedNNModelTrainConfig
from util.metric import transformers_aprf_metrics


class QuickBertTrainer(BertTrainerBase):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig,
            bert_model_config: BertModelConfig
    ):
        super(QuickBertTrainer, self).__init__(
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
            eval_dataset=self.valid_dataset if eval_data is not None else None,  # evaluation dataset
            compute_metrics=transformers_aprf_metrics
        )
        self.trainer.train()
        self.trainer.evaluate()

    def evaluate(self, eval_dataset):
        self.trainer.evaluate(eval_dataset)

    def predict(self, test_dataset):
        self.trainer.predict(test_dataset)


class BertTrainer(BertTrainerBase, torch.nn.Module):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig,
            bert_model_config: BertModelConfig
    ):
        super(BertTrainer, self).__init__(
            train_config=train_config,
            bert_model_config=bert_model_config
        )

        dim_in, h, dim_out = 768, 50, 2
        self.bert = BertModel(config=self.bert_config)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim_in, h),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(h, dim_out)
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

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
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

    def fit(self, train_data, valid_data=None):
        super().fit(train_data, valid_data)

        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler,
                                                       batch_size=self.train_config.train_batch_size)

        optimizer = AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,  # Default learning rate
            eps=1e-8  # Default epsilon value
        )

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=10000)

        print("Start training...\n")
        for epoch_i in range(self.train_config.epoch):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.train()
            self.to(self.device)

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = (batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device))

                # Zero out any previously calculated gradients
                self.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = self(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = self.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            if valid_data is not None:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy = self.evaluate(valid_data)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-" * 70)
            print("\n")

        print("Training complete!")

    def evaluate(self, eval_data):
        super().evaluate(eval_data)

        eval_sampler = torch.utils.data.RandomSampler(self.eval_dataset)
        eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, sampler=eval_sampler,
                                                       batch_size=self.train_config.eval_batch_size)

        self.eval()
        # Tracking variables
        eval_accuracy = []
        eval_loss = []

        # For each batch in our validation set...
        for batch in eval_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = (
            batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device))

            # Compute logits
            with torch.no_grad():
                logits = self(b_input_ids, b_attn_mask)

            # Compute loss
            loss = self.loss_fn(logits, b_labels)
            eval_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            eval_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(eval_loss)
        val_accuracy = np.mean(eval_accuracy)

        return val_loss, val_accuracy