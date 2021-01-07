import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from models.torch.nlp.dataset import TextDataset
from models.trainer import SupervisedNNModelTrainConfig, Trainer
from util.metric import precision_recall_f1_score


class PyTorchTrainer(Trainer):
    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig,
            model: torch.nn.Module,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=None
    ):
        super(PyTorchTrainer, self).__init__(train_config=train_config)

        self.model = model

        self.loss_fn = nn.BCEWithLogitsLoss() if train_config.binary_out else loss_fn

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config.learning_rate,  # Default learning rate
            eps=1e-8  # Default epsilon value
        ) if optimizer is None else optimizer

        if train_config.device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")


    def fit(self, train_data, eval_data=None):
        super().fit(train_data=train_data, eval_data=eval_data)

        xs_train, ys_train = train_data
        train_dataset = TextDataset(xs_train, ys_train)
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.train_config.train_batch_size
        )

        if self.train_config.fp16:
            scaler = torch.cuda.amp.GradScaler()

        with tqdm(total=self.train_config.epoch) as t:
            for epoch_i in range(self.train_config.epoch):

                # Measure the elapsed time of each epoch
                t0_epoch, t0_batch = time.time(), time.time()

                # Reset tracking variables at the beginning of each epoch
                total_loss, batch_loss, batch_counts = 0, 0, 0

                # Put the model into the training mode
                self.model.train()
                self.model.to(self.device)

                # For each batch of training data...
                for step, (data, label) in enumerate(train_dataloader):

                    batch_counts += 1
                    self.optimizer.zero_grad()

                    data, y_true = data.to(self.device), label.to(self.device)

                    # Perform a forward pass. This will return logits.
                    if self.train_config.fp16:
                        with torch.cuda.amp.autocast():
                            logits = self.model(data)
                    else:
                        logits = self.model(data)

                    if self.train_config.binary_out:
                        logits = logits.squeeze(1)

                    loss = self.loss_fn(logits, y_true.float())

                    batch_loss += loss.item()
                    total_loss += loss.item()

                    # Perform a backward pass to calculate gradients
                    if self.train_config.fp16:
                        loss = scaler.scale(loss)
                    loss.backward()

                    # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    # Update parameters and the learning rate
                    if self.train_config.fp16:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()

                # Calculate the average loss over the entire training data
                avg_train_loss = total_loss / len(train_dataloader)

                if eval_data is not None:
                    # After the completion of each training epoch, measure the model's performance
                    # on our validation set.
                    eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1score = self.evaluate(
                        eval_data=eval_data)

                    # Print performance over the entire training data
                    time_elapsed = time.time() - t0_epoch

                    t.set_postfix(
                        loss=avg_train_loss,
                        eval_loss=eval_loss,
                        eval_accuracy=eval_accuracy,
                        eval_precision=eval_precision,
                        eval_recall=eval_recall,
                        eval_f1score=eval_f1score,
                        time_elapsed=time_elapsed
                    )
                    t.update()

        print("Training complete!")

    def evaluate(self, eval_data):
        super().evaluate(eval_data=eval_data)

        xs_eval, ys_eval = eval_data
        eval_dataset = TextDataset(xs_eval, ys_eval)
        eval_sampler = torch.utils.data.RandomSampler(eval_dataset)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, sampler=eval_sampler,
                                                      batch_size=self.train_config.eval_batch_size)

        self.model.eval()
        loss, y_preds, y_true = 0.0, [], []

        # For each batch in our validation set...
        for batch in eval_dataloader:
            batch_data, batch_labels = batch[0].to(self.device), batch[1].to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = self.model(batch_data)

            if self.train_config.binary_out:
                logits = logits.squeeze(1)

            # Compute loss of current batch
            batch_loss = self.loss_fn(logits, batch_labels.float())
            loss += batch_loss.item()

            # Get the predictions
            if self.train_config.binary_out:
                batch_preds = torch.round(torch.sigmoid(logits))
            else:
                batch_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).flatten()

            y_preds.extend([i for i in batch_preds.int().cpu()])
            y_true.extend([i for i in batch_labels.int().cpu()])

        # Compute the average accuracy and loss over the validation set.
        loss = loss / len(eval_dataloader)
        accuracy = (np.asarray(y_preds) == np.asarray(y_true)).mean() * 100
        precision, recall, f1 = precision_recall_f1_score(y_true, y_preds)

        return loss, accuracy, precision, recall, f1

    def predict(self, test_data):
        pass


