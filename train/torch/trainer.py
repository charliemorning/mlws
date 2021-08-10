import time

import torch
import torch.nn as nn
import numpy as np


from train.torch.nlp.dataset import TextMCCDataset
from train.dataset import TextDataset
from train.trainer import SupervisedNNModelTrainConfig, Trainer
from util.metric import get_confusion_matrix, report_metrics, precision_recall_f1_score


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self,
                 patience=7,
                 verbose=False,
                 delta=0,
                 path='checkpoint.pt',
                 trace_func=print
                 ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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
            eps=1e-8,  # Default epsilon value
        ) if optimizer is None else optimizer

        # TODO: add scheduler arguments
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, min_lr=0.0005)

        if train_config.device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def fit(self,
            train_data: TextDataset,
            eval_data: TextDataset = None) -> tuple:
        super().fit(train_data=train_data, eval_data=eval_data)

        xs_train, ys_train = train_data
        # TODO: here initiate a multi-class classification dataset default, which should be more options
        train_dataset = TextMCCDataset(xs_train, ys_train)
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.train_config.train_batch_size
        )

        if self.train_config.fp16:
            scaler = torch.cuda.amp.GradScaler()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.train_config.patience, verbose=True)

        for epoch_i in range(self.train_config.epoch):

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_train_loss = 0
            # TODO: initiate tensor to target shape
            preds = torch.tensor([], dtype=torch.int, device=self.device)
            trues = torch.tensor([], dtype=torch.int, device=self.device)

            # Put the model into the training mode
            self.model.train()
            self.model.to(self.device)

            # For each batch of training data...
            for step, (data, label) in enumerate(train_dataloader):

                data, y_true = data.to(self.device), label.to(self.device)
                trues = torch.cat((trues, y_true))

                # Perform a forward pass. This will return logits.
                if self.train_config.fp16:
                    with torch.cuda.amp.autocast():
                        logits = self.model(data)
                else:
                    logits = self.model(data)

                if self.train_config.binary_out:
                    logits = logits.squeeze(1)

                # batch_preds = torch.argmax(logits, dim=1).flatten()
                # Get the predictions
                if self.train_config.binary_out:
                    batch_preds = torch.round(torch.sigmoid(logits))
                else:
                    batch_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).flatten()
                preds = torch.cat((preds, batch_preds))


                if self.train_config.binary_out:
                    assert y_true.dtype is torch.float\
                           or y_true.dtype is torch.float32 or y_true.dtype is torch.float64
                    loss = self.loss_fn(logits, y_true.float())
                else:
                    if y_true.dtype is torch.int32:
                        loss = self.loss_fn(logits, y_true.long())
                    else:
                        assert y_true.dtype is torch.long or y_true.dtype is torch.int64
                        loss = self.loss_fn(logits, y_true)

                total_train_loss += loss.item()

                self.optimizer.zero_grad()
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
                    self.scheduler.step(total_train_loss)

            # Calculate the average loss over the entire training data
            avg_train_loss = total_train_loss / len(train_dataloader)

            preds = preds.cpu().detach().numpy()
            if self.train_config.binary_out:
                trues = torch.argmax(trues, dim=2).cpu().detach().numpy().reshape(-1)
            else:
                trues = trues.cpu().detach().numpy().reshape(-1)

            train_acc = (np.asarray(preds) == np.asarray(trues)).mean()
            train_prec, train_recall, train_f1 = precision_recall_f1_score(trues, preds)

            time_elapsed = time.time() - t0_epoch
            print(f"The {epoch_i}th epoch train completed, cost {time_elapsed:.3f} seconds.")
            report_metrics(avg_train_loss, train_acc, train_prec, train_recall, train_f1)

            if eval_data is not None:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                eval_loss, eval_acc, eval_prec, eval_recall, eval_f1 = self.evaluate(eval_data=eval_data)

                # Print performance over the entire training data
                print("Start evaluation:")
                report_metrics(eval_loss, eval_acc, eval_prec, eval_recall, eval_f1)

                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(eval_loss, self.model)
                if early_stopping.early_stop:
                    print("Early stop triggered by eval loss.")
                    best_loss, best_acc, best_prec, best_recall, best_f1 = eval_loss, eval_acc, eval_prec, eval_recall, eval_f1
                    break
                else:
                    # FIXME: record the best metrics
                    best_loss, best_acc, best_prec, best_recall, best_f1 = eval_loss, eval_acc, eval_prec, eval_recall, eval_f1
            else:
                early_stopping(avg_train_loss, self.model)
                if early_stopping.early_stop:
                    print("Early stop triggered by eval loss.")
                    best_loss = eval_loss
                    break

        print("Training complete!")
        return best_loss, best_acc, best_prec, best_recall, best_f1

    def evaluate(self,
                 eval_data: TextDataset) -> tuple:
        super().evaluate(eval_data=eval_data)

        xs_eval, ys_eval = eval_data
        eval_dataset = TextMCCDataset(xs_eval, ys_eval)
        eval_sampler = torch.utils.data.RandomSampler(eval_dataset)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, sampler=eval_sampler,
                                                      batch_size=self.train_config.eval_batch_size)

        self.model.eval()
        loss = 0.0
        y_preds = torch.tensor([], dtype=torch.int, device=self.device)
        y_trues = torch.tensor([], dtype=torch.int, device=self.device)

        # For each batch in our validation set...
        for batch in eval_dataloader:
            batch_data, batch_labels = batch[0].to(self.device), batch[1].to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = self.model(batch_data)

            if self.train_config.binary_out:
                logits = logits.squeeze(1)

            # Compute loss of current batch
            if batch_labels.dtype is torch.int32:
                batch_loss = self.loss_fn(logits, batch_labels.long())
            else:
                assert batch_labels.dtype is torch.long or batch_labels.dtype is torch.int64
                batch_loss = self.loss_fn(logits, batch_labels)
            loss += batch_loss.item()

            # Get the predictions
            if self.train_config.binary_out:
                batch_preds = torch.round(torch.sigmoid(logits))
            else:
                batch_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).flatten()

            y_preds = torch.cat((y_preds, batch_preds))
            y_trues = torch.cat((y_trues, batch_labels))

        # Compute the average accuracy and loss over the validation set.
        loss = loss / len(eval_dataloader)

        if self.train_config.binary_out:
            y_trues = torch.argmax(y_trues, dim=2).cpu().detach().numpy().reshape(-1)
        else:
            y_trues = y_trues.cpu().detach().numpy()
        y_preds = y_preds.cpu().detach().numpy().reshape(-1)

        confusion_matrix = get_confusion_matrix(y_trues, y_preds)

        accuracy = (np.asarray(y_preds) == np.asarray(y_trues)).mean()
        precision, recall, f1 = precision_recall_f1_score(y_trues, y_preds)

        return loss, accuracy, precision, recall, f1

    def predict(self, test_data):

        self.model.eval()

        test_dataset = TextMCCDataset(test_data)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.train_config.eval_batch_size
        )

        # FIXME: how to get the output size
        logits = torch.zeros((len(test_data), 35), device=self.device, dtype=torch.float32)

        # TODO: initiate tensor to target shape
        preds = torch.tensor([], device=self.device, dtype=torch.int)

        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                data = data.to(self.device)
                batch_logits = self.model(data)
                batch_preds = torch.argmax(torch.softmax(batch_logits, dim=1), dim=1).flatten()
                logits[i:i + len(data)] = batch_logits
                preds = torch.cat((preds, batch_preds))

        return logits.detach().cpu().numpy(), preds.detach().cpu().numpy()




