import time

import torch
import torch.nn as nn
import numpy as np


from train.torch.nlp.dataset import TextMCCDataset
from train.dataset import TextDataset
from train.trainer import SupervisedNNModelTrainConfig, Trainer
from util.metric import get_confusion_matrix, report_metrics, precision_recall_f1_score


def sort_sequences(inputs: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """sort_sequences
    Sort sequences according to lengths descendingly.

    :param inputs (Tensor): input sequences, size [B, T, D]
    :param lengths (Tensor): length of each sequence, size [B]
    """
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    _, unsorted_idx = sorted_idx.sort()
    return inputs[sorted_idx], lengths_sorted, sorted_idx, unsorted_idx


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
            eps=1e-8  # Default epsilon value
        ) if optimizer is None else optimizer

        # TODO: add scheduler arguments
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, min_lr=0.001)

        if train_config.device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def __forward(self, data, lens, mode="train"):
        if self.train_config.fp16:
            with torch.cuda.amp.autocast():
                logits = self.model(data)
        else:
            if mode is "train":
                logits = self.model(data, lens)
            elif mode is "eval" or mode is "predict":
                with torch.no_grad():
                    logits = self.model(data)
            else:
                raise TypeError()

        if self.train_config.binary_out:
            logits = logits.squeeze(1)

        return logits

    def __get_predicts(self, logits):
        if self.train_config.binary_out:
            return torch.round(torch.sigmoid(logits))
        else:
            return torch.argmax(torch.softmax(logits, dim=1), dim=1).flatten()

    def __get_loss(self, logits, y_trues):
        if self.train_config.binary_out:
            assert y_trues.dtype is torch.float \
                   or y_trues.dtype is torch.float32 or y_trues.dtype is torch.float64
            loss = self.loss_fn(logits, y_trues.float())
        else:
            if y_trues.dtype is torch.int32:
                loss = self.loss_fn(logits, y_trues.long())
            else:
                assert y_trues.dtype is torch.long or y_trues.dtype is torch.int64
                loss = self.loss_fn(logits, y_trues)

        return loss

    @staticmethod
    def __get_metrics(y_trues, y_preds):
        accuracy = (np.asarray(y_preds) == np.asarray(y_trues)).mean()
        precision, recall, f1 = precision_recall_f1_score(y_trues, y_preds)
        return accuracy, precision, recall, f1

    def fit(self,
            train_data: Tuple[Any, list],
            eval_data: Tuple[Any, list] = None,
            *args,
            **kwargs) -> Tuple[float, float, float, float, float]:

        best_loss, best_acc, best_prec, best_recall, best_f1 = super().fit(train_data=train_data, eval_data=eval_data)

        # unpack train data and labels
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

            y_preds = torch.zeros(len(train_dataset), dtype=torch.int, device=self.device)
            y_trues = torch.zeros(len(train_dataset), dtype=torch.int, device=self.device)

            # Put the model into the training mode
            self.model.train()
            self.model.to(self.device)

            # for each batch
            for batch_i, (batch_data, lens, batch_labels) in enumerate(train_dataloader):

                start_index = batch_i * self.train_config.train_batch_size

                batch_data, lens, sorted_index, unsorted_index = sort_sequences(batch_data, lens)
                batch_labels = batch_labels[sorted_index]

                batch_data, batch_y_trues = batch_data.to(self.device), batch_labels.to(self.device)
                y_trues[start_index: start_index + len(batch_data)] = batch_y_trues

                # forward
                logits = self.__forward(batch_data, lens, mode="train")

                # predict
                batch_y_preds = self.__get_predicts(logits)
                y_preds[start_index: start_index + len(batch_data)] = batch_y_preds

                # loss
                loss = self.__get_loss(logits, batch_y_trues)

                total_train_loss += loss.item()

                self.optimizer.zero_grad()
                # Perform a backward pass to calculate gradients
                if self.train_config.fp16:
                    loss = scaler.scale(loss)
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # update parameters and the learning rate
                if self.train_config.fp16:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                    self.scheduler.step(total_train_loss)

            # average train loss per sample
            avg_train_loss = total_train_loss / len(train_dataloader)

            y_preds = y_preds.cpu().detach().numpy()

            if self.train_config.binary_out:
                y_trues = torch.argmax(y_trues, dim=2).cpu().detach().numpy().reshape(-1)
            else:
                y_trues = y_trues.cpu().detach().numpy().reshape(-1)

            train_acc = (np.asarray(preds) == np.asarray(trues)).mean()
            train_prec, train_recall, train_f1 = precision_recall_f1_score(trues, preds)

            time_elapsed = time.time() - t0_epoch
            print(f"The {epoch_i}th epoch train completed, cost {time_elapsed:.3f} seconds.")
            report_metrics(avg_train_loss, train_acc, train_prec, train_recall, train_f1)

            # to evaluate
            if eval_data is not None:

                print("Start evaluation:")
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
                    best_loss = avg_train_loss
                    break
                else:
                    best_loss, best_acc, best_prec, best_recall, best_f1 = avg_train_loss, train_acc, train_prec, train_recall, train_f1

        print("Training complete!")
        return best_loss, best_acc, best_prec, best_recall, best_f1

    def evaluate(self,
                 eval_data: Tuple[Any, list],
                 *args,
                 **kwargs) -> Tuple[float, float, float, float, float]:

        total_loss, accuracy, precision, recall, f1 = super().evaluate(eval_data=eval_data)

        # unpack evaluation data
        xs_eval, ys_eval = eval_data
        eval_dataset = TextMCCDataset(xs_eval, ys_eval)
        eval_sampler = torch.utils.data.RandomSampler(eval_dataset)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      sampler=eval_sampler,
                                                      batch_size=self.train_config.eval_batch_size)

        self.model.eval()
        y_preds = torch.zeros(len(eval_data), dtype=torch.int, device=self.device)
        y_trues = torch.tensor(len(eval_data), dtype=torch.int, device=self.device)

        for batch_i, (batch_data, lens, batch_labels) in enumerate(eval_dataloader):

            start_i = batch_i * self.train_config.eval_batch_size

            batch_data, lens, sorted_index, unsorted_index = sort_sequences(batch_data, lens)
            batch_labels = batch_labels[sorted_index]

            batch_data, batch_y_trues = batch_data.to(self.device), batch_labels.to(self.device)

            logits = self.__forward(batch_data, lens, mode="eval")

            batch_y_preds = self.__get_predicts(logits)

            loss = self.__get_loss(logits, batch_y_trues)

            total_loss += loss.item()

            y_preds[start_i: start_i + len(batch_data)] = batch_y_preds
            y_trues[start_i: start_i + len(batch_data)] = batch_y_trues

        # Compute the average accuracy and loss over the validation set.
        total_loss = total_loss / len(eval_dataloader)

        if self.train_config.binary_out:
            y_trues = torch.argmax(y_trues, dim=2).cpu().detach().numpy().reshape(-1)
        else:
            y_trues = y_trues.cpu().detach().numpy()
        y_preds = y_preds.cpu().detach().numpy().reshape(-1)

        confusion_matrix = get_confusion_matrix(y_trues, y_preds)

        accuracy, precision, recall, f1 = PyTorchTrainer.__get_metrics(y_trues, y_preds)

        return total_loss, accuracy, precision, recall, f1

    def predict(self, test_data: Tuple[Any, list]) -> Tuple[np.array, np.array]:

        self.model.eval()

        test_dataset = TextMCCDataset(test_data)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.train_config.eval_batch_size
        )

        logits = torch.zeros((len(test_data[0]), 35), device=self.device, dtype=torch.float32)

        preds = torch.zeros(len(test_data[0]), device=self.device, dtype=torch.int)

        with torch.no_grad():
            for i, (data, lens) in enumerate(test_dataloader):

                start_index = i * self.train_config.eval_batch_size

                data, lens, _, unsorted_index = sort_sequences(data, lens)

                data = data.to(self.device)

                batch_logits = self.model(data, lens)

                batch_preds = torch.argmax(torch.softmax(batch_logits, dim=1), dim=1).flatten()

                logits[start_index: start_index + len(data)] = batch_logits

                preds[start_index:start_index + len(data)] = batch_preds[unsorted_index]

        return logits.detach().cpu().numpy(), preds.detach().cpu().numpy()




