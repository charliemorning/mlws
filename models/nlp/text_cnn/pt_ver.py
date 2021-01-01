import time

from tqdm import tqdm
import torch
import numpy as np

from models.nlp.framework import PyTorchTrainFramework, SupervisedNNModelTrainConfig, TextDataset
from models.nlp.text_cnn import TextCNNModelConfig, PoolingType

from util.metric import precision_recall_f1_score
from util.nn import calculate_conv_output_dim, GlobalMaxPool1D, GlobalAvgPool1D


class TextCNNTrainer(PyTorchTrainFramework, torch.nn.Module):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig,
            cnn_model_config: TextCNNModelConfig,
            embedding_matrix=None
    ):

        super(TextCNNTrainer, self).__init__(train_config=train_config)
        self.cnn_model_config = cnn_model_config

        # if cnn_model_config.without_pretrained:
        self.embedding = torch.nn.Embedding(
            num_embeddings=cnn_model_config.max_features,
            embedding_dim=cnn_model_config.embedding_size
        )

        torch.nn.init.xavier_uniform_(self.embedding.weight)

        # self.embedding.weight = torch.nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # self.embedding.weight.requires_grad = True

        self.conv_list = torch.nn.ModuleList([torch.nn.Conv1d(
            in_channels=cnn_model_config.embedding_size,
            out_channels=cnn_model_config.filters[i],
            kernel_size=cnn_model_config.kernel_size[i],
            stride=cnn_model_config.stride[i],
            padding=cnn_model_config.padding[i],
            dilation=cnn_model_config.dilation[i]
        ) for i, _ in enumerate(cnn_model_config.kernel_size)])

        def init_xavier_uniform(m):
            if type(m) == torch.nn.Conv1d:
                torch.nn.init.xavier_uniform_(m.weight)
        self.conv_list.apply(init_xavier_uniform)

        conv_dim_out_list = [calculate_conv_output_dim(
            dim_in=cnn_model_config.max_seq_length,
            padding=cnn_model_config.padding[i],
            dilation=cnn_model_config.dilation[i],
            kernel_size=cnn_model_config.kernel_size[i],
            stride=cnn_model_config.stride[i],
        ) for i, _ in enumerate(cnn_model_config.kernel_size)]

        self.dropout = torch.nn.Dropout(0.5)

        pool_padding = 0
        pool_dilation = 1
        pool_kernel_size = 3
        pool_stride = 1
        pool_out_dim_list = [calculate_conv_output_dim(
            dim_in=conv_dim_out,
            padding=pool_padding,
            dilation=pool_dilation,
            kernel_size=pool_kernel_size,
            stride=pool_stride,
        ) for conv_dim_out in conv_dim_out_list]

        if cnn_model_config.global_pooling:
            if cnn_model_config.pooling_type == PoolingType.MAX_POOLING:
                self.pool = GlobalMaxPool1D()
            elif cnn_model_config.pooling_type == PoolingType.AVG_POOLING:
                self.pool = GlobalAvgPool1D()
            else:
                raise TypeError()

            fc_in = np.sum(np.asarray(cnn_model_config.filters))
        else:
            if cnn_model_config.pooling_type == PoolingType.MAX_POOLING:
                self.pool = torch.nn.MaxPool1d(
                    kernel_size=pool_kernel_size,
                    stride=pool_stride,
                    padding=pool_padding,
                    dilation=pool_dilation
                )
            elif cnn_model_config.pooling_type == PoolingType.AVG_POOLING:
                self.pool = torch.nn.AvgPool1d(
                    kernel_size=pool_kernel_size,
                    stride=pool_stride,
                    padding=pool_padding
                )
            else:
                raise TypeError()

            fc_in = np.sum(np.asarray(pool_out_dim_list) * np.asarray(cnn_model_config.filters))

        self.fc = torch.nn.Linear(fc_in, cnn_model_config.dim_out)
        torch.nn.init.uniform_(self.fc.weight)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.train_config.learning_rate,  # Default learning rate
            eps=1e-8  # Default epsilon value
        )

        if cnn_model_config.dim_out == 1:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # (batch_size, max_seq_length) -> (batch_size, max_seq_length, embedding_size)
        embedding = self.embedding(x)

        # (batch_size, max_seq_length, embedding_size) -> (batch_size, embedding_size, max_seq_length)
        embedding = embedding.permute(0, 2, 1)

        # conv_out = conv(embedding): (batch_size, embedding_size, max_seq_length) -> (batch_size, num_filters, _)
        #
        # pool(conv_out):
        #   if global pool: (batch_size, num_filters)
        #   otherwise: (batch_size, num_filters, _)
        #
        conv_pool_out = [self.pool(torch.tanh_(conv(embedding))).flatten(1) for conv in self.conv_list]

        conv_pool_out = torch.cat(conv_pool_out, dim=1)
        conv_pool_out = self.dropout(conv_pool_out)
        logit = self.fc(conv_pool_out)
        return logit

    def fit(self, train_data, eval_data=None):
        print("Start training...\n")
        super().fit(train_data=train_data, eval_data=eval_data)

        xs, ys = train_data
        self.train_dataset = TextDataset(xs, ys)
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler,
                                                       batch_size=self.train_config.train_batch_size)

        with tqdm(total=self.train_config.epoch) as t:
            for epoch_i in range(self.train_config.epoch):

                # Measure the elapsed time of each epoch
                t0_epoch, t0_batch = time.time(), time.time()

                # Reset tracking variables at the beginning of each epoch
                total_loss, batch_loss, batch_counts = 0, 0, 0

                # Put the model into the training mode
                self.train()
                self.to(self.device)

                # For each batch of training data...
                for step, (data, label) in enumerate(train_dataloader):

                    batch_counts += 1
                    self.optimizer.zero_grad()

                    data, y_true = data.to(self.device), label.to(self.device)

                    # Perform a forward pass. This will return logits.
                    # with torch.cuda.amp.autocast():
                    logits = self(data)

                    if self.cnn_model_config.dim_out == 1:
                        logits = logits.squeeze(1)

                    loss = self.loss_fn(logits, y_true.float())

                    batch_loss += loss.item()
                    total_loss += loss.item()

                    # Perform a backward pass to calculate gradients
                    # scaler.scale(loss).backward()
                    loss.backward()

                    # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                    # Update parameters and the learning rate
                    # scaler.step(self.optimizer)
                    self.optimizer.step()

                    # scaler.update()

                # Calculate the average loss over the entire training data
                avg_train_loss = total_loss / len(train_dataloader)

                if eval_data is not None:
                    # After the completion of each training epoch, measure the model's performance
                    # on our validation set.
                    xs_dev, ys_dev = eval_data

                    eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1score = self.evaluate(eval_data=(xs_dev, ys_dev))

                    # Print performance over the entire training data
                    time_elapsed = time.time() - t0_epoch

                    t.set_postfix(
                        loss=avg_train_loss,
                        eval_loss=eval_loss,
                        eval_accuracy=eval_accuracy,
                        eval_precision= eval_precision,
                        eval_recall=eval_recall,
                        eval_f1score=eval_f1score,
                        time_elapsed=time_elapsed
                    )
                    t.update()

        print("Training complete!")

    def evaluate(self, eval_data):
        super().evaluate(eval_data)

        xs, ys = eval_data

        self.eval_dataset = TextDataset(xs, ys)
        eval_sampler = torch.utils.data.RandomSampler(self.eval_dataset)
        eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, sampler=eval_sampler,
                                                      batch_size=self.train_config.eval_batch_size)

        # enable evaluation mode
        self.eval()

        loss = 0.0
        y_preds = []
        y_true = []

        # For each batch in our validation set...
        for batch in eval_dataloader:
            batch_data, batch_labels = batch[0].to(self.device), batch[1].to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = self(batch_data)

            if self.cnn_model_config.dim_out == 1:
                logits = logits.squeeze(1)

            # Compute loss of current batch
            batch_loss = self.loss_fn(logits, batch_labels.float())
            loss += batch_loss.item()

            # Get the predictions
            if self.cnn_model_config.dim_out == 1:
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

