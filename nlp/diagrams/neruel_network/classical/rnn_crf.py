from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass()
class SeqClsModelConfig:
    max_features: int
    dim_out: int


class LSTMSeqCls(nn.Module):

    def __init__(
            self,
            config: SeqClsModelConfig
    ):
        super(LSTMSeqCls, self).__init__()

        c = config

        self.embedding_layer = nn.Embedding(c.max_features, 300)

        self.encoder_layer = nn.LSTM(300, 100, batch_first=True)

        self.linear_layer = nn.Linear(100, c.dim_out)

        # self.decoder_layer = CRF(train_config.dim_out)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4)



        def loss_fn(outputs, labels):
            # reshape labels to give a flat vector of length batch_size*seq_len
            labels = labels.view(-1)

            # mask out 'PAD' tokens
            mask = (labels >= 0).float()

            # the number of tokens is the sum of elements in mask
            num_tokens = int(torch.sum(mask).item())

            # pick the values corresponding to labels and multiply by mask
            range_tensor = torch.range(0, outputs.shape[0], device=outputs.device).int()
            # outputs = outputs[range_tensor, labels.int()] * mask

            # cross entropy loss for all non 'PAD' tokens
            return -torch.sum(outputs) / num_tokens

        self.loss_fn = loss_fn

    def forward(self, x):

        embedding = self.embedding_layer(x)

        encodes, (hidden_out, cell_out) = self.encoder_layer(embedding)

        encodes = encodes.reshape(-1, encodes.shape[2])

        tag = self.linear_layer(encodes)

        return torch.nn.functional.log_softmax(tag, dim=1)


