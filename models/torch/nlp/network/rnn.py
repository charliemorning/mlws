from dataclasses import dataclass

import torch
import torch.nn as nn

from framework.torch.layers.rnn import LSTM
from framework.torch.layers.linear import Dense


@dataclass
class TextRNNModelConfig:
    max_features: int  # vocabulary size
    max_seq_length: int
    embedding_size: int
    dim_out: int
    without_pretrained: bool = None
    freeze_pretrained: bool = False


class TextRNN(nn.Module):

    def __init__(
            self,
            model_config: TextRNNModelConfig,
            embedding_matrix=None
    ):
        super(TextRNN, self).__init__()
        self.model_config = model_config

        self.embedding = nn.Embedding(
            num_embeddings=model_config.max_features,
            embedding_dim=model_config.embedding_size,
            _weight=torch.tensor(embedding_matrix, dtype=torch.float)
        )
        self.embedding.weight.requires_grad = False

        self.rnn = nn.LSTM(
            input_size=model_config.embedding_size,
            hidden_size=100,
            num_layers=1,
            # dropout=0.5,
            bidirectional=True,
            batch_first=True
        )

        # self.rnn = LSTM(model_config.embedding_size, 100, device=self.device)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(200, model_config.dim_out)
        # self.fc = Dense(100, model_config.dim_out)

    def forward(self, x):

        embedding = self.embedding(x)
        rnn_encode, (hidden, cell) = self.rnn(embedding)
        logits = self.fc(self.dropout(torch.cat((hidden[0], hidden[1]), dim=1)))
        # logits = self.fc(self.dropout(hidden))

        return logits
