from dataclasses import dataclass

import torch
import torch.nn as nn

from playground.nlp.framework.torch.layers.rnn import LSTM
from playground.nlp.framework.torch.layers.linear import Linear
from nlp.diagrams.neruel_network.classical.attention import Attention


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

        if self.model_config.without_pretrained:
            self.embedding = nn.Embedding(
                num_embeddings=model_config.max_features,
                embedding_dim=model_config.embedding_size
            )
        else:
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

    def forward(self, x, x_len):

        embedding = self.embedding(x)
        rnn_encode, (hidden, cell) = self.rnn(embedding)
        logits = self.fc(self.dropout(torch.cat((hidden[0], hidden[1]), dim=1)))
        # logits = self.fc(self.dropout(hidden))
        return logits



class TextRNN2(nn.Module):

    def __init__(
            self,
            model_config: TextRNNModelConfig,
            embedding_matrix=None
    ):
        super(TextRNN2, self).__init__()
        self.model_config = model_config

        if self.model_config.without_pretrained:
            self.embedding = nn.Embedding(
                num_embeddings=model_config.max_features,
                embedding_dim=model_config.embedding_size
            )
        else:
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
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(200, model_config.dim_out)
        # self.fc = Dense(100, model_config.dim_out)

    def forward(self, x, x_len):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        x, (hidden, cell) = self.rnn(x)
        # x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        logits = self.fc(self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return logits


class TextRNNAtt(nn.Module):

    def __init__(
            self,
            model_config: TextRNNModelConfig,
            embedding_matrix=None
    ):
        super(TextRNNAtt, self).__init__()
        self.model_config = model_config

        if self.model_config.without_pretrained:
            self.embedding = nn.Embedding(
                num_embeddings=model_config.max_features,
                embedding_dim=model_config.embedding_size
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=model_config.max_features,
                embedding_dim=model_config.embedding_size,
                _weight=torch.tensor(embedding_matrix, dtype=torch.float)
            )
            self.embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(p=0.5)

        hidden_size = 100
        lstm_layer = 2

        self.lstm1 = nn.LSTM(
            input_size=model_config.embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.atten1 = Attention(hidden_size * 2, batch_first=True)

        self.lstm2 = nn.LSTM(input_size=hidden_size * 2,
                             hidden_size=hidden_size,
                             num_layers=1,
                             bidirectional=True)

        self.atten2 = Attention(hidden_size * 2, batch_first=True)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(200, model_config.dim_out)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size * lstm_layer * 2, hidden_size * lstm_layer * 2),
                                 nn.BatchNorm1d(hidden_size * lstm_layer * 2),
                                 nn.ReLU())
        self.fc2 = nn.Linear(hidden_size * lstm_layer * 2, model_config.dim_out)

    def forward(self, x, x_len):

        # unpack x and lengths
        # x, x_len = x

        x = self.embedding(x)
        x = self.dropout(x)

        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out1, (h_n, c_n) = self.lstm1(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        x, _ = self.atten1(x, lengths)  # skip connect

        out2, (h_n, c_n) = self.lstm2(out1)
        y, lengths = nn.utils.rnn.pad_packed_sequence(out2, batch_first=True)
        y, _ = self.atten2(y, lengths)

        z = torch.cat([x, y], dim=1)
        z = self.fc1(self.dropout(z))
        z = self.fc2(self.dropout(z))
        return z

