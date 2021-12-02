from dataclasses import dataclass

import torch
import torch.nn as nn

from playground.nlp.framework.torch.layers.embedding import PositionalEncoding


@dataclass
class TransformerConfig:
    vocab_size: int
    n_model: int
    n_layer: int
    n_head: int
    dropout: float
    dim_out: int


class Transformers(nn.Module):

    def __init__(self,
                 config: TransformerConfig,
                 embedding_matrix):
        super(Transformers, self).__init__()

        c = config
        self.c = c

        self.embedding = nn.Embedding(
            c.vocab_size,
            c.n_model,
            _weight=torch.tensor(embedding_matrix, dtype=torch.float)
        )

        self.embedding.weight.requires_grad = False
        self.pos_embedding = PositionalEncoding(c.n_model, c.dropout, c.n_model)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(c.n_model, c.n_head)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, c.n_layer)

        self.fc = nn.Linear(c.n_model, c.dim_out)

    def forward(self, x):

        embedding = self.embedding(x)
        embedding = self.pos_embedding(embedding)

        encode = self.transformer_encoder(embedding)

        encode = torch.mean(encode, dim=1)

        logits = self.fc(encode)
        return logits
