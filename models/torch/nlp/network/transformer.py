from dataclasses import dataclass

import torch
import torch.nn as nn

from models.torch.nlp.network.pos_embed import PositionalEncoding

@dataclass
class TransformerConfig:
    max_feature: int


class Transformers(nn.Module):

    def __init__(self,
                 config: TransformerConfig,
                 embedding_matrix):
        super(Transformers, self).__init__()

        self.embedding = nn.Embedding(config.max_feature, 300, _weight=torch.tensor(embedding_matrix, dtype=torch.float)
            )
        self.embedding.weight.requires_grad = False
        self.pos_embedding = PositionalEncoding(300, 0.1, 300)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(300, 4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, 2)

        self.fc = nn.Linear(300, 1)


    def forward(self, x):

        embedding = self.embedding(x)
        embedding = self.pos_embedding(embedding)

        encode = self.transformer_encoder(embedding)

        encode = torch.mean(encode, dim=1)

        logits = self.fc(encode)

        return logits
