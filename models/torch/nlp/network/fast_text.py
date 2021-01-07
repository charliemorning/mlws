from dataclasses import dataclass

import torch.nn as nn


@dataclass
class FastTextConfig:
    max_feature: int  # vocabulary size
    embedding_size: int
    dim_out: int
    without_pretrained: bool = None
    freeze_pretrained: bool = False


class FastText(nn.Module):

    def __init__(self,
                 config: FastTextConfig):

        super(FastText, self).__init__()

        self.config = config

        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=config.max_feature,
            embedding_dim=config.embedding_size,
            mode="mean",
            sparse=False
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(in_features=config.embedding_size,
                            out_features=config.dim_out)

    def forward(self, x):

        embedding = self.embedding_bag(x)

        logits = self.fc(self.dropout(embedding))

        return logits
