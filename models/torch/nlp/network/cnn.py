from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from util.nn import calculate_conv_output_dim, GlobalMaxPool1D, GlobalAvgPool1D


class PoolingType(Enum):
    MAX_POOLING = 1
    AVG_POOLING = 2


@dataclass
class TextCNNModelConfig:
    max_features: int  # vocabulary size
    max_seq_length: int
    embedding_size: int
    filters: list
    kernel_size: list
    padding: list
    stride:  list
    dilation: list
    dim_out: int
    pooling_type: PoolingType
    global_pooling: bool
    without_pretrained: bool = None
    freeze_pretrained: bool = False


class TextCNN(nn.Module):

    def __init__(
            self,
            cnn_model_config: TextCNNModelConfig,
            embedding_matrix=None
    ):

        super(TextCNN, self).__init__()
        self.cnn_model_config = cnn_model_config

        if cnn_model_config.without_pretrained:
            self.embedding = torch.nn.Embedding(
                num_embeddings=cnn_model_config.max_features,
                embedding_dim=cnn_model_config.embedding_size,
                padding_idx=0
            )
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        else:
            self.embedding = torch.nn.Embedding(
                num_embeddings=cnn_model_config.max_features,
                embedding_dim=cnn_model_config.embedding_size,
                _weight=torch.tensor(embedding_matrix, dtype=torch.float)
            )
            if cnn_model_config.freeze_pretrained:
                self.embedding.weight.requires_grad = False

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




