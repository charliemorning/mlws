from dataclasses import dataclass
from typing import Union
from enum import Enum

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