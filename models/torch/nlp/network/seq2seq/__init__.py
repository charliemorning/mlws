from dataclasses import dataclass


@dataclass
class ModelConfig:
    max_features: int  # vocabulary size
    max_seq_length: int
    embedding_size: int