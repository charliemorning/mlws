from dataclasses import dataclass


@dataclass()
class BertModelConfig:
    tokenizer_path: str
    config_path: str
    model_path: str
    cache_dir: str
    freeze_pretrained_model_while_training: bool = False

