from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from nlp.dataset import TextDataset


@dataclass
class SupervisedNNModelTrainConfig:
    binary_out: bool = False
    epoch: int = 1  # total # of training epochs
    train_batch_size: int = 64  # batch size during training
    eval_batch_size: int = 64  # batch size for evaluation
    learning_rate: float = 5e-5  # learning rate
    weight_decay: float = 0.01  # strength of weight decay
    max_input_length: int = 128  # max length of input
    logging_dir: str = None
    output_dir: str = None  # directory for storing logs
    warmup_steps: int = None  # number of warmup steps for learning rate scheduler
    multi_label: bool = False  # if multi-label
    device: str = "gpu"
    fp16: bool = False
    patience: int = 7  # patience for early stop
    delta: float = 0.3  # delta of early stop


class Trainer(object):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig
    ):
        super(Trainer, self).__init__()
        self.train_config = train_config

    def fit(self,
            train_data: Tuple[Any, list],
            eval_data: Tuple[Any, list] = None,
            *args,
            **kwargs) -> Tuple[float, float, float, float, float]:

        if type(train_data) is not tuple:
            raise TypeError()

        if type(train_data) is tuple\
                and len(train_data) != 2:
            raise TypeError()

        if eval_data is not None and len(eval_data) != 2:
            raise TypeError()

        return 0.0, 0.0, 0.0, 0.0, 0.0

    def evaluate(self,
                 eval_data: Tuple[Any, list],
                 *args,
                 **kwargs) -> Tuple[float, float, float, float, float]:

        if eval_data is None:
            raise TypeError()

        if len(eval_data) != 2:
            raise TypeError()

        return 0.0, 0.0, 0.0, 0.0, 0.0

    def predict(self,
                test_data: Tuple[Any, list],
                *args,
                **kwargs) -> Tuple[np.array, np.array]:
        pass
