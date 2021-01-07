from dataclasses import dataclass


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


class Trainer(object):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig
    ):
        super(Trainer, self).__init__()
        self.train_config = train_config

    def fit(self, train_data, eval_data=None):

        if type(train_data) is not tuple:
            raise TypeError()

        if type(train_data) is tuple\
                and len(train_data) != 2:
            raise TypeError()

        if eval_data is not None and len(eval_data) != 2:
            raise TypeError()

    def evaluate(self, eval_data):

        if eval_data is None:
            raise TypeError()

        if len(eval_data) != 2:
            raise TypeError()

    def predict(self, test_data):
        pass
