from dataclasses import dataclass
import string
import torch
import nltk

from preprocess.feature.transform import transform_token_seqs_to_word_index_seqs


@dataclass
class SupervisedNNModelTrainConfig:
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


class TrainFramework(object):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig
    ):
        super(TrainFramework, self).__init__()
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


class PyTorchTrainFramework(TrainFramework):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig
    ):
        super(PyTorchTrainFramework, self).__init__(train_config=train_config)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def fit(self, train_data, eval_data=None):
        super().fit(train_data=train_data, eval_data=eval_data)

    def evaluate(self, eval_data):
        super().evaluate(eval_data=eval_data)

    def predict(self, test_data):
        pass


class KerasTrainFramework(object):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig
    ):
        self.train_config = train_config

    def fit(self, xs_train, ys_train, validation_data=None, callbacks=None, verbose=2):
        self.model.fit(xs_train, ys_train,
                       batch_size=self.train_config.train_batch_size,
                       epochs=self.train_config.epoch,
                       validation_data=validation_data,
                       callbacks=callbacks,
                       verbose=verbose)

    def evaluate(self, xs_test, ys_test):
        self.model.evaluate(xs_test, ys_test, self.train_config.eval_batch_size)

    def predict(self, xs_test):
        return self.model.predict(xs_test)

    def load_model(self, model_file_path):
        self.model.load_weights(model_file_path)


class TensorFlowEstimatorTrainFramework(object):

    def __init__(self,
                 train_config: SupervisedNNModelTrainConfig
                 ):
        self.train_config = train_config

    def __input_fn_builder(self, xs_test, ys_test=None):
        pass

    def __model_fn_builder(self):
        pass

    def fit(self, xs_train, ys_train):
        input_fn = self.__input_fn_builder(xs_train, ys_train)
        self.estimator.train(input_fn=input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None)

    def evaluate(self, xs_valid, ys_valid):
        input_fn = self.__input_fn_builder(xs_valid, ys_valid)
        self.estimator.evaluate(input_fn=input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None)

    def predict(self, xs_test):
        input_fn = self.__input_fn_builder(xs_test)
        self.estimator.predict(input_fn=input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None)

    def load_model(self, model_file_path):
        self.estimator.export_saved_model(model_file_path,
                                          # serving_input_receiver_fn,
                                          assets_extra=None,
                                          as_text=False,
                                          checkpoint_path=None)


class TextDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            xs: list,
            ys: list
    ):
        self.data = torch.tensor(xs, dtype=torch.long)
        self.labels = torch.tensor(ys)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]