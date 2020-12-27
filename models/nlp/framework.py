from dataclasses import dataclass


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


class TrainFramework(object):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig
    ):
        super(TrainFramework, self).__init__()
        self.train_config = train_config

    def fit(self, train_data, valid_data=None):
        pass

    def evaluate(self, eval_dataset):
        pass

    def predict(self, test_dataset):
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
