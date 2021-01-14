from models.torch.trainer import SupervisedNNModelTrainConfig, Trainer


class KerasTrainer(Trainer):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig
    ):
        super(KerasTrainer, self).__init__(train_config)

    def fit(self, train_data, eval_data=None, callbacks=None, verbose=2):
        super(KerasTrainer, self).fit(train_data=train_data, eval_data=eval_data)
        xs_train, ys_train = train_data
        self.model.fit(xs_train, ys_train,
                       batch_size=self.train_config.train_batch_size,
                       epochs=self.train_config.epoch,
                       validation_data=eval_data,
                       callbacks=callbacks,
                       verbose=verbose)

    def evaluate(self, eval_data):
        super(KerasTrainer, self).evaluate(eval_data=eval_data)
        xs_test, ys_test = eval_data
        self.model.evaluate(xs_test, ys_test, self.train_config.eval_batch_size)

    def predict(self, xs_test):
        return self.model.predict(xs_test)

    def load_model(self, model_file_path):
        self.model.load_weights(model_file_path)


class TensorFlowEstimatorTrainer(Trainer):

    def __init__(self,
                 train_config: SupervisedNNModelTrainConfig
                 ):
        super(TensorFlowEstimatorTrainer, self).__init__()
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
