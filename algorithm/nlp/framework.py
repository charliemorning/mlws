class SupervisedNNModelTrainConfig:

    def __init__(self,
                 epoch,
                 batch_size
                 ):
        self.epoch = epoch
        self.batch_size = batch_size


class KerasTrainFramework(object):

    def __init__(self,
                 config: SupervisedNNModelTrainConfig
                 ):
        self.config = config

    def fit(self, xs_train, ys_train, validation_data=None, callbacks=None, verbose=2):
        self.model.fit(xs_train, ys_train,
                       batch_size=self.config.batch_size,
                       epochs=self.config.epoch,
                       validation_data=validation_data,
                       callbacks=callbacks,
                       verbose=verbose)

    def evaluate(self, xs_test, ys_test):
        self.model.evaluate(xs_test, ys_test, self.config.batch_size)

    def predict(self, xs_test):
        return self.model.predict(xs_test)

    def load_model(self, model_file_path):
        self.model.load_weights(model_file_path)