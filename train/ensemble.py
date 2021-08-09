from collections import Counter

import numpy as np

from train.trainer import Trainer


class Ensembler(object):
    def __init__(self):
        pass

    def fit(self, xs_train, ys_train):
        pass


class Stacker(object):
    def __init__(self):
        pass


class Voter(object):

    def __init__(self,
                 models: list[Trainer]
                 ):
        self.models = models

    def vote(self, test_data):

        n = len(self.models)

        stack_preds = np.zeros((n, len(test_data)))
        preds = np.zeros(len(test_data))

        xs_test = test_data.get_sequences()
        for i, model in enumerate(self.models):
            _, i_preds = model.predict(xs_test)
            stack_preds[i] = i_preds

        stack_preds = stack_preds.transpose()
        for i in range(len(stack_preds)):
            pred = stack_preds[i]
            label_counter = Counter(pred)
            preds[i] = label_counter.most_common(1)[0][0]

        return preds

