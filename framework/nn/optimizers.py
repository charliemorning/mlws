import numpy as np


class Optimizer(object):

    def __init__(self):
        pass

    def update(self, param, grads):
        pass


class SGD:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = 0.01

    def update(self, parameters, gradients):
        return parameters - self.learning_rate * gradients