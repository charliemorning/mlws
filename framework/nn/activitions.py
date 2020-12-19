import numpy as np


class Activition(object):

    def __init__(self):
        self.y = None

    def forward(self, z):
        pass

    def backward(self, d_j):
        pass


def sigmoid(z):
    return (1 + np.exp(-z)) ** -1


def softmax(z):
    """

    :param z:
    :return:
    """

    # to avoid overflow
    z -= np.max(z)
    return np.exp(z) / np.sum(np.exp(z))


class Sigmoid(Activition):

    def forward(self, z):
        self.y = sigmoid(z)
        return self.y

    def backward(self, d_j):
        return d_j * self.y * (1 - self.y)


class Softmax:

    def __init__(self):
        self.y = None

    def forward(self, z):
        self.y = softmax(z)
        return self.y

    def backward(self, d_j):
        return d_j * self.y * (1 - self.y)


class Relu:

    def __init__(self):

    def foward(self, z):
        pass

    def backward(self, j):
        pass
