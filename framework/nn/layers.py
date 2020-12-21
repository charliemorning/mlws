import numpy as np


class Layer(object):

    def __init__(self):
        pass

    def forward(self, _X):
        pass

    def backward(self):
        pass


class FullConnectedLayer(Layer):

    def __init__(self, input_size, num_cells):
        self._W = np.random.rand(input_size, num_cells)
        self._b = np.random.rand(num_cells)

    def forward(self, _X):

        if _X.ndim == 1:
            _X.reshape(1, len(_X))

        return np.dot(_X, self.W) + self._b

    def backward(self, _J):
        pass


