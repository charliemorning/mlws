import numpy as np


class Network:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, optimizer, epoch, batch_size):
        pass


class ComputationalGraph:

    def __init__(self):
        pass