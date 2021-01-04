import numpy as np

from framework.nn.activitions import sigmoid, tanh

class LSTM:

    def __init__(self,
                 in_dim,
                 hidden_dim):

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # parameter of input in input gate
        self.W_ii = np.random.randn(in_dim, hidden_dim)
        self.b_ii = np.random.randn(hidden_dim)

        # parameters of hidden state in input gate
        self.W_hi = np.random.randn(hidden_dim, hidden_dim)
        self.b_hi = np.random.randn(hidden_dim)

        # the parameter of input in forget gate
        self.W_if = np.random.randn(in_dim, hidden_dim)
        self.b_if = np.random.randn(hidden_dim)

        # the parameter of hidden state in forget gate
        self.W_hf = np.random.randn(hidden_dim, hidden_dim)
        self.b_hf = np.random.randn(hidden_dim)

        # the parameter of input in cell
        self.W_ig = np.random.randn(in_dim, hidden_dim)
        self.b_ig = np.random.randn(hidden_dim)

        # the parameter of hidden state in cell
        self.W_hg = np.random.randn(hidden_dim, hidden_dim)
        self.b_hg = np.random.randn(hidden_dim)

        # the parameter of input in output
        self.W_io = np.random.randn(in_dim, hidden_dim)
        self.b_io = np.random.randn(out_dim)

        # the parameter of hidden state in output
        self.W_ho = np.random.randn(hidden_dim, hidden_dim)
        self.b_ho = np.random.randn(hidden_dim)


    def init_weight(self):
        self.W_ii *= (1.0 / np.sqrt(hidden_dim))



    def _forward(self, x, hidden_state_last, cell_state_last):
        """

        :param x: shape=(batch_size, embed_size)
        :param h: shape=(batch, hidden_dim)
        :param c: shape=(batch_size, hidden_dim)
        :return:
        """

        # i_t = σ(x * W_ii + + b_ii + h * W_hi + b_hi)
        input_gate = sigmoid(np.dot(x, self.W_ii) + self.b_ii + np.dot(hidden_state_last, self.W_hi) + self.b_hi)

        forget_gate = sigmoid(np.dot(x, self.W_if) + self.b_if + np.dot(hidden_state_last, self.W_hf) + self.b_hf)

        cell_gate = tanh(np.dot(x, self.W_ig) + self.b_ig + np.dot(hidden_state_last, self.W_hg) + self.b_hg)

        output_gate = sigmoid(np.dot(x, self.W_io) + self.b_io + np.dot(hidden_state_last * self.W_ho) + self.b_ho)

        cell_state = forget_gate * cell_state_last + input_gate * cell_gate

        hidden_state = output_gate * tanh(cell_state)

        return hidden_state, cell_state


    def forawrd(self, X, masks):

        hidden_state_last = np.random.randn(self.hidden_dim)
        cell_state_last = np.random.randn(self.hidden_dim)

        hidden_states = []
        for x in X:
            hidden_state_last, cell_state_last = self._forward(x, hidden_state_last, cell_state_last)
            hidden_states.append(hidden_state_last)

        return hidden_states, hidden_state_last, cell_state_last