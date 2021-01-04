import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 device
                 ):

        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        u = torch.tensor(1.0) / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float))

        weights_and_bias = lambda h, w: (nn.Parameter(torch.randn((h, w), dtype=torch.float, device=device) * u, requires_grad=True),
                                            nn.Parameter(torch.randn(w, dtype=torch.float, device=device) * u, requires_grad=True))

        self.w_ii, self.b_ii = weights_and_bias(input_size, hidden_size)
        self.w_hi, self.b_hi = weights_and_bias(hidden_size, hidden_size)
        self.w_if, self.b_if = weights_and_bias(input_size, hidden_size)
        self.w_hf, self.b_hf = weights_and_bias(hidden_size, hidden_size)
        self.w_ig, self.b_ig = weights_and_bias(input_size, hidden_size)
        self.w_hg, self.b_hg = weights_and_bias(hidden_size, hidden_size)
        self.w_io, self.b_io = weights_and_bias(input_size, hidden_size)
        self.w_ho, self.b_ho = weights_and_bias(hidden_size, hidden_size)

    def _forward(self, x, h, c):

        # input gate
        i = F.sigmoid(x @ self.w_ii + self.b_ii + h @ self.w_hi + self.b_hi)

        # forget gate
        f = F.sigmoid(x @ self.w_if + self.b_if + h @ self.w_hf + self.b_hf)

        # cell gate
        g = torch.tanh(x @ self.w_ig + self.b_ig + h @ self.w_hg + self.b_hg)

        # output gate
        o = F.sigmoid((x @ self.w_io + self.b_io + h @ self.w_ho + self.b_ho))

        c_t = f * c + i * g

        h_t = o * torch.tanh(c_t)

        return h_t, c_t

    def forward(self, X):

        h = torch.zeros(self.hidden_size, device=self.device)
        c = torch.zeros(self.hidden_size, device=self.device)

        states = torch.zeros((X.shape[0], X.shape[1], self.hidden_size), device=self.device)

        for i in range(X.shape[1]):

            x = X[:, i, :]

            h, c = self._forward(x, h, c)

            states[:, i,: ] = h

        return states, (h, c)
