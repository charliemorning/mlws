import math
import torch
import torch.nn.functional as functional


def calculate_conv_output_dim(dim_in, padding, dilation, kernel_size, stride):
    return math.floor((dim_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class GlobalMaxPool1D(torch.nn.Module):

    def __init__(self):
        super(GlobalMaxPool1D, self).__init__()

    def forward(self, x):
        """
        x shape: (batch_size, channel, seq_len)
        return shape: (batch_size, channel, 1)
        """
        return functional.max_pool1d(x, kernel_size=x.shape[2])


class GlobalAvgPool1D(torch.nn.Module):

    def __init__(self):
        super(GlobalAvgPool1D, self).__init__()

    def forward(self, x):
        """
        x shape: (batch_size, channel, seq_len)
        return shape: (batch_size, channel, 1)
        """
        return functional.avg_pool1d(x, kernel_size=x.shape[2])