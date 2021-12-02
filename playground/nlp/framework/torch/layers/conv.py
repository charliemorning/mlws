import torch
import torch.nn as nn
import torch.nn.functional as functional


class Conv1D(nn.Module):

    def __init__(self,
                 num_channels,
                 num_filters,
                 kernels,
                 strides,
                 paddings,
                 dialates
                 ):

        super(Conv1D, self).__init__()

        self.w = torch.nn.Parameter(
            torch.tensor(num_filters, num_channels, kernels)
        )

