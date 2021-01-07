import torch
import torch.nn as nn


class Linear(nn.Module):

    def __init__(self,
                 dim_in: int,
                 dim_out: int):
        super(Linear, self).__init__()


        self.w = nn.Parameter(
            torch.randn((dim_in, dim_out), dtype=torch.float)
        )

        torch.nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(
            torch.randn(dim_out, dtype=torch.float)
        )


    def forward(self, x):
        """
        x: size=(batch_size, input_lenth)
        """
        # (batch_size, input_length) * (input_length, dim_out) = (batch_size, dim_out)
        h = x @ self.w + self.b
        return h



