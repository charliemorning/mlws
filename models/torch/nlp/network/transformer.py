from dataclasses import dataclass

import torch.nn as nn


@dataclass
class TransformerConfig:
    pass


class Transformers(nn.Module):

    def __init__(self):
        super(Transformers, self).__init__()
        pass
