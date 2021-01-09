from itertools import product
import torch
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self,
                 embedding_num,
                 embedding_size):
        super(Embedding, self).__init__()

        self.embedding_num = embedding_num
        self.embedding_size = embedding_size
        self.w = torch.nn.Parameter(torch.randn((embedding_num, embedding_size)))

    def forward(self, x):
        """
        x: shape=(batch_size, seq_length)

        """
        # shpae=(batch_size, seq_length, embedding_num)
        onehot = torch.zeros((x.shape[0], x.shape[1], self.embedding_num), dtype=torch.float, device=x.device)

        # shape=(batch_size, seq_length)
        index = torch.tensor([_ for _ in product(torch.arange(0, x.shape[0]),  torch.arange(0, x.shape[1]))], device=x.device)

        index = (index.T[0], index.T[1], x.view(-1))
        onehot = torch.index_put(onehot, indices=index, values=torch.tensor(1, dtype=torch.float, device=x.device))
        return onehot @ self.w


if __name__ == '__main__':

    embedding_num = 1000
    embedding_size = 300

    x = torch.zeros((64, 128), dtype=torch.long)

    embedding = Embedding(embedding_num, embedding_size)
    print(embedding(x).shape)