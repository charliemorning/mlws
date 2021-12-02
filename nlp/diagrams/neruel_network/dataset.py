import torch


def sort_sequences(inputs, lengths):
    """sort_sequences
    Sort sequences according to lengths descendingly.

    :param inputs (Tensor): input sequences, size [B, T, D]
    :param lengths (Tensor): length of each sequence, size [B]
    """
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    _, unsorted_idx = sorted_idx.sort()
    return inputs[sorted_idx], lengths_sorted, unsorted_idx


class TextMCCDataset(torch.utils.data.Dataset):
    """
    Text Multi-class classification dataset
    """

    def __init__(
            self,
            xs: list,
            ys: list = None
    ):
        # FIXME
        self.data = torch.tensor(xs[0], dtype=torch.long)
        self.lens = torch.tensor(xs[1], dtype=torch.int32)

        if ys is not None:
            self.labels = torch.tensor(ys)
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.labels is None:
            return self.data[index], self.lens[index]
        else:
            return self.data[index], self.lens[index], self.labels[index]


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            xs: list,
            ys: list
    ):
        self.data = torch.tensor(xs, dtype=torch.long)
        self.labels = torch.tensor(ys, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
