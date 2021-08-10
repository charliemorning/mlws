import torch


class TextMCCDataset(torch.utils.data.Dataset):
    """
    Text Multi-class classification dataset
    """

    def __init__(
            self,
            xs: list,
            ys: list = None
    ):
        self.data = torch.tensor(xs, dtype=torch.long)
        if ys is not None:
            self.labels = torch.tensor(ys)
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.labels is None:
            return self.data[index]
        else:
            return self.data[index], self.labels[index]


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
