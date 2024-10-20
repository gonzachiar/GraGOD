# print all the list and numpy arrays
import numpy as np
import torch
from torch.utils.data import Dataset

np.set_printoptions(threshold=np.inf)


class TimeDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        edge_index: torch.Tensor,
        is_train: bool = False,
        config: dict | None = None,
    ):
        self.config = config
        self.edge_index = edge_index
        self.is_train = is_train
        data = data.T

        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(
        self, data: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [
            self.config[k] for k in ["slide_win", "slide_stride"]
        ]

        node_num, total_time_len = data.shape
        rang = (
            range(slide_win, total_time_len, slide_stride)
            if self.is_train
            else range(slide_win, total_time_len)
        )

        for i in rang:

            ft = data[:, i - slide_win : i]
            tar = data[:, i]

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index
