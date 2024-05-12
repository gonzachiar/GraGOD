from typing import Optional

import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,  # TODO: Check if a tensor is a numpy ArrayLike
        window: int,
        target_dim: Optional[int] = None,
        horizon: int = 1,
    ):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window
