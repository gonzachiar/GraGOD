import torch
from torch.utils.data import Dataset


class TimeDataset(Dataset):
    """
    A custom Dataset class for time series data.

    This dataset is designed to handle time series data with sliding windows,
    supporting both training and evaluation modes.
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        edge_index: torch.Tensor,
        config: dict,
        is_train: bool = False,
    ):
        """
        Initialize the TimeDataset.

        Args:
            data: Input time series data.
            labels: Labels corresponding to the input data.
            edge_index: Edge indices for graph structure.
            is_train: Whether the dataset is for training. Defaults to False.
            config: Configuration dictionary. Defaults to None.
        """
        self.config = config
        self.edge_index = edge_index
        self.is_train = is_train
        data = data.T

        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.x)

    def process(
        self, data: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the input data and labels into sliding windows.

        Args:
            data: Input time series data.
            labels: Labels corresponding to the input data.

        Returns:
            Processed features, targets, and labels.
        """
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [
            self.config[k] for k in ["slide_win", "slide_stride"]
        ]

        _, total_time_len = data.shape
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
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing feature, target, label, and edge index tensors.
        """
        feature = self.x[idx].double()
        y = self.y[idx].double()
        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index
