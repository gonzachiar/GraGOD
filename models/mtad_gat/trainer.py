import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torch.utils.tensorboard import SummaryWriter

from gragod import PathType
from models.mtad_gat.model import MTAD_GAT


class Trainer:
    """
    Trainer class for MTAD-GAT model.

    Args:
        model: MTAD-GAT model
        optimizer: Optimizer used to minimize the loss function
        window_size: Length of the input sequence
        n_features: Number of input features
        target_dims: dimension of input features to forecast and reconstruct
        n_epochs: Number of iterations/epochs
        batch_size: Number of windows in a single batch
        init_lr: Initial learning rate of the module
        forecast_criterion: Loss to be used for forecasting.
        recon_criterion: Loss to be used for reconstruction.
        boolean use_cuda: To be run on GPU or not
        dload: Download directory where models are to be dumped
        log_dir: Directory where SummaryWriter logs are written to
        print_every: At what epoch interval to print losses
        log_tensorboard: Whether to log loss++ to tensorboard
        args_summary: Summary of args that will also be written to tensorboard if
            log_tensorboard
    """

    def __init__(
        self,
        model: MTAD_GAT,
        optimizer: torch.optim.Optimizer,
        window_size: int,
        n_features: int,
        target_dims: Optional[int] = None,
        n_epochs: int = 200,
        batch_size: int = 256,
        init_lr: float = 0.001,
        forecast_criterion: torch.nn.Module = nn.MSELoss(),  # Check type
        recon_criterion: torch.nn.Module = nn.MSELoss(),  # Check type
        use_cuda: bool = False,
        use_mps: bool = True,
        dload: str = "",
        log_dir: str = "output/",
        print_every: int = 1,
        log_tensorboard: bool = True,
        args_summary: str = "",
        *args,
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        if use_mps:
            self.device = "mps"
        elif use_cuda:
            self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()
        elif self.device == "mps":
            self.model.to("mps")

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        """
        Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        Args:
            train_loader: train loader of input data
            val_loader: validation loader of input data
        """

        init_train_loss = self.evaluate(train_loader)
        print(f"Init total train loss: {init_train_loss[2]:5f}")

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            print(f"Init total val loss: {init_val_loss[2]:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []
            recon_b_losses = []

            # if self.scheduler:
            # self.scheduler()

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                preds, recons = self.model(x)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)
                    preds = preds[..., self.target_dims].squeeze(-1)
                    recons = recons[..., self.target_dims].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))
                loss = forecast_loss + recon_loss

                loss.backward()
                self.optimizer.step()

                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())

            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses**2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses**2).mean())

            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            # Evaluate on validation set
            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"
            if val_loader is not None:
                forecast_val_loss, recon_val_loss, total_val_loss = self.evaluate(
                    val_loader
                )
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_total"].append(total_val_loss)

                if total_val_loss <= self.losses["val_total"][-1]:
                    self.save("model.pt")

            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )

                s += f" [{epoch_time:.1f}s]"
                print(s)

        if val_loader is None:
            self.save("model.pt")

        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader: torch.utils.data.DataLoader):
        """
        Evaluate model

        Args:
            data_loader: data loader of input data
        Returns:
            forecasting loss, reconstruction loss, total loss
        """

        self.model.eval()

        forecast_losses = []
        recon_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                preds, recons = self.model(x)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)
                    preds = preds[..., self.target_dims].squeeze(-1)
                    recons = recons[..., self.target_dims].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))

                forecast_losses.append(forecast_loss.item())
                recon_losses.append(recon_loss.item())

        forecast_losses = np.array(forecast_losses)
        recon_losses = np.array(recon_losses)

        forecast_loss = np.sqrt((forecast_losses**2).mean())
        recon_loss = np.sqrt((recon_losses**2).mean())

        total_loss = forecast_loss + recon_loss

        return forecast_loss, recon_loss, total_loss

    def save(self, file_name: str):
        """
        Pickles the model parameters to be retrieved later

        Args:
            file_name: the filename to be saved as,`dload`
                serves as the download directory
        """
        PATH = os.path.join(self.dload, file_name)
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.model.state_dict(), PATH)

    def load(self, path: PathType):
        """
        Loads the model's parameters from the path mentioned

        Args:
            Path: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def write_loss(self, epoch: int):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)
