import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.dataloader
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from gragod import PathType


class MTAD_GAT_PLModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        model_params: dict,
        target_dims: int | None = None,
        init_lr: float = 0.001,
        forecast_criterion: torch.nn.Module = nn.MSELoss(),
        recon_criterion: torch.nn.Module = nn.MSELoss(),
        checkpoint_cb: ModelCheckpoint | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model = model

        self.model_params = model_params
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.target_dims = target_dims
        self.best_model_score = None
        self.checkpoint_cb = checkpoint_cb
        self.best_metrics = None

        self.save_hyperparameters(ignore=["model"])

    def _register_best_metrics(self):
        if self.global_step != 0:
            self.best_metrics = {
                "epoch": self.trainer.current_epoch,
                "train_loss": self.trainer.callback_metrics["Total_loss/train"],
                "train_recon_loss": self.trainer.callback_metrics["Recon_loss/train"],
                "train_forecast_loss": self.trainer.callback_metrics[
                    "Forecast_loss/train"
                ],
                "val_loss": self.trainer.callback_metrics["Total_loss/val"],
                "val_recon_loss": self.trainer.callback_metrics["Recon_loss/val"],
                "val_forecast_loss": self.trainer.callback_metrics["Forecast_loss/val"],
            }

    def forward(self, x):
        return self.model(x)

    def call_logger(
        self,
        loss: torch.Tensor,
        recon_loss: torch.Tensor,
        forecast_loss: torch.Tensor,
        step_type: str,
    ):
        self.log(
            f"Recon_loss/{step_type}",
            recon_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=True,
            logger=True,
        )
        self.log(
            f"Forecast_loss/{step_type}",
            forecast_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=True,
            logger=True,
        )
        self.log(
            f"Total_loss/{step_type}",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            logger=True,
        )

    def shared_step(self, batch, batch_idx):
        x, y = batch
        preds, recons = self(x)

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

        return loss, recon_loss, forecast_loss

    def training_step(self, batch, batch_idx):
        loss, recon_loss, forecast_loss = self.shared_step(batch, batch_idx)
        self.call_logger(loss, recon_loss, forecast_loss, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, forecast_loss = self.shared_step(batch, batch_idx)
        self.call_logger(loss, recon_loss, forecast_loss, "val")
        return loss

    def on_train_epoch_start(self):
        if (
            self.checkpoint_cb is not None
            and self.checkpoint_cb.best_model_score is not None
        ):
            if self.best_model_score is None:
                self.best_model_score = float(self.checkpoint_cb.best_model_score)
                self._register_best_metrics()
            elif (
                self.checkpoint_cb.mode == "min"
                and float(self.checkpoint_cb.best_model_score) < self.best_model_score
            ) or (
                self.checkpoint_cb.mode == "max"
                and float(self.checkpoint_cb.best_model_score) > self.best_model_score
            ):
                self.best_model_score = float(self.checkpoint_cb.best_model_score)
                self._register_best_metrics()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        return optimizer


class TrainerPL:
    def __init__(
        self,
        model: nn.Module,
        model_params: dict,
        n_epochs: int = 200,
        batch_size: int = 256,
        target_dims: int | None = None,
        init_lr: float = 0.001,
        forecast_criterion: torch.nn.Module = nn.MSELoss(),
        recon_criterion: torch.nn.Module = nn.MSELoss(),
        device: str = "cpu",
        log_dir: str = "output/",
        print_every: int = 1,
        log_tensorboard: bool = True,
        callbacks: list = [],
        *args,
        **kwargs,
    ):
        # callbacks
        self.early_stop = EarlyStopping(
            monitor="Total_loss/val",
            min_delta=0.0001,
            patience=2,
            verbose=True,
            mode="min",
        )
        self.checkpoint = ModelCheckpoint(
            monitor="Total_loss/val",
            dirpath=log_dir,
            filename="{epoch}-{Total_loss_val:.2f}",
            save_top_k=1,
            mode="min",
        )
        self.lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [self.early_stop, self.checkpoint, self.lr_monitor]

        self.lightning_module = MTAD_GAT_PLModule(
            model=model,
            model_params=model_params,
            init_lr=init_lr,
            forecast_criterion=forecast_criterion,
            recon_criterion=recon_criterion,
            batch_size=batch_size,
            target_dims=target_dims,
            n_epochs=n_epochs,
            device=device,
            callbacks=callbacks,
            checkpoint_cb=self.checkpoint,
        )
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard
        self.callbacks = callbacks

        if self.log_tensorboard:
            self.logger = TensorBoardLogger(
                save_dir=log_dir, name="mtad_gat", default_hp_metric=False
            )

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        args_summary: dict = {},
    ):
        trainer = pl.Trainer(
            max_epochs=self.n_epochs,
            accelerator=self.device,
            logger=self.logger if self.log_tensorboard else None,
            log_every_n_steps=self.print_every,
            callbacks=self.callbacks,
        )

        trainer.fit(self.lightning_module, train_loader, val_loader)

        best_metrics = {
            k: v
            for k, v in self.lightning_module.best_metrics.items()  # type: ignore
            if "epoch" in k
        }
        self.logger.log_hyperparams(params=args_summary, metrics=best_metrics)

    # def save(self, file_name: str):
    #     PATH = os.path.join(self.dload, file_name)
    #     if not os.path.exists(self.dload):
    #         os.makedirs(self.dload)
    #     torch.save(self.lightning_module.model.state_dict(), PATH)

    def load(self, path: PathType):
        self.lightning_module.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )

    def evaluate(self, data_loader: torch.utils.data.DataLoader):
        trainer = pl.Trainer(accelerator=self.device)
        results = trainer.test(self.lightning_module, data_loader)
        return (
            results[0]["test_loss"],
            results[0]["test_forecast_loss"],
            results[0]["test_recon_loss"],
        )