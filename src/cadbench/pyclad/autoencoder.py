import numpy as np
import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, random_split

from pyclad.models.model import Model


class Autoencoder(Model):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, lr: float = 1e-3, threshold: float = 0.5, epochs: int = 20
    ):
        self.module = AutoencoderModule(encoder, decoder, lr)
        self.threshold = threshold
        self.epochs = epochs
        self.scaler = StandardScaler()

    def fit(self, data: np.ndarray):
        data = self.scaler.fit_transform(data)
        dataset = TensorDataset(torch.Tensor(data))
        val_size = max(1, int(0.1 * len(dataset)))
        train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=256, num_workers=8)
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        trainer = pl.Trainer(max_epochs=self.epochs, callbacks=[TQDMProgressBar(refresh_rate=1), early_stopping])
        trainer.fit(self.module, train_dl, val_dl)

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        data = self.scaler.transform(data)
        self.module.eval()
        with torch.no_grad():
            x = torch.Tensor(data).to(self.module.device)
            x_hat = self.module(x).cpu()
        rec_error = ((data - x_hat.numpy()) ** 2).mean(axis=1)
        binary_predictions = (rec_error > self.threshold).astype(int)
        return binary_predictions, rec_error

    def name(self) -> str:
        return "Autoencoder"

    def additional_info(self):
        return {
            "threshold": self.threshold,
            "encoder": str(self.module.encoder),
            "decoder": str(self.module.decoder),
            "lr": self.module.lr,
            "epochs": self.epochs,
        }


class AutoencoderModule(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr

        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch[0]
        loss = self.loss(self(x), x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        loss = self.loss(self(x), x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
