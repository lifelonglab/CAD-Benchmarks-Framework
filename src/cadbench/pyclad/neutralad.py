"""NeuTraL-AD adapted for the pyclad Model interface.

Original: Qiu et al., ICML 2021 — Neural Transformation Learning for Anomaly Detection.
Partially adapted from https://github.com/boschresearch/NeuTraL-AD (AGPL-3.0 license).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from pyclad.models.model import Model


class NeuTraLAD(Model):
    """Neural Transformation Learning-based Anomaly Detection (ICML 2021)."""

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        n_trans: int = 11,
        trans_type: str = "residual",
        temp: float = 0.1,
        rep_dim: int = 128,
        hidden_dims: str = "100,50",
        trans_hidden_dims: int = 50,
        act: str = "LeakyReLU",
        bias: bool = False,
        contamination: float = 0.05,
        seed: int = None,
        freeze_preprocessing: bool = True,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_trans = n_trans
        self.trans_type = trans_type
        self.temp = temp
        self.rep_dim = rep_dim
        self.hidden_dims = hidden_dims
        self.trans_hidden_dims = trans_hidden_dims
        self.act = act
        self.bias = bias
        self.contamination = contamination
        self.seed = seed
        self._net: _TabNeuTraLADNet = None
        self._net_n_features: int = None
        self._criterion: _DCL = None
        self._scaler = StandardScaler()
        self.freeze_preprocessing = freeze_preprocessing

    def fit(self, data: np.ndarray):
        if len(data) > 500_000:
            rng = np.random.default_rng(self.seed)
            data = data[rng.choice(len(data), size=500_000, replace=False)]
        # Scores are computed in the scaler's coordinates: refitting it per concept
        # would redefine them and erase performance on past concepts.
        if self.freeze_preprocessing and hasattr(self._scaler, "mean_"):
            data = self._scaler.transform(data)
        else:
            data = self._scaler.fit_transform(data)
        n_features = data.shape[1]
        # Warm-start: keep weights across concepts; only (re)build when absent or the
        # feature dimension changed.
        if self._net is None or self._net_n_features != n_features:
            self._net = _TabNeuTraLADNet(
                n_features=n_features,
                n_trans=self.n_trans,
                trans_type=self.trans_type,
                enc_hidden_dims=self.hidden_dims,
                trans_hidden_dims=self.trans_hidden_dims,
                activation=self.act,
                bias=self.bias,
                rep_dim=self.rep_dim,
            )
            self._net_n_features = n_features
        if self._criterion is None:
            self._criterion = _DCL(temperature=self.temp)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(data)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self._net.train()
        for _ in trange(self.epochs, desc="NeuTraLAD"):
            for (batch,) in loader:
                z = self._net(batch)
                loss = self._criterion(z)
                if getattr(self, "_loss_penalty", None) is not None:
                    loss = loss + self._loss_penalty(self._net, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self._anomaly_scores(data)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        labels = (scores >= threshold).astype(int)
        return labels, scores

    def _anomaly_scores(self, data: np.ndarray) -> np.ndarray:
        data = self._scaler.transform(data)
        self._net.eval()
        self._criterion.reduction = "none"
        parts = []
        with torch.no_grad():
            for (batch,) in DataLoader(TensorDataset(torch.FloatTensor(data)), batch_size=256, shuffle=False):
                z = self._net(batch)
                s = self._criterion(z)
                parts.append(s.numpy())
        self._criterion.reduction = "mean"
        return np.concatenate(parts)

    def name(self) -> str:
        return "NeuTraLAD"

    def additional_info(self):
        return {
            "epochs": self.epochs,
            "n_trans": self.n_trans,
            "trans_type": self.trans_type,
            "temp": self.temp,
            "rep_dim": self.rep_dim,
            "contamination": self.contamination,
        }


# ── internal network classes ──────────────────────────────────────────────────

class _TabNeuTraLADNet(nn.Module):
    def __init__(self, n_features, n_trans=11, trans_type="residual",
                 enc_hidden_dims="24,24,24,24", trans_hidden_dims=24,
                 rep_dim=24, activation="ReLU", bias=False):
        super().__init__()
        self.enc = _MLPNet(
            n_features=n_features,
            n_hidden=enc_hidden_dims,
            n_output=rep_dim,
            activation=activation,
            bias=bias,
        )
        self.trans = nn.ModuleList([
            _MLPNet(
                n_features=n_features,
                n_hidden=trans_hidden_dims,
                n_output=n_features,
                activation=activation,
                bias=bias,
            )
            for _ in range(n_trans)
        ])
        self.n_trans = n_trans
        self.trans_type = trans_type
        self.z_dim = rep_dim

    def forward(self, x):
        x_transform = torch.empty(x.shape[0], self.n_trans, x.shape[-1]).to(x)
        for i in range(self.n_trans):
            mask = self.trans[i](x)
            if self.trans_type == "forward":
                x_transform[:, i] = mask
            elif self.trans_type == "mul":
                x_transform[:, i] = torch.sigmoid(mask) * x
            else:  # residual
                x_transform[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1), x_transform], dim=1)
        zs = self.enc(x_cat.reshape(-1, x.shape[-1]))
        return zs.reshape(x.shape[0], self.n_trans + 1, self.z_dim)

    def hidden_repr(self, x):
        """The encoder's last hidden activation for the plain (untransformed) input.

        Used only for LwF distillation: ``forward()``'s output feeds ``_DCL``'s
        contrastive loss directly (over original vs. transformed views), so
        distilling it would pin that comparison to the teacher's. This is one
        layer earlier — the encoder's own pre-final-projection activation — and
        computed only for the plain input, upstream of the transform comparisons.
        """
        return self.enc.network[:-1](x)


class _DCL(nn.Module):
    def __init__(self, temperature=0.1, reduction="mean"):
        super().__init__()
        self.temp = temperature
        self.reduction = reduction

    def forward(self, z):
        z = F.normalize(z, p=2, dim=-1)
        z_ori = z[:, 0]       # (n, z_dim)
        z_trans = z[:, 1:]    # (n, n_trans, z_dim)
        batch_size, n_trans, z_dim = z.shape

        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1)) / self.temp)  # (n, k, k)
        mask = (torch.ones_like(sim_matrix) - torch.eye(n_trans, device=z.device).unsqueeze(0)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, n_trans, -1)
        trans_matrix = sim_matrix[:, 1:, 1:].sum(-1)  # (n, k-1): exclude column 0 (original)

        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), dim=-1) / self.temp)  # (n, k-1)
        K = n_trans - 1
        scale = 1 / np.abs(K * np.log(1.0 / K))

        loss = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale
        loss = loss.sum(1)  # (n,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"


class _MLPNet(nn.Module):
    def __init__(self, n_features, n_hidden="500,100", n_output=20,
                 activation="ReLU", bias=False):
        super().__init__()
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        elif isinstance(n_hidden, str):
            n_hidden = [int(h) for h in n_hidden.split(",")]

        dims = [n_features] + n_hidden + [n_output]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2:
                layers.append(getattr(nn, activation)())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
