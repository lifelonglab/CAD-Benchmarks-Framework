"""DAGMM adapted for pyclad Model interface.
Original: Daniel Stanley Tan (https://github.com/danieltan07/dagmm), Zong et al. (2018).
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from pyclad.models.model import Model


class _AutoencoderModule(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden = max(latent_dim * 2, input_dim // 4, 8)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x, return_latent=False):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        if return_latent:
            return dec, enc
        return dec


class _DAGMMModule(nn.Module):
    def __init__(self, autoencoder: _AutoencoderModule, n_gmm: int, latent_dim: int):
        super().__init__()
        self.add_module("autoencoder", autoencoder)
        self.estimation = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1),
        )
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))

    def _relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / torch.clamp(a.norm(2, dim=1), min=1e-10)

    def forward(self, x):
        flat = x.view(x.shape[0], -1)
        dec, enc = self.autoencoder(x, return_latent=True)
        dec_flat = dec.view(dec.shape[0], -1)
        rec_cosine = F.cosine_similarity(flat, dec_flat, dim=1)
        rec_euclidean = self._relative_euclidean_distance(flat, dec_flat)
        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
        gamma = self.estimation(z)
        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        sum_gamma = torch.sum(gamma, dim=0)
        phi = sum_gamma / N
        self.phi = phi.data
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov

        k, d, _ = cov.size()
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-6
        for i in range(k):
            cov_k = cov[i] + torch.eye(d) * eps
            pinv = np.linalg.pinv(cov_k.detach().numpy())
            cov_inverse.append(torch.from_numpy(pinv).unsqueeze(0))

            eigvals = np.linalg.eigvals(cov_k.detach().cpu().numpy() * (2 * np.pi))
            det_cov.append(np.prod(np.clip(eigvals, a_min=sys.float_info.epsilon, a_max=None)))

            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0).float()
        det_cov = torch.from_numpy(np.float32(np.array(det_cov)))

        exp_term_tmp = -0.5 * torch.sum(
            torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1
        )
        max_val = torch.max(exp_term_tmp.clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov) + eps).unsqueeze(0), dim=1) + eps
        )

        if size_average:
            sample_energy = torch.mean(sample_energy)
        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x.view(*x_hat.shape) - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag


class DAGMM(Model):
    """Deep Autoencoding Gaussian Mixture Model (Zong et al., 2018)."""

    def __init__(
        self,
        num_epochs: int = 10,
        lambda_energy: float = 0.1,
        lambda_cov_diag: float = 0.005,
        lr: float = 1e-3,
        batch_size: int = 50,
        gmm_k: int = 3,
        latent_dim: int = None,
        contamination: float = 0.05,
        seed: int = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.num_epochs = num_epochs
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.lr = lr
        self.batch_size = batch_size
        self.gmm_k = gmm_k
        self.latent_dim = latent_dim  # None = auto: 5 + n_features // 20
        self.contamination = contamination
        self.seed = seed
        self._module: _DAGMMModule = None
        self._scaler = StandardScaler()

    def fit(self, data: np.ndarray):
        data = self._scaler.fit_transform(data)
        n_features = data.shape[1]
        latent_dim = self.latent_dim if self.latent_dim is not None else 5 + n_features // 20
        self._module = _DAGMMModule(
            autoencoder=_AutoencoderModule(n_features, latent_dim),
            n_gmm=self.gmm_k,
            latent_dim=latent_dim + 2,
        )
        optimizer = torch.optim.Adam(self._module.parameters(), lr=self.lr)
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(data)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        for _ in trange(self.num_epochs):
            self._module.train()
            for (batch,) in loader:
                _, dec, z, gamma = self._module(batch)
                loss, _, _, _ = self._module.loss_function(
                    batch, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag
                )
                optimizer.zero_grad()
                torch.clamp(loss, max=1e7).backward()
                torch.nn.utils.clip_grad_norm_(self._module.parameters(), 5)
                optimizer.step()

        # Compute GMM params over the full training set so inference is stable
        self._refit_gmm(data)

    def _refit_gmm(self, data: np.ndarray):
        """One eval-mode pass to compute GMM params from all training data."""
        self._module.eval()
        all_z, all_gamma = [], []
        with torch.no_grad():
            for (batch,) in DataLoader(TensorDataset(torch.FloatTensor(data)), batch_size=256, shuffle=False):
                _, _, z, gamma = self._module(batch)
                all_z.append(z)
                all_gamma.append(gamma)
        self._module.compute_gmm_params(torch.cat(all_z), torch.cat(all_gamma))

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        scores = self._anomaly_scores(data)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        labels = (scores >= threshold).astype(int)
        return labels, scores

    def _anomaly_scores(self, data: np.ndarray) -> np.ndarray:
        data = self._scaler.transform(data)
        self._module.eval()
        parts = []
        with torch.no_grad():
            for (batch,) in DataLoader(TensorDataset(torch.FloatTensor(data)), batch_size=256, shuffle=False):
                _, _, z, _ = self._module(batch)
                energy, _ = self._module.compute_energy(z, size_average=False)
                parts.append(energy.numpy())
        return np.concatenate(parts)

    def name(self) -> str:
        return "DAGMM"

    def additional_info(self):
        return {
            "num_epochs": self.num_epochs,
            "gmm_k": self.gmm_k,
            "latent_dim": self.latent_dim,
            "contamination": self.contamination,
        }