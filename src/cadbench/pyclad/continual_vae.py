import numpy as np
import torch
from sklearn.utils.validation import check_array

from pyod.models.vae import VAE, VAEModel
from pyod.utils.torch_utility import TorchDataset


class ContinualVAE(VAE):
    """VAE that warm-starts from existing weights on repeated fit() calls.

    Identical to PyOD's VAE except:
    - build_model() skips model creation if a model is already present,
      enabling continual fine-tuning across concepts. The optimizer is still
      reset each call (fresh Adam state), but the network weights carry over.
    - With freeze_preprocessing (default), X_mean/X_std are computed once on
      the first concept and reused afterwards. decision_function() standardizes
      with these stats, so recomputing them per concept redefines the anomaly
      score and erases performance on past concepts regardless of the weights.
    - Optional gradient norm clipping (``max_grad_norm``, default ``None`` —
      off, matching prior behaviour) before each optimizer step, as a NaN
      safety net: this benchmark's heavy-tailed rate features (Flow Bytes/s,
      IAT columns, ...) can put individual rows tens of std devs out even
      within their own concept, and worse under the frozen scaler on later
      concepts; an unclipped outlier batch can in principle produce a
      squared-error gradient large enough to diverge the network to NaN
      weights. Clipping the gradient norm bounds the size of any one
      optimizer step without altering what the network actually sees, unlike
      clipping the inputs themselves (tried first, rejected: this benchmark's
      later concepts routinely exceed even generous input-clip thresholds
      under the frozen scaler by design, so clipping inputs flattens large,
      legitimate chunks of a later concept's distribution). Left off by
      default because even grad-norm clipping measurably shifts training
      dynamics on small runs (non-monotonic effect on current-task fit in
      testing, not a clean win) and a NaN crash wasn't reproduced locally —
      turn it on explicitly (e.g. ``max_grad_norm=20``) if you hit NaN and
      want to try this as a mitigation, and re-validate your specific run.
    """

    def __init__(self, *args, freeze_preprocessing: bool = True, max_grad_norm: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_preprocessing = freeze_preprocessing
        self.max_grad_norm = max_grad_norm

    def build_model(self):
        if getattr(self, 'model', None) is not None:
            return
        self.model = VAEModel(
            self.feature_size,
            encoder_neuron_list=self.encoder_neuron_list,
            decoder_neuron_list=self.decoder_neuron_list,
            latent_dim=self.latent_dim,
            hidden_activation_name=self.hidden_activation_name,
            output_activation_name=self.output_activation_name,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
            logvar_clip=self.logvar_clip,
        )

    def training_forward(self, batch_data):
        # Copy of pyod VAE.training_forward with one change: gradient norm
        # clipping before optimizer.step(), see class docstring.
        x = batch_data
        x = x.to(self.device)
        self.optimizer.zero_grad()
        x_recon, z_mu, z_logvar = self.model(x)
        loss = self.criterion(x, x_recon, z_mu, z_logvar,
                               beta=self.beta, capacity=self.capacity,
                               logvar_clip=self.logvar_clip)
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def fit(self, X, y=None):
        # Copy of BaseDeepLearningDetector.fit (pyod 2.1.0) with one change:
        # X_mean/X_std are only computed when absent or freezing is disabled.
        X = check_array(X)
        self._set_n_classes(y)

        self.data_num, self.feature_size = X.shape
        self.build_model()
        self.training_prepare()

        if self.preprocessing:
            if getattr(self, 'X_mean', None) is None or not self.freeze_preprocessing:
                self.X_mean = np.mean(X, axis=0)
                # Features constant in this concept may vary in later ones; without
                # the guard TorchDataset's eps would blow them up by ~1e8.
                X_std = np.std(X, axis=0)
                self.X_std = np.where(X_std == 0, 1.0, X_std)
            train_set = TorchDataset(X=X, y=None,
                                     mean=self.X_mean, std=self.X_std)
        else:
            train_set = TorchDataset(X=X, y=None)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=self.batch_size,
            shuffle=True, drop_last=True)

        self.train(train_loader)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self
