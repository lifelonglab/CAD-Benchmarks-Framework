import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array
from torch.utils.data import TensorDataset, DataLoader

from pyod.models.deep_svdd import DeepSVDD, InnerDeepSVDD, optimizer_dict


class ContinualDeepSVDD(DeepSVDD):
    """DeepSVDD adapted for continual fine-tuning, with upstream bugs fixed.

    Changes vs PyOD's DeepSVDD:
    - loss.backward() was commented out (no training occurred) — fixed.
    - Center c was always 0.0 (stored on model_.c but read from self.c) — fixed.
    - w_d was a pre-loop constant so contributed nothing to gradients — removed.
    - Model is preserved across fit() calls, enabling warm-start fine-tuning.
    - With freeze_preprocessing/freeze_center (default), the scaler and center c
      are estimated once on the first concept and reused afterwards. The anomaly
      score is the distance to c in the fixed input scale, so re-estimating either
      on each concept redefines the score and erases performance on past concepts
      regardless of how well the network weights are preserved.
    - use_ae=True was broken upstream: with a decoder attached, the full forward
      pass returns a reconstruction (input-dim), but the distance-to-center loss
      and score need the embedding (dim hidden_neurons[-1]) that c lives in,
      causing a shape-mismatch crash. Fixed here by reading the embedding off a
      forward hook (see _embed) so it's available alongside the reconstruction
      regardless of use_ae.
    """

    def __init__(self, *args, freeze_preprocessing: bool = True, freeze_center: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_preprocessing = freeze_preprocessing
        self.freeze_center = freeze_center

    def _embed(self, x):
        """Return (full forward output, embedding) for *x*.

        With use_ae=False these are the same tensor (the network already stops
        at the embedding). With use_ae=True the full forward continues through
        a decoder back to input space, but c and the distance-to-center
        loss/score live in embedding space (the same layer _init_c reads off) —
        captured here via a forward hook so both are available in one pass.
        """
        layer_name = f"hidden_activation_e{len(self.hidden_neurons)}"
        captured = {}
        handle = self.model_.model._modules[layer_name].register_forward_hook(
            lambda m, i, o: captured.__setitem__("out", o)
        )
        try:
            outputs = self.model_(x)
        finally:
            handle.remove()
        return outputs, captured["out"]

    def decision_function(self, X):
        X = check_array(X)
        X_norm = self.scaler_.transform(X) if self.preprocessing else np.copy(X)
        X_norm = torch.tensor(X_norm, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            _, embedding = self._embed(X_norm)
            dist = torch.sum((embedding - self.c) ** 2, dim=-1)
        return dist.numpy()

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)
        self.n_samples_, self.n_features_ = X.shape

        if self.preprocessing:
            if getattr(self, "scaler_", None) is None or not self.freeze_preprocessing:
                self.scaler_ = StandardScaler()
                self.scaler_.fit(X)
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        if np.min(self.hidden_neurons) > self.n_features_ and self.use_ae:
            raise ValueError(
                "The number of neurons should not exceed the number of features"
            )

        first_fit = self.model_ is None
        if first_fit:
            self.model_ = InnerDeepSVDD(
                self.n_features,
                use_ae=self.use_ae,
                hidden_neurons=self.hidden_neurons,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
                dropout_rate=self.dropout_rate,
                l2_regularizer=self.l2_regularizer,
            )

        if first_fit or not self.freeze_center:
            X_shuffled = torch.tensor(
                np.random.permutation(X_norm), dtype=torch.float32
            )
            self.model_._init_c(X_shuffled)
            # _init_c runs a forward pass without no_grad, so model_.c is a non-leaf
            # tensor that pins the whole init graph and breaks deepcopy (LwF teacher).
            self.model_.c = self.model_.c.detach()
            self.c = self.model_.c  # sync: _init_c writes model_.c, loss reads self.c

        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        dataloader = DataLoader(
            TensorDataset(X_tensor, X_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = optimizer_dict[self.optimizer](
            self.model_.parameters(), weight_decay=self.l2_regularizer
        )

        best_loss = float('inf')
        best_model_dict = None

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                outputs, embedding = self._embed(batch_x)
                dist = torch.sum((embedding - self.c) ** 2, dim=-1)
                if self.use_ae:
                    loss = torch.mean(dist) + torch.mean(
                        torch.square(outputs - batch_x)
                    )
                else:
                    loss = torch.mean(dist)
                if getattr(self, "_loss_penalty", None) is not None:
                    loss = loss + self._loss_penalty(self.model_, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_dict = self.model_.state_dict()
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.6f}")

        self.best_model_dict = best_model_dict

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self
