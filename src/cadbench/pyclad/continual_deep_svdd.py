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
    - Model is preserved across fit() calls; center is re-estimated from the
      current network on each new concept's data, enabling warm-start fine-tuning.
    """

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)
        self.n_samples_, self.n_features_ = X.shape

        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        if np.min(self.hidden_neurons) > self.n_features_ and self.use_ae:
            raise ValueError(
                "The number of neurons should not exceed the number of features"
            )

        if self.model_ is None:
            self.model_ = InnerDeepSVDD(
                self.n_features,
                use_ae=self.use_ae,
                hidden_neurons=self.hidden_neurons,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
                dropout_rate=self.dropout_rate,
                l2_regularizer=self.l2_regularizer,
            )

        # Re-estimate center from current network on the new concept's data.
        X_shuffled = torch.tensor(
            np.random.permutation(X_norm), dtype=torch.float32
        )
        self.model_._init_c(X_shuffled)
        self.c = self.model_.c.detach()  # sync + detach: _init_c writes model_.c, loss reads self.c

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
                outputs = self.model_(batch_x)
                dist = torch.sum((outputs - self.c) ** 2, dim=-1)
                if self.use_ae:
                    loss = torch.mean(dist) + torch.mean(
                        torch.square(outputs - batch_x)
                    )
                else:
                    loss = torch.mean(dist)
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
