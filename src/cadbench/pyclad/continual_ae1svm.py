import numpy as np
import torch
from sklearn.utils.validation import check_array

from pyod.models.ae1svm import AE1SVM, InnerAE1SVM, TorchDataset
from pyod.utils.stat_models import pairwise_distances_no_broadcast


class ContinualAE1SVM(AE1SVM):
    """AE1SVM adapted for continual fine-tuning.

    Identical to PyOD's AE1SVM except the network is preserved across fit()
    calls, enabling warm-start fine-tuning on new concepts. The preprocessing
    scaler and optimizer are still reset each call (adapt to new data statistics;
    fresh Adam state).
    """

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)

        n_samples, n_features = X.shape
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            self.std[self.std == 0] = 1e-6
            train_set = TorchDataset(X=X, mean=self.mean, std=self.std, return_idx=True)
        else:
            train_set = TorchDataset(X=X, return_idx=True)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            drop_last=self.batch_norm,  # BatchNorm fails on single-sample batches
        )

        if self.model is None:
            self.model = InnerAE1SVM(
                n_features=n_features,
                encoding_dim=32,
                rff_dim=self.kernel_approx_features,
                sigma=self.sigma,
                hidden_neurons=self.hidden_neurons,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                hidden_activation=self.hidden_activation,
            ).to(self.device)

        self._train_autoencoder(train_loader)

        if self.best_model_dict is not None:
            self.model.load_state_dict(self.best_model_dict)
        else:
            raise ValueError('Training failed, no valid model state found')

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ['model', 'best_model_dict'])
        X = check_array(X)
        dataset = (
            TorchDataset(X=X, mean=self.mean, std=self.std, return_idx=True)
            if self.preprocessing
            else TorchDataset(X=X, return_idx=True)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
        self.model.eval()
        outlier_scores = np.zeros(X.shape[0])
        with torch.no_grad():
            for data, data_idx in dataloader:
                data = data.to(self.device).float()
                reconstructions, _ = self.model(data)
                scores = pairwise_distances_no_broadcast(
                    data.cpu().numpy(), reconstructions.cpu().numpy()
                )
                outlier_scores[data_idx.cpu().numpy()] = scores  # fix: tensor → numpy index
        return outlier_scores
