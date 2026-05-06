import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

from pyod.models.dif import DIF, MLPnet


class ContinualDIF(DIF):
    """DIF adapted for continual fine-tuning.

    On the first fit() call the random networks and MinMaxScaler are
    initialized and then frozen for all subsequent calls.  Only the
    IsolationForests are refit on each new task's representations, so
    the representation space stays consistent across tasks.
    """

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)
        n_features = X.shape[1]

        if self.minmax_scaler is None:
            self.minmax_scaler = MinMaxScaler()
            self.minmax_scaler.fit(X)

            network_params = {
                "n_features": n_features,
                "n_hidden": self.hidden_neurons,
                "n_output": self.representation_dim,
                "activation": self.hidden_activation,
                "skip_connection": self.skip_connection,
            }
            ensemble_seeds = np.random.randint(0, 100000, self.n_ensemble)
            self.net_lst = []
            self.iForest_lst = [
                IsolationForest(
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples,
                    random_state=int(ensemble_seeds[i]),
                )
                for i in range(self.n_ensemble)
            ]
            for i in range(self.n_ensemble):
                net = MLPnet(**network_params).to(self.device)
                torch.manual_seed(int(ensemble_seeds[i]))
                for name, param in net.named_parameters():
                    if name.endswith("weight"):
                        torch.nn.init.normal_(param, mean=0.0, std=1.0)
                self.net_lst.append(net)

        X_scaled = self.minmax_scaler.transform(X)
        self.x_reduced_lst = []
        for i in range(self.n_ensemble):
            x_reduced = self._deep_representation(self.net_lst[i], X_scaled)
            self.x_reduced_lst.append(x_reduced)
            self.iForest_lst[i].fit(x_reduced)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self