from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.base import ClusterMixin


@dataclass
class ClusteringConfig:
    feature_columns: list[str]
    min_normal_samples: int
    min_anomalous_samples: int
    min_clusters: int
    sampling_size: int | None = None

    def min_samples(self):
        return self.min_normal_samples + self.min_anomalous_samples


ClusteringMethod = Callable[[pd.DataFrame, Callable[[ClusteringConfig], ClusterMixin], ClusteringConfig], pd.DataFrame | None]
