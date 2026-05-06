import numpy as np

from pyclad.models.model import Model


class RandomModel(Model):
    """Baseline model that assigns uniformly random anomaly scores, independent of input data."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def fit(self, data: np.ndarray):
        pass

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self._rng.uniform(0.0, 1.0, size=len(data))
        labels = (scores >= 0.5).astype(int)
        return labels, scores

    def name(self) -> str:
        return "RandomModel"

    def additional_info(self):
        return {"seed": self._seed}