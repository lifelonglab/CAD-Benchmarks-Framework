from typing import Callable, Dict

import numpy as np

from pyclad.models.model import Model
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptAwareStrategy,
    ConceptIncrementalStrategy,
)


class CumulativeNMStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy, ConceptAgnosticStrategy):
    def __init__(self, model_creation_fn: Callable[[], Model]):
        self._model_creation_fn = model_creation_fn
        self._replay = []
        self._model: Model = model_creation_fn()

    def learn(self, data: np.ndarray, *args, **kwargs) -> None:
        self._replay.append(data)
        self._model = self._model_creation_fn()
        self._model.fit(np.concatenate(self._replay))

    def predict(self, data: np.ndarray, *args, **kwargs) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def name(self) -> str:
        return "CumulativeNM"

    def additional_info(self) -> Dict:
        return {"model": self._model.name(), "buffer_size": len(np.concatenate(self._replay))}