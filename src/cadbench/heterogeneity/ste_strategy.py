from collections.abc import Callable

import logging
from pyclad.models.model import Model
from pyclad.strategies.strategy import ConceptIncrementalStrategy, ConceptAwareStrategy

logger = logging.getLogger(__name__)


class SingleTaskExpertStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    def __init__(self, model_fn: Callable[[], Model]):
        self.model_fn = model_fn
        self.model = None

    def learn(self, data, concept_id=None):
        self.model = self.model_fn()
        logger.info(f"Created new model for concept {concept_id}")
        self.model.fit(data)

    def predict(self, data, concept_id=None):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call learn() before predict().")
        return self.model.predict(data)

    def name(self) -> str:
        return "SingleTaskExpert"
