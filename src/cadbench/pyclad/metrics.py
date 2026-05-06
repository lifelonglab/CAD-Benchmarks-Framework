import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pyclad.metrics.base.base_metric import BaseMetric


class PrAuc(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        scores = np.asarray(anomaly_scores, dtype=np.float64)

        if not np.all(np.isfinite(scores)):
            finite = scores[np.isfinite(scores)]

            if finite.size == 0:
                return np.nan

            max_finite = np.max(finite)
            min_finite = np.min(finite)

            scores = np.nan_to_num(
                scores,
                nan=max_finite,
                posinf=max_finite,
                neginf=min_finite,
            )
        return average_precision_score(y_true=y_true, y_score=scores)

    def name(self) -> str:
        return "PR-AUC"


class NormalizedPrAuc(BaseMetric):
    """PR-AUC normalized by the random-classifier baseline (class prevalence).

    normalized = (PR-AUC - prevalence) / (1 - prevalence)

    Ranges from 0 (random) to 1 (perfect). Can be negative for below-random classifiers.
    Returns NaN when all labels are the same class (prevalence is 0 or 1).
    """

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        scores = np.asarray(anomaly_scores, dtype=np.float64)

        if not np.all(np.isfinite(scores)):
            finite = scores[np.isfinite(scores)]

            if finite.size == 0:
                return np.nan

            max_finite = np.max(finite)
            min_finite = np.min(finite)

            scores = np.nan_to_num(
                scores,
                nan=max_finite,
                posinf=max_finite,
                neginf=min_finite,
            )
        prevalence = np.mean(y_true)
        if prevalence == 0.0 or prevalence == 1.0:
            return float("nan")
        pr_auc = average_precision_score(y_true=y_true, y_score=scores)
        return (pr_auc - prevalence) / (1.0 - prevalence)

    def name(self) -> str:
        return "Normalized-PR-AUC"

class RocAucRobust(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        scores = np.asarray(anomaly_scores, dtype=np.float64)

        if not np.all(np.isfinite(scores)):
            finite = scores[np.isfinite(scores)]

            if finite.size == 0:
                return np.nan

            max_finite = np.max(finite)
            min_finite = np.min(finite)

            scores = np.nan_to_num(
                scores,
                nan=max_finite,
                posinf=max_finite,
                neginf=min_finite,
            )
        return roc_auc_score(y_true=y_true, y_score=scores)

    def name(self) -> str:
        return "ROC-AUC"
