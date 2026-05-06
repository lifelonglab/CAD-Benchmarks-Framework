import json
import logging
import pathlib
from typing import TypeAlias

MetricMatrix: TypeAlias = dict[str, dict[str, float]]

logger = logging.getLogger(__name__)


def load_selected_ste_results(ste_results_path: pathlib.Path, selected_alg: str = None,
                              selected_metric: str = 'ROC-AUC') -> MetricMatrix:
    matrices = load_ste_results(ste_results_path)[selected_metric]
    if selected_alg is not None:
        return matrices[selected_alg]
    return average_results(matrices)


def load_ste_results(ste_results_path: pathlib.Path) -> dict[
    str, dict[str, MetricMatrix]]:  # metric name -> [alg -> metric matrix]
    values = json.load(ste_results_path.open())
    # backward compatibility
    if 'ROC-AUC' not in values:
        logger.info("fixing backward compatibility for STE results file")
        values = {'ROC-AUC': values}
    #Filter out random model
    return {
        metric: {alg: matrix for alg, matrix in algs.items() if alg != 'Random' and alg != 'COPOD' and alg != 'HBOS' and alg != 'DeepSVDD'}
        for metric, algs in values.items()
    }


def average_results(alg_metrics: dict[str, MetricMatrix]) -> MetricMatrix:
    matrices = list(alg_metrics.values())
    concepts = matrices[0].keys()
    return {
        trained_concept: {
            eval_concept: sum(m[trained_concept][eval_concept] for m in matrices) / len(matrices)
            for eval_concept in concepts
        }
        for trained_concept in concepts
    }
