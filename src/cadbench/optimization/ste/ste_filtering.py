import logging

import pandas as pd

from cadbench.optimization.ste.utils import MetricMatrix

logger = logging.getLogger(__name__)



def filter_matrices(
        matrices: dict[str, MetricMatrix], concepts_to_keep: list[str]) -> dict[str, MetricMatrix]:
    return {
        alg: {
            trained: {eval_concept: scores[eval_concept] for eval_concept in scores.keys() if
                      eval_concept in concepts_to_keep}
            for trained, scores in matrix.items()
            if trained in concepts_to_keep
        }
        for alg, matrix in matrices.items()
    }


def filter_dataset(df: pd.DataFrame, concepts_to_keep: list[str]) -> pd.DataFrame:
    return df.loc[df['concept_name'].isin(concepts_to_keep)]

