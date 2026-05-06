import pandas as pd

from cadbench.paths import logger


def reorder_concepts(df: pd.DataFrame, concepts_order: list[str]) -> pd.DataFrame:
    """
    Reorder the concepts in the DataFrame according to the provided order.

    :param df: DataFrame with a column 'concept' containing concept names.
    :param concepts_order: List of concept names in the desired order.
    :return: DataFrame with concepts reordered.
    """
    concept_to_new_id = {concept: i for i, concept in enumerate(concepts_order)}
    logger.info(f'Reordering concepts according to the following order: {concepts_order}')
    df['concept_id'] = df['concept_name'].map(concept_to_new_id)
    df = df.sort_values('concept_id')

    return df