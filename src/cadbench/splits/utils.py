import pandas as pd
from sklearn.model_selection import train_test_split


def split_normal_train_test(concept_df: pd.DataFrame, test_size=0.2, random_state=42) -> pd.DataFrame:
    """
    Splits the normal samples of a concept into train and test parts, while keeping all anomalous samples in the test part.
    :return: pd.DataFrame with additional column 'concept_split' indicating 'train' or 'test'
    """
    normal_idx = concept_df[concept_df["label"] == 0].index
    train_normal_idx, _ = train_test_split(normal_idx, test_size=test_size, random_state=random_state)
    concept_df['concept_split'] = 'test'
    concept_df.loc[train_normal_idx, 'concept_split'] = 'train'
    return concept_df
