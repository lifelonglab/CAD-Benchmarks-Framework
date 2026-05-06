import json
import logging
from pathlib import Path

import pandas as pd
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.concepts_readers import read_concepts_from_df

from cadbench.optimization.processing import reorder_concepts


def load_ordered_dataset(ordering_json: Path, ordering_key: str) -> tuple[ConceptsDataset, str]:
    """Load a dataset from an orderings JSON produced by optimize_by_ste.py.

    Reads the source CSV, applies the concept order for *ordering_key*, and
    returns a (ConceptsDataset, name) tuple where name encodes both the source
    stem and the ordering key.

    :param ordering_json: Path to a ``*_orderings.json`` file.
    :param ordering_key: One of the keys in ``orderings``, e.g. ``"generalization_desc"``.
    :raises KeyError: If *ordering_key* is not present in the JSON.
    """
    with open(ordering_json) as f:
        spec = json.load(f)

    if ordering_key not in spec["orderings"]:
        raise KeyError(f"Ordering key '{ordering_key}' not found. Available: {list(spec['orderings'].keys())}")

    source_path = Path(spec["source"])
    df = pd.read_csv(source_path)
    if 'task_id' in df.columns:
        df = df.rename(columns={'task_id': 'concept_id', 'task_name': 'concept_name', 'task_split': 'concept_split'})
    concepts_order = spec["orderings"][ordering_key]
    df = reorder_concepts(df, concepts_order)
    df = df.dropna(subset=["concept_id"])
    df["concept_id"] = df["concept_id"].astype(int)

    source_stem = source_path.stem
    name = f"{source_stem}__{ordering_key}"
    return load_dataset_from_df(df, name=name), name


def load_dataset_from_df(df: pd.DataFrame, name='dataset') -> ConceptsDataset:
    expected_columns = ['concept_id', 'concept_name', 'label', 'concept_split']
    if not all(col in df.columns for col in expected_columns):
        for col in expected_columns:
            if col not in df.columns:
                logging.error(f"missing {col} column")
        raise ValueError(f"Dataset DataFrame should have the following columns: {expected_columns}")

    train_df = df[df['concept_split'] == 'train']
    test_df = df[df['concept_split'] == 'test']

    train_df.drop(columns=['concept_split', 'concept_part'], inplace=True, errors='ignore')
    test_df.drop(columns=['concept_split', 'concept_part'], inplace=True, errors='ignore')
    logging.info(f"Train sizes: {len(train_df)}")
    return ConceptsDataset(
        name=name,
        train_concepts=read_concepts_from_df(train_df),
        test_concepts=read_concepts_from_df(test_df)
    )
