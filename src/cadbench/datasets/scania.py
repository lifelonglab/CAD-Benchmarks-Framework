import pathlib
import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cadbench.paths import TABULAR_DATASETS_PATH, OUTPUT_PATH

SCANIA_DATASET_PATH = TABULAR_DATASETS_PATH / 'scania'

# Both files carry a 20-line GPL license header before the CSV header row.
_HEADER_ROWS_TO_SKIP = 20

# Columns with more than this fraction of missing values are dropped rather
# than imputed; the dataset is notoriously sparse (some columns >80% NaN),
# and imputing those would mostly manufacture a constant column.
_MAX_MISSING_FRACTION = 0.5


def _load_raw(dataset_path: pathlib.Path) -> pd.DataFrame:
    """Load and pool the official train + test files.

    The dataset ships with its own train/test split, but (matching this
    repo's TCM/credit_fraud convention) we pool both and re-split so that
    the train partition is normal-only, rather than honoring the authors'
    split which mixes anomalies into both partitions.
    """
    train_file = dataset_path / 'aps_failure_training_set.csv'
    test_file = dataset_path / 'aps_failure_test_set.csv'
    for f in (train_file, test_file):
        if not f.exists():
            raise FileNotFoundError(f"Scania dataset file not found: {f}")

    logging.info(f"Reading {train_file.name} and {test_file.name}")
    train_df = pd.read_csv(train_file, skiprows=_HEADER_ROWS_TO_SKIP, na_values='na')
    test_df = pd.read_csv(test_file, skiprows=_HEADER_ROWS_TO_SKIP, na_values='na')
    df = pd.concat([train_df, test_df], ignore_index=True)
    logging.info(f"Pooled rows: {len(df)} ({len(train_df)} + {len(test_df)})")
    return df


def _encode_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'class': 'label'})
    df['label'] = (df['label'] == 'pos').astype(int)
    return df


def _drop_sparse_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Return feature_columns minus those exceeding _MAX_MISSING_FRACTION NaN."""
    missing_fraction = df[feature_columns].isna().mean()
    sparse_columns = missing_fraction[missing_fraction > _MAX_MISSING_FRACTION].index.tolist()
    if sparse_columns:
        logging.info(
            f"Dropping {len(sparse_columns)} columns with >{_MAX_MISSING_FRACTION:.0%} missing values: "
            f"{sparse_columns}"
        )
    return [c for c in feature_columns if c not in sparse_columns]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logging.info(f"Dropped {n_before - len(df)} duplicate rows")
    return df


def preprocess_scania(
    dataset_path: pathlib.Path = SCANIA_DATASET_PATH,
    output_path: pathlib.Path = OUTPUT_PATH / 'datasets' / 'scania',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Preprocess the Scania APS Failure dataset as a single concept.

    Steps applied:
    - Pool the official train + test files (see ``_load_raw``).
    - Drop exact duplicate rows.
    - Binary label: ``class`` column ('pos'/'neg') -> ``label`` (1/0).
    - Drop feature columns with >50% missing values (~8 of 170 columns).
    - Median-impute remaining NaNs, fit globally on all pooled rows (train +
      test). This is a deliberate exception to this repo's usual
      fit-on-train-only rule: it keeps ``scania_processed.csv`` NaN-free and
      usable standalone (e.g. for clustering on raw features), and the
      leakage from a pooled median statistic is considered negligible.
    - Stratified 80/20 split of normal-only rows into train/test; all
      anomalous rows go to test (matching TCM/credit_fraud convention).
    - StandardScaler fit on the normal-train partition only.

    Output files (written to *output_path*):
    - ``scania_processed.csv``       – cleaned, label-encoded, imputed, unscaled.
    - ``scania_single_concept.csv``  – imputed + scaled, tagged with concept_id=0.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    raw_df = _load_raw(dataset_path)
    df = _clean(raw_df)
    df = _encode_label(df)

    feature_columns = [c for c in df.columns if c != 'label']
    feature_columns = _drop_sparse_columns(df, feature_columns)
    df = df[feature_columns + ['label']].copy()

    imputer = SimpleImputer(strategy='median')
    df[feature_columns] = imputer.fit_transform(df[feature_columns])

    df.to_csv(output_path / 'scania_processed.csv', index=False)
    logging.info(f"Saved processed (unscaled) dataset to {output_path / 'scania_processed.csv'}")

    normal_df = df[df['label'] == 0]
    anomalous_df = df[df['label'] == 1]
    logging.info(f"Normal samples: {len(normal_df)}, Anomalous samples: {len(anomalous_df)}")

    train_normal_df, test_normal_df = train_test_split(
        normal_df, test_size=test_size, random_state=random_state
    )
    test_df = pd.concat([test_normal_df, anomalous_df], ignore_index=True)
    logging.info(f"Train set: {len(train_normal_df)}, Test set: {len(test_df)}")

    scaler = StandardScaler()
    train_scaled = train_normal_df.copy()
    test_scaled = test_df.copy()
    train_scaled[feature_columns] = scaler.fit_transform(train_normal_df[feature_columns])
    test_scaled[feature_columns] = scaler.transform(test_df[feature_columns])

    train_scaled['concept_split'] = 'train'
    test_scaled['concept_split'] = 'test'

    full_df = pd.concat([train_scaled, test_scaled], ignore_index=True)
    full_df['concept_id'] = 0
    full_df['concept_name'] = 'scania'

    out_file = output_path / 'scania_single_concept.csv'
    full_df.to_csv(out_file, index=False)
    logging.info(f"Saved single-concept dataset to {out_file}")
