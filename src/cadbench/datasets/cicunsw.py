import pathlib
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cadbench.paths import RESOURCES_PATH, OUTPUT_PATH, create_path

CICUNSW_DATASET_PATH = RESOURCES_PATH / 'datasets' / 'tabular' / 'cicunsw'

# Metadata columns produced by CICFlowMeter that are not network-traffic features
_DROP_COLUMNS = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']


def _load_raw(dataset_path: pathlib.Path) -> pd.DataFrame:
    """Load CICFlowMeter_out.csv from *dataset_path*."""
    csv_file = dataset_path / 'CICFlowMeter_out.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f"CICFlowMeter_out.csv not found in {dataset_path}")

    logging.info(f"Reading {csv_file.name}")
    df = pd.read_csv(csv_file, low_memory=False)
    df.columns = df.columns.str.strip()
    logging.info(f"Rows loaded: {len(df)}")
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Inf/NaN rows, drop metadata columns, and drop duplicates."""
    drop_cols = [c for c in _DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna()
    logging.info(f"Dropped {n_before - len(df)} rows containing NaN/Inf values")

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logging.info(f"Dropped {n_before - len(df)} duplicate rows")

    return df


def _encode_label(df: pd.DataFrame) -> pd.DataFrame:
    df['Label'] = df['Label'].str.strip().apply(lambda v: 0 if v.lower() == 'benign' else 1)
    df.rename(columns={'Label': 'label'}, inplace=True)
    return df


def _scale(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on *train*, apply to both splits."""
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[columns] = scaler.fit_transform(train[columns])
    test_scaled[columns] = scaler.transform(test[columns])
    return train_scaled, test_scaled


def split_cicunsw_by_day(
    dataset_path: pathlib.Path = CICUNSW_DATASET_PATH,
    output_path: pathlib.Path = OUTPUT_PATH / 'datasets' / 'cicunsw',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Split the CIC-UNSW dataset into concepts by calendar day.

    The Timestamp column is parsed to extract the date before being dropped.
    Each unique date becomes one concept. Within each concept the normal
    samples are split into train/test; all anomalous samples go to the test
    partition.

    Output files (written to *output_path*):
    - ``cicunsw_by_day.csv`` – one concept per calendar day.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    raw_df = _load_raw(dataset_path)
    raw_df['_date'] = pd.to_datetime(
        raw_df['Timestamp'], dayfirst=True
    ).dt.date

    dates = sorted(raw_df['_date'].unique())
    logging.info(f"Found {len(dates)} unique days: {dates}")

    per_day: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []

    for date in dates:
        day_df = raw_df[raw_df['_date'] == date].drop(columns=['_date'])
        day_df = _clean(day_df)
        day_df = _encode_label(day_df)

        normal_df = day_df[day_df['label'] == 0]
        anomalous_df = day_df[day_df['label'] == 1]
        name = str(date).replace('-', '_')
        logging.info(f"  {name}: {len(normal_df)} normal, {len(anomalous_df)} anomalous")

        train_normal, test_normal = train_test_split(
            normal_df, test_size=test_size, random_state=random_state
        )
        test_df = pd.concat([test_normal, anomalous_df], ignore_index=True)
        per_day.append((train_normal, test_df, name))

    concept_dfs: list[pd.DataFrame] = []
    concept_id = 0
    for train_normal, test_df, name in per_day:
        n_test_normal = (test_df['label'] == 0).sum()
        n_test_anomalous = (test_df['label'] == 1).sum()

        if n_test_anomalous == 0:
            logging.warning(f"  Skipping concept '{name}': test set has no anomalous samples")
            continue
        if n_test_normal == 0:
            logging.warning(f"  Skipping concept '{name}': test set has no normal samples")
            continue
        if n_test_anomalous > n_test_normal:
            logging.warning(
                f"  Concept '{name}': test anomalies ({n_test_anomalous}) exceed "
                f"normal samples ({n_test_normal}), subsampling anomalies to match"
            )
            test_df = pd.concat([
                test_df[test_df['label'] == 0],
                test_df[test_df['label'] == 1].sample(n=n_test_normal, random_state=random_state),
            ], ignore_index=True)
            n_test_anomalous = n_test_normal

        train_normal = train_normal.copy()
        train_normal['concept_split'] = 'train'
        test_df['concept_split'] = 'test'

        concept_df = pd.concat([train_normal, test_df], ignore_index=True)
        concept_df['concept_id'] = concept_id
        concept_df['concept_name'] = name
        concept_dfs.append(concept_df)
        logging.info(
            f"  Concept {concept_id} '{name}': "
            f"{len(train_normal)} train, {len(test_df)} test "
            f"({n_test_normal} normal / {n_test_anomalous} anomalous)"
        )
        concept_id += 1

    full_df = pd.concat(concept_dfs, ignore_index=True)
    out_file = output_path / 'cicunsw_by_day.csv'
    full_df.to_csv(out_file, index=False)
    logging.info(f"Saved day-split dataset ({concept_id} / {len(per_day)} concepts) to {out_file}")


def preprocess_cicunsw(
    dataset_path: pathlib.Path = CICUNSW_DATASET_PATH,
    output_path: pathlib.Path = OUTPUT_PATH / 'datasets' / 'cicunsw',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Preprocess the CIC-UNSW dataset and write output CSV files.

    Steps applied:
    - Strip whitespace from column names (CICFlowMeter artifact).
    - Drop metadata columns (Flow ID, Src IP, Dst IP, Timestamp).
    - Replace ±Inf with NaN and drop affected rows.
    - Drop exact duplicate rows.
    - Binary label: Benign → 0, any attack → 1.
    - Stratified 80/20 train/test split (normal-only train set).
    - StandardScaler fit on train only, applied to both splits.

    Output files (written to *output_path*):
    - ``cicunsw_processed.csv``                    – cleaned, unscaled.
    - ``cicunsw_single_concept_standardscaler.csv`` – scaled, tagged with concept_id=0.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    raw_df = _load_raw(dataset_path)
    clean_df = _clean(raw_df)
    df = _encode_label(clean_df)
    df.to_csv(output_path / 'cicunsw_processed.csv', index=False)
    logging.info(f"Saved preprocessed dataset to {output_path / 'cicunsw_processed.csv'}")

    normal_df = df.loc[df['label'] == 0]
    anomalous_df = df.loc[df['label'] == 1]
    logging.info(f"Normal samples: {len(normal_df)}, Anomalous samples: {len(anomalous_df)}")

    train_normal_df, test_normal_df = train_test_split(
        normal_df,
        test_size=test_size,
        random_state=random_state,
    )

    test_df = pd.concat([test_normal_df, anomalous_df], ignore_index=True)
    logging.info(f"Train set: {len(train_normal_df)}, Test set: {len(test_df)}")

    numerical_columns = [col for col in df.columns if col != 'label']
    train_scaled, test_scaled = _scale(train_normal_df, test_df, columns=numerical_columns)
    train_scaled['concept_part'] = 'train'
    test_scaled['concept_part'] = 'test'
    full_df = pd.concat([train_scaled, test_scaled], ignore_index=True)
    full_df['concept_id'] = 0
    full_df['concept_name'] = 'cicunsw'
    full_df.to_csv(output_path / 'cicunsw_single_concept_standardscaler.csv', index=False)
    logging.info(f"Saved single-concept scaled dataset to {output_path / 'cicunsw_single_concept_standardscaler.csv'}")
