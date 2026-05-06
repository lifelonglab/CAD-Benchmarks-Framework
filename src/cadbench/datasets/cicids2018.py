import pathlib
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cadbench.paths import RESOURCES_PATH, OUTPUT_PATH

CICIDS2018_DATASET_PATH = RESOURCES_PATH / 'datasets' /  'cicids2018'

# Metadata columns not useful as network-traffic features
_DROP_COLUMNS = ['Timestamp']


def _load_file(f: pathlib.Path) -> pd.DataFrame:
    """Load and clean a single CICIDS2018 CSV file."""
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP'] + _DROP_COLUMNS, errors='ignore')
    df = df[df['Label'].str.strip() != 'Label'].copy()
    feature_cols = [c for c in df.columns if c != 'Label']
    df = df.assign(**{col: pd.to_numeric(df[col], errors='coerce') for col in feature_cols})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _load_raw(dataset_path: pathlib.Path) -> pd.DataFrame:
    """Concatenate all CSV files in *dataset_path* into one DataFrame."""
    csv_files = sorted(dataset_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    daily_dfs = []
    for f in csv_files:
        logging.info(f"Reading {f.name}")
        df = _load_file(f)
        logging.info(f"{f.name}: {len(df)} rows after cleaning")
        daily_dfs.append(df)

    raw_df = pd.concat(daily_dfs, ignore_index=True)
    logging.info(f"Rows after concatenation: {len(raw_df)}")
    return raw_df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate; per-file cleaning already done in _load_raw."""
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


def _concept_name(path: pathlib.Path) -> str:
    # e.g. "02-14-2018.csv" → "02_14_2018", "Thursday-WorkingHours.csv" → "thursday_workinghours"
    stem = path.stem.replace('-', '_').replace(' ', '_').lower()
    skip = {'workinghours', 'working', 'hours'}
    parts = [p for p in stem.split('_') if p not in skip]
    return '_'.join(parts)


def split_cicids2018_by_day(
    dataset_path: pathlib.Path = CICIDS2018_DATASET_PATH,
    output_path: pathlib.Path = OUTPUT_PATH / 'datasets' / 'cicids2018',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Split the CICIDS2018 dataset into concepts by source file (day/session).

    Each CSV file in *dataset_path* becomes one concept. Within each concept
    the normal samples are split into train/test; all anomalous samples go to
    the test partition.

    Output files (written to *output_path*):
    - ``cicids2018_by_day.csv`` – one concept per source file.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(dataset_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    per_file: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []

    for f in csv_files:
        logging.info(f"Reading {f.name}")
        df = _load_file(f)
        df = _encode_label(df)

        normal_df = df.loc[df['label'] == 0]
        anomalous_df = df.loc[df['label'] == 1]
        logging.info(f"  {f.name}: {len(normal_df)} normal, {len(anomalous_df)} anomalous")

        train_normal, test_normal = train_test_split(
            normal_df, test_size=test_size, random_state=random_state
        )
        test_df = pd.concat([test_normal, anomalous_df], ignore_index=True)
        per_file.append((train_normal, test_df, _concept_name(f)))

    concept_dfs: list[pd.DataFrame] = []
    concept_id = 0
    for train_normal, test_df, name in per_file:
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
    out_file = output_path / 'cicids2018_by_day.csv'
    full_df.to_csv(out_file, index=False)
    logging.info(f"Saved day-split dataset ({concept_id} / {len(per_file)} concepts) to {out_file}")


def preprocess_cicids2018(
    dataset_path: pathlib.Path = CICIDS2018_DATASET_PATH,
    output_path: pathlib.Path = OUTPUT_PATH / 'datasets' / 'cicids2018',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Preprocess the CICIDS2018 dataset and write output CSV files.

    Steps applied:
    - Strip whitespace from column names (CICFlowMeter artifact).
    - Drop metadata columns (Timestamp).
    - Remove rows where Label == 'Label' (repeated header artifact in CICIDS2018).
    - Convert feature columns to numeric after header-row removal.
    - Replace ±Inf with NaN and drop affected rows.
    - Drop exact duplicate rows.
    - Binary label: Benign → 0, any attack → 1.
    - Normal-only train set; test = remaining normal + all anomalous.
    - StandardScaler fit on train only, applied to both splits.

    Output files (written to *output_path*):
    - ``cicids2018_processed.csv``                    – cleaned, unscaled.
    - ``cicids2018_single_concept_standardscaler.csv`` – scaled, tagged with concept_id=0.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    raw_df = _load_raw(dataset_path)
    clean_df = _clean(raw_df)
    df = _encode_label(clean_df)
    df.to_csv(output_path / 'cicids2018_processed.csv', index=False)
    logging.info(f"Saved preprocessed dataset to {output_path / 'cicids2018_processed.csv'}")

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
    full_df['concept_name'] = 'cicids2018'
    full_df.to_csv(output_path / 'cicids2018_single_concept_standardscaler.csv', index=False)
    logging.info(
        f"Saved single-concept scaled dataset to "
        f"{output_path / 'cicids2018_single_concept_standardscaler.csv'}"
    )
