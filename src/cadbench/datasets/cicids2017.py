import pathlib
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from cadbench.paths import RESOURCES_PATH, OUTPUT_PATH, create_path

CICIDS2017_DATASET_PATH = RESOURCES_PATH / 'datasets' / 'cicids2017'

# Metadata columns produced by CICFlowMeter that are not network-traffic features
_DROP_COLUMNS = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']


def _load_raw(dataset_path: pathlib.Path) -> pd.DataFrame:
    """Concatenate all CSV files in *dataset_path* into one DataFrame."""
    csv_files = list(dataset_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    daily_dfs = []
    for f in csv_files:
        logging.info(f"Reading {f.name}")
        df = pd.read_csv(f, low_memory=False)
        # CICFlowMeter often writes column names with leading/trailing spaces
        df.columns = df.columns.str.strip()
        daily_dfs.append(df)

    raw_df = pd.concat(daily_dfs, ignore_index=True)
    logging.info(f"Rows after concatenation: {len(raw_df)}")
    return raw_df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Inf/NaN rows, and drop duplicates."""
    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna()
    logging.info(f"Dropped {n_before - len(df)} rows containing NaN/Inf values")

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logging.info(f"Dropped {n_before - len(df)} duplicate rows")

    return df


def _encode_label(df: pd.DataFrame) -> pd.DataFrame:
    df['Label'] = df['Label'].str.strip().str.upper().apply(lambda v: 0 if v == 'BENIGN' else 1)
    df.rename(columns={'Label': 'label'}, inplace=True)
    return df


def _scale(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on *train*, apply to both splits. Returns (train_scaled, test_scaled)."""
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[columns] = scaler.fit_transform(train[columns])
    test_scaled[columns] = scaler.transform(test[columns])
    return train_scaled, test_scaled


def _add_split_marker(features: pd.DataFrame, label: pd.Series, part: str) -> pd.DataFrame:
    return pd.concat([features, label], axis=1).assign(concept_part=part)


def split_cicids2017_by_day(
    dataset_path: pathlib.Path = CICIDS2017_DATASET_PATH,
    output_path: pathlib.Path = OUTPUT_PATH / 'datasets' / 'cicids2017',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Split the CICIDS2017 dataset into concepts by source file (day/session).

    Each CSV file in *dataset_path* becomes one concept. Within each concept
    the normal samples are split into train/test; all anomalous samples go to
    the test partition. A StandardScaler is fit on the combined training
    (normal) data from all concepts and applied to every concept.

    Output files (written to *output_path*):
    - ``cicids2017_by_day_standardscaler.csv`` – scaled, one concept per source file.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(dataset_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    def _concept_name(path: pathlib.Path) -> str:
        # e.g. "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" → "friday_afternoon_ddos"
        stem = path.name.replace('.pcap_ISCX.csv', '').replace('.csv', '')
        parts = stem.replace('-', '_').lower().split('_')
        # Remove generic tokens that add no information
        skip = {'workinghours', 'workinghours'.lower(), 'working', 'hours'}
        parts = [p for p in parts if p not in skip]
        return '_'.join(parts)

    # --- Pass 1: clean every file and collect train partitions for scaler fitting ---
    per_file: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []  # (train_normal, test_df, name)
    all_train_normals: list[pd.DataFrame] = []

    for f in csv_files:
        logging.info(f"Reading {f.name}")
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        df = _clean(df)
        df = _encode_label(df)

        normal_df = df.loc[df['label'] == 0]
        anomalous_df = df.loc[df['label'] == 1]
        logging.info(
            f"  {f.name}: {len(normal_df)} normal, {len(anomalous_df)} anomalous"
        )

        train_normal, test_normal = train_test_split(
            normal_df, test_size=test_size, random_state=random_state
        )
        test_df = pd.concat([test_normal, anomalous_df], ignore_index=True)

        per_file.append((train_normal, test_df, _concept_name(f)))
        all_train_normals.append(train_normal)

    combined_train = pd.concat(all_train_normals, ignore_index=True)
    numerical_columns = [col for col in combined_train.columns if col != 'label']


    # --- Pass 2: assign concept metadata ---
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
    out_file = output_path / 'cicids2017_by_day.csv'
    full_df.to_csv(out_file, index=False)
    logging.info(f"Saved day-split dataset ({concept_id} / {len(per_file)} concepts) to {out_file}")


def preprocess_cicids2017(
    dataset_path: pathlib.Path = CICIDS2017_DATASET_PATH,
    output_path: pathlib.Path = OUTPUT_PATH / 'datasets',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Preprocess the CICIDS2017 dataset and write three output CSV files.

    Best-practice steps applied:
    - Strip whitespace from column names (CICFlowMeter artifact).
    - Drop metadata columns (Flow ID, Source IP, Destination IP, Timestamp).
    - Replace ±Inf with NaN and drop affected rows.
    - Drop exact duplicate rows (well-documented issue in CICIDS2017).
    - Binary label: BENIGN → 0, any attack → 1.
    - Stratified 80/20 train/test split (CICIDS2017 has no predefined split).
    - StandardScaler fit on train only, applied to both splits.

    Output files (written to *output_path*):
    - ``cicids2017_processed_raw.csv``         – unscaled; all features are numeric so no encoding needed.
    - ``cicids2017_processed_raw.csv``         – StandardScaler applied to all feature columns.
    - ``cicids2017_single_concept.csv`` – scaled version tagged with concept_id=0.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    raw_df = _load_raw(dataset_path)
    clean_df = _clean(raw_df)
    df = _encode_label(clean_df)
    df.to_csv(output_path / 'cicids2017_processed.csv', index=False)
    logging.info(f'Saved dataset after basic preprocessing to {output_path}/cicids2017_processed.csv')

    # Scale and save as one concept
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
    numerical_columns = [col for col in df.columns if col not in ['Label']]
    train_scaled, test_scaled = _scale(train_normal_df, test_df, columns=numerical_columns)
    train_scaled['concept_part'] = 'train'
    test_scaled['concept_part'] = 'test'
    full_df = pd.concat([train_scaled, test_scaled], ignore_index=True)
    full_df['concept_id'] = 0
    full_df['concept_name'] = 'cicids2017'
    full_df.to_csv(output_path / 'cicids2017_single_concept_standardscaler.csv', index=False)


