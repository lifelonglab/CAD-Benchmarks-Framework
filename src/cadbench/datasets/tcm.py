import pathlib
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cadbench.paths import TABULAR_DATASETS_PATH, OUTPUT_PATH

TCM_DATASET_PATH = TABULAR_DATASETS_PATH / 'tcm'

# The 16 boolean fault-type indicators; combined into a single binary `label`
# and dropped from the feature set (they are the ground truth, not sensor readings).
_ANOMALY_COLUMNS = (
    ['Anomaly_Reduction']
    + [f'Anomaly_Electric_{i}' for i in range(1, 6)]
    + [f'Anomaly_Bearing_{i}' for i in range(1, 6)]
    + [f'Anomaly_WorkRoll_{i}' for i in range(1, 6)]
)


def _discover_files(dataset_path: pathlib.Path) -> list[pathlib.Path]:
    files = sorted(dataset_path.glob('tcm5_dataset_*.csv'))
    if not files:
        raise FileNotFoundError(f"No tcm5_dataset_*.csv files found in {dataset_path}")
    return files


def _load_raw(csv_file: pathlib.Path) -> pd.DataFrame:
    logging.info(f"Reading {csv_file.name}")
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    logging.info(f"Rows loaded: {len(df)}")
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna()
    logging.info(f"Dropped {n_before - len(df)} rows containing NaN/Inf values")

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logging.info(f"Dropped {n_before - len(df)} duplicate rows")
    return df


def _encode_label(df: pd.DataFrame) -> pd.DataFrame:
    """Combine the 16 per-fault-type boolean columns into a single binary `label`."""
    df = df.copy()
    df['label'] = df[_ANOMALY_COLUMNS].any(axis=1).astype(int)
    df = df.drop(columns=_ANOMALY_COLUMNS)
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


def split_tcm_by_file(
    dataset_path: pathlib.Path = TCM_DATASET_PATH,
    output_path: pathlib.Path = OUTPUT_PATH / 'datasets' / 'tcm',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Split the TCM dataset into concepts by source file.

    Each ``tcm5_dataset_N.csv`` is one simulation run and becomes one concept.
    Rows within each file are ordered by simulation time (work-roll mileage
    accumulates monotonically, and datasets 5-6 contain deliberate concept
    drift over the run), so unlike this repo's other tabular datasets the
    train/test split here is **chronological, not random**: the first
    ``1 - test_size`` fraction of each file (filtered to normal-only rows)
    becomes train, and the trailing fraction (normal + anomalous) becomes
    test. This preserves the mileage/drift structure the dataset was
    designed to have and matches the split used in the dataset authors'
    own reference code (pdm-steel-datasets/ML_sample.py).

    A single StandardScaler is fit on the combined normal-train data across
    all concepts and applied to every concept.

    Output files (written to *output_path*):
    - ``tcm_by_file.csv`` – one concept per source file.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    files = _discover_files(dataset_path)
    logging.info(f"Found {len(files)} source files: {[f.name for f in files]}")

    feature_columns: list[str] | None = None

    # --- Pass 1: clean per file, chronological split, collect train partitions for scaler ---
    per_file: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    all_train_normals: list[pd.DataFrame] = []

    for csv_file in files:
        df = _load_raw(csv_file)
        df = _clean(df)
        df = _encode_label(df)

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != 'label']

        name = csv_file.stem
        split_idx = int(len(df) * (1 - test_size))
        train_part = df.iloc[:split_idx]
        test_part = df.iloc[split_idx:]

        train_normal = train_part[train_part['label'] == 0]
        test_df = test_part
        n_dropped_anomalous = (train_part['label'] == 1).sum()
        logging.info(
            f"  {name}: {len(train_normal)} train (normal, chronological head), "
            f"{len(test_df)} test ({(test_df['label'] == 0).sum()} normal / "
            f"{(test_df['label'] == 1).sum()} anomalous), "
            f"{n_dropped_anomalous} anomalous rows dropped from train head"
        )

        per_file.append((train_normal, test_df, name))
        all_train_normals.append(train_normal)

    combined_train = pd.concat(all_train_normals, ignore_index=True)

    scaler = StandardScaler()
    scaler.fit(combined_train[feature_columns])

    # --- Pass 2: assign concept metadata, skip degenerate concepts ---
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

        train_scaled = train_normal.copy()
        test_scaled = test_df.copy()
        train_scaled[feature_columns] = scaler.transform(train_normal[feature_columns])
        test_scaled[feature_columns] = scaler.transform(test_df[feature_columns])

        train_scaled['concept_split'] = 'train'
        test_scaled['concept_split'] = 'test'

        concept_df = pd.concat([train_scaled, test_scaled], ignore_index=True)
        concept_df['concept_id'] = concept_id
        concept_df['concept_name'] = name
        concept_dfs.append(concept_df)
        logging.info(
            f"  Concept {concept_id} '{name}': "
            f"{len(train_scaled)} train, {len(test_scaled)} test "
            f"({n_test_normal} normal / {n_test_anomalous} anomalous)"
        )
        concept_id += 1

    full_df = pd.concat(concept_dfs, ignore_index=True)
    out_file = output_path / 'tcm_by_file.csv'
    full_df.to_csv(out_file, index=False)
    logging.info(f"Saved file-split dataset ({concept_id} / {len(per_file)} concepts) to {out_file}")


def preprocess_tcm(
    dataset_path: pathlib.Path = TCM_DATASET_PATH,
    output_path: pathlib.Path = OUTPUT_PATH / 'datasets' / 'tcm',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Preprocess the TCM dataset as a single concept (baseline sanity check).

    Steps applied:
    - Load and concatenate all ``tcm5_dataset_N.csv`` source files.
    - Replace +/-Inf with NaN and drop affected rows.
    - Drop exact duplicate rows.
    - Binary label: any of the 16 per-fault-type columns True -> 1, else 0.
    - Stratified 80/20 train/test split (normal-only train set).
    - StandardScaler fit on train only, applied to both splits.

    Note: unlike ``split_tcm_by_file``, this collapses all 6 simulation runs
    (including the drifting runs 5-6) into one blob with a random split, so
    it is only meant as a quick sanity check, not the primary benchmark split.

    Output files (written to *output_path*):
    - ``tcm_processed.csv``                    – cleaned, unscaled.
    - ``tcm_single_concept_standardscaler.csv`` – scaled, tagged with concept_id=0.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    files = _discover_files(dataset_path)
    raw_df = pd.concat([_load_raw(f) for f in files], ignore_index=True)
    clean_df = _clean(raw_df)
    df = _encode_label(clean_df)
    df.to_csv(output_path / 'tcm_processed.csv', index=False)
    logging.info(f"Saved preprocessed dataset to {output_path / 'tcm_processed.csv'}")

    normal_df = df[df['label'] == 0]
    anomalous_df = df[df['label'] == 1]
    logging.info(f"Normal samples: {len(normal_df)}, Anomalous samples: {len(anomalous_df)}")

    train_normal_df, test_normal_df = train_test_split(
        normal_df, test_size=test_size, random_state=random_state
    )
    test_df = pd.concat([test_normal_df, anomalous_df], ignore_index=True)
    logging.info(f"Train set: {len(train_normal_df)}, Test set: {len(test_df)}")

    feature_columns = [c for c in df.columns if c != 'label']
    train_scaled, test_scaled = _scale(train_normal_df, test_df, columns=feature_columns)
    train_scaled['concept_split'] = 'train'
    test_scaled['concept_split'] = 'test'

    full_df = pd.concat([train_scaled, test_scaled], ignore_index=True)
    full_df['concept_id'] = 0
    full_df['concept_name'] = 'tcm'
    full_df.to_csv(output_path / 'tcm_single_concept_standardscaler.csv', index=False)
    logging.info(f"Saved single-concept scaled dataset to {output_path / 'tcm_single_concept_standardscaler.csv'}")
