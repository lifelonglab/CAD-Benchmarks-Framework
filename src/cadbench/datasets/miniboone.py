import pathlib
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cadbench.paths import TABULAR_DATASETS_PATH, OUTPUT_PATH

MINIBOONE_DATASET_PATH = TABULAR_DATASETS_PATH / 'miniboone' / 'MiniBooNE_PID.txt'

_NUM_FEATURES = 50
_FEATURE_COLUMNS = [f'feature_{i:02d}' for i in range(_NUM_FEATURES)]

# Rows where every feature equals this value are a known sentinel in this
# UCI dump, marking an invalid/undetermined particle reconstruction; they
# are not real measurements and are dropped rather than imputed.
_SENTINEL_VALUE = -999.0

# Separately, a handful of rows carry a single catastrophically corrupted
# feature (e.g. one row has feature_19 = 16,000,900 against a column median
# of 0.79) without hitting the exact -999 sentinel on every column. Flagged
# via a robust (median/MAD) per-feature z-score: real values (including
# heavy-tailed physics ones) top out around z~110, while the corrupted rows
# sit at z~32,000-53,000,000 -- a 300x gap, so this threshold cleanly
# separates sensor glitches from genuine (if skewed) measurements.
_ROBUST_Z_OUTLIER_THRESHOLD = 1000.0


def _load_raw(dataset_path: pathlib.Path) -> pd.DataFrame:
    """Load the fixed-width MiniBooNE txt file and assign labels by block.

    The file's first line holds two integers, the number of signal
    (electron neutrino) rows followed by the number of background (muon
    neutrino) rows; the data is laid out as all signal rows first, then
    all background rows, with no header/label column of its own.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"MiniBooNE dataset not found at {dataset_path}")

    logging.info(f"Reading {dataset_path.name}")
    with open(dataset_path) as f:
        n_signal, n_background = (int(x) for x in f.readline().split())

    df = pd.read_csv(dataset_path, sep=r'\s+', skiprows=1, header=None, names=_FEATURE_COLUMNS)
    if len(df) != n_signal + n_background:
        raise ValueError(
            f"Row count {len(df)} does not match header counts {n_signal} + {n_background}"
        )
    logging.info(f"Rows loaded: {len(df)} ({n_signal} signal, {n_background} background)")

    # Signal (electron neutrino, the rare/interesting event) is the anomaly;
    # background (muon neutrino) is normal.
    df['label'] = 0
    df.loc[: n_signal - 1, 'label'] = 1
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)

    sentinel_mask = (df[_FEATURE_COLUMNS] == _SENTINEL_VALUE).all(axis=1)
    logging.info(f"Dropping {sentinel_mask.sum()} rows that are all sentinel value ({_SENTINEL_VALUE:.0f})")
    df = df[~sentinel_mask].reset_index(drop=True)

    median = df[_FEATURE_COLUMNS].median()
    mad = (df[_FEATURE_COLUMNS] - median).abs().median().replace(0, np.nan)
    robust_z = (df[_FEATURE_COLUMNS] - median).div(mad, axis=1).abs()
    outlier_mask = (robust_z > _ROBUST_Z_OUTLIER_THRESHOLD).any(axis=1)
    if outlier_mask.any():
        bad_columns = sorted(robust_z.columns[(robust_z > _ROBUST_Z_OUTLIER_THRESHOLD).any(axis=0)].tolist())
        logging.info(
            f"Dropping {outlier_mask.sum()} rows with a single-feature robust z-score > "
            f"{_ROBUST_Z_OUTLIER_THRESHOLD:.0f} (corrupted columns: {bad_columns})"
        )
    df = df[~outlier_mask].reset_index(drop=True)

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logging.info(f"Dropped {n_before - len(df)} duplicate rows")

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    logging.info(f"Dropped {n_before - len(df)} rows containing NaN/Inf values")
    return df


def preprocess_miniboone(
    dataset_path: pathlib.Path = MINIBOONE_DATASET_PATH,
    output_path: pathlib.Path = OUTPUT_PATH / 'datasets' / 'miniboone',
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Preprocess the MiniBooNE particle-ID dataset as a single concept.

    Steps applied:
    - Load the fixed signal-then-background block layout; label signal
      (electron neutrino) rows as anomalous (1), background (muon
      neutrino) rows as normal (0).
    - Drop rows that are all ``-999`` (a known invalid-reconstruction
      sentinel in this file, ~0.4% of rows) and any resulting duplicates.
    - Normal-only 80/20 train/test split; all anomalous (signal) rows go
      to test, kept at their natural ~28% prevalence (no downsampling).
    - StandardScaler fit on the normal-train partition only.

    Output files (written to *output_path*):
    - ``miniboone_processed.csv``       – cleaned, label-encoded, unscaled.
    - ``miniboone_single_concept.csv``  – scaled, tagged with concept_id=0.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    raw_df = _load_raw(dataset_path)
    df = _clean(raw_df)
    df.to_csv(output_path / 'miniboone_processed.csv', index=False)
    logging.info(f"Saved processed (unscaled) dataset to {output_path / 'miniboone_processed.csv'}")

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
    train_scaled[_FEATURE_COLUMNS] = scaler.fit_transform(train_normal_df[_FEATURE_COLUMNS])
    test_scaled[_FEATURE_COLUMNS] = scaler.transform(test_df[_FEATURE_COLUMNS])

    train_scaled['concept_split'] = 'train'
    test_scaled['concept_split'] = 'test'

    full_df = pd.concat([train_scaled, test_scaled], ignore_index=True)
    full_df['concept_id'] = 0
    full_df['concept_name'] = 'miniboone'

    out_file = output_path / 'miniboone_single_concept.csv'
    full_df.to_csv(out_file, index=False)
    logging.info(f"Saved single-concept dataset to {out_file}")
