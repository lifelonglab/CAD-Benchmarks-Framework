import logging
import pathlib

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table with normal/anomaly counts and ratio per concept split."""
    grouped = (
        df.groupby(["concept_id", "concept_name", "concept_split", "label"])
        .size()
        .reset_index(name="count")
    )

    # Pivot so normal (0) and anomaly (1) become columns
    pivoted = grouped.pivot_table(
        index=["concept_id", "concept_name", "concept_split"],
        columns="label",
        values="count",
        fill_value=0,
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={0: "normal", 1: "anomaly"})

    for col in ("normal", "anomaly"):
        if col not in pivoted.columns:
            pivoted[col] = 0

    pivoted["total"] = pivoted["normal"] + pivoted["anomaly"]
    pivoted["anomaly_ratio"] = pivoted["anomaly"] / pivoted["total"].replace(0, float("nan"))
    pivoted["imbalanced"] = (pivoted["concept_split"] == "test") & (pivoted["anomaly"] > pivoted["normal"])

    return pivoted.sort_values(["concept_id", "concept_split"])


def print_ratio_report(ratios: pd.DataFrame, filepath: pathlib.Path) -> None:
    logger.info("\n%s", "="*70)
    logger.info("File: %s", filepath.name)
    logger.info("%s", "="*70)
    logger.info(
        "\n%s",
        ratios.to_string(
            index=False,
            columns=["concept_id", "concept_name", "concept_split", "normal", "anomaly", "total", "anomaly_ratio", "imbalanced"],
            float_format=lambda x: f"{x:.3f}",
        ),
    )
    imbalanced_test = ratios[(ratios["concept_split"] == "test") & ratios["imbalanced"]]
    if imbalanced_test.empty:
        logger.info("[OK] All test splits have anomaly <= normal.")
    else:
        logger.info("[!] %d test concept(s) have more anomalies than normals.", len(imbalanced_test))


def rebalance(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    For each concept's test split, if anomalies > normals, randomly drop
    anomaly rows until anomaly count == normal count.

    Training rows and non-test rows are left untouched.
    """
    rng = pd.np.random.default_rng(seed) if hasattr(pd, "np") else __import__("numpy").random.default_rng(seed)

    parts = []
    for (concept_id, split), group in df.groupby(["concept_id", "concept_split"]):
        if split != "test":
            parts.append(group)
            continue

        normals = group[group["label"] == 0]
        anomalies = group[group["label"] == 1]
        n_normal = len(normals)
        n_anomaly = len(anomalies)

        if n_anomaly > n_normal:
            keep_idx = rng.choice(anomalies.index, size=n_normal, replace=False)
            anomalies = anomalies.loc[keep_idx]

        parts.append(pd.concat([normals, anomalies]))

    return pd.concat(parts).sort_index()
