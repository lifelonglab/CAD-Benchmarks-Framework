"""
Analyze normal/anomaly ratio per concept in clustering CSV files and optionally
rebalance so that the number of anomalies in each concept's test set does not
exceed the number of normal samples in that test set.
"""
import argparse
import logging
import pathlib

import pandas as pd

from cadbench.logger import setup_logger
from cadbench.paths import create_path
from cadbench.splits.rebalancing import compute_ratios, print_ratio_report, rebalance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze normal/anomaly ratio per concept in clustering CSV files.\n"
            "Optionally rebalance test splits so anomaly count <= normal count."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input',
        type=pathlib.Path,
        default=None,
        required=True,
        help=(
            "CSV file(s) or directory to scan for *.csv files. "
        ),
    )
    parser.add_argument(
        "--rebalance",
        action="store_true",
        default=False,
        help=(
            "Drop excess anomaly rows from test splits where anomaly > normal. "
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help=(
            "Directory to write rebalanced files into (preserves original filenames). "
        ),
    )
    return parser.parse_args()


def collect_csv_files(inputs: pathlib.Path) -> list[pathlib.Path]:
    if inputs.is_file() and inputs.suffix == ".csv":
        return [inputs]
    elif inputs.is_dir():
        found = sorted(inputs.glob("*.csv"))
        if not found:
            logger.warning("No CSV files found in %s", inputs)
        return found
    else:
        logger.warning("Path not found or not a CSV: %s", inputs)
    raise ValueError("Path not found or not a CSV")


def main() -> None:
    setup_logger()
    args = parse_args()

    csv_files = collect_csv_files(args.input)

    total_imbalanced = 0

    for filepath in csv_files:
        df = pd.read_csv(filepath)

        required = {"label", "concept_id", "concept_name", "concept_split"}
        missing = required - set(df.columns)
        if missing:
            logger.warning("Skipping %s: missing columns %s", filepath.name, missing)
            continue

        ratios = compute_ratios(df)
        print_ratio_report(ratios, filepath)
        total_imbalanced += ratios["imbalanced"].sum()

        if args.rebalance:
            rebalanced = rebalance(df)

            if args.output is not None:
                args.output.mkdir(parents=True, exist_ok=True)
                out_path = args.output / filepath.name
            else:
                out_path = create_path(filepath.parent.parent / 'tasks_balanced') / filepath.name

            rebalanced.to_csv(out_path, index=False)
            logger.info("Rebalanced file written to: %s", out_path)

            new_ratios = compute_ratios(rebalanced)
            still_bad = new_ratios[(new_ratios["concept_split"] == "test") & new_ratios["imbalanced"]]
            if still_bad.empty:
                logger.info("[OK] All test splits now satisfy anomaly <= normal.")
            else:
                logger.info("[!] %d test concept(s) still imbalanced (this should not happen).", len(still_bad))

    logger.info("Summary: %d file(s) analyzed, %d imbalanced test concept(s) total.", len(csv_files), int(total_imbalanced))


if __name__ == "__main__":
    main()