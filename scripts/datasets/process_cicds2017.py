import argparse
import pathlib

import pandas as pd

from cadbench.datasets.cicids2017 import preprocess_cicids2017, split_cicids2017_by_day
from cadbench.logger import setup_logger
from cadbench.paths import OUTPUT_PATH, create_path, RESOURCES_PATH
from cadbench.splits.clustering.clustering import cluster_dataset
from cadbench.splits.clustering.types import ClusteringConfig


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess and/or cluster CICIDS2017 dataset")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Run CICIDS2017 preprocessing (creates cicids2017_single_concept_minmax.csv and related files)",
    )
    parser.add_argument(
        "--split-by-day",
        action="store_true",
        default=False,
        help="Split dataset into concepts by source file (one concept per day/session CSV)",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=False,
        help="Run concept clustering",
    )
    parser.add_argument(
        "--input-file",
        type=pathlib.Path,
        default=None,
        help="Input CSV used for clustering (required with --cluster)",
    )

    parser.add_argument(
        "--sampled-size",
        type=int,
        default=None,
        help="If provided, sample the dataset to this size for cluster fitting (only for --cluster)",
    )

    args = parser.parse_args()

    has_input = args.input_file is not None
    if args.cluster != has_input:
        parser.error("--cluster and --input_file must be provided together")

    return args


def _cluster_cicids2017(data_path: pathlib.Path, dataset_path: pathlib.Path, sampled_size: int | None = None):
    raw_df = pd.read_csv(data_path)
    feature_columns = [
        col
        for col in raw_df.columns
        if col not in ["label", "concept_part", "concept_id", "concept_name"]
    ]
    output_path = create_path(dataset_path / "clustering")
    clustering_config = ClusteringConfig(
        feature_columns=feature_columns,
        min_normal_samples=1500,
        min_anomalous_samples=1000,
        min_clusters=10,
        sampling_size=sampled_size
    )

    cluster_dataset(
        raw_df,
        config=clustering_config,
        output_path=output_path,
        dataset_name="cicids",
    )


if __name__ == '__main__':
    args = parse_arguments()
    setup_logger()

    dataset_path = create_path(OUTPUT_PATH / "datasets" / "cicids2017")

    if args.preprocess:
        preprocess_cicids2017(RESOURCES_PATH / 'datasets' / "tabular" / 'cicids2017', output_path=dataset_path)

    if args.split_by_day:
        split_cicids2017_by_day(
            RESOURCES_PATH / 'datasets' / 'tabular' / 'cicids2017',
            output_path=dataset_path,
        )

    if args.cluster:
        input_file = args.input_file

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        _cluster_cicids2017(input_file, dataset_path, sampled_size=args.sampled_size)
