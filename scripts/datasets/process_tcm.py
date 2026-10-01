import argparse
import pathlib

import pandas as pd

from cadbench.datasets.tcm import preprocess_tcm, split_tcm_by_file
from cadbench.logger import setup_logger
from cadbench.paths import OUTPUT_PATH, create_path, RESOURCES_PATH
from cadbench.splits.clustering.clustering import CLUSTERING_ALGORITHMS, cluster_dataset
from cadbench.splits.clustering.types import ClusteringConfig


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess and/or cluster TCM dataset")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Run TCM preprocessing (single-concept baseline)",
    )
    parser.add_argument(
        "--split-by-file",
        action="store_true",
        default=False,
        help="Split TCM into one concept per source file (tcm5_dataset_N.csv)",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for clustering/anomaly-assignment randomness (--cluster). "
             "Default 42 matches prior (unparameterized) behavior.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        choices=list(CLUSTERING_ALGORITHMS),
        help=f"Restrict --cluster to these clustering algorithms (default: all of {list(CLUSTERING_ALGORITHMS)})",
    )
    parser.add_argument(
        "--cluster-output",
        type=pathlib.Path,
        default=None,
        help="Override the output directory for clustered CSVs (default: "
             "OUTPUT_PATH/datasets/tcm/clustering/). Use a unique directory per "
             "concurrent invocation (e.g. per seed) -- the default location is shared "
             "and NOT safe to write to from parallel runs.",
    )

    args = parser.parse_args()

    has_input = args.input_file is not None
    if args.cluster != has_input:
        parser.error("--cluster and --input_file must be provided together")

    return args


def _cluster_tcm(data_path: pathlib.Path, dataset_path: pathlib.Path, sampled_size: int | None = None,
                 seed: int = 42, algorithms: list[str] | None = None,
                 output_override: pathlib.Path | None = None):
    raw_df = pd.read_csv(data_path)
    feature_columns = [
        col
        for col in raw_df.columns
        if col not in ["label", "concept_part", "concept_split", "concept_id", "concept_name"]
    ]
    output_path = create_path(output_override) if output_override is not None else create_path(dataset_path / "clustering")
    clustering_config = ClusteringConfig(
        feature_columns=feature_columns,
        min_normal_samples=1500,
        min_anomalous_samples=1000,
        min_clusters=10,
        sampling_size=sampled_size,
        random_state=seed,
    )
    restricted_algorithms = (
        {name: CLUSTERING_ALGORITHMS[name] for name in algorithms} if algorithms is not None else None
    )

    cluster_dataset(
        raw_df,
        config=clustering_config,
        output_path=output_path,
        dataset_name="tcm",
        algorithms=restricted_algorithms,
    )


if __name__ == '__main__':
    args = parse_arguments()
    setup_logger()

    dataset_path = create_path(OUTPUT_PATH / "datasets" / "tcm")

    if args.preprocess:
        preprocess_tcm(
            RESOURCES_PATH / 'datasets' / 'tabular' / 'tcm',
            output_path=dataset_path,
        )

    if args.split_by_file:
        split_tcm_by_file(
            RESOURCES_PATH / 'datasets' / 'tabular' / 'tcm',
            output_path=dataset_path,
        )

    if args.cluster:
        input_file = args.input_file

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        _cluster_tcm(input_file, dataset_path, sampled_size=args.sampled_size,
                    seed=args.seed, algorithms=args.algorithms, output_override=args.cluster_output)
