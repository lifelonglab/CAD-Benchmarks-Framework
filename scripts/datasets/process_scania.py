import argparse
import pathlib

import pandas as pd

from cadbench.datasets.scania import preprocess_scania
from cadbench.logger import setup_logger
from cadbench.paths import OUTPUT_PATH, create_path
from cadbench.splits.clustering.clustering import CLUSTERING_ALGORITHMS, cluster_dataset
from cadbench.splits.clustering.types import ClusteringConfig

_METADATA_COLUMNS = {"label", "concept_split", "concept_id", "concept_name"}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess and/or cluster the Scania APS Failure dataset")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Run preprocessing (creates scania_single_concept.csv)",
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
        help="Random seed for the normal/test split (--preprocess) and clustering/anomaly-assignment "
             "randomness (--cluster). Default 42 matches prior (unparameterized) behavior.",
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
             "OUTPUT_PATH/datasets/scania/clustering/). Use a unique directory per "
             "concurrent invocation (e.g. per seed) -- the default location is shared "
             "and NOT safe to write to from parallel runs.",
    )

    args = parser.parse_args()

    has_input = args.input_file is not None
    if args.cluster != has_input:
        parser.error("--cluster and --input-file must be provided together")

    return args


def _cluster_scania(data_path: pathlib.Path, dataset_path: pathlib.Path, sampled_size: int | None = None,
                    seed: int = 42, algorithms: list[str] | None = None,
                    output_override: pathlib.Path | None = None):
    raw_df = pd.read_csv(data_path)
    feature_columns = [col for col in raw_df.columns if col not in _METADATA_COLUMNS]
    output_path = create_path(output_override) if output_override is not None else create_path(dataset_path / "clustering")
    clustering_config = ClusteringConfig(
        feature_columns=feature_columns,
        min_normal_samples=1500,
        min_anomalous_samples=100,
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
        dataset_name="scania",
        algorithms=restricted_algorithms,
    )


if __name__ == '__main__':
    args = parse_arguments()
    setup_logger()

    dataset_path = create_path(OUTPUT_PATH / "datasets" / "scania")

    if args.preprocess:
        preprocess_scania(output_path=dataset_path, random_state=args.seed)

    if args.cluster:
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        _cluster_scania(args.input_file, dataset_path, sampled_size=args.sampled_size,
                        seed=args.seed, algorithms=args.algorithms, output_override=args.cluster_output)
