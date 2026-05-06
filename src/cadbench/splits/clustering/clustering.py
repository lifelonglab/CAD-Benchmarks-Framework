import logging
import pathlib
from typing import Callable

import pandas as pd
from sklearn.base import ClusterMixin
from sklearn.cluster import HDBSCAN, SpectralClustering, KMeans
from sklearn.mixture import GaussianMixture

from cadbench.splits.clustering.methods import cluster_normal_random_anomalies, cluster_both_classes, \
    cluster_separate_assign_nearest
from cadbench.splits.clustering.types import ClusteringConfig, ClusteringMethod
from cadbench.splits.clustering.utils import transform_to_split_concepts

logger = logging.getLogger(__name__)

CLUSTERING_METHODS: dict[str, ClusteringMethod] = {
    'RandomAnomalies': cluster_normal_random_anomalies,
    'Closest': cluster_separate_assign_nearest,
    'BothClasses': cluster_both_classes
}



CLUSTERING_ALGORITHMS: dict[str, tuple[Callable[[ClusteringConfig], ClusterMixin], int | None]] = {
    # algorithm name -> (algorithm constructor, max size of dataset to run without sampling)
    'KMeans': (lambda config: KMeans(n_clusters=config.min_clusters, random_state=42), None),
    'GaussianMixture': (lambda _: GaussianMixture(n_components=10, covariance_type='diag', reg_covar=1e-3), None),
    'SpectralClustering': (lambda config: SpectralClustering(n_clusters=config.min_clusters, random_state=42,
                                                              affinity='nearest_neighbors'), 100_000),
    # 'HDBSCAN': (lambda config: HDBSCAN(min_cluster_size=config.min_samples()), 10_000),
    # HDBSCAN can be slow on large datasets
    # 'DBSCAN': (lambda config: HDBSCAN(min_cluster_size=config.min_samples(), cluster_selection_method='leaf'), 10_000)
    # DBSCAN can be slow on large datasets
}


def cluster_dataset(raw_df: pd.DataFrame,
                    config: ClusteringConfig,
                    output_path: pathlib.Path,
                    dataset_name: str):

    logger.info(f"Starting clustering for dataset {dataset_name}; clustering config: {config}")
    for method_name, clustering_method in CLUSTERING_METHODS.items():
        for alg_name, (algorithm, max_ds_size) in CLUSTERING_ALGORITHMS.items():
            logger.info(f"Clustering using {method_name} and algorithm {alg_name}")
            logger.info(f"Clustering with algorithm: {alg_name}")
            if max_ds_size is not None and len(raw_df) > max_ds_size and (
                    config.sampling_size is None or config.sampling_size > max_ds_size):
                logger.info(f"Dataset size {len(raw_df)} exceeds max size {max_ds_size} for {alg_name}. Skipping execution")
                continue

            try:
                clustered_df = clustering_method(raw_df.copy(), algorithm, config)
                concepts_df = transform_to_split_concepts(clustered_df, concept_name=dataset_name)

                sampled_postfix = f"_sampled_{config.sampling_size}" if config.sampling_size is not None else ""
                out_file = output_path / f"{dataset_name}_clustered_{method_name}_{alg_name}{sampled_postfix}.csv"
                concepts_df.to_csv(out_file, index=False)
                logger.info(f"Saved clustered dataset to {out_file}")
            except Exception as e:
                logger.error(f"Error processing algorithm {alg_name}: {e}", exc_info=e)

