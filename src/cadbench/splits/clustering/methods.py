import logging
from typing import Callable

import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import ClusterMixin

from cadbench.splits.clustering.filtering import filter_clusters
from cadbench.splits.clustering.types import ClusteringConfig
from cadbench.splits.clustering.utils import describe_clustered_dataset

logger = logging.getLogger(__name__)


def cluster_both_classes(raw_df: pd.DataFrame,
                         algorithm_fn: Callable[[ClusteringConfig], ClusterMixin],
                         config: ClusteringConfig
                         ) -> pd.DataFrame | None:
    """ Cluster both normal and anomalous data together."""
    logger.info("Clustering both normal and anomalous samples together.")
    algorithm = algorithm_fn(config)
    clustered_df = _cluster_data(raw_df, algorithm, config)
    clustered_df = filter_clusters(clustered_df, config.min_normal_samples, config.min_anomalous_samples)
    return clustered_df


def cluster_normal_random_anomalies(raw_df: pd.DataFrame,
                                    algorithm_fn: Callable[[ClusteringConfig], ClusterMixin],
                                    config: ClusteringConfig
                                    ) -> pd.DataFrame | None:
    """ Cluster normal data, assign anomalies randomly to normal clusters"""
    logger.info("Clustering normal samples and assigning random anomalies to clusters.")
    normal_df = raw_df[raw_df['label'] == 0]
    anomalous_df = raw_df[raw_df['label'] == 1]

    # cluster normal
    normal_df = _cluster_data(normal_df, algorithm_fn(config), config)
    describe_clustered_dataset(normal_df)
    normal_df = filter_clusters(normal_df, config.min_normal_samples, 0)

    valid_clusters = normal_df['cluster'].unique()
    if len(valid_clusters) == 0:
        logger.warning("No valid clusters found with the given minimum normal samples. No anomalies will be assigned.")
        return None

    # assign random anomalies (the right size)
    anomalies_per_cluster = int(len(anomalous_df) // len(valid_clusters))
    if config.min_anomalous_samples > anomalies_per_cluster:
        logger.warning(
            f"Minimum anomalous samples per cluster ({config.min_anomalous_samples}) is greater than the maximum possible ({anomalies_per_cluster}). Ignoring the indicated minimum.")

    result_df = normal_df.copy()

    for cluster_id in valid_clusters:
        sampled_anomalies = anomalous_df.sample(n=anomalies_per_cluster, random_state=42)
        anomalous_df = anomalous_df.drop(sampled_anomalies.index)
        sampled_anomalies['cluster'] = cluster_id
        result_df = pd.concat([result_df, sampled_anomalies], axis=0)

    return result_df


def cluster_separate_assign_nearest(raw_df: pd.DataFrame,
                                    algorithm_fn: Callable[[ClusteringConfig], ClusterMixin],
                                    config: ClusteringConfig
                                    ) -> pd.DataFrame | None:
    logger.info("Clustering normal samples and assigning each anomaly to its nearest normal cluster centroid.")
    normal_df = raw_df[raw_df['label'] == 0].copy()
    anomalous_df = raw_df[raw_df['label'] == 1].copy()

    normal_df = _cluster_data(normal_df, algorithm_fn(config), config)
    normal_df = filter_clusters(normal_df, config.min_normal_samples, 0)

    if normal_df is None or len(normal_df['cluster'].unique()) == 0:
        logger.warning("No valid normal clusters found.")
        return None

    normal_centroids = normal_df.groupby('cluster')[config.feature_columns].mean()
    anomalous_df = anomalous_df.reset_index(drop=True)
    dists = cdist(anomalous_df[config.feature_columns].values, normal_centroids.values)
    anomalous_df['cluster'] = normal_centroids.index.values[dists.argmin(axis=1)]

    # Move anomalies from surplus clusters to deficit clusters (nearest first)
    for i, cluster_id in enumerate(normal_centroids.index):
        deficit = config.min_anomalous_samples - (anomalous_df['cluster'] == cluster_id).sum()
        if deficit <= 0:
            continue
        surplus_clusters = anomalous_df['cluster'].value_counts()
        surplus_clusters = surplus_clusters[surplus_clusters > config.min_anomalous_samples].index
        candidates = anomalous_df.index[anomalous_df['cluster'].isin(surplus_clusters)]
        if len(candidates) == 0:
            logger.warning(f"Cluster {cluster_id}: cannot reach minimum of {config.min_anomalous_samples} anomalies.")
            continue
        to_move = candidates[dists[candidates, i].argsort()[:deficit]]
        anomalous_df.loc[to_move, 'cluster'] = cluster_id
        logger.info(f"Cluster {cluster_id}: moved {len(to_move)} anomalies from surplus clusters.")

    return pd.concat([normal_df, anomalous_df], axis=0)


def _cluster_data(df: pd.DataFrame,
                  algorithm,
                  config: ClusteringConfig) -> pd.DataFrame:
    """
    Supports clustering data either by clustering the entire dataset at once or by clustering a random sample and
    then assigning the remaining data to clusters based on the fitted model.
    The latter can be useful for large datasets where clustering the entire dataset at once may be computationally expensive.
    """
    if config.sampling_size is None or len(df) <= config.sampling_size:
        df['cluster'] = algorithm.fit_predict(df[config.feature_columns])
        return df
    else:
        sample_df = df.sample(n=config.sampling_size, random_state=42)
        remaining_df = df.drop(sample_df.index)

        sample_df['cluster'] = algorithm.fit_predict(sample_df[config.feature_columns])
        assert len(remaining_df) > 0, "Remaining dataframe is empty after sampling, cannot assign clusters to remaining data."

        if hasattr(algorithm, 'predict'):  # not all algorithms support predict on new data
            logger.info(
                f"Algorithm {algorithm.__class__.__name__} supports predict, assigning clusters to remaining data.")
            remaining_df['cluster'] = algorithm.predict(remaining_df[config.feature_columns])
        else:
            logger.info(
                f"Algorithm {algorithm.__class__.__name__} does not support predict, assigning clusters to remaining data based on nearest cluster centroid.")
            centroid_df = sample_df.groupby('cluster')[config.feature_columns].mean()
            dists = cdist(remaining_df[config.feature_columns].values, centroid_df.values)
            remaining_df['cluster'] = centroid_df.index.values[dists.argmin(axis=1)]
        return pd.concat([sample_df, remaining_df])
