import logging
import pandas as pd

logger = logging.getLogger(__name__)

def filter_clusters(clustered_df: pd.DataFrame, min_normal_samples: int, min_anomalous_samples: int) -> pd.DataFrame:
    clusters_with_min_samples = filter_clusters_min_size(clustered_df,
                                                         min_normal=min_normal_samples,
                                                         min_anomalies=min_anomalous_samples)
    logger.info(
        f"Filtering concepts: number of concepts with at least {min_normal_samples} normal and {min_anomalous_samples} anomalous samples: {len(clusters_with_min_samples)}")

    filtered_df = clustered_df[clustered_df['cluster'].isin(clusters_with_min_samples)]
    return filtered_df


def filter_clusters_min_size(clustered_df, min_normal, min_anomalies):
    valid_clusters = []
    for cluster_id, cluster_df in clustered_df.groupby('cluster'):
        normal_count = (cluster_df['label'] == 0).sum()
        anomaly_count = (cluster_df['label'] == 1).sum()
        if normal_count >= min_normal and anomaly_count >= min_anomalies:
            valid_clusters.append(cluster_id)
    return valid_clusters
