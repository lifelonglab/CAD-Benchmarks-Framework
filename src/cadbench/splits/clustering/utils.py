import logging
import pandas as pd

from cadbench.splits.utils import split_normal_train_test

logger = logging.getLogger(__name__)

def transform_to_split_concepts(df: pd.DataFrame, concept_name: str = 'C') -> pd.DataFrame:
    concepts = []
    for concept_id, cluster_id in enumerate(df['cluster'].unique()):
        concept_df = split_normal_train_test(df[df['cluster'] == cluster_id],
                                             test_size=0.2,
                                             random_state=42)
        concept_df['concept_id'] = concept_id
        concept_df['concept_name'] = f"{concept_name}_{concept_id}"
        concepts.append(concept_df)

        logger.info(
            f"Concept {concept_id} (cluster {cluster_id}): {len(concept_df)} samples, {concept_df['label'].sum()} anomalies")
    concepts_df = pd.concat(concepts, axis=0)
    concepts_df = concepts_df.drop(columns=['cluster'])
    return concepts_df


def describe_clustered_dataset(clustered_df: pd.DataFrame):
    cluster_stats = clustered_df.groupby('cluster')['label'].agg(['count', 'sum'])
    cluster_stats = cluster_stats.rename(columns={'count': 'total_samples', 'sum': 'anomalies'})
    cluster_stats['normal_samples'] = cluster_stats['total_samples'] - cluster_stats['anomalies']
    logger.info(f"Clustered dataset stats:\n{cluster_stats}")
