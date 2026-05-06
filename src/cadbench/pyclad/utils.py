from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from torch import nn

from cadbench.pyclad.autoencoder import Autoencoder


def get_metric(metric: ConceptMetricCallback):
    return metric.info()['concept_metric_callback_ROC-AUC']['metrics']['ContinualAverage']


def get_metric_matrix(metric: ConceptMetricCallback):
    metric_info = metric.info()
    metric_key = [key for key in metric_info.keys() if key.startswith('concept_metric_callback')][0]
    return metric.info()[metric_key]['metric_matrix']

def get_concepts_order(metric: ConceptMetricCallback):
    metric_info = metric.info()
    metric_key = [key for key in metric_info.keys() if key.startswith('concept_metric_callback')][0]
    return metric.info()[metric_key]['concepts_order']


def create_autoencoder(input_features, epochs=20):
    encoder = nn.Sequential(
        nn.Linear(input_features, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
    )

    decoder = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, input_features),
        # nn.Sigmoid(),
    )
    return Autoencoder(encoder, decoder, epochs=epochs)
