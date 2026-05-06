import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.concepts_readers import read_concepts_from_df
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.baselines.mste import MSTE

from cadbench.logger import setup_logger
from cadbench.pyclad.metrics import NormalizedPrAuc
from cadbench.pyclad.utils import create_autoencoder, get_metric


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process a dataset and save results to an output directory"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to the input dataset file",
        required=True
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=False,
        help="Output directory for results"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        parser.error(f"Dataset file not found: {dataset_path}")

    if args.output is None:
        output_dir = dataset_path.parent / 'pyclad' / 'single_concept'
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    return dataset_path, output_dir


if __name__ == '__main__':
    dataset_path, output_dir = parse_arguments()
    setup_logger(logs_path=output_dir / "logs.log")

    logging.info(f"Dataset path: {dataset_path}")
    logging.info(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    df = df.rename(columns={'Label': 'label'})
    train_df = df[df['concept_split'] == 'train']
    test_df = df[df['concept_split'] == 'test']
    train_df.drop(columns=['concept_split'], inplace=True)
    test_df.drop(columns=['concept_split'], inplace=True)

    train_concepts = read_concepts_from_df(train_df)
    test_concepts = read_concepts_from_df(test_df)
    logging.info(f"Read {len(train_concepts)} train concepts and {len(test_concepts)} test concepts")

    dataset = ConceptsDataset(name='single_concept_dataset', train_concepts=train_concepts, test_concepts=test_concepts)
    input_features = train_concepts[0].data.shape[1]

    models = [
        IsolationForestAdapter,
        lambda: create_autoencoder(input_features)
    ]

    results = {}
    results2 = {}

    for model_fn in models:
        model_name = model_fn().name()
        logging.info(f"Running scenario with model: {model_name}")
        strategy = MSTE(model_fn)
        metric = ConceptMetricCallback(
            base_metric=RocAuc(),
            metrics=[ContinualAverage()],
        )
        metric2 = ConceptMetricCallback(
            base_metric=NormalizedPrAuc(),
            metrics=[ContinualAverage()],
        )
        callbacks = [metric, metric2, TimeEvaluationCallback()]
        scenario = ConceptAwareScenario(dataset, strategy, callbacks)
        scenario.run()

        results[model_name] = get_metric(metric)
        results2[model_name] = metric2.info()['concept_metric_callback_Normalized-PR-AUC']['metrics']['ContinualAverage']

        model_out_dir = output_dir / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        output_writer = JsonOutputWriter(model_out_dir / "output.json")
        output_writer.write([dataset, strategy, *callbacks])

    logging.info(f"Final results: {results}")
    logging.info(f"Final results: {results2}")
    json.dump(results, open(output_dir / "results.json", "w"))
    json.dump(results2, open(output_dir / "results2.json", "w"))
