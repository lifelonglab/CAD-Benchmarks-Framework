import argparse
import json
import logging
import pathlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pyclad.analysis.scenario_heatmap import plot_metric_heatmap
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyod.models.ae1svm import AE1SVM
from pyod.models.alad import ALAD
from pyod.models.anogan import AnoGAN
from pyod.models.deep_svdd import DeepSVDD
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter, LocalOutlierFactorAdapter, COPODAdapter, ECODAdapter, PyODAdapter
from pyod.models.hbos import HBOS
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lunar import LUNAR
from pyod.models.pca import PCA
from pyod.models.so_gaal import SO_GAAL

from cadbench.heterogeneity.ste_strategy import SingleTaskExpertStrategy
from cadbench.logger import setup_logger
from cadbench.paths import OUTPUT_PATH, create_path, OUTPUT_DATASETS_PATH
from cadbench.pyclad.continual_ae1svm import ContinualAE1SVM
from cadbench.pyclad.data_loader import load_dataset_from_df
from cadbench.pyclad.metrics import PrAuc, NormalizedPrAuc
from cadbench.pyclad.random_model import RandomModel
from cadbench.pyclad.utils import get_metric_matrix, create_autoencoder

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Single Task Expert for multiple validation strategies and models"
    )
    parser.add_argument("--input", type=str, help="Path to a CSV file or directory to scan recursively for CSV files", required=True)
    parser.add_argument("--output", type=str, help="Output directory for results (if not given, "
                                                   "it will be created automatically as subdir in the parent of csv file",
                        required=False, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the torch/numpy global RNG state and for Isolation Forest.")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = args.output

    if not input_path.exists():
        parser.error(f"Input path not found: {input_path}")

    return input_path, output_dir, args.seed


def collect_datasets(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.csv"))


def _run_ste_for_dataset(dataset_path: Path, output_dir: Path, seed: int = 42):
    logging.info(f"Processing dataset path: {dataset_path}")
    logging.info(f"Output directory: {output_dir}")

    df = pd.read_csv(dataset_path)
    dataset = load_dataset_from_df(df, name=dataset_path.stem)
    input_features = dataset.train_concepts()[0].data.shape[1]

    models = {
        "IsolationForest": lambda: IsolationForestAdapter(random_state=seed),
        "Autoencoder": lambda: create_autoencoder(input_features),
        'AE1_SVM': lambda: PyODAdapter(ContinualAE1SVM(epochs=20, batch_size=128, kernel_approx_features=512), model_name='AE1_SVM')



    }

    results = defaultdict(dict)
    for model_name, model_fn in models.items():
        try:
            strategy = SingleTaskExpertStrategy(model_fn)
            logger.info(f"Running scenario for model: {model_name}, strategy: {strategy.name()}")

            roc_auc_metric = ConceptMetricCallback(
                base_metric=RocAuc(),
                metrics=[ContinualAverage()],
            )
            pr_auc_metric = ConceptMetricCallback(
                base_metric=PrAuc(),
                metrics=[ContinualAverage()],
            )

            normalized_pr_auc_metric = ConceptMetricCallback(
                base_metric=NormalizedPrAuc(),
                metrics=[ContinualAverage()]
            )

            callbacks = [roc_auc_metric, pr_auc_metric, normalized_pr_auc_metric, TimeEvaluationCallback()]
            scenario = ConceptAwareScenario(dataset, strategy, callbacks)
            scenario.run()

            results['ROC-AUC'][model_name] = get_metric_matrix(roc_auc_metric)
            results['PR-AUC'][model_name] = get_metric_matrix(pr_auc_metric)
            results['nPR-AUC'][model_name] = get_metric_matrix(normalized_pr_auc_metric)

            out_filepath = output_dir / f"{model_name}__{strategy.name()}.json"
            logger.info(f"Saving results to {out_filepath}")
            output_writer = JsonOutputWriter(out_filepath)
            output_writer.write([dataset, strategy, *callbacks])
        except Exception as e:
            logger.error(f'Error while processing {model_name}', exc_info=e)

    logger.info(f"Saving summary results to {output_dir / 'results.json'}")
    json.dump(results, open(output_dir / "results.json", mode='w'), indent=4)

    # plotting the results
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, metric_results in results.items():
        for model_name, metric_matrix in metric_results.items():
            ax = plot_metric_heatmap(metric_matrix,
                                      concepts_order=list(metric_matrix.keys()),
                                      title=f"{model_name} {metric_name} Heatmap",
                                      annotate=True,
                                      figsize=(12, 12),
                                      output_path=plot_dir / f"{model_name}_{metric_name}_heatmap.png",
                                      ignore_upper_diagonal=False)
            plt.close(ax.get_figure())

if __name__ == "__main__":
    dataset_path, output_dir, seed = parse_arguments()
    if output_dir is None:
        output_dir = dataset_path if dataset_path.is_dir() else dataset_path.parent
    else:
        output_dir = pathlib.Path(output_dir)

    torch.manual_seed(seed)
    np.random.seed(seed)

    files = collect_datasets(dataset_path)
    for file in files:
        file_output_path = create_path(output_dir / file.stem / 'ste')
        setup_logger(logs_path=file_output_path / "logs.log")
        _run_ste_for_dataset(file, file_output_path, seed=seed)
