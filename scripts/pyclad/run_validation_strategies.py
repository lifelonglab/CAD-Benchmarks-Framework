import argparse
import logging
from pathlib import Path
from typing import Any, Generator

import pandas as pd
from matplotlib import pyplot as plt
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter, LocalOutlierFactorAdapter, OneClassSVMAdapter, \
    PyODAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from pyclad.strategies.baselines.mste import MSTE
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection
from pyod.models.anogan import AnoGAN
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.lunar import LUNAR
from pyod.models.vae import VAE
import json
from cadbench.logger import setup_logger
from cadbench.paths import create_path
from cadbench.pyclad.continual_deep_svdd import ContinualDeepSVDD
from cadbench.pyclad.continual_dif import ContinualDIF
from cadbench.pyclad.continual_vae import ContinualVAE
from cadbench.pyclad.cumulative_nm import CumulativeNMStrategy
from cadbench.pyclad.dagmm import DAGMM
from cadbench.pyclad.data_loader import load_dataset_from_df, load_ordered_dataset
from cadbench.pyclad.lwf import LwFStrategy
from cadbench.pyclad.metrics import NormalizedPrAuc, PrAuc, RocAucRobust
from cadbench.pyclad.neutralad import NeuTraLAD
from cadbench.pyclad.plots import plot_metric_heatmap
from cadbench.pyclad.utils import create_autoencoder, get_metric_matrix, get_concepts_order


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run ConceptAwareScenario for multiple validation strategies and models"
    )
    parser.add_argument("--input", type=str, help="Path to a CSV/JSON file or directory to scan", required=True)
    parser.add_argument("--output", type=str, required=False, default=None, help="Output directory for results. If not provided, defaults will be used")
    parser.add_argument("--mode", type=str, choices=["raw", "ordering"], default="raw",
                        help="'raw' scans for CSV files; 'orderings' scans for *_orderings.json files")
    parser.add_argument("--model", type=str, choices=["DeepSVDD", "Autoencoder", "VAE", "DIF", "LUNAR", 'AnoGAN', 'DAGMM', "NeutralAD"], default=None,
                        help="Run only the specified model. If not provided, all models are run.")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        parser.error(f"Input path not found: {input_path}")

    return input_path, args.output, args.mode, args.model


def collect_datasets(input_path: Path, mode: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if mode == "ordering":
        return sorted(input_path.glob("*_orderings.json"))
    return sorted(input_path.glob("*.csv"))


def _load_raw_datasets(dataset_path: Path) -> Generator[tuple[ConceptsDataset, str, int], Any, None]:
    df = pd.read_csv(dataset_path)
    dataset = load_dataset_from_df(df, name=dataset_path.stem)
    input_features = dataset.train_concepts()[0].data.shape[1]
    yield dataset, dataset_path.stem, input_features


def _load_ordered_datasets(dataset_path: Path) -> Generator[tuple[ConceptsDataset, str, int], Any, None]:
    with open(dataset_path) as f:
        spec = json.load(f)
    for ordering_key in spec["orderings"]:
        dataset, name = load_ordered_dataset(dataset_path, ordering_key)
        input_features = dataset.train_concepts()[0].data.shape[1]
        yield dataset, name, input_features


def run_experiment(dataset_path: Path, output_dir: Path, mode: str, model: str | None = None):
    logging.info(f"Running experiment for dataset: {dataset_path}")

    loader = _load_ordered_datasets if mode == "ordering" else _load_raw_datasets
    for dataset, name, input_features in loader(dataset_path):
        experiment_dir = create_path(output_dir / name)
        _run_single_experiment(dataset, name, input_features, experiment_dir, model)


def _run_single_experiment(dataset, name: str, input_features: int, output_dir: Path, model: str | None = None):
    logging.info(f"Running scenario for dataset: {name}")

    models = {
        "DeepSVDD": lambda: PyODAdapter(ContinualDeepSVDD(n_features=input_features, epochs=20, batch_size=128, verbose=0), model_name="DeepSVDD"),
        "Autoencoder": lambda: create_autoencoder(input_features),
        "VAE": lambda: PyODAdapter(ContinualVAE(epoch_num=20, batch_size=128, verbose=0), model_name="VAE"),
        "DIF": lambda: PyODAdapter(ContinualDIF(n_ensemble=20), model_name="DIF"),
        "DAGMM": lambda: DAGMM(batch_size=128, num_epochs=20),
        "NeutralAD": lambda: NeuTraLAD(epochs=20, batch_size=128)
    }
    if model is not None:
        models = {model: models[model]}
    strategies = [
        lambda model_fn: NaiveStrategy(model_fn()),
        lambda model_fn: MSTE(model_fn),
        # lambda model_fn: CumulativeStrategy(model_fn()),
        lambda model_fn: ReplayEnhancedStrategy(model_fn(), buffer=AdaptiveBalancedReplayBuffer(max_size=25_000, selection_method=RandomSelection())),
        # lambda model_fn: LwFStrategy(model_fn(), epochs=20),
        lambda model_fn: CumulativeNMStrategy(model_fn)
    ]

    for model_name, model_fn in models.items():
        for strategy_fn in strategies:
            try:
                strategy = strategy_fn(model_fn)
                out_filepath = output_dir / f"{model_name}__{strategy.name()}_pyclad_validation.json"
                if out_filepath.exists():
                    logging.info(f"Skipping running scenario for model: {model_name}, strategy: {strategy.name()}; results file already exists")
                    continue
                logging.info(f"Running scenario for model: {model_name}, strategy: {strategy.name()}")

                roc_auc_metric = ConceptMetricCallback(
                    base_metric=RocAucRobust(),
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

                logging.info(f"Saving results to {out_filepath}")
                output_writer = JsonOutputWriter(out_filepath)
                output_writer.write([dataset, strategy, *callbacks])

                # plotting the results
                plot_dir = output_dir / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)

                logging.info(f'Plotting concepts order {get_concepts_order(roc_auc_metric)}')


                for metric_name, metric in [('ROC-AUC', roc_auc_metric), ('nPR-AUC', normalized_pr_auc_metric)]:
                    ax = plot_metric_heatmap(get_metric_matrix(metric),
                                                 concepts_order=get_concepts_order(metric),
                                                 title=f"{model_name} - {strategy.name()} {metric_name} Heatmap",
                                                 annotate=True,
                                                 figsize=(12, 12),
                                                 output_path=plot_dir / f"{metric_name}_{model_name}_{strategy.name()}_heatmap.png",
                                                 ignore_upper_diagonal=False)
                    plt.close(ax.get_figure())
            except Exception as e:
                logging.error(f'Error while processing {model_name}', exc_info=e)



if __name__ == "__main__":
    input_path, output_dir, mode, model = parse_arguments()
    if output_dir is None:
        output_dir = input_path.parent if not input_path.is_dir() else input_path
        output_dir = create_path(output_dir / 'validation')
    else:
        output_dir = Path(output_dir)

    setup_logger(logs_path=output_dir / "pyclad_validations.log")
    logging.info(f"Input path: {input_path}; Output directory: {output_dir}; Mode: {mode}")

    dataset_paths = collect_datasets(input_path, mode)
    logging.info(f"Found {len(dataset_paths)} dataset(s)")

    for dataset_path in dataset_paths:
        setup_logger(logs_path=output_dir / f"pyclad_validations_model_{model}.log")
        ds_out_dir = create_path(output_dir / dataset_path.name)
        run_experiment(dataset_path, ds_out_dir, mode, model)
