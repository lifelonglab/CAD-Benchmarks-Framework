import argparse
import json
import logging
import pathlib
from pathlib import Path

import pandas as pd

from cadbench.logger import setup_logger
from cadbench.optimization.ste.filtering_criteria import apply_all_filtering_criteria, FilteringStats
from cadbench.optimization.ste.split_selection import SplitResult, select_best_split, borda_aggregate
from cadbench.optimization.ste.ste_filtering import filter_matrices, filter_dataset
from cadbench.optimization.ste.ste_optimization import (
    order_by_curriculum, order_by_generalization,
    order_by_smooth_drift, order_by_abrupt_drift,
)
from cadbench.optimization.ste.utils import load_ste_results, average_results
from cadbench.paths import create_path

logger = logging.getLogger(__name__)

ALGORITHMS: list[tuple[str, callable]] = [
    ("curriculum_asc",      lambda m: order_by_curriculum(m, descending=False)),
    ("curriculum_desc",     lambda m: order_by_curriculum(m, descending=True)),
    ("gen_paper_desc",      lambda m: order_by_generalization(m, descending=True)),
    ("gen_paper_asc",       lambda m: order_by_generalization(m, descending=False)),
    ("smooth_drift",        order_by_smooth_drift),
    ("abrupt_drift",        order_by_abrupt_drift),
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Reorder concepts in a dataset using all optimization algorithms"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a CSV file or directory to scan recursively for CSV files")
    parser.add_argument("--output", type=str, required=False, default=None,
                        help="Output directory for the reordered dataset CSV files. If not provided, then the defaults are used")
    parser.add_argument("--selected-model", type=str, required=False, default=None,
                        help="Model name to use for optimization. If not provided, all models are used (averaged).")
    parser.add_argument("--selected-metric", type=str, required=False, default='ROC-AUC',
                        help="Metric name to use for optimization.")
    parser.add_argument("--filtering", type=bool, required=False, default=False,
                        help="Should filter concepts that seem to be mostly confusing")
    parser.add_argument("--min-tasks", type=int, required=False, default=None,
                        help="Minimum number of tasks that must remain after filtering to proceed with the scenario")
    parser.add_argument("--select-best", action="store_true", default=False,
                        help="Select the single best split across all ordering types and models")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        parser.error(f"Input path not found: {input_path}")

    return input_path, args.output, args.selected_model, args.selected_metric, args.filtering, args.min_tasks, args.select_best


def collect_datasets(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.csv"))


def _find_ste_results(dataset_path: Path) -> Path:
    return dataset_path.parent / dataset_path.stem / "ste" / "results.json"


def fix_concept_ids(df: pd.DataFrame) -> pd.DataFrame:
    sorted_ids = sorted(df["concept_id"].unique())
    id_mapping = {old: new for new, old in enumerate(sorted_ids)}
    df["concept_id"] = df["concept_id"].map(id_mapping)
    return df


def _load_dataset(dataset_path: Path, selected_metric: str) -> tuple[pd.DataFrame, dict] | None:
    ste_results_path = _find_ste_results(dataset_path)
    if not ste_results_path.exists():
        logger.warning(f"STE results not found for {dataset_path} (expected at {ste_results_path}), skipping.")
        return None

    logger.info(f"Dataset: {dataset_path}; STE results: {ste_results_path}; Output: {output_dir}; "
                f"Selected model: {selected_model}; Filtering: {should_filter}; Min tasks: {min_tasks}")

    # df = pd.read_csv(dataset_path)
    df = pd.DataFrame()
    ste_results = load_ste_results(ste_results_path)[selected_metric]

    return df, ste_results


def _filter(ste_results, df):
    n_initial_tasks = len(next(iter(ste_results.values())))
    if min_tasks is not None and n_initial_tasks < min_tasks:
        logger.warning(f"Only {n_initial_tasks} tasks in {dataset_path} (min required: {min_tasks}), skipping.")
        return None

    if should_filter:
        logger.info("Filtering uninformative concepts")
        # concepts_to_keep = filter_concepts(ste_results)
        concepts_to_keep, filtering_stats = apply_all_filtering_criteria(ste_results)
        ste_results = filter_matrices(ste_results, concepts_to_keep)
        df = filter_dataset(df, concepts_to_keep)
        stats_path = output_dir / f"{dataset_path.stem}_filtering_stats.json"
        with open(stats_path, "w") as f:
            json.dump(filtering_stats.to_dict(), f, indent=2)
        logger.info(f"Saved filtering stats to {stats_path}")

    n_tasks = len(df['concept_id'].unique())
    if min_tasks is not None and n_tasks < min_tasks:
        logger.warning(f"Only {n_tasks} tasks remaining (min required: {min_tasks}), skipping {dataset_path}.")
        return None
    return df, ste_results

def _build_results_json(ste_dir: Path) -> None:
    results: dict[str, dict[str, dict]] = {}
    for json_path in sorted(ste_dir.glob("*__SingleTaskExpert.json")):
        model_name = json_path.stem.split("__")[0]
        data = json.loads(json_path.read_text())
        for key, val in data.items():
            if not key.startswith("concept_metric_callback_"):
                continue
            metric_name = val["base_metric_name"]
            results.setdefault(metric_name, {})[model_name] = val["metric_matrix"]
    if results:
        out = ste_dir / "results.json"
        out.write_text(json.dumps(results, indent=2))
        logger.info(f"Built {out}")


def _run_for_dataset(dataset_path: Path, output_dir: Path, selected_model: str | None, should_filter: bool, min_tasks: int | None, metric: str = 'ROC-AUC'):
    ste_results_path = _find_ste_results(dataset_path)
    _build_results_json(ste_results_path.parent)
    if not ste_results_path.exists():
        logger.warning(f"STE results not found for {dataset_path} (expected at {ste_results_path}), skipping.")
        return None

    logger.info(f"Dataset: {dataset_path}; STE results: {ste_results_path}; Output: {output_dir}; "
                f"Selected model: {selected_model}; Filtering: {should_filter}; Min tasks: {min_tasks}")

    # df = pd.read_csv(dataset_path)
    df = pd.DataFrame()
    ste_results = load_ste_results(ste_results_path)[metric]

    n_initial_tasks = len(next(iter(ste_results.values())))
    if min_tasks is not None and n_initial_tasks < min_tasks:
        logger.warning(f"Only {n_initial_tasks} tasks in {dataset_path} (min required: {min_tasks}), skipping.")
        return None

    if should_filter:
        logger.info("Filtering uninformative concepts")
        # concepts_to_keep = filter_concepts(ste_results)
        concepts_to_keep, filtering_stats = apply_all_filtering_criteria(ste_results)
        ste_results = filter_matrices(ste_results, concepts_to_keep)
        # df = filter_dataset(df, concepts_to_keep)
        stats_path = output_dir / f"{dataset_path.stem}_filtering_stats.json"
        with open(stats_path, "w") as f:
            json.dump(filtering_stats.to_dict(), f, indent=2)
        logger.info(f"Saved filtering stats to {stats_path}")

    n_tasks = len(concepts_to_keep)
    # n_tasks = len(df['concept_id'].unique())
    if min_tasks is not None and n_tasks < min_tasks:
        logger.warning(f"Only {n_tasks} tasks remaining (min required: {min_tasks}), skipping {dataset_path}.")
        return None

    metric_matrix = ste_results[selected_model] if selected_model is not None else average_results(ste_results)

    orderings = {}
    for key, alg_fn in ALGORITHMS:
        ordered = alg_fn(metric_matrix)
        concepts_order = [concept for concept, _ in ordered]
        orderings[key] = concepts_order
        logger.info(f"[{key}] Optimized concept order: {concepts_order}")

    output = {
        "source": str(dataset_path.resolve()),
        "orderings": orderings,
    }
    output_path = output_dir / f"{dataset_path.stem}_orderings.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved orderings to {output_path}")

    return ste_results, dataset_path, orderings


def _compute_per_model_orderings(
    results: list[tuple[dict, Path, dict] | None],
) -> dict[Path, dict[str, dict[str, list[str]]]]:
    """
    For each split and each anomaly detection model, run every ordering algorithm
    on that model's metric matrix and record the resulting concept order.

    Returns: dataset_path -> model_name -> algorithm_key -> [concept, ...]
    """
    per_model_orderings = {}
    for entry in results:
        if entry is None:
            continue
        ste_results, dataset_path, _ = entry
        per_model_orderings[dataset_path] = {}
        for model, matrix in ste_results.items():
            per_model_orderings[dataset_path][model] = {
                key: [concept for concept, _ in alg_fn(matrix)]
                for key, alg_fn in ALGORITHMS
            }
    return per_model_orderings


if __name__ == "__main__":
    input_path, output_dir, selected_model, selected_metric, should_filter, min_tasks, select_best = parse_arguments()
    if output_dir is None:
        output_dir = create_path(input_path.parent / 'scenarios_optimized')
    else:
        output_dir = create_path(pathlib.Path(output_dir))

    setup_logger(logs_path=output_dir / "optimize_by_ste.log")

    dataset_paths = collect_datasets(input_path)
    logger.info(f"Found {len(dataset_paths)} dataset(s)")

    results = []
    for dataset_path in dataset_paths:
        try:
            ds_results = _run_for_dataset(dataset_path, output_dir, selected_model, should_filter, min_tasks, metric=selected_metric)
            if ds_results is not None:
                results.append(ds_results)
        except Exception as e:
            logger.error(f'Error while processing dataset {dataset_path}', exc_info=e)
            pass


    ordering_ranks = _compute_per_model_orderings(results)

    valid_results = [r for r in results if r is not None]
    split_results = [
        SplitResult(
            path=dataset_path,
            alg_metrics=ste_results,
            filtering_stats=None,
            n_tasks=len(next(iter(ste_results.values()))),
            orderings=ordering_ranks[dataset_path],
        )
        for ste_results, dataset_path, orderings in valid_results
        if dataset_path in ordering_ranks
    ]
    if len(split_results) > 1:
        best, selection_report = select_best_split(split_results)
        report_path = output_dir / "split_selection_report.json"
        with open(report_path, "w") as f:
            json.dump(selection_report, f, indent=2)
        logger.info(f"Best split: {best.path}. Report saved to {report_path}")

        consensus_orderings = borda_aggregate(best.orderings)
        consensus_path = output_dir / "best_split_orderings.json"
        with open(consensus_path, "w") as f:
            json.dump({"source": str(best.path), "orderings": consensus_orderings}, f, indent=2)
        logger.info(f"Consensus orderings saved to {consensus_path}")
    if len(split_results) == 0:
        logger.info("Did not find any dataset with min tasks")
