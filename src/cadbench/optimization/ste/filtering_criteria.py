import logging
from dataclasses import dataclass, field

import numpy as np

from cadbench.optimization.ste.utils import MetricMatrix

logger = logging.getLogger(__name__)


def filter_self_learnability(matrices: dict[str, MetricMatrix], threshold: float = 0.75) -> list[str]:
    """
    Removes concepts that cannot be handled by its own specialists by any AD model.
    """
    concepts = next(iter(matrices.values())).keys()
    concepts_to_remove = []

    for concept in concepts:
        if max([m[concept][concept] for m in matrices.values()]) < threshold:
            concepts_to_remove.append(concept)
            logger.info(f'Filtering out concept {concept} due to being impossible to learn by any AD model')

    return concepts_to_remove


def filter_coverage_by_other_tasks(matrices: dict[str, MetricMatrix], margin_pct: float = 0.95, max_models_no: int = 2):
    concepts = next(iter(matrices.values())).keys()
    concepts_to_remove = []

    for concept in concepts:
        best_specialist = max(matrix[concept][concept] for matrix in matrices.values())
        solved_by_no_max = 0
        for matrix in matrices.values():
            solved_by = sum(
                matrix[c][concept] / best_specialist >= margin_pct
                for c in concepts if c != concept
            )
            solved_by_no_max = max(solved_by_no_max, solved_by)

        if solved_by_no_max > max_models_no:
            concepts_to_remove.append(concept)
            logger.info(
                f'Filtering out concept {concept} due to being solved by {solved_by_no_max} other specialists '
                f'at >={margin_pct:.0%} of best specialist ({best_specialist:.3f})'
            )

    return concepts_to_remove


def filter_dominance_over_other_tasks(matrices: dict[str, MetricMatrix], margin_pct: float = 0.95, max_tasks_no: int = 2):
    concepts = next(iter(matrices.values())).keys()
    concepts_to_remove = []

    for concept in concepts:
        best_specialist_on_c = {c: max(m[c][c] for m in matrices.values()) for c in concepts if c != concept}
        solving_max = 0
        for matrix in matrices.values():
            solving = sum(
                matrix[concept][c] / best_specialist_on_c[c] >= margin_pct
                for c in concepts if c != concept
            )
            solving_max = max(solving_max, solving)

        if solving_max > max_tasks_no:
            concepts_to_remove.append(concept)
            logger.info(
                f'Filtering out concept {concept} due to solving {solving_max} tasks '
                f'at >={margin_pct:.0%} of their best specialist (more than {max_tasks_no} tasks)'
            )

    return concepts_to_remove


def filter_performance_discriminativeness_across_stes(matrices: dict[str, MetricMatrix], threshold: float = 0.05):
    concepts = next(iter(matrices.values())).keys()
    concepts_to_remove = []

    for concept in concepts:
        for matrix in matrices.values():
            std_val = np.std([matrix[c][concept] for c in concepts])
            if std_val < threshold:
                concepts_to_remove.append(concept)
                logger.info(f'Filtering out concept {concept} due to not enough discriminativeness across STEs ({std_val} < {threshold})')
                break

    return concepts_to_remove


def filter_pairwise_redundancy(matrices: dict[str, MetricMatrix], threshold: float = 0.75):
    concepts = next(iter(matrices.values())).keys()
    concepts_to_merge = []

    for concept1 in concepts:
        for concept2 in concepts:
            if concept1 == concept2:
                continue

            diff_means = []
            for matrix in matrices.values():
                diff_means.append(np.mean(np.abs(
                    np.array(list(matrix[concept1].values())) - np.array(list(matrix[concept2].values()))
                )))
            total_mean = np.mean(diff_means)
            if total_mean < threshold:
                concepts_to_merge.append((concept1, concept2))
                logger.info(f'Suggesting merging concept {concept1} with concept {concept2}')

    return concepts_to_merge

def filter_pairwise_redundancy_remove(matrices: dict[str, MetricMatrix], threshold: float = 0.75):
    concepts = next(iter(matrices.values())).keys()
    concepts_to_remove = []

    for concept1 in concepts:
        for concept2 in concepts:
            if concept1 == concept2:
                continue

            diff_means = []
            for matrix in matrices.values():
                diff_means.append(np.mean(np.abs(
                    np.array(list(matrix[concept1].values())) - np.array(list(matrix[concept2].values()))
                )))
            total_mean = np.mean(diff_means)
            if total_mean < threshold:
                concepts_to_remove.append(concept2)
                logger.info(f'Removing {concept1} due to similarity to {concept2}')

    return concepts_to_remove


def filter_column_profile_redundancy(matrices: dict[str, MetricMatrix], threshold: float = 0.75):
    """
    Removes concepts whose eval-target profile is redundant with another concept's.
    Two concepts are column-redundant if all specialists score similarly on both,
    meaning they're interchangeable as eval targets in the benchmark.
    """
    concepts = list(next(iter(matrices.values())).keys())
    concepts_to_remove = []

    for concept1 in concepts:
        for concept2 in concepts:
            if concept1 == concept2:
                continue

            diff_means = []
            for matrix in matrices.values():
                col1 = np.array([matrix[c][concept1] for c in concepts])
                col2 = np.array([matrix[c][concept2] for c in concepts])
                diff_means.append(np.mean(np.abs(col1 - col2)))
            if np.mean(diff_means) < threshold:
                concepts_to_remove.append(concept2)
                logger.info(f'Removing concept {concept2} due to column-profile similarity to {concept1}')

    return concepts_to_remove


@dataclass
class FilteringResult:
    concepts_to_remove: list[str] = field(default_factory=list)
    concepts_to_merge: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class FilteringStats:
    initial_tasks: int
    final_tasks: int
    filtered_by_self_learnability: list[str]
    filtered_by_coverage_by_other_tasks: list[str]
    filtered_by_dominance_over_other_tasks: list[str]
    filtered_by_performance_discriminativeness: list[str]
    filtered_by_pairwise_redundancy: list[str]
    filtered_by_column_profile_redundancy: list[str]
    params: dict = field(default_factory=dict)

    @property
    def filtered_total(self) -> int:
        return self.initial_tasks - self.final_tasks

    def to_dict(self) -> dict:
        return {
            "initial_tasks": self.initial_tasks,
            "final_tasks": self.final_tasks,
            "filtered_total": self.filtered_total,
            "filtered_by_self_learnability": self.filtered_by_self_learnability,
            "filtered_by_coverage_by_other_tasks": self.filtered_by_coverage_by_other_tasks,
            "filtered_by_dominance_over_other_tasks": self.filtered_by_dominance_over_other_tasks,
            "filtered_by_performance_discriminativeness": self.filtered_by_performance_discriminativeness,
            "filtered_by_pairwise_redundancy": self.filtered_by_pairwise_redundancy,
            "filtered_by_column_profile_redundancy": self.filtered_by_column_profile_redundancy,
            "params": self.params,
        }


def _drop_concepts(matrices: dict[str, MetricMatrix], to_remove: set[str]) -> dict[str, MetricMatrix]:
    return {
        alg: {
            trained: {ec: v for ec, v in evals.items() if ec not in to_remove}
            for trained, evals in matrix.items() if trained not in to_remove
        }
        for alg, matrix in matrices.items()
    }


def apply_all_filtering_criteria(
    matrices: dict[str, MetricMatrix],
    self_learnability_threshold: float = 0.75,
    coverage_margin_pct: float = 0.9,
    coverage_max_models_no: int = 2,
    dominance_margin_pct: float = 0.9,
    dominance_max_tasks_no: int = 2,
    discriminativeness_threshold: float = 0.05,
    redundancy_threshold: float = 0.1,
    column_redundancy_threshold: float = 0.1,
) -> tuple[list[str], FilteringStats]:
    initial_tasks = len(next(iter(matrices.values())))
    current = matrices

    all_by_self_learnability: list[str] = []
    all_by_coverage: list[str] = []
    all_by_dominance: list[str] = []
    all_by_discriminativeness: list[str] = []
    all_by_redundancy: list[str] = []
    all_by_column_redundancy: list[str] = []

    while True:
        by_self_learnability = filter_self_learnability(current, threshold=self_learnability_threshold)
        by_coverage = filter_coverage_by_other_tasks(current, margin_pct=coverage_margin_pct, max_models_no=coverage_max_models_no)
        by_dominance = filter_dominance_over_other_tasks(current, margin_pct=dominance_margin_pct, max_tasks_no=dominance_max_tasks_no)
        by_discriminativeness = [] #filter_performance_discriminativeness_across_stes(current, threshold=discriminativeness_threshold)
        by_redundancy = filter_pairwise_redundancy_remove(current, threshold=redundancy_threshold)
        by_column_redundancy = filter_column_profile_redundancy(current, threshold=column_redundancy_threshold)

        criteria_lists = [by_self_learnability, by_coverage, by_dominance, by_discriminativeness, by_redundancy, by_column_redundancy]
        flagged_counts: dict[str, int] = {}
        for lst in criteria_lists:
            for c in lst:
                flagged_counts[c] = flagged_counts.get(c, 0) + 1

        if not flagged_counts:
            break

        # drop the single most-flagged concept; tie-break by first appearance in criteria priority order
        def _sort_key(concept: str) -> tuple[int, int]:
            first_criterion = next(i for i, lst in enumerate(criteria_lists) if concept in lst)
            return (flagged_counts[concept], -first_criterion)

        worst = max(flagged_counts, key=_sort_key)

        if worst in by_self_learnability:
            all_by_self_learnability.append(worst)
        if worst in by_coverage:
            all_by_coverage.append(worst)
        if worst in by_dominance:
            all_by_dominance.append(worst)
        if worst in by_discriminativeness:
            all_by_discriminativeness.append(worst)
        if worst in by_redundancy:
            all_by_redundancy.append(worst)
        if worst in by_column_redundancy:
            all_by_column_redundancy.append(worst)

        current = _drop_concepts(current, {worst})

    concepts_to_keep = list(next(iter(current.values())).keys())

    stats = FilteringStats(
        initial_tasks=initial_tasks,
        final_tasks=len(concepts_to_keep),
        filtered_by_self_learnability=all_by_self_learnability,
        filtered_by_coverage_by_other_tasks=all_by_coverage,
        filtered_by_dominance_over_other_tasks=all_by_dominance,
        filtered_by_performance_discriminativeness=all_by_discriminativeness,
        filtered_by_pairwise_redundancy=all_by_redundancy,
        filtered_by_column_profile_redundancy=all_by_column_redundancy,
        params={
            "self_learnability_threshold": self_learnability_threshold,
            "coverage_margin_pct": coverage_margin_pct,
            "coverage_max_models_no": coverage_max_models_no,
            "dominance_margin_pct": dominance_margin_pct,
            "dominance_max_tasks_no": dominance_max_tasks_no,
            "discriminativeness_threshold": discriminativeness_threshold,
            "redundancy_threshold": redundancy_threshold,
            "column_redundancy_threshold": column_redundancy_threshold,
        },
    )

    return concepts_to_keep, stats