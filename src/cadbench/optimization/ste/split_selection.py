import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cadbench.optimization.ste.filtering_criteria import FilteringStats
from cadbench.optimization.ste.utils import MetricMatrix

logger = logging.getLogger(__name__)

# One representative key per ordering family. asc/desc variants produce identical
# Kendall's W because reversing all rankings uniformly preserves concordance.
ALGORITHM_FAMILIES = {
    "curriculum": "curriculum_asc",
    "generalization": "gen_paper_asc",
    "smooth_drift": "smooth_drift",
    "abrupt_drift": "abrupt_drift",
}


@dataclass
class SplitResult:
    path: Path
    alg_metrics: dict[str, MetricMatrix]           # model_name -> MetricMatrix
    filtering_stats: FilteringStats | None
    n_tasks: int
    # model_name -> algorithm_key -> [concept, ...] in ordering order
    orderings: dict[str, dict[str, list[str]]] = field(default_factory=dict)


def _kendalls_w(rankings: list[list[int]]) -> float:
    """
    Kendall's W (coefficient of concordance) for k rankers over n items.

    rankings: list of k rankings, each a list of n 1-based rank integers.
    Returns W in [0, 1]: 0 = no agreement, 1 = perfect agreement.

    Formula: W = 12 * SS / (k^2 * (n^3 - n))
    where SS = sum over items of (sum_of_ranks - mean_sum_of_ranks)^2.
    The normalization by k^2*(n^3-n) makes W comparable across different n,
    which is why it is suitable for comparing splits with different task counts.
    """
    k = len(rankings)
    n = len(rankings[0])
    if k < 2 or n < 2:
        return 0.0
    # rank_sums[i] = sum of ranks assigned to item i across all rankers
    rank_sums = [sum(r[i] for r in rankings) for i in range(n)]
    # Under perfect agreement all rank_sums equal k*(n+1)/2
    mean_rank_sum = k * (n + 1) / 2
    ss = sum((rs - mean_rank_sum) ** 2 for rs in rank_sums)
    # 12 normalizes W to [0,1]: the maximum possible SS under perfect agreement is
    # k^2 * sum_{i=1}^{n}(i - (n+1)/2)^2 = k^2 * (n^3-n)/12, so dividing ss by
    # that maximum gives W = 12*ss / (k^2*(n^3-n)), with W=1 at perfect agreement.
    return 12 * ss / (k ** 2 * (n ** 3 - n))


def _family_w(result: SplitResult, algorithm_key: str) -> float:
    """
    Compute Kendall's W across anomaly detection models for one algorithm.

    Each model's ordering is a list of concepts in the order that algorithm
    places them. Position in the list is the rank (position 0 = rank 1).
    W measures how consistently models agree on that ordering.

    High W means the concept ordering is robust across models — it reflects
    genuine concept structure rather than a model-specific artifact.
    """
    models = list(result.orderings.keys())
    if len(models) < 2:
        # W requires at least 2 rankers
        return 0.0

    # Use the first model's concept list as a fixed reference order so that
    # all per-model rank lists are indexed consistently for _kendalls_w.
    reference = result.orderings[models[0]][algorithm_key]

    rankings = []
    for model in models:
        # rank[c] = 1-based position of concept c in this model's ordering
        rank = {c: i + 1 for i, c in enumerate(result.orderings[model][algorithm_key])}
        rankings.append([rank[c] for c in reference])

    return _kendalls_w(rankings)


def borda_aggregate(per_model_orderings: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    """
    Aggregate per-model concept orderings into a consensus ordering per algorithm via Borda count.

    For each algorithm and each concept, sum the concept's 1-based rank across all
    anomaly detection models. Concepts with lower total rank are placed first —
    they were consistently ranked early across models.

    This is the ordering counterpart to Kendall's W selection: W picks the split
    where models agree most; Borda count produces the consensus ordering that
    reflects that agreement.

    Returns: algorithm_key -> [concept, ...] in consensus order.
    """
    if not per_model_orderings:
        return {}

    algorithm_keys = list(next(iter(per_model_orderings.values())).keys())
    result = {}

    for key in algorithm_keys:
        concepts = list(next(iter(per_model_orderings.values()))[key])
        rank_sums = {c: 0 for c in concepts}
        for model_orderings in per_model_orderings.values():
            for rank, concept in enumerate(model_orderings[key], start=1):
                rank_sums[concept] += rank
        result[key] = sorted(concepts, key=lambda c: rank_sums[c])

    return result


def select_best_split(
    results: list[SplitResult],
) -> tuple[SplitResult, dict]:
    """
    Select the split with the highest mean Kendall's W across algorithm families.

    Intuition: if anomaly detection models consistently agree on how concepts
    should be ordered (high W), the ordering reflects genuine structural
    properties of the concept space. A split where models disagree (low W)
    has no robust ordering — any choice is model-specific noise.

    W is averaged across four qualitatively distinct families so that each
    structural hypothesis (difficulty, generalization, smooth drift, abrupt drift)
    contributes equally regardless of how many directional variants it has.
    """
    scores: dict[Path, float] = {}
    family_ws: dict[Path, dict] = {r.path: {} for r in results}

    for r in results:
        w_values = []
        for family_name, key in ALGORITHM_FAMILIES.items():
            w = _family_w(r, key)
            family_ws[r.path][family_name] = round(w, 4)
            w_values.append(w)
        # Mean W across families — each family weighted equally
        scores[r.path] = float(np.mean(w_values))

    # n_tasks breaks ties: prefer larger splits when W is equal
    best = max(results, key=lambda r: (scores[r.path], r.n_tasks))

    selection_report = {
        "selected": str(best.path),
        "splits": {
            str(r.path): {
                "n_tasks": r.n_tasks,
                "score": scores[r.path],
                "family_w": family_ws[r.path],
            }
            for r in results
        },
    }
    logger.info(f"Selected best split: {best.path} (W={scores[best.path]:.4f}, n_tasks={best.n_tasks})")
    return best, selection_report