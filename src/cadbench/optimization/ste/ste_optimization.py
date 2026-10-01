import numpy as np
from typing_extensions import deprecated

from cadbench.optimization.ste.utils import MetricMatrix


@deprecated('deprecated')
def optimize_by_ste_difficulty(metric_matrix: MetricMatrix, descending=False) -> list[tuple[str, float]]:
    """
    Reorder concepts by their average performance across the metric matrix (column-wise average).
    """
    concept_avgs = []
    for eval_concept in metric_matrix.keys():
        scores = [metric_matrix[c][eval_concept] for c in metric_matrix.keys()]
        concept_avgs.append((eval_concept, np.mean(scores)))
    return sorted(concept_avgs, key=lambda x: x[1], reverse=descending)


@deprecated('deprecated')
def optimize_by_ste_generalization(metric_matrix: MetricMatrix, descending=True) -> list[tuple[str, float]]:
    """
    Find the order of concepts that optimizes generalization by sorting them based on the average of their metric values across all concepts after learning a given concept.
    """
    concept_avgs = [(concept, float(np.mean(list(values.values())))) for concept, values in metric_matrix.items()]
    return sorted(concept_avgs, key=lambda x: x[1], reverse=descending)


# --- NEW ordering strategies ---

def order_by_curriculum(metric_matrix: MetricMatrix, descending: bool = False) -> list[tuple[str, float]]:
    """
    Curriculum ordering: sort by task difficulty d_i = 1 - M[i][i].
    Ascending (default): easy-to-hard. Descending: hard-to-easy.
    """
    difficulties = [(c, 1.0 - metric_matrix[c][c]) for c in metric_matrix]
    return sorted(difficulties, key=lambda x: x[1], reverse=descending)


def order_by_generalization(metric_matrix: MetricMatrix, descending: bool = True) -> list[tuple[str, float]]:
    """
    Generalization ordering: sort by g_i = mean off-diagonal row values M[i][j], j != i.
    Descending (default): best generalizers first. Ascending: worst first.
    """
    concepts = list(metric_matrix.keys())
    K = len(concepts)
    if K < 2:
        # No other concept to generalize to -- score is undefined, not zero.
        return [(c, float("nan")) for c in concepts]
    scores = [
        (c, sum(metric_matrix[c][other] for other in concepts if other != c) / (K - 1))
        for c in concepts
    ]
    return sorted(scores, key=lambda x: x[1], reverse=descending)


def _build_distance_matrix(metric_matrix: MetricMatrix) -> tuple[list[str], np.ndarray]:
    """
    Symmetric distance matrix from metric matrix:
        D[i,j] = 1 - (M[i,j] + M[j,i]) / 2
    High distance = low cross-task generalization = dissimilar tasks.
    """
    concepts = list(metric_matrix.keys())
    K = len(concepts)
    D = np.zeros((K, K))
    for i, ci in enumerate(concepts):
        for j, cj in enumerate(concepts):
            D[i, j] = 1.0 - (metric_matrix[ci][cj] + metric_matrix[cj][ci]) / 2.0
    return concepts, D


def _greedy_tsp(D: np.ndarray, maximize: bool) -> tuple[list[int], float]:
    """
    Greedy nearest/farthest-neighbour heuristic over all starting points.
    Returns the index ordering and its total consecutive distance.
    """
    K = len(D)
    best_order, best_cost = None, None
    for start in range(K):
        visited = [False] * K
        order = [start]
        visited[start] = True
        cost = 0.0
        for _ in range(K - 1):
            curr = order[-1]
            next_idx = min(
                (j for j in range(K) if not visited[j]),
                key=lambda j: -D[curr, j] if maximize else D[curr, j],
            )
            cost += D[curr, next_idx]
            order.append(next_idx)
            visited[next_idx] = True
        if best_cost is None or (maximize and cost > best_cost) or (not maximize and cost < best_cost):
            best_cost, best_order = cost, order
    return best_order, best_cost


def order_by_smooth_drift(metric_matrix: MetricMatrix) -> list[tuple[str, float]]:
    """
    Smooth-drift ordering: minimise sum of consecutive inter-task distances
        pi* = argmin_pi  sum_{t=1}^{K-1} D[pi_t, pi_{t+1}]
    Uses a greedy nearest-neighbour heuristic over all starting points.
    """
    concepts, D = _build_distance_matrix(metric_matrix)
    order, _ = _greedy_tsp(D, maximize=False)
    return [(concepts[order[t]], D[order[t], order[t + 1]] if t < len(order) - 1 else 0.0)
            for t in range(len(order))]


def order_by_abrupt_drift(metric_matrix: MetricMatrix) -> list[tuple[str, float]]:
    """
    Abrupt-drift ordering: maximise sum of consecutive inter-task distances
        pi* = argmax_pi  sum_{t=1}^{K-1} D[pi_t, pi_{t+1}]
    Uses a greedy farthest-neighbour heuristic over all starting points.
    """
    concepts, D = _build_distance_matrix(metric_matrix)
    order, _ = _greedy_tsp(D, maximize=True)
    return [(concepts[order[t]], D[order[t], order[t + 1]] if t < len(order) - 1 else 0.0)
            for t in range(len(order))]
