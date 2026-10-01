import pytest
import numpy as np

from cadbench.optimization.ste.filtering_criteria import (
    filter_self_learnability,
    filter_coverage_by_other_tasks,
    filter_dominance_over_other_tasks,
    filter_performance_discriminativeness_across_stes,
    filter_pairwise_redundancy,
    filter_column_profile_redundancy,
)


def make_matrix(concepts: list[str], values: list[list[float]]) -> dict:
    return {concepts[i]: {concepts[j]: values[i][j] for j in range(len(concepts))} for i in range(len(concepts))}


# ── filter_self_learnability ──────────────────────────────────────────────────

class TestFilterSelfLearnability:
    def test_removes_concept_below_threshold_in_all_matrices(self):
        m1 = make_matrix(["a", "b"], [[0.9, 0.4], [0.3, 0.5]])
        m2 = make_matrix(["a", "b"], [[0.8, 0.3], [0.2, 0.6]])
        result = filter_self_learnability({"alg1": m1, "alg2": m2}, threshold=0.75)
        assert result == ["b"]

    def test_keeps_concept_above_threshold_in_at_least_one_matrix(self):
        m1 = make_matrix(["a", "b"], [[0.9, 0.4], [0.3, 0.5]])
        m2 = make_matrix(["a", "b"], [[0.8, 0.3], [0.2, 0.8]])
        result = filter_self_learnability({"alg1": m1, "alg2": m2})
        assert "b" not in result

    def test_empty_result_when_all_concepts_learnable(self):
        m = make_matrix(["a", "b"], [[0.9, 0.4], [0.3, 0.8]])
        assert filter_self_learnability({"alg": m}) == []

    def test_all_concepts_filtered(self):
        m = make_matrix(["a", "b"], [[0.5, 0.4], [0.3, 0.6]])
        result = filter_self_learnability({"alg": m}, threshold=0.75)
        assert set(result) == {"a", "b"}

    def test_single_concept_above_threshold(self):
        m = make_matrix(["a"], [[0.9]])
        assert filter_self_learnability({"alg": m}) == []

    def test_single_concept_below_threshold(self):
        m = make_matrix(["a"], [[0.5]])
        assert filter_self_learnability({"alg": m}) == ["a"]

    def test_threshold_boundary_equal_not_filtered(self):
        # condition is strict <, so equal to threshold is kept
        m = make_matrix(["a"], [[0.75]])
        assert filter_self_learnability({"alg": m}, threshold=0.75) == []


# ── filter_coverage_by_other_tasks ───────────────────────────────────────────

class TestFilterCoverageByOtherTasks:
    def test_removes_concept_covered_by_many_others(self):
        # best_specialist['a']=0.9; 'b' (0.87/0.9=0.967) and 'c' (0.86/0.9=0.956) both >=0.95 → filtered
        m = make_matrix(["a", "b", "c"], [
            [0.9, 0.1, 0.1],
            [0.87, 0.8, 0.1],
            [0.86, 0.1, 0.9],
        ])
        result = filter_coverage_by_other_tasks({"alg": m}, margin_pct=0.95, max_models_no=1)
        assert "a" in result

    def test_keeps_concept_covered_by_few_others(self):
        # best_specialist['a']=0.9; 'b' (0.88/0.9=0.978) >=0.95, 'c' (0.7/0.9=0.778) not → kept
        m = make_matrix(["a", "b", "c"], [
            [0.9, 0.1, 0.1],
            [0.88, 0.8, 0.1],
            [0.7, 0.1, 0.9],
        ])
        result = filter_coverage_by_other_tasks({"alg": m}, margin_pct=0.95, max_models_no=2)
        assert "a" not in result

    def test_empty_result_when_no_concept_over_covered(self):
        m = make_matrix(["a", "b"], [
            [0.9, 0.1],
            [0.2, 0.9],
        ])
        result = filter_coverage_by_other_tasks({"alg": m}, margin_pct=0.95, max_models_no=1)
        assert result == []

    def test_uses_max_across_multiple_matrices(self):
        # 'a' not covered in m1 but covered by 'b' and 'c' in m2 → max coverage = 2 → filtered
        m1 = make_matrix(["a", "b", "c"], [
            [0.9, 0.1, 0.1],
            [0.3, 0.9, 0.1],
            [0.3, 0.1, 0.9],
        ])
        m2 = make_matrix(["a", "b", "c"], [
            [0.9, 0.1, 0.1],
            [0.88, 0.9, 0.1],
            [0.87, 0.1, 0.9],
        ])
        result = filter_coverage_by_other_tasks({"alg1": m1, "alg2": m2}, margin_pct=0.95, max_models_no=1)
        assert "a" in result

    def test_not_filtered_when_others_score_below_pct_of_specialist(self):
        # best_specialist['a']=0.95; others at 0.88 → 0.88/0.95=0.926 < 0.95 → not covered → kept
        m = make_matrix(["a", "b", "c"], [
            [0.95, 0.1, 0.1],
            [0.88, 0.9, 0.1],
            [0.87, 0.1, 0.9],
        ])
        result = filter_coverage_by_other_tasks({"alg": m}, margin_pct=0.95, max_models_no=1)
        assert "a" not in result


# ── filter_dominance_over_other_tasks ────────────────────────────────────────

class TestFilterDominanceOverOtherTasks:
    def test_removes_concept_dominating_many_tasks(self):
        # best_specialist['b']=0.9, best_specialist['c']=0.9
        # 'a' on 'b': 0.87/0.9=0.967 >=0.95; 'a' on 'c': 0.86/0.9=0.956 >=0.95 → dominates 2 → filtered
        m = make_matrix(["a", "b", "c"], [
            [0.9, 0.87, 0.86],
            [0.1, 0.9,  0.1 ],
            [0.1, 0.1,  0.9 ],
        ])
        result = filter_dominance_over_other_tasks({"alg": m}, margin_pct=0.95, max_tasks_no=1)
        assert "a" in result

    def test_keeps_concept_not_dominating_enough_tasks(self):
        # 'a' on 'b': 0.88/0.9=0.978 >=0.95; 'a' on 'c': 0.5/0.9=0.556 <0.95 → dominates 1 → kept
        m = make_matrix(["a", "b", "c"], [
            [0.9, 0.88, 0.5],
            [0.1, 0.9,  0.1],
            [0.1, 0.1,  0.9],
        ])
        result = filter_dominance_over_other_tasks({"alg": m}, margin_pct=0.95, max_tasks_no=1)
        assert "a" not in result

    def test_empty_result_when_no_dominant_concept(self):
        m = make_matrix(["a", "b"], [
            [0.9, 0.2],
            [0.2, 0.9],
        ])
        result = filter_dominance_over_other_tasks({"alg": m}, margin_pct=0.95, max_tasks_no=1)
        assert result == []

    def test_uses_max_across_multiple_matrices(self):
        # best 'a'→'b' across matrices = 0.87 (m2), best 'a'→'c' = 0.86 (m2);
        # both ≥ 95% of best specialist (0.9) → solving=2 > max_tasks_no=1 → filtered
        m1 = make_matrix(["a", "b", "c"], [
            [0.9, 0.2,  0.2 ],
            [0.1, 0.9,  0.1 ],
            [0.1, 0.1,  0.9 ],
        ])
        m2 = make_matrix(["a", "b", "c"], [
            [0.9, 0.87, 0.86],
            [0.1, 0.9,  0.1 ],
            [0.1, 0.1,  0.9 ],
        ])
        result = filter_dominance_over_other_tasks({"alg1": m1, "alg2": m2}, margin_pct=0.95, max_tasks_no=1)
        assert "a" in result

    def test_dominated_tasks_are_counted_per_matrix_not_pooled(self):
        # 'a' dominates only 'b' in m1 and only 'c' in m2 → per-matrix count is 1 in both
        # (2 if pooled across matrices) → max=1, not > max_tasks_no=1 → kept
        m1 = make_matrix(["a", "b", "c"], [
            [0.9, 0.87, 0.2 ],
            [0.1, 0.9,  0.1 ],
            [0.1, 0.1,  0.9 ],
        ])
        m2 = make_matrix(["a", "b", "c"], [
            [0.9, 0.2,  0.86],
            [0.1, 0.9,  0.1 ],
            [0.1, 0.1,  0.9 ],
        ])
        result = filter_dominance_over_other_tasks({"alg1": m1, "alg2": m2}, margin_pct=0.95, max_tasks_no=1)
        assert "a" not in result

    def test_not_filtered_when_domination_below_pct_of_specialist(self):
        # best_specialist['b']=0.95; 'a' on 'b': 0.88/0.95=0.926 <0.95 → does not dominate → kept
        m = make_matrix(["a", "b", "c"], [
            [0.9, 0.88, 0.88],
            [0.1, 0.95, 0.1 ],
            [0.1, 0.1,  0.95],
        ])
        result = filter_dominance_over_other_tasks({"alg": m}, margin_pct=0.95, max_tasks_no=1)
        assert "a" not in result


# ── filter_performance_discriminativeness_across_stes ────────────────────────

class TestFilterPerformanceDiscriminativeness:
    def test_removes_concept_with_low_std(self):
        # all models score ~0.8 on 'a' → std ≈ 0 < threshold
        m = make_matrix(["a", "b", "c"], [
            [0.8, 0.1, 0.9],
            [0.8, 0.9, 0.1],
            [0.8, 0.5, 0.7],
        ])
        result = filter_performance_discriminativeness_across_stes({"alg": m}, threshold=0.05)
        assert "a" in result

    def test_keeps_concept_with_high_std(self):
        # scores on 'b': 0.1, 0.9, 0.5 → std ≈ 0.33
        m = make_matrix(["a", "b", "c"], [
            [0.9, 0.1, 0.5],
            [0.1, 0.9, 0.5],
            [0.5, 0.5, 0.9],
        ])
        result = filter_performance_discriminativeness_across_stes({"alg": m}, threshold=0.05)
        assert "b" not in result

    def test_removes_concept_if_low_std_in_any_matrix(self):
        # 'a' has high std in m1 but low std in m2 → removed
        m1 = make_matrix(["a", "b"], [[0.9, 0.1], [0.1, 0.9]])
        m2 = make_matrix(["a", "b"], [[0.8, 0.1], [0.8, 0.9]])
        result = filter_performance_discriminativeness_across_stes({"alg1": m1, "alg2": m2}, threshold=0.05)
        assert "a" in result

    def test_keeps_concept_if_high_std_in_all_matrices(self):
        m1 = make_matrix(["a", "b"], [[0.9, 0.1], [0.1, 0.9]])
        m2 = make_matrix(["a", "b"], [[0.8, 0.2], [0.2, 0.8]])
        result = filter_performance_discriminativeness_across_stes({"alg1": m1, "alg2": m2}, threshold=0.05)
        assert "a" not in result

    def test_empty_result_when_all_discriminative(self):
        m = make_matrix(["a", "b"], [[0.9, 0.1], [0.1, 0.9]])
        assert filter_performance_discriminativeness_across_stes({"alg": m}, threshold=0.05) == []

    def test_no_duplicate_concepts_in_result(self):
        # concept filtered by first matrix should not be added again by second
        m1 = make_matrix(["a", "b"], [[0.5, 0.1], [0.5, 0.9]])
        m2 = make_matrix(["a", "b"], [[0.5, 0.2], [0.5, 0.8]])
        result = filter_performance_discriminativeness_across_stes({"alg1": m1, "alg2": m2}, threshold=0.05)
        assert result.count("a") == 1


# ── filter_pairwise_redundancy ────────────────────────────────────────────────

class TestFilterPairwiseRedundancy:
    def test_suggests_merging_nearly_identical_concepts(self):
        m = make_matrix(["a", "b", "c"], [
            [0.9, 0.8, 0.3],
            [0.9, 0.8, 0.3],  # 'b' row identical to 'a'
            [0.1, 0.1, 0.9],
        ])
        result = filter_pairwise_redundancy({"alg": m}, threshold=0.1)
        assert ("a", "b") in result or ("b", "a") in result

    def test_does_not_suggest_merging_distinct_concepts(self):
        m = make_matrix(["a", "b"], [
            [0.9, 0.1],
            [0.1, 0.9],
        ])
        result = filter_pairwise_redundancy({"alg": m}, threshold=0.1)
        assert result == []

    def test_no_self_pairs_in_result(self):
        m = make_matrix(["a", "b"], [
            [0.9, 0.8],
            [0.8, 0.9],
        ])
        result = filter_pairwise_redundancy({"alg": m}, threshold=0.75)
        assert not any(c1 == c2 for c1, c2 in result)

    def test_aggregates_across_multiple_matrices(self):
        # 'a' and 'b' are similar in both matrices → merged
        m1 = make_matrix(["a", "b", "c"], [
            [0.9, 0.8, 0.3],
            [0.9, 0.8, 0.3],
            [0.1, 0.1, 0.9],
        ])
        m2 = make_matrix(["a", "b", "c"], [
            [0.7, 0.6, 0.2],
            [0.7, 0.6, 0.2],
            [0.1, 0.1, 0.8],
        ])
        result = filter_pairwise_redundancy({"alg1": m1, "alg2": m2}, threshold=0.1)
        assert ("a", "b") in result or ("b", "a") in result


# ── filter_column_profile_redundancy ─────────────────────────────────────────

class TestFilterColumnProfileRedundancy:
    def test_removes_concept_with_similar_column_profile(self):
        # column 'a': [0.9, 0.9, 0.1]; column 'b': [0.9, 0.9, 0.1] → identical → 'b' removed
        m = make_matrix(["a", "b", "c"], [
            [0.9, 0.9, 0.1],
            [0.9, 0.9, 0.1],
            [0.1, 0.1, 0.9],
        ])
        result = filter_column_profile_redundancy({"alg": m}, threshold=0.1)
        assert "b" in result or "a" in result

    def test_keeps_concept_with_distinct_column_profile(self):
        # column 'a': [0.9, 0.1, 0.1]; column 'b': [0.1, 0.9, 0.1] → distinct
        m = make_matrix(["a", "b", "c"], [
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.9],
        ])
        result = filter_column_profile_redundancy({"alg": m}, threshold=0.1)
        assert result == []

    def test_distinct_from_row_redundancy(self):
        # rows of 'a' and 'b' are identical → row-redundant
        # columns of 'a' and 'b' differ → NOT column-redundant
        m = make_matrix(["a", "b", "c"], [
            [0.9, 0.8, 0.3],  # row 'a'
            [0.9, 0.8, 0.3],  # row 'b' identical to 'a'
            [0.1, 0.5, 0.9],
        ])
        result = filter_column_profile_redundancy({"alg": m}, threshold=0.1)
        assert "a" not in result and "b" not in result

    def test_aggregates_across_multiple_matrices(self):
        m1 = make_matrix(["a", "b", "c"], [
            [0.9, 0.9, 0.1],
            [0.9, 0.9, 0.1],
            [0.1, 0.1, 0.9],
        ])
        m2 = make_matrix(["a", "b", "c"], [
            [0.8, 0.8, 0.2],
            [0.8, 0.8, 0.2],
            [0.2, 0.2, 0.8],
        ])
        result = filter_column_profile_redundancy({"alg1": m1, "alg2": m2}, threshold=0.1)
        assert "a" in result or "b" in result