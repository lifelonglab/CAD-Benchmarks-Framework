# CAD-Benchmarks: Principled Scenarios for Continual Anomaly Detection

This repository provides the code and experimental pipeline accompanying the paper:

**Towards Principled Continual Anomaly Detection: A Systematic Framework and Benchmark Scenarios**

The repository implements an end-to-end framework for transforming existing tabular anomaly-detection datasets into validated **continual anomaly detection (CAD)** scenarios. Rather than relying on arbitrary chronological or heuristic task splits, the framework discovers candidate tasks, evaluates their learnability and cross-task structure, filters unsuitable tasks, derives principled task orderings, and validates the resulting scenarios under multiple continual-learning strategies.

---

## Overview

Continual anomaly detection studies anomaly detectors that must adapt to a sequence of changing data regimes while retaining useful knowledge from previously observed regimes. 
In tabular data, however, meaningful task boundaries are rarely given. 
A naive split may produce tasks that are unlearnable, redundant, dominated by another task, or insufficiently informative for evaluating continual-learning behavior.

This repository addresses this problem through a reproducible scenario-construction pipeline.

The framework supports:

- **Dataset preprocessing**
- **Candidate task discovery** using natural metadata boundaries and clustering-based decompositions.
- **Single-Task Expert (STE) evaluation** to estimate task learnability, transfer, overlap, and redundancy.
- **Task filtering criteria** that remove unsuitable candidate tasks before benchmark construction.
- **Principled task orderings** that instantiate different continual-learning dynamics.
- **Scenario selection** using cross-model ordering agreement and consensus aggregation.
- **Scenario validation** under multiple anomaly-detection models and continual-learning strategies.
- **Ready-to-use CAD scenarios** with task statistics, orderings, validation results, and reproducible scripts.

---

## Benchmark Scenarios

The paper instantiates the framework on six tabular datasets from three domains (network intrusion detection, particle physics, and predictive maintenance) and derives eight validated CAD scenarios.

| Scenario | Domain | # Features | # Tasks | # Train samples | # Test samples | # Test anomalies | Test anomaly ratio (range over tasks) |
|---|---|---:|---:|---:|---:|---:|---:|
| `CAD-CICIDS2017` | Intrusion detection | 78 | 6 | 1,588,098 | 488,750 | 91,723 | 18.8% (12.9–50.0) |
| `CAD-CICIDS2018` | Intrusion detection | 78 | 5 | 1,925,233 | 665,538 | 186,647 | 28.0% (2.1–50.0) |
| `CAD-CICUNSW` | Intrusion detection | 79 | 5 | 843,281 | 241,647 | 30,825 | 12.8% (8.1–50.0) |
| `CAD-MiniBooNE` | Particle physics | 50 | 5 | 34,647 | 31,469 | 22,805 | 72.5% (49.8–89.0) |
| `CAD-Scania` | Truck maintenance | 162 | 3 | 40,205 | 10,879 | 825 | 7.6% (3.4–28.9) |
| `CAD-TCM` | Steel manufacturing | 51 | 8 | 83,275 | 25,597 | 4,776 | 18.7% (9.7–38.4) |
| `MCAD-CIC-3x1` | Intrusion detection (3 datasets, one task each) | 77 | 3 | 12,036,713 | 5,878,856 | 1,866,616 | 31.8% (9.6–44.8) |
| `MCAD-CIC-3xN` | Intrusion detection (3 datasets, retained tasks) | 77 | 13 | 2,667,154 | 914,638 | 250,247 | 27.4% (2.1–50.0) |

The two multi-dataset scenarios use the 77 features shared by the three intrusion-detection sources after harmonizing their attribute names.

Each scenario is associated with six principled task orderings:

1. `curriculum_asc` — easy-to-hard ordering,
2. `curriculum_desc` — hard-to-easy ordering,
3. `generalization_asc` — low-to-high transfer ordering,
4. `generalization_desc` — high-to-low transfer ordering,
5. `smooth_drift` — gradual transition between tasks,
6. `abrupt_drift` — large distributional shifts between consecutive tasks.

These orderings are intended to expose different forms of continual-learning behavior, including forgetting, transfer, adaptation difficulty, robustness to ordering effects, and sensitivity to concept drift.

The scenarios are available [on Hugging Face](https://huggingface.co/collections/lifelonglab/tabular-cad-benchmarks).

---

## Repository Structure

```
.
├── requirements.txt
├── docs/                               # Dataset-specific notes (MiniBooNE, Scania, TCM)
├── tests/                              # Unit tests of the filtering criteria
├── src/
│   └── cadbench/
│       ├── logger.py                   # Logging setup
│       ├── paths.py                    # Path resolution via environment variables
│       ├── datasets/                   # Dataset loading and preprocessing
│       │   ├── cicids2017.py
│       │   ├── cicids2018.py
│       │   ├── cicunsw.py
│       │   ├── miniboone.py
│       │   ├── scania.py
│       │   └── tcm.py
│       ├── optimization/
│       │   ├── processing.py           # Concept reordering utilities
│       │   └── ste/
│       │       ├── filtering_criteria.py  # Concept filtering (learnability, redundancy, …)
│       │       ├── split_selection.py     # Kendall's W-based split selection, Borda aggregation
│       │       ├── ste_filtering.py       # Concept/matrix filtering helpers
│       │       ├── ste_optimization.py    # Ordering strategies (curriculum, generalization, drift)
│       ├── pyclad/                     # pyclad integration and model implementations
│       ├── splits/
│       │   ├── clustering/             # Clustering-based concept splitting
│       │   ├── rebalancing.py          # Test-split anomaly rebalancing
│       │   └── utils.py                # Normal/anomaly train-test splitting
│       └── heterogeneity/
│           └── ste_strategy.py         # SingleTaskExpertStrategy (pyclad strategy)
└── scripts/
    ├── datasets/
    │   ├── process_cicds2017.py        # Preprocess / cluster CICIDS2017
    │   ├── process_cicids2018.py       # Preprocess / cluster CICIDS2018
    │   ├── process_cicunsw.py          # Preprocess / cluster CIC-UNSW-NB15
    │   ├── process_miniboone.py        # Preprocess / cluster MiniBooNE
    │   ├── process_scania.py           # Preprocess / cluster APS Failure at Scania Trucks
    │   └── process_tcm.py              # Preprocess / split by file / cluster TCM
    ├── optimization/
    │   └── optimize_by_ste.py          # STE-based concept ordering & split selection
    ├── pyclad/
    │   ├── run_single_concept.py       # Single-concept baseline evaluation
    │   ├── run_single_task_expert.py   # STE evaluation (produces metric matrices)
    │   └── run_validation_strategies.py  # Full continual learning strategy evaluation
    └── tasks/
        └── rebalance_tasks.py          # Analyze and rebalance concept test splits
```

---

## Installation
Install dependencies:

```bash
pip install -r requirements.txt
```

Add the repository source directory to `PYTHONPATH`:

```bash
export PYTHONPATH=src:$PYTHONPATH
```

Run the unit tests of the filtering criteria:

```bash
pytest
```



---

## Full Experimental Workflow

After installation, the full benchmark pipeline can be reproduced through the following stages:

1. configure paths,
2. prepare raw datasets,
3. preprocess datasets,
4. generate candidate task splits,
5. optionally rebalance task test partitions,
6. run Single-Task Expert evaluation,
7. filter tasks and select final scenarios,
8. generate task orderings,
9. validate continual-learning strategies,
10. inspect results.

The commands below show a minimal end-to-end example for `CAD-CICIDS2017`. The other datasets follow the same stages with their own `scripts/datasets/process_*.py` script; dataset-specific instructions for MiniBooNE, Scania, and TCM are given in [`docs/`](docs/).

All randomized stages are seeded: the clustering and single-task expert scripts accept a `--seed` option (default 42).

### 1. Configure paths

```bash
export RESOURCES_PATH=/path/to/resources
export OUTPUT_PATH=/path/to/output
export PYTHONPATH=src:$PYTHONPATH
```

### 2. Prepare raw datasets

Place datasets raw CSV files under:

```text
$RESOURCES_PATH/datasets/tabular/cicids2017/
```

### 3. Preprocess the dataset

```bash
python scripts/datasets/process_cicds2017.py --preprocess
```

### 4. Generate natural task splits

```bash
python scripts/datasets/process_cicds2017.py --split-by-day
```

### 5. Generate clustering-based candidate task splits

```bash
python scripts/datasets/process_cicds2017.py \
    --cluster \
    --input-file $OUTPUT_PATH/datasets/cicids2017/cicids2017_processed.csv
```

For large-scale clustering experiments, use subsampling during cluster fitting:

```bash
python scripts/datasets/process_cicds2017.py \
    --cluster \
    --input-file $OUTPUT_PATH/datasets/cicids2017/cicids2017_by_day.csv \
    --sampled-size 100000
```

### 6. Optionally rebalance task test partitions

```bash
python scripts/tasks/rebalance_tasks.py \
    --input $OUTPUT_PATH/datasets/cicids2017/clustering/ \
    --rebalance \
    --output $OUTPUT_PATH/datasets/cicids2017/tasks_balanced/
```

### 7. Run Single-Task Expert evaluation

```bash
python scripts/pyclad/run_single_task_expert.py \
    --input $OUTPUT_PATH/datasets/cicids2017/clustering/
```

This produces cross-task performance matrices in which entry `(i, j)` measures the performance of a model trained on task `i` and evaluated on task `j`.

### 8. Filter tasks, select the final scenario, and generate orderings

```bash
python scripts/optimization/optimize_by_ste.py \
    --input $OUTPUT_PATH/datasets/cicids2017/clustering/ \
    --filtering True \
    --select-best \
    --min-tasks 5
```

Expected outputs include:

```text
*_orderings.json
split_selection_report.json
best_split_orderings.json
```

### 9. Validate continual-learning strategies

```bash
python scripts/pyclad/run_validation_strategies.py \
    --input $OUTPUT_PATH/datasets/cicids2017/scenarios_optimized/ \
    --mode ordering
```

To evaluate a single model:

```bash
python scripts/pyclad/run_validation_strategies.py \
    --input $OUTPUT_PATH/datasets/cicids2017/scenarios_optimized/ \
    --mode ordering \
    --model Autoencoder
```

The validation script reads both the orderings produced by `optimize_by_ste.py` and the `orderings.json` files distributed with the released scenarios.

A valid scenario should satisfy three conditions:

- **Feasibility**: the scenario is learnable by at least one strong reference strategy.
- **Non-triviality**: naive sequential training does not already solve the scenario.
- **Forgetting**: naive sequential training exhibits measurable degradation on previous tasks.

---
