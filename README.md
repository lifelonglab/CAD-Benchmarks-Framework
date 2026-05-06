# CAD-Benchmarks: Principled Scenarios for Continual Anomaly Detection

This repository provides the code and experimental pipeline accompanying the paper:

**Towards Principled Continual Anomaly Detection Benchmarks: A Systematic Framework and Validated Scenarios**

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

The paper instantiates the framework on three large-scale tabular cybersecurity datasets and derives several validated CAD scenarios.

| Scenario | Description | # Tasks | # Samples | Test anomaly ratio |
|---|---:|---:|---:|---:|
| `CAD-CICIDS2017` | Single-dataset CAD scenario from CICIDS2017 | 6 | 2,076,848 | 18.77% |
| `CAD-CICIDS2018` | Single-dataset CAD scenario from CICIDS2018 | 5 | 2,590,771 | 28.04% |
| `CAD-CICUNSW` | Single-dataset CAD scenario from CIC-UNSW-NB15 | 5 | 1,084,928 | 12.76% |
| `MCAD-CIC-3x1` | Multi-dataset scenario with one task per dataset | 3 | 17,915,569 | 10.42% |
| `MCAD-CIC-3xN` | Multi-dataset scenario combining retained tasks across datasets | 13 | 3,581,792 | 27.36% |

Each scenario is associated with six principled task orderings:

1. `curriculum_asc` — easy-to-hard ordering,
2. `curriculum_desc` — hard-to-easy ordering,
3. `generalization_asc` — low-to-high transfer ordering,
4. `generalization_desc` — high-to-low transfer ordering,
5. `smooth_drift` — gradual transition between tasks,
6. `abrupt_drift` — large distributional shifts between consecutive tasks.

These orderings are intended to expose different forms of continual-learning behavior, including forgetting, transfer, adaptation difficulty, robustness to ordering effects, and sensitivity to concept drift.

Data is available [on HuggingFace](https://huggingface.co/anonymizeddb)

---

## Repository Structure

```
.
├── requirements.txt
├── src/
│   └── cadbench/
│       ├── logger.py                   # Logging setup
│       ├── paths.py                    # Path resolution via environment variables
│       ├── datasets/                   # Dataset loading and preprocessing
│       │   ├── cicids2017.py
│       │   ├── cicids2018.py
│       │   ├── cicunsw.py
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
    │   └── process_cicunsw.py          # Preprocess / cluster CIC-UNSW-NB15
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

The commands below show a minimal end-to-end example for `CAD-CICIDS2017`.

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

A valid scenario should satisfy three conditions:

- **Feasibility**: the scenario is learnable by at least one strong reference strategy.
- **Non-triviality**: naive sequential training does not already solve the scenario.
- **Forgetting**: naive sequential training exhibits measurable degradation on previous tasks.

---
