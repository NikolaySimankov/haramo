# haramo

**H**olistic **A**utoML-driven **R**obust pipeline for **A**pplied **M**ulti-**O**mics

Authors: Nikolay Simankov & Helene Soyeurt

---

## Overview

haramo is an AutoML pipeline for binary classification on high-dimensional tabular data, particularly multi-omics datasets. It automates feature selection, scaling, algorithm selection, and hyperparameter tuning inside a nested cross-validation framework, and reports a comprehensive set of classification metrics on the outer folds.

---

## Installation

```bash
pip install haramo
pip install greedyboruta
```

---

## How it works

The pipeline has four sequential steps optimized jointly by Optuna:

```
VarianceThreshold → Feature Selector → Scaler → Classifier
```

**Outer loop** (4-fold `StratifiedKFold` or `StratifiedGroupKFold`): splits data into train/test sets for unbiased evaluation.

**Inner loop** (4-fold `StratifiedKFold`): runs Optuna search on the training split of each outer fold.

haramo is designed to be run in two stages:

### Stage 1 — Pipeline structure search (`hyperparameters="default"`)

Uses `GridSampler` to exhaustively evaluate all combinations of algorithm, scaler, and feature selector with default hyperparameters. With all options open this covers up to 240 configurations (15 algorithms × 4 scalers × 4 feature selector combinations) per outer fold. The goal is to identify which pipeline structures work best on the data.

### Stage 2 — Hyperparameter optimization (`hyperparameters="optimize"`)

Takes the top pipeline structures identified in stage 1 and runs Bayesian (`TPESampler`, multivariate) hyperparameter search on each one, tuning every step of the pipeline jointly.

**Model selection**: the best pipeline across outer folds is selected by maximum MCC.

---

## Pipeline overview

End-to-end view of what `magic_now` does on each call.

```
magic_now(X, y, scoring="PR AUC", calibration="auto"|"isotonic"|"sigmoid"|None,
          optimize_threshold=True|False, pos_weight_factor=1.0, ...)
│
├── [Optional] Dataset combo selection      (only when X is a list/dict of DataFrames)
│      └── beam search over candidate DataFrames → concatenated best combo
│
├── Compute sample_weight
│      └── balanced class weights × pos_weight_factor on the positive class
│
├── train()                                  ← outer CV: 4-fold Stratified[Group]KFold (random_state)
│      │
│      └── For each outer fold (in parallel):
│             │
│             ├── Phase 1 — Feature-selection HPO
│             │      └── 5/10/15 Optuna trials × inner 3-fold CV
│             │             → fit-fixed (StandardScaler + default LGBM) → best FS pipeline
│             │
│             └── Phase 2 — Scaler + model HPO
│                    └── n_trials Optuna trials × inner 3-fold CV
│                           → tunes scaler, algorithm (if list ⇒ categorical), hyperparams
│                           → study.best_params
│
│      Output: pipelines = {fold_1: pipeline, fold_2: pipeline, fold_3: pipeline, fold_4: pipeline}
│              studies   = {fold_i: optuna study}
│
├── nested_crossval()                        ← per-fold reduction-pct sweep + dual evaluation
│      │
│      ├── Build TWO CV grids:
│      │      ├── tts_splits  (random_state)        ← same grid train() used; each fold's "true" held-out fold
│      │      └── cfc_splits  (cfc_random_state)    ← independent neutral grid for cross-fold comparison
│      │
│      ├── For each valid reduction pct (10%–100%, filtered by ≥2000 samples & SVM cap):
│      │      │
│      │      ├── CFC fits  — every fold_key × every cfc_split → pred_store_cfc[fk, si, pct]
│      │      ├── TTS fits  — every fold_key on its own tts_split[i] → pred_store_tts[fk, pct]
│      │      │
│      │      ├── Combined report per fold_key:
│      │      │      ├── CFC report  ⟶ "{metric} CFC" columns (concatenated across cfc_splits)
│      │      │      └── TTS report  ⟶ "{metric} TTS" columns (single OOF held-out fold)
│      │      │
│      │      └── Early stopping per fold_key on "{scoring} CFC"  (2-strike rule)
│      │
│      └── For each fold_key:
│             ├── Pick best pct by "{scoring} CFC"            ← per-fold, NOT cross-fold winner
│             ├── Refit pipeline on full data at best pct
│             └── Surface OOF preds:  {"cfc": concat over cfc_splits, "tts": single tts_split}
│
│      Output: reduction_validation     (MultiIndex (fold, reduction); CFC+TTS columns)
│              pipeline_by_fold         (dict {fold_key: refitted Pipeline})
│              predictions_by_fold      (dict {fold_key: {"cfc": df, "tts": df}})
│
├── [Optional] Score-level calibration + threshold tuning
│      │      (zero extra model fits — operates entirely on the OOF scores already produced)
│      │
│      └── For each fold:
│             ├── Pick calibration method:
│             │      ├── calibration="auto"     → pick_best_score_calibration_method (k-fold on CFC OOF, Brier)
│             │      ├── calibration="isotonic" → ScoreCalibrator(method="isotonic")
│             │      ├── calibration="sigmoid"  → ScoreCalibrator(method="sigmoid")
│             │      └── calibration=None       → no calibrator
│             │
│             ├── Fit ScoreCalibrator on CFC OOF (y_true, y_score)
│             ├── Apply calibrator to CFC scores and TTS scores  ⟶ calibrated probabilities
│             │
│             ├── If optimize_threshold:
│             │      └── find_best_threshold on calibrated CFC scores
│             │             ├── metric = scoring   (or "FNFP Loss" fallback for threshold-free)
│             │             └── lower/upper-better aware (Brier, FNFP Loss → argmin)
│             │
│             ├── Build per-variant rows in the final validation table:
│             │      ├── "raw model"                          — raw scores, threshold 0.5
│             │      ├── "calibrated model"                   — calibrated scores, threshold 0.5
│             │      └── "calibrated model with threshold"    — calibrated + tuned threshold
│             │      (every row carries both "{metric} CFC" and "{metric} TTS" columns)
│             │
│             └── Wrap (pipeline, calibrator, threshold) in CalibratedThresholdedClassifier
│
├── Persist
│      ├── models/pipelines{tag}.pkl            ← {fold_key: CalibratedThresholdedClassifier | Pipeline}
│      ├── trials/studies{tag}.pkl              ← Optuna studies
│      ├── results/best_params{tag}.tsv         ← per-fold HPO winner (algorithm + hyperparams)
│      ├── results/validation{tag}.tsv          ← per-fold per-variant; CFC + TTS columns
│      ├── results/reduction_validation{tag}.tsv← full nested_crossval table
│      └── results/dataset_selection{tag}.tsv   ← (if dataset combo selection ran)
│
└── Plots (if plots=True):
       ├── plots/pr_curve{tag}.pdf              ← one CFC curve per fold + mean ± std
       ├── plots/roc_curve{tag}.pdf
       ├── plots/ks_statistic{tag}.pdf
       └── plots/calibration_curve_{fold}{tag}.pdf
              (uncalibrated vs isotonic vs sigmoid, Brier in legend, per outer fold)


Deployment / inference
======================
art = pickle.load(open("pipelines{tag}.pkl"))["fold_1"]
y_proba = art.predict_proba(X_new)[:, 1]      ← pipeline.predict_proba → ScoreCalibrator → clip[0,1]
y_pred  = art.predict(X_new)                  ← (y_proba >= art.threshold).astype(int)
```

### Key invariants

- **HPO loop is calibration-free.** Phase 1 + Phase 2 inside `train()` never see calibration or threshold tuning. The scorer (e.g. PR AUC, FNFP Loss) drives Optuna directly. Keeps per-trial cost stable and predictable.
- **CFC drives selection, TTS is diagnostic.** Reduction-pct selection, early stopping, and calibration-method picking all key off CFC (more data per evaluation, lower variance). TTS metrics appear in the same row for comparison but never determine winners.
- **Per-fold, not cross-fold.** Every outer fold's pipeline is refit at its own best pct, calibrated against its own OOF, gets its own threshold, and is shipped as one entry in the pickled dict. There is no single "winner" — N independent deployable artifacts, plus a validation table that lets you pick by mean(CFC,TTS), stability, or any other rule downstream.
- **`scoring` controls four things:** the Optuna objective in Phase 1/Phase 2 HPO, the early-stopping criterion in `nested_crossval`, the per-fold best-pct selection in `nested_crossval`, and the threshold-tuning metric in `find_best_threshold` (with `"FNFP Loss"` fallback when `scoring` is threshold-free).
- **`pos_weight_factor` threads everywhere `sample_weight` is used:** HPO inner CV fits, nested_crossval fits, calibrator fit on OOF scores. Default `1.0` is balanced; higher values prioritize sensitivity.

---

## Quick start

```python
from pathlib import Path
import pandas as pd
from haramo.classification import magic_now

X = pd.read_csv("features.csv", index_col=0)
y = pd.read_csv("labels.csv", index_col=0).squeeze()

# --- Stage 1: find the best pipeline structures ---
_, _, studies = magic_now(
    X=X,
    y=y,
    hyperparameters="default",
    output_dir=Path("results/stage1"),
    tag="_stage1",
)

# Extract top 3 structures from any fold (they are consistent across folds)
top3 = sorted(studies["fold_1"].trials, key=lambda t: t.value, reverse=True)[:3]
for i, trial in enumerate(top3):
    print(f"#{i+1}: {trial.params}  score={trial.value:.4f}")

# --- Stage 2: optimize hyperparameters of the best structure ---
# Example: stage 1 identified LGBM + robust scaler + boruta as best
validation, pipeline, studies = magic_now(
    X=X,
    y=y,
    algorithm="LGBM",
    scaler="robust",
    feature_selector="boruta",
    hyperparameters="optimize",
    n_trials=200,
    output_dir=Path("results/stage2"),
    tag="_lgbm_robust_boruta",
)

print(validation)
```

---

## API reference

### `magic_now`

```python
from haramo.classification import magic_now

validation, pipeline, studies = magic_now(
    X,
    y,
    scoring="balanced_accuracy",
    task="classification",
    feature_selector="optimize",
    scaler="optimize",
    algorithm="optimize",
    hyperparameters="optimize",
    random_state=42,
    n_trials=100,
    output_dir=None,
    tag="",
    groups=None,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | `pd.DataFrame` | — | Feature matrix. Must be a DataFrame with named columns. |
| `y` | `pd.Series` | — | Binary target vector. |
| `scoring` | `str` or callable | `"balanced_accuracy"` | Scoring metric for the inner Optuna objective. Any sklearn scorer name or callable. |
| `task` | `str` | `"classification"` | Task type. Currently only `"classification"` is supported. |
| `feature_selector` | `str` or `list` | `"optimize"` | Feature selection strategy. See [Feature selection](#feature-selection). |
| `scaler` | `str` or `list` | `"optimize"` | Scaling strategy. See [Scalers](#scalers). |
| `algorithm` | `str` or `list` | `"optimize"` | Classifier strategy. See [Algorithms](#algorithms). |
| `hyperparameters` | `str` | `"optimize"` | `"optimize"` tunes all hyperparameters via Optuna. `"default"` fixes classifier hyperparameters and grid-searches only the pipeline structure. |
| `random_state` | `int` | `42` | Random seed for reproducibility. |
| `n_trials` | `int` | `100` | Number of Optuna trials per outer fold. Ignored when `hyperparameters="default"`. |
| `output_dir` | `Path` | `None` | **Required.** Root directory for all outputs. Created automatically with subdirectories. |
| `tag` | `str` | `""` | Optional suffix appended to all output filenames. |
| `groups` | array-like | `None` | Group labels for samples. When provided, uses `StratifiedGroupKFold` instead of `StratifiedKFold` for the outer loop, preventing data leakage across groups (e.g. repeated measures, patients). |

**Returns**

| Name | Type | Description |
|---|---|---|
| `validation` | `pd.DataFrame` | Metrics per outer fold (rows = folds). Columns: MCC, F1-score, Kappa, Bal. Acc., Precision, Sensitivity, Selectivity. |
| `pipeline` | `sklearn.pipeline.Pipeline` | Best pipeline selected by maximum MCC across outer folds. |
| `studies` | `dict` | Optuna `Study` objects keyed by fold name (`"fold_1"`, ..., `"fold_4"`). |

**Output files**

```
output_dir/
    results/  validation{tag}.tsv    # validation metrics per fold
    models/   pipelines{tag}.pkl     # best fitted pipeline
    trials/   studies{tag}.pkl       # Optuna study objects
```

---

### `train`

Lower-level function that runs the Optuna search on each outer fold and returns unfitted best pipelines.

```python
from haramo.classification import train

pipelines, studies = train(
    X, y,
    scoring="balanced_accuracy",
    task="classification",
    feature_selector="optimize",
    scaler="optimize",
    algorithm="optimize",
    hyperparameters="optimize",
    random_state=42,
    n_trials=100,
    groups=None,
)
```

Returns `pipelines` (dict of `Pipeline` keyed by fold) and `studies` (dict of Optuna `Study`).

---

### `nested_crossval`

Retrains a set of pipelines on the outer folds and collects predictions for evaluation.

```python
from haramo.classification import nested_crossval

validation, best_pipeline = nested_crossval(
    X, y,
    pipelines=pipelines,
    random_state=42,
    groups=None,
)
```

Returns the validation `DataFrame` and the best pipeline by MCC.

---

## Feature selection

Controlled by the `feature_selector` parameter.

| Value | Behaviour |
|---|---|
| `"optimize"` | Optuna decides whether to apply a p-value filter and/or a GreedyBoruta filter (all four combinations are searchable). |
| `"pvalue"` | Only p-value filter (Pearson correlation, threshold optimized by Optuna). |
| `"boruta"` | Only GreedyBoruta filter. |
| `None` | No feature selection (identity). |
| `list` | Optuna picks from the provided list, e.g. `["pvalue", "boruta", None]`. |

A `VarianceThreshold` step always precedes the feature selector in the pipeline (threshold is also optimized).

**GreedyBoruta hyperparameters** (when `hyperparameters="optimize"`):

- `perc` — percentile threshold (80–100, step 10)
- `boruta_max_leaf_nodes` — max leaf nodes of the internal RF (10–50, step 10)

---

## Scalers

Controlled by the `scaler` parameter.

| Value | Behaviour |
|---|---|
| `"optimize"` | Optuna selects among `None`, `"standard"`, `"minmax"`, `"robust"`. |
| `"standard"` | `StandardScaler` (`with_mean` and `with_std` optimized). |
| `"minmax"` | `MinMaxScaler` (feature range optimized: `(0,1)` or `(-1,1)`). |
| `"robust"` | `RobustScaler` (centering, scaling, quantile range optimized). |
| `None` | No scaling (identity). |
| `list` | Optuna picks from the provided list, e.g. `["standard", "robust"]`. |

---

## Algorithms

Controlled by the `algorithm` parameter.

| Key | Model |
|---|---|
| `"LSVM"` | `SVC` with linear or polynomial kernel |
| `"RBFSVM"` | `SVC` with RBF kernel |
| `"NuLSVM"` | `NuSVC` with linear or polynomial kernel |
| `"NuRBFSVM"` | `NuSVC` with RBF kernel |
| `"SGD"` | `SGDClassifier` |
| `"MLP"` | `MLPClassifier` |
| `"RF"` | `RandomForestClassifier` |
| `"ET"` | `ExtraTreesClassifier` |
| `"LGBM"` | `LGBMClassifier` |
| `"XGB"` | `XGBClassifier` |
| `"CatB"` | `CatBoostClassifier` |
| `"KNN"` | `KNeighborsClassifier` |
| `"ENet"` | `LogisticRegression` (Elastic Net, saga solver) |
| `"PLR"` | `LogisticRegression` (L2 primal) |
| `"DLR"` | `LogisticRegression` (L2 dual, liblinear) |
| `"Ridge"` | `RidgeClassifier` |
| `"LDA"` | `LinearDiscriminantAnalysis` |

Pass `algorithm="optimize"` to let Optuna select among all of the above, or pass a list to restrict the search space:

```python
magic_now(..., algorithm=["LGBM", "RF", "RBFSVM"])
```

All classifiers are instantiated with `class_weight="balanced"` to handle class imbalance.

---

## Usage examples

### Two-stage workflow (recommended)

```python
from pathlib import Path
import pandas as pd
from haramo.classification import magic_now

X = pd.read_csv("features.csv", index_col=0)
y = pd.read_csv("labels.csv", index_col=0).squeeze()

# Stage 1: exhaustive grid over all pipeline structures
_, _, studies = magic_now(
    X=X,
    y=y,
    algorithm="optimize",       # all 15 algorithms
    scaler="optimize",          # all 4 scalers
    feature_selector="optimize", # all 4 feature selector combinations
    hyperparameters="default",
    output_dir=Path("results/stage1"),
)

# Inspect top 3 structures from fold 1
# Params keys: "algorithm", "scaling_method", "add_pvalue_filter", "add_boruta_filter"
top3 = sorted(studies["fold_1"].trials, key=lambda t: t.value, reverse=True)[:3]
for i, t in enumerate(top3):
    print(f"#{i+1} score={t.value:.4f}  params={t.params}")

# Stage 2: run the top 3 structures with full hyperparameter optimization
# Repeat for each of the top 3 identified structures
for i, trial in enumerate(top3):
    p = trial.params
    # Map add_pvalue_filter / add_boruta_filter back to feature_selector
    if p.get("add_pvalue_filter") and p.get("add_boruta_filter"):
        fs = "optimize"
    elif p.get("add_pvalue_filter"):
        fs = "pvalue"
    elif p.get("add_boruta_filter"):
        fs = "boruta"
    else:
        fs = None

    validation, pipeline, studies2 = magic_now(
        X=X,
        y=y,
        algorithm=p["algorithm"],
        scaler=p.get("scaling_method"),
        feature_selector=fs,
        hyperparameters="optimize",
        n_trials=200,
        output_dir=Path("results/stage2"),
        tag=f"_top{i+1}",
    )
    print(f"\nTop {i+1} — {p['algorithm']} / {p.get('scaling_method')} / fs={fs}")
    print(validation)
```

### Restrict search space in stage 1

```python
# Only consider a subset of algorithms
_, _, studies = magic_now(
    X=X,
    y=y,
    algorithm=["LGBM", "RF", "RBFSVM", "ENet"],
    scaler="optimize",
    feature_selector="optimize",
    hyperparameters="default",
    output_dir=Path("results/stage1_subset"),
)
```

### Group-aware cross-validation (e.g. repeated measures, patients)

```python
# Prevents samples from the same group appearing in both train and test splits
_, _, studies = magic_now(
    X=X,
    y=y,
    groups=subject_ids,        # array-like of group labels, same length as y
    hyperparameters="default",
    output_dir=Path("results/stage1_grouped"),
)
```

### Custom scoring metric

```python
from sklearn.metrics import make_scorer, matthews_corrcoef

mcc_scorer = make_scorer(matthews_corrcoef)

validation, pipeline, studies = magic_now(
    X=X,
    y=y,
    algorithm="LGBM",
    scaler="robust",
    feature_selector="boruta",
    scoring=mcc_scorer,
    hyperparameters="optimize",
    n_trials=200,
    output_dir=Path("results/stage2_mcc"),
)
```

---

## Output metrics

The `validation` DataFrame contains one row per outer fold:

| Metric | Description |
|---|---|
| MCC | Matthews Correlation Coefficient |
| F1-score | Binary F1 score |
| Kappa | Cohen's Kappa |
| Bal. Acc. | Balanced accuracy |
| Precision | Positive predictive value |
| Sensitivity | True positive rate (recall) |
| Selectivity | True negative rate |

---

## License

MIT License. See `LICENSE` for details.
