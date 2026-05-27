# -*- coding: utf-8 -*-

###########
# Imports #
###########

import pandas as pd
import numpy as np

import pickle

from joblib import Parallel, delayed
from typing import Union, Callable, Dict, List
from os import PathLike

from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
)

from math import sqrt

from sklearn.pipeline import Pipeline

from optuna import (
    Trial,
    create_study,
)

from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import GridSampler

from ._instantiation import instantiate_pipeline
from ..utils import (
    classification_report,
    reduce_dataset,
    resolve_scorer,
    scoring_to_metric_column,
    metric_is_lower_better,
    plot_pr_curve,
    plot_roc_curve,
    plot_ks_statistic,
    fit_score_calibrator,
    pick_best_score_calibration_method,
    find_best_threshold,
    CalibratedThresholdedClassifier,
)
from ..feature_selection import select_best_dataset_combo, select_best_feature_selector

#############
# Functions #
#############


def _balanced_sample_weight(y, pos_weight_factor: float = 1.0) -> pd.Series:
    """Compute sklearn's "balanced" sample weights, then scale the positive
    class by ``pos_weight_factor``.

    With ``pos_weight_factor=1.0`` the result is identical to
    ``compute_sample_weight(class_weight="balanced", y=y)``. Higher values
    push the model to weight false negatives more heavily; lower values
    relax the balanced correction toward the natural class prior.
    """
    sw = pd.Series(compute_sample_weight(class_weight="balanced", y=y), index=y.index)
    if pos_weight_factor != 1.0:
        pos_mask = y == 1
        sw.loc[pos_mask] = sw.loc[pos_mask] * float(pos_weight_factor)
    return sw


def _score_fold(
    pipeline,
    X,
    y,
    scoring,
    train_index,
    test_index,
    sample_weight: Union[np.ndarray, pd.DataFrame] = None,
    random_state: int = 42,
    reduced_train_index=None,
):
    """Fit a cloned pipeline on one CV fold and return the score."""
    pipe = clone(pipeline)
    scorer = resolve_scorer(scoring)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    if reduced_train_index is None:
        reduced_index = reduce_dataset(
            X=X_train,
            y=y_train,
            target_size=2000,
            stage2_shrink=1,
            class_weight="balanced",
            random_state=random_state,
            verbose=False,
        )
    else:
        reduced_index = reduced_train_index

    X_reduced = X_train.loc[reduced_index]
    y_reduced = y_train.loc[reduced_index]

    if sample_weight is not None:
        w_train = sample_weight.loc[y_train.index]
        sample_weight_reduced = w_train.loc[reduced_index]
        try:
            pipe.fit(X_reduced, y_reduced, model__sample_weight=sample_weight_reduced)
        except:
            pipe.fit(X_reduced, y_reduced)
    else:
        pipe.fit(X_reduced, y_reduced)

    return scorer(pipe, X_test, y_test)


def pipeline_cross_val(
    pipeline,
    X,
    y,
    scoring,
    cv,
    sample_weight: Union[np.ndarray, pd.DataFrame] = None,
    random_state: int = 42,
    n_jobs: int = 1,
    pre_reduced_indices=None,
):
    """
    Custom cross-validation function that maintains DataFrame format.

    Parameters:
    -----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : pd.DataFrame
        The data to fit.
    y : pd.Series
        The target variable to try to predict in the case of supervised learning.
    scoring : Union[str, callable]
        A scorer callable object / function with signature scorer(estimator, X, y) or a string.
    cv : iterable
        Cross-validation splitting strategy.
    sample_weight : pd.Series, optional
        Sample weights to be used in training.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs for cross-validation folds.

    Returns:
    --------
    scores : list of float
        Array of scores of the estimator for each run of the cross-validation.
    """
    splits = list(cv)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_score_fold)(
            pipeline,
            X,
            y,
            scoring,
            train_idx,
            test_idx,
            sample_weight,
            random_state,
            reduced_train_index=(
                pre_reduced_indices[i] if pre_reduced_indices is not None else None
            ),
        )
        for i, (train_idx, test_idx) in enumerate(splits)
    )

    return scores


def objective(
    trial: Trial,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series, list],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series, list],
    sample_weight: Union[np.ndarray, pd.DataFrame] = None,
    scoring: Union[str, Callable] = "balanced_accuracy",
    task: str = "classification",
    feature_selector: Union[str, list] = "optimize",
    scaler: Union[str, list] = "optimize",
    algorithm: Union[str, list] = "optimize",
    hyperparameters: str = "optimize",
    random_state: int = 42,
    n_cv_jobs: int = 1,
    model_jobs: int = 1,
    inner_cv_groups: Union[np.ndarray, pd.Series, list] = None,
    inner_splits=None,
    inner_reduced_indices=None,
):
    """
    Objective function for hyperparameter optimization using cross-validation.
    Parameters:
    -----------
    trial : Trial
        An Optuna trial object for suggesting hyperparameters.
    X : Union[np.ndarray, pd.DataFrame],
        Feature matrix for training the model.
    y : Union[np.ndarray, pd.Series, list],
        Target vector for training the model.
    scoring : Union[str, Callable], default="balanced_accuracy"
        Scoring method to evaluate the predictions on the test set.
    algorithm : str, default="LSVM"
        The machine learning algorithm to be used.
    random_state : int, default=42
        Random seed for reproducibility.
    sample_weight : Union[np.ndarray, pd.DataFrame], optional
        Sample weights to be used in training.
    n_cv_jobs : int, default=1
        Number of parallel jobs for inner cross-validation folds.
    model_jobs : int, default=1
        Number of parallel jobs for the model's native parallelism.
    inner_cv_groups : Union[np.ndarray, pd.Series, list], optional
        Group labels for StratifiedGroupKFold in the inner CV. When provided,
        inner CV ensures no group appears in both train and test folds, so
        hyperparameters are optimized for generalization to unseen groups.
    Returns:
    --------
    float
        Mean cross-validation score.
    """

    pipeline = instantiate_pipeline(
        trial,
        task=task,
        feature_selector=feature_selector,
        scaler=scaler,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        random_state=random_state,
        n_jobs=model_jobs,
    )

    if inner_splits is not None:
        cv_iter = inner_splits
    elif inner_cv_groups is not None:
        cv_iter = StratifiedGroupKFold(n_splits=3).split(
            X_train, y_train.astype("str"), groups=inner_cv_groups
        )
    else:
        cv_iter = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=random_state
        ).split(X_train, y_train.astype("str"))

    scores = pipeline_cross_val(
        pipeline,
        X=X_train,
        y=y_train,
        sample_weight=sample_weight,
        scoring=scoring,
        cv=cv_iter,
        random_state=random_state,
        n_jobs=n_cv_jobs,
        pre_reduced_indices=inner_reduced_indices,
    )

    scores = pd.Series(scores, dtype=object).fillna(0.01).tolist()

    return float(np.mean(scores))


def get_search_space(feature_selector, scaler, algorithm):
    search_space = {}
    n_trials = 1

    if feature_selector == "optimize":
        search_space["add_pvalue_filter"] = [True, False]
        search_space["add_boruta_filter"] = [True, False]
        n_trials *= len(search_space["add_pvalue_filter"]) * len(
            search_space["add_boruta_filter"]
        )
    elif isinstance(feature_selector, list):
        search_space["feature_selection_method"] = feature_selector
        n_trials *= len(feature_selector)

    if scaler == "optimize":
        search_space["scaling_method"] = [None, "standard", "minmax", "robust"]
        n_trials *= len(search_space["scaling_method"])
    elif isinstance(scaler, list):
        search_space["scaling_method"] = scaler
        n_trials *= len(scaler)

    if algorithm == "optimize":
        search_space["algorithm"] = [
            "LSVM",
            "RBFSVM",
            "NuLSVM",
            "NuRBFSVM",
            "SGD",
            "MLP",
            "RF",
            "ET",
            "LGBM",
            "KNN",
            "ENet",
            "PLR",
            "DLR",
            "Ridge",
            "LDA",
        ]
        n_trials *= len(search_space["algorithm"])
    elif isinstance(algorithm, list):
        search_space["algorithm"] = algorithm
        n_trials *= len(algorithm)

    return search_space, n_trials


def _train_fold(
    fold_idx,
    X,
    y,
    sample_weight,
    train_index,
    test_index,
    scoring,
    task,
    feature_selector,
    scaler,
    algorithm,
    hyperparameters,
    random_state,
    n_trials,
    n_cv_jobs,
    inner_cv_groups=None,
):
    """
    Run two-phase Optuna optimisation for a single outer fold.

    Phase 1 – Feature-selection HPO
        A small study (5 / 10 / 15 trials) finds the best feature-selection
        configuration using a fixed default pipeline (StandardScaler + LGBM).
        All available CPUs are forwarded to boruta's random-forest so the main
        bottleneck benefits fully from parallelism.  The winning selector is
        fitted on X_train and used to transform both splits before phase 2.

    Phase 2 – Scaler + model HPO
        The main study optimises scaler and algorithm hyperparameters on the
        already-selected feature matrix (feature_selector=None), so the search
        space and compute are entirely dedicated to the model.

    The final pipeline merges both phases as a standard sklearn Pipeline so
    nested_crossval can clone and refit it correctly on fresh splits.
    """

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    w_train = sample_weight.loc[y_train.index]
    if inner_cv_groups is not None:
        groups_s = (
            inner_cv_groups
            if isinstance(inner_cv_groups, pd.Series)
            else pd.Series(inner_cv_groups, index=y.index)
        )
        groups_train = groups_s.iloc[train_index]
    else:
        groups_train = None

    # ------------------------------------------------------------------ #
    # Phase 1 – feature-selection HPO                                    #
    # n_cv_jobs flows entirely to boruta's RF; trials run sequentially.  #
    # ------------------------------------------------------------------ #

    fs_pipeline = select_best_feature_selector(
        X_train=X_train,
        y_train=y_train,
        feature_selector=feature_selector,
        task=task,
        scoring=scoring,
        random_state=random_state,
        inner_cv_groups=groups_train,
        n_jobs=n_cv_jobs,
    )
    X_train_sel = fs_pipeline.transform(X_train)
    X_test_sel = fs_pipeline.transform(X_test)

    # ------------------------------------------------------------------- #
    # Phase 2 – scaler + model HPO on pre-selected features               #
    # feature_selector=None: X is already filtered, no FS in search space #
    # ------------------------------------------------------------------- #
    if hyperparameters == "default":
        search_space, n_trials = get_search_space(None, scaler, algorithm)
        sampler = GridSampler(search_space)
    else:
        sampler = TPESampler(
            seed=random_state,
            multivariate=True,
        )

    study = create_study(
        direction="maximize",
        pruner=SuccessiveHalvingPruner(reduction_factor=2),
        sampler=sampler,
    )

    if n_cv_jobs < 2:
        _model_jobs = n_cv_jobs
        _inner_jobs = 1
    elif n_cv_jobs == 3:
        _model_jobs = 1
        _inner_jobs = n_cv_jobs
    elif n_cv_jobs < 6:
        _model_jobs = n_cv_jobs
        _inner_jobs = 1
    else:
        _model_jobs = n_cv_jobs // 3
        _inner_jobs = 3

    # Pre-compute inner CV splits and reduced indices once; reused across all trials
    if groups_train is not None:
        _p2_splits = list(
            StratifiedGroupKFold(n_splits=3).split(
                X_train_sel, y_train.astype("str"), groups=groups_train
            )
        )
    else:
        _p2_splits = list(
            StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state).split(
                X_train_sel, y_train.astype("str")
            )
        )
    _p2_reduced = [
        reduce_dataset(
            X=X_train_sel.iloc[tr],
            y=y_train.iloc[tr],
            target_size=2000,
            stage2_shrink=1,
            class_weight="balanced",
            random_state=random_state,
            verbose=False,
        )
        for tr, _ in _p2_splits
    ]

    study.optimize(
        lambda trial: objective(
            trial,
            X_train=X_train_sel,
            y_train=y_train,
            X_test=X_test_sel,
            y_test=y_test,
            sample_weight=w_train,
            scoring=scoring,
            task=task,
            feature_selector=None,  # already applied in phase 1
            scaler=scaler,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            random_state=random_state,
            n_cv_jobs=_inner_jobs,
            model_jobs=_model_jobs,
            inner_cv_groups=groups_train,
            inner_splits=_p2_splits,
            inner_reduced_indices=_p2_reduced,
        ),
        n_trials=n_trials,
        n_jobs=1,
    )

    # ------------------------------------------------------------------ #
    # Build final pipeline: FS steps (phase 1) + scaler + model (phase 2)#
    # ------------------------------------------------------------------ #
    phase2_pipeline = instantiate_pipeline(
        trial=study.best_trial,
        feature_selector=None,
        scaler=scaler,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )
    final_pipeline = Pipeline(
        fs_pipeline.steps
        + [s for s in phase2_pipeline.steps if s[0] in ("scaler", "model")]
    )

    # ------------------------------------------------------------------ #
    # Inner-OOF pct sweep — drives all downstream supervised selection   #
    # (best_pct, calibration method, calibrator fit, threshold) without  #
    # ever touching this fold's outer-test rows. Reuses `_p2_splits`     #
    # (3-fold CV inside this outer fold's training data) so the inner    #
    # CV grid matches HPO's.                                              #
    # ------------------------------------------------------------------ #
    _all_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    _min_inner_train = min(len(tr) for tr, _ in _p2_splits)
    _MIN_TRAIN = 2000
    _MAX_TRAIN = 10_000

    if _min_inner_train < _MIN_TRAIN:
        _inner_pcts = [0.8, 0.9, 1.0]
    else:
        _inner_pcts = [
            p for p in _all_percentages if int(p * _min_inner_train) >= _MIN_TRAIN
        ]
    # Universal max-train cap (applies to every algorithm).
    _inner_pcts = [p for p in _inner_pcts if int(p * _min_inner_train) <= _MAX_TRAIN]
    if not _inner_pcts:
        _inner_pcts = [round(_MAX_TRAIN / _min_inner_train, 4)]
    _inner_pcts = sorted(_inner_pcts)

    # Early stopping criterion — same metric the HPO optimised.
    _metric_col = scoring_to_metric_column(scoring, default="MCC")
    _lower_better = metric_is_lower_better(_metric_col)
    _is_better = (
        (lambda new, cur: new <= cur)
        if _lower_better
        else (lambda new, cur: new >= cur)
    )
    _worst_init = float("inf") if _lower_better else float("-inf")
    _cmp = ">" if _lower_better else "<"

    _best_score = _worst_init
    _flagged = False
    _active = True

    inner_oof_per_pct = {}
    for _pct in _inner_pcts:
        if not _active:
            break
        _rows = []
        for _inner_train_idx, _inner_test_idx in _p2_splits:
            _X_tr_full = X_train.iloc[_inner_train_idx]
            _y_tr_full = y_train.iloc[_inner_train_idx]
            _w_tr_full = w_train.iloc[_inner_train_idx]
            _X_te = X_train.iloc[_inner_test_idx]
            _y_te = y_train.iloc[_inner_test_idx]

            if _pct < 1.0:
                _red_idx = reduce_dataset(
                    X=_X_tr_full,
                    y=_y_tr_full,
                    target_size=int(_pct * len(_inner_train_idx)),
                    stage2_shrink=1,
                    class_weight="balanced",
                    random_state=random_state,
                    verbose=False,
                )
                _X_tr = _X_tr_full.loc[_red_idx]
                _y_tr = _y_tr_full.loc[_red_idx]
                _w_tr = _w_tr_full.loc[_red_idx]
            else:
                _X_tr, _y_tr, _w_tr = _X_tr_full, _y_tr_full, _w_tr_full

            _pipe = clone(final_pipeline)
            try:
                _pipe.fit(_X_tr, _y_tr, model__sample_weight=_w_tr)
            except Exception:
                _pipe.fit(_X_tr, _y_tr)

            _predicted = _pipe.predict(_X_te)
            if hasattr(_pipe, "predict_proba"):
                _proba = _pipe.predict_proba(_X_te)
                _score = (
                    _proba[:, 1]
                    if _proba.ndim == 2 and _proba.shape[1] >= 2
                    else _proba.ravel()
                )
            elif hasattr(_pipe, "decision_function"):
                _score = _pipe.decision_function(_X_te)
            else:
                _score = np.full(len(_y_te), np.nan)

            _rows.append(
                pd.DataFrame(
                    {
                        "true": np.asarray(_y_te),
                        "predicted": _predicted,
                        "score": _score,
                    },
                    index=_y_te.index,
                )
            )
        _oof_df = pd.concat(_rows, axis=0)
        inner_oof_per_pct[_pct] = _oof_df

        # 2-strike early stop on the chosen scoring metric.
        _rep = classification_report(
            _oof_df["true"],
            _oof_df["predicted"],
            y_score=_oof_df["score"] if "score" in _oof_df.columns else None,
        )
        _score_val = _rep[_metric_col]
        if not _flagged:
            if _is_better(_score_val, _best_score):
                _best_score = _score_val
            else:
                _flagged = True
                print(
                    f"[_train_fold {fold_idx}] dip at {int(_pct * 100)}% "
                    f"({_metric_col}={_score_val:.4f} {_cmp} best={_best_score:.4f})"
                    " — giving one more chance …"
                )
        else:
            if _is_better(_score_val, _best_score):
                _flagged = False
                _best_score = _score_val
            else:
                _active = False
                print(
                    f"[_train_fold {fold_idx}] early stop at {int(_pct * 100)}% "
                    f"({_metric_col}={_score_val:.4f} {_cmp} best={_best_score:.4f})"
                )

    fold_name = f"fold_{fold_idx}"
    return fold_name, final_pipeline, study, inner_oof_per_pct


def train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, list],
    scoring: Union[str, Callable] = "PR AUC",
    task: str = "classification",
    feature_selector: Union[str, list] = "optimize",
    scaler: Union[str, list] = "optimize",
    algorithm: Union[str, list] = "optimize",
    hyperparameters: str = "optimize",
    random_state: int = 42,
    n_trials: int = 100,
    outer_cv_groups: Union[np.ndarray, pd.Series, list] = None,
    inner_cv_groups: Union[np.ndarray, pd.Series, list] = None,
    n_jobs: int = 16,
    pos_weight_factor: float = 1.0,
):
    """
    Train a model using stratified k-fold cross-validation and hyperparameter optimization.

    Parameters:
    -----------
    X : Union[np.ndarray, pd.DataFrame],
        Feature matrix. Can be a NumPy array or a pandas DataFrame.
    y : Union[np.ndarray, pd.Series, list],
        Target vector. Can be a NumPy array, pandas Series, or a list.
    scoring : Union[str, Callable], default="PR AUC"
        Scoring method to evaluate the predictions on the test set.
        Internal aliases:  "PR AUC", "ROC AUC", "MCC". Any sklearn
        scorer string or a callable is also accepted.
    task : str, default="classification"
        The type of task to perform. Currently supports "classification".
    feature_selector : Union[str, list], default="optimize"
        Feature selection method(s) to be used.
    scaler : Union[str, list], default="optimize"
        Scaling method(s) to be used.
    algorithm : Union[str, list], default="optimize"
        Algorithm(s) to be used.
    hyperparameters : str, default="optimize"
        Hyperparameters to be optimized.
    random_state : int, default=42
        Random seed for reproducibility.
    n_trials : int, default=100
        Number of trials for hyperparameter optimization. igored if hyperparameters = "default"
    n_jobs : int, default=16
        Total number of parallel CPUs. Split as 4 outer folds × (n_jobs // 4) inner CV jobs.

    Returns:
    --------
    models : dict[str, Pipeline]
        One unfitted final pipeline per outer fold, keyed by ``"fold_{i}"``.
    studies : dict[str, optuna.Study]
        Phase-2 Optuna study per outer fold.
    inner_oof : dict[str, dict[float, pd.DataFrame]]
        Per outer fold: ``{pct: DataFrame[true, predicted, score]}`` indexed by
        outer-train row indices. Produced by an inner-CV pct sweep on the
        winning pipeline and used by ``nested_crossval`` / ``magic_now`` as the
        honest selection-OOF set (never touches the outer-test rows).
    """

    sample_weight = _balanced_sample_weight(y, pos_weight_factor=pos_weight_factor)

    if outer_cv_groups is not None:
        strat_kfold_outer = StratifiedGroupKFold(n_splits=4)
    else:
        strat_kfold_outer = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=random_state,
        )

    splits = list(strat_kfold_outer.split(X, y.astype("str"), groups=outer_cv_groups))
    n_outer_folds = len(splits)
    n_cv_jobs = max(1, n_jobs // n_outer_folds)

    # When `algorithm` is a list, Optuna handles it as a categorical search
    # space inside the single per-fold study (see ``instantiate_pipeline``).
    flat = Parallel(n_jobs=n_outer_folds)(
        delayed(_train_fold)(
            fold_idx=fold_idx,
            X=X,
            y=y,
            sample_weight=sample_weight,
            train_index=train_index,
            test_index=test_index,
            scoring=scoring,
            task=task,
            feature_selector=feature_selector,
            scaler=scaler,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            random_state=random_state,
            n_trials=n_trials,
            n_cv_jobs=n_cv_jobs,
            inner_cv_groups=inner_cv_groups,
        )
        for fold_idx, (train_index, test_index) in enumerate(splits, start=1)
    )

    models = {name: model for name, model, _, _ in flat}
    studies = {name: study for name, _, study, _ in flat}
    inner_oof = {name: oof for name, _, _, oof in flat}

    return models, studies, inner_oof


def _fit_and_predict(
    pipeline, X, y, sample_weight, train_index, test_index, reduced_train_index
):
    """Fit a cloned pipeline on a (possibly reduced) split and return predictions.

    Parameters
    ----------
    reduced_train_index : pandas Index
        Pre-computed subset of training rows to fit on.  Pass the full
        training index for no reduction (100 %).  The same index is shared
        across all pipelines for a given (split, pct) so that reduction noise
        does not confound model comparisons.  Scoring always uses the full
        test split.
    """
    pipe = clone(pipeline)
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    X_train = X.loc[reduced_train_index]
    y_train = y.loc[reduced_train_index]
    w_train = sample_weight.loc[reduced_train_index]

    try:
        pipe.fit(X_train, y_train, model__sample_weight=w_train)
    except Exception:
        pipe.fit(X_train, y_train)

    predicted = pipe.predict(X_test)

    # Positive-class score for AUC metrics — mirrors the predict_proba /
    # decision_function fallback in utils/_dataset_reducer.py.
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
        score = (
            proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        )
    elif hasattr(pipe, "decision_function"):
        score = pipe.decision_function(X_test)
    else:
        score = np.full(len(y_test), np.nan)

    return pd.DataFrame(
        {"true": np.asarray(y_test), "predicted": predicted, "score": score},
        index=y_test.index,
    )


def _refit_pipeline(
    pipeline, X, y, sample_weight, pct=1.0, random_state=42, max_svm_samples=10_000
):
    """Refit a cloned pipeline, optionally on a reduced dataset.

    Parameters
    ----------
    pct : float, default 1.0
        Fraction of rows to use (the winning reduction percentage from
        nested_crossval).  When 1.0 the full dataset is used.
    max_svm_samples : int, default 10_000
        Hard cap on training size for kernel SVM pipelines (O(n²) models).
    """
    model = clone(pipeline)

    model_step = model.named_steps.get("model")
    is_svm = model_step is not None and hasattr(model_step, "kernel")

    target = int(pct * len(X))
    if is_svm:
        target = min(target, max_svm_samples)

    if target < len(X):
        final_index = reduce_dataset(
            X=X,
            y=y,
            target_size=target,
            stage2_shrink=1,
            class_weight="balanced",
            random_state=random_state,
            verbose=False,
        )
        X_fit = X.loc[final_index]
        y_fit = y.loc[final_index]
        w_fit = sample_weight.loc[final_index]
    else:
        X_fit, y_fit, w_fit = X, y, sample_weight

    try:
        model.fit(X_fit, y_fit, model__sample_weight=w_fit)
    except Exception:
        model.fit(X_fit, y_fit)
    return model


def nested_crossval(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, list],
    pipelines: dict,
    inner_oof: dict,
    scoring: Union[str, Callable] = "MCC",
    random_state: int = 42,
    cfc_random_state: int = 2024,
    outer_cv_groups: Union[np.ndarray, pd.Series, list] = None,
    n_jobs: int = 16,
    max_svm_samples: int = 10_000,
    pos_weight_factor: float = 1.0,
):
    """
    Perform nested cross-validation jointly optimising over trained model
    (fold), dataset reduction percentage, and — when requested — algorithm.

    Reduction sizes
    ---------------
    The pct sweep is performed in ``_train_fold`` on inner CV (see its
    inner-OOF block); ``nested_crossval`` only evaluates at each fold's
    chosen ``best_pct``.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    pipelines : dict
        Mapping ``"fold_{i}"`` → fitted Pipeline (one per outer fold).
    inner_oof : dict[str, dict[float, pd.DataFrame]]
        Inner-OOF predictions from ``train()`` — ``{fold_key: {pct: df}}``,
        each df indexed by outer-train rows with columns
        ``["true", "predicted", "score"]``. Drives best_pct selection and
        is forwarded to ``magic_now`` for calibrator/threshold fitting,
        keeping every supervised choice clean of the outer-test fold.
    scoring : str or callable, default ``"MCC"``
        Metric used to pick each fold's best reduction pct from the inner
        OOF. CFC + TTS metrics are reported in the validation table but
        never drive selection.
    random_state : int, default 42
        Drives the TTS splits, which match train()'s outer CV grid.
    cfc_random_state : int, default 2024
        Independent fixed seed for the CFC evaluation grid.
    outer_cv_groups : array-like, optional
        Group labels for the outer StratifiedGroupKFold (used by both
        TTS and CFC grids).
    n_jobs : int, default 16

    Returns
    -------
    validation : pd.DataFrame
        Per-fold classification report at each fold's chosen ``best_pct``.
        MultiIndex (fold, reduction); every metric column appears with both
        ``" CFC"`` and ``" TTS"`` suffixes.
    best_pipelines : dict[str, Pipeline]
        One refitted pipeline per fold_key, at its own best pct, fit on
        full ``X``.
    best_predictions : dict[str, dict[str, pd.DataFrame]]
        ``{fold_key: {"inner": ..., "cfc": ..., "tts": ...}}``. ``inner`` is
        the inner-OOF DataFrame at the chosen pct (clean of outer-test);
        ``cfc`` and ``tts`` are evaluation predictions at the chosen pct.
    reduction_validation : pd.DataFrame
        Inner-OOF classification report at every ``(fold, pct)`` evaluated.
        Diagnostic view of the reduction-pct selection.
    """
    sample_weight = _balanced_sample_weight(y, pos_weight_factor=pos_weight_factor)

    # The early-stopping criterion and the final (fold, pct) ranking use
    # the same metric the HPO optimised. Falls back to MCC for callable
    # scorers or unrecognised strings.
    metric_col = scoring_to_metric_column(scoring, default="MCC")
    lower_better = metric_is_lower_better(metric_col)
    _idxbest = (lambda s: s.idxmin()) if lower_better else (lambda s: s.idxmax())

    # TTS splits — identical to train()'s outer CV. fold_key "fold_{i}"
    # was originally held out from tts_splits[i-1][1].
    if outer_cv_groups is not None:
        tts_kf = StratifiedGroupKFold(n_splits=4)
    else:
        tts_kf = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=random_state,
        )
    tts_splits = list(tts_kf.split(X, y.astype("str"), groups=outer_cv_groups))

    # CFC splits — separate fixed seed so the evaluation grid is independent
    # of train()'s outer CV.
    if outer_cv_groups is not None:
        cfc_kf = StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=cfc_random_state
        )
    else:
        cfc_kf = StratifiedKFold(
            n_splits=4, shuffle=True, random_state=cfc_random_state
        )
    cfc_splits = list(cfc_kf.split(X, y.astype("str"), groups=outer_cv_groups))
    n_cfc_splits = len(cfc_splits)

    def _tts_split_idx(fold_key):
        """0-indexed outer-fold position from a fold_key like ``"fold_3"``."""
        return int(fold_key.split("_")[1]) - 1

    # ------------------------------------------------------------------ #
    # Step 1 — pick best_pct per fold from inner OOF. Honest selector:   #
    # inner OOF lives inside outer-train, never touches TTS.             #
    # ------------------------------------------------------------------ #
    best_pct_per_fold: dict = {}
    inner_reports: dict = {}
    for fold_key, pct_to_df in inner_oof.items():
        pct_scores = {}
        for pct, df in pct_to_df.items():
            rep = classification_report(
                df["true"],
                df["predicted"],
                y_score=df["score"] if "score" in df.columns else None,
            )
            inner_reports[(fold_key, pct)] = rep
            pct_scores[pct] = rep[metric_col]
        best_pct = _idxbest(pd.Series(pct_scores))
        best_pct_per_fold[fold_key] = best_pct
        print(
            f"[nested_crossval] {fold_key}: best reduction={int(best_pct * 100)}% "
            f"(inner-OOF {metric_col}={pct_scores[best_pct]:.4f})"
        )

    reduction_validation = pd.DataFrame(
        list(inner_reports.values()),
        index=pd.MultiIndex.from_tuples(
            list(inner_reports.keys()), names=["fold", "reduction"]
        ),
    )

    # ------------------------------------------------------------------ #
    # Step 2 — refit+predict at each fold's best_pct on CFC and TTS.     #
    # CFC: n_cfc_splits fits per fold_key (~16 total).                   #
    # TTS: 1 fit per fold_key (~4 total).                                #
    # ------------------------------------------------------------------ #
    cfc_reduced: dict = {}
    tts_reduced: dict = {}

    def _reduce_for(train_index, pct, key, store):
        if (key, pct) in store:
            return
        X_train_ = X.iloc[train_index]
        y_train_ = y.iloc[train_index]
        if pct == 1.0:
            store[(key, pct)] = X_train_.index
        else:
            store[(key, pct)] = reduce_dataset(
                X=X_train_,
                y=y_train_,
                target_size=int(pct * len(train_index)),
                stage2_shrink=1,
                class_weight="balanced",
                random_state=random_state,
                verbose=False,
            )

    for fk in pipelines:
        pct = best_pct_per_fold[fk]
        for cfc_si in range(n_cfc_splits):
            _reduce_for(cfc_splits[cfc_si][0], pct, cfc_si, cfc_reduced)
        tts_si = _tts_split_idx(fk)
        _reduce_for(tts_splits[tts_si][0], pct, tts_si, tts_reduced)

    cfc_tasks = [(fk, cfc_si) for fk in pipelines for cfc_si in range(n_cfc_splits)]
    tts_tasks = [(fk, _tts_split_idx(fk)) for fk in pipelines]

    cfc_delayed = [
        delayed(_fit_and_predict)(
            pipelines[fk],
            X,
            y,
            sample_weight,
            cfc_splits[cfc_si][0],
            cfc_splits[cfc_si][1],
            cfc_reduced[(cfc_si, best_pct_per_fold[fk])],
        )
        for fk, cfc_si in cfc_tasks
    ]
    tts_delayed = [
        delayed(_fit_and_predict)(
            pipelines[fk],
            X,
            y,
            sample_weight,
            tts_splits[tts_si][0],
            tts_splits[tts_si][1],
            tts_reduced[(tts_si, best_pct_per_fold[fk])],
        )
        for fk, tts_si in tts_tasks
    ]
    results = Parallel(n_jobs=n_jobs)(cfc_delayed + tts_delayed)
    n_cfc = len(cfc_tasks)

    pred_store_cfc: dict = {}
    for (fk, cfc_si), df in zip(cfc_tasks, results[:n_cfc]):
        pred_store_cfc[(fk, cfc_si)] = df
    pred_store_tts: dict = {}
    for (fk, _), df in zip(tts_tasks, results[n_cfc:]):
        pred_store_tts[fk] = df

    # ------------------------------------------------------------------ #
    # Step 3 — refit deployment pipeline on full X at each fold's pct.   #
    # ------------------------------------------------------------------ #
    best_pipelines: dict = {}
    best_predictions: dict = {}
    for fk in pipelines:
        pct = best_pct_per_fold[fk]
        print(
            f"[nested_crossval] {fk}: refitting on full data at " f"{int(pct * 100)}%"
        )
        best_pipelines[fk] = _refit_pipeline(
            pipelines[fk],
            X,
            y,
            sample_weight,
            pct=pct,
            random_state=random_state,
            max_svm_samples=max_svm_samples,
        )
        cfc_by_split = {si: pred_store_cfc[(fk, si)] for si in range(n_cfc_splits)}
        cfc_oof = pd.concat([cfc_by_split[si] for si in range(n_cfc_splits)], axis=0)
        best_predictions[fk] = {
            "inner": inner_oof[fk][pct],
            "cfc": cfc_oof,
            "cfc_by_split": cfc_by_split,
            "tts": pred_store_tts[fk],
        }

    # ------------------------------------------------------------------ #
    # Step 4 — validation table: raw CFC + TTS at each fold's pct.       #
    # (Calibration / threshold variants live in magic_now.)              #
    # ------------------------------------------------------------------ #
    validation_rows: list = []
    validation_index: list = []
    for fk in pipelines:
        pct = best_pct_per_fold[fk]
        cfc_oof = best_predictions[fk]["cfc"]
        tts_oof = best_predictions[fk]["tts"]
        cfc_rep = classification_report(
            cfc_oof["true"],
            cfc_oof["predicted"],
            y_score=cfc_oof["score"] if "score" in cfc_oof.columns else None,
        )
        tts_rep = classification_report(
            tts_oof["true"],
            tts_oof["predicted"],
            y_score=tts_oof["score"] if "score" in tts_oof.columns else None,
        )
        combined = pd.concat([cfc_rep.add_suffix(" CFC"), tts_rep.add_suffix(" TTS")])
        validation_rows.append(combined)
        validation_index.append((fk, pct))

    validation = pd.DataFrame(
        validation_rows,
        index=pd.MultiIndex.from_tuples(validation_index, names=["fold", "reduction"]),
    )

    return validation, best_pipelines, best_predictions, reduction_validation


def magic_now(
    X: Union[np.ndarray, pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    y: Union[np.ndarray, pd.Series, list],
    scoring: Union[str, Callable] = "PR AUC",
    task: str = "classification",
    feature_selector: Union[str, list] = "optimize",
    scaler: Union[str, list] = "optimize",
    algorithm: Union[str, list] = "optimize",
    hyperparameters: str = "optimize",
    random_state: int = 42,
    n_trials: int = 100,
    output_dir: Union[str, "PathLike[str]"] = None,
    outer_cv_groups: Union[np.ndarray, pd.Series, list] = None,
    inner_cv_groups: Union[np.ndarray, pd.Series, list] = None,
    tag: str = "",
    n_jobs: int = 12,
    plots: bool = True,
    calibration: Union[str, None] = None,
    optimize_threshold: bool = False,
    pos_weight_factor: float = 1.0,
    cfc_random_state: int = 2024,
):

    if not output_dir:
        raise ValueError("Output directory must be specified.")

    if calibration not in (None, "isotonic", "sigmoid", "auto"):
        raise ValueError(
            f"calibration must be None, 'isotonic', 'sigmoid', or 'auto' "
            f"(got {calibration!r})"
        )

    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    trials_dir = output_dir / "trials"
    trials_dir.mkdir(exist_ok=True)

    plots_dir = output_dir / "plots"
    if plots:
        plots_dir.mkdir(exist_ok=True)

    if isinstance(X, (list, dict)):
        if isinstance(X, list):
            if len(X) == 0:
                raise ValueError("X must contain at least one DataFrame.")
            datasets = {f"dataset_{i}": df for i, df in enumerate(X)}
        else:
            if len(X) == 0:
                raise ValueError("X must contain at least one DataFrame.")
            datasets = X

        best_combo_name, X, scores_series = select_best_dataset_combo(
            datasets=datasets,
            y=y,
            scoring=scoring,
            task=task,
            random_state=random_state,
            inner_cv_groups=inner_cv_groups,
            n_jobs=n_jobs,
        )

        print(
            f"[Dataset Selection] Best combination : {best_combo_name!r} "
            f"(score = {scores_series[best_combo_name]:.4f})"
        )
        print("[Dataset Selection] Full ranking:")
        print(scores_series.to_string())

        scores_series.to_csv(
            results_dir / f"dataset_selection{tag}.tsv",
            sep="\t",
            header=True,
        )

    pipelines, studies, inner_oof = train(
        X=X,
        y=y,
        scoring=scoring,
        task=task,
        feature_selector=feature_selector,
        scaler=scaler,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        random_state=random_state,
        n_trials=n_trials,
        outer_cv_groups=outer_cv_groups,
        inner_cv_groups=inner_cv_groups,
        n_jobs=n_jobs,
        pos_weight_factor=pos_weight_factor,
    )

    validation, pipeline_by_fold, predictions_by_fold, reduction_validation = (
        nested_crossval(
            X=X,
            y=y,
            pipelines=pipelines,
            inner_oof=inner_oof,
            scoring=scoring,
            outer_cv_groups=outer_cv_groups,
            n_jobs=n_jobs,
            pos_weight_factor=pos_weight_factor,
            cfc_random_state=cfc_random_state,
        )
    )

    # Keep an uncalibrated reference for the calibration-variants diagnostic
    # plot; calibrated_artifacts below wraps each pipeline in
    # CalibratedThresholdedClassifier when calibration/threshold are on.
    uncalibrated_pipeline = dict(pipeline_by_fold)

    sample_weight = _balanced_sample_weight(y, pos_weight_factor=pos_weight_factor)

    # ---------------------------------------------------------------------#
    # Optional calibration + threshold tuning per outer-fold model.        #
    # Score-level: ScoreCalibrator is fit on the INNER OOF scores produced #
    # inside outer-train (never touches TTS rows). Calibrator and threshold#
    # are then applied to CFC + TTS scores for honest evaluation, and to   #
    # the inner OOF itself for the threshold scan.                          #
    # ---------------------------------------------------------------------#
    # Map each fold_key to its chosen reduction pct (taken from the
    # nested_crossval `validation` table's MultiIndex). Used to surface
    # the reduction column in the per-variant evaluation table.
    best_pct_per_fold = {fk: pct for fk, pct in validation.index}

    evaluation_rows: list = []
    evaluation_index: list = []
    calibrated_artifacts: dict = {}

    if calibration is not None or optimize_threshold:
        for fold_key, fold_pipeline in pipeline_by_fold.items():
            oof = predictions_by_fold.get(fold_key)
            if oof is None:
                continue
            inner_df = oof["inner"]
            cfc_df = oof["cfc"]
            tts_df = oof["tts"]

            inner_true = inner_df["true"].to_numpy().astype(int)
            inner_score_raw = inner_df["score"].to_numpy(dtype=float)
            inner_weights = sample_weight.loc[inner_df.index].to_numpy()

            cfc_true = cfc_df["true"].to_numpy().astype(int)
            cfc_pred_raw = cfc_df["predicted"].to_numpy().astype(int)
            cfc_score_raw = cfc_df["score"].to_numpy(dtype=float)

            tts_true = tts_df["true"].to_numpy().astype(int)
            tts_pred_raw = tts_df["predicted"].to_numpy().astype(int)
            tts_score_raw = tts_df["score"].to_numpy(dtype=float)

            # ----- pick + fit calibrator on INNER OOF (honest, no leak) -----
            calibrated_method = None
            if calibration == "auto":
                calibrated_method, brier = pick_best_score_calibration_method(
                    inner_true,
                    inner_score_raw,
                    sample_weight=inner_weights,
                    random_state=random_state,
                )
                print(
                    f"[magic_now] {fold_key}: auto-picked calibration="
                    f"{calibrated_method!r} (Brier={brier:.4f})."
                )
            elif calibration in ("isotonic", "sigmoid"):
                calibrated_method = calibration

            if calibrated_method is not None:
                calibrator = fit_score_calibrator(
                    inner_true,
                    inner_score_raw,
                    method=calibrated_method,
                    sample_weight=inner_weights,
                )
                inner_score_cal = calibrator.predict(inner_score_raw)
                cfc_score_cal = calibrator.predict(cfc_score_raw)
                tts_score_cal = calibrator.predict(tts_score_raw)
                print(f"[magic_now] {fold_key}: calibrated ({calibrated_method}).")
            else:
                calibrator = None
                inner_score_cal = inner_score_raw
                cfc_score_cal = cfc_score_raw
                tts_score_cal = tts_score_raw

            cfc_pred_cal_05 = (cfc_score_cal >= 0.5).astype(int)
            tts_pred_cal_05 = (tts_score_cal >= 0.5).astype(int)

            # ----- threshold tuned on calibrated INNER OOF (still honest) ---
            calibrated_threshold = 0.5
            if optimize_threshold:
                calibrated_threshold, threshold_score, used_metric = (
                    find_best_threshold(inner_true, inner_score_cal, metric=scoring)
                )
                print(
                    f"[magic_now] {fold_key}: threshold={calibrated_threshold:.3f} "
                    f"({used_metric}={threshold_score:.4f})."
                )

            cfc_pred_cal_thr = (cfc_score_cal >= calibrated_threshold).astype(int)
            tts_pred_cal_thr = (tts_score_cal >= calibrated_threshold).astype(int)

            # Each variant carries its own (method, threshold) — raw model
            # has neither calibrator nor tuned threshold; calibrated@0.5 has
            # the calibrator but the default cutoff; calibrated+threshold
            # has both.
            variant_specs = [
                {
                    "label": "raw model",
                    "cfc_pred": cfc_pred_raw,
                    "cfc_score": cfc_score_raw,
                    "tts_pred": tts_pred_raw,
                    "tts_score": tts_score_raw,
                    "method": None,
                    "threshold": 0.5,
                },
                {
                    "label": "calibrated model",
                    "cfc_pred": cfc_pred_cal_05,
                    "cfc_score": cfc_score_cal,
                    "tts_pred": tts_pred_cal_05,
                    "tts_score": tts_score_cal,
                    "method": calibrated_method,
                    "threshold": 0.5,
                },
            ]
            if optimize_threshold:
                variant_specs.append(
                    {
                        "label": "calibrated model with threshold",
                        "cfc_pred": cfc_pred_cal_thr,
                        "cfc_score": cfc_score_cal,
                        "tts_pred": tts_pred_cal_thr,
                        "tts_score": tts_score_cal,
                        "method": calibrated_method,
                        "threshold": calibrated_threshold,
                    }
                )

            fold_best_pct = best_pct_per_fold[fold_key]
            for v in variant_specs:
                cfc_report = classification_report(
                    cfc_true, v["cfc_pred"], y_score=v["cfc_score"]
                )
                tts_report = classification_report(
                    tts_true, v["tts_pred"], y_score=v["tts_score"]
                )
                combined = pd.concat(
                    [cfc_report.add_suffix(" CFC"), tts_report.add_suffix(" TTS")]
                )
                combined["label"] = v["label"]
                combined["reduction"] = fold_best_pct
                combined["calibration_method"] = v["method"]
                combined["threshold"] = v["threshold"]
                evaluation_rows.append(combined)
                evaluation_index.append(fold_key)

            calibrated_artifacts[fold_key] = CalibratedThresholdedClassifier(
                pipeline=fold_pipeline,
                calibrator=calibrator,
                threshold=calibrated_threshold,
            )

        validation = pd.DataFrame(evaluation_rows, index=evaluation_index)
        validation.index.name = "fold"
    else:
        # nested_crossval already produced the raw CFC+TTS validation table
        # for each fold's best pct; lift the reduction from the MultiIndex
        # into a column and label the rows so the schema matches the
        # calibration-on branch.
        validation = validation.copy().reset_index(level="reduction")
        validation["label"] = "raw model"
        validation["calibration_method"] = None
        validation["threshold"] = 0.5

    # ---------------------------------------------------------------------#
    # Persist artefacts. All models saved as all_pipelines_{tag}.pkl      #
    # Select best model using sqrt(FNFPCFC**2+FNFPTTS**2) metric          #
    # Save best model as pipeline_{tag}.pkl                               #
    # Both CalibratedThresholdedClassifier and bare pipelines supported.  #
    # ---------------------------------------------------------------------#
    model_payload = calibrated_artifacts if calibrated_artifacts else pipeline_by_fold
    with open(models_dir / f"all_pipelines{tag}.pkl", "wb") as handle:
        pickle.dump(model_payload, handle)

    # Select best model using sqrt(FNFP CFC**2 + FNFP TTS**2) from validation table
    best_fold_key = None
    best_label = None

    if "FNFP CFC" in validation.columns and "FNFP TTS" in validation.columns:
        metrics = (validation["FNFP CFC"] ** 2 + validation["FNFP TTS"] ** 2) ** 0.5
        best_pos = int(np.argmin(metrics.to_numpy())) if len(metrics) else None
        if best_pos is not None:
            best_fold_key = validation.index[best_pos]
            best_row = validation.iloc[best_pos]
            best_label = best_row.get("label")

    # Save best model (fold + variant aware)
    if best_fold_key is not None:
        if best_label == "raw model" or not best_label:
            best_model = pipeline_by_fold[best_fold_key]
        elif best_label == "calibrated model":
            calibrator = calibrated_artifacts.get(best_fold_key)
            if calibrator is None:
                best_model = pipeline_by_fold[best_fold_key]
            else:
                best_model = CalibratedThresholdedClassifier(
                    pipeline=pipeline_by_fold[best_fold_key],
                    calibrator=calibrator.calibrator,
                    threshold=0.5,
                )
        elif best_label == "calibrated model with threshold":
            best_model = calibrated_artifacts.get(
                best_fold_key, pipeline_by_fold[best_fold_key]
            )
        else:
            best_model = model_payload[best_fold_key]

        with open(models_dir / f"pipeline{tag}.pkl", "wb") as handle:
            pickle.dump(best_model, handle)

    with open(
        trials_dir / f"studies{tag}.pkl",
        "wb",
    ) as handle:
        pickle.dump(studies, handle)

    # ---------------------------------------------------------------------#
    # Plots: Best model only — PR / ROC / KS for CFC (mean±std) and TTS  #
    # CFC is plotted with mean and standard deviation across splits       #
    # TTS is plotted as single curve                                       #
    # ---------------------------------------------------------------------#
    if plots and best_fold_key is not None:
        best_cfc_df = predictions_by_fold[best_fold_key]["cfc"]
        best_tts_df = predictions_by_fold[best_fold_key]["tts"]

        # Apply calibrator only for calibrated variants
        if best_label in ("calibrated model", "calibrated model with threshold"):
            art = calibrated_artifacts.get(best_fold_key)
            if art is not None and art.calibrator is not None:
                best_tts_df = best_tts_df.copy()
                best_tts_df["score"] = art.calibrator.predict(
                    best_tts_df["score"].to_numpy(dtype=float)
                )
                best_cfc_df = best_cfc_df.copy()
                best_cfc_df["score"] = art.calibrator.predict(
                    best_cfc_df["score"].to_numpy(dtype=float)
                )

        plot_pr_curve(
            best_cfc_df,
            best_tts_df,
            plots_dir / f"pr_curve{tag}.pdf",
            title=f"PR — Best Model {best_fold_key}",
        )
        plot_roc_curve(
            best_cfc_df,
            best_tts_df,
            plots_dir / f"roc_curve{tag}.pdf",
            title=f"ROC — Best Model {best_fold_key}",
        )
        plot_ks_statistic(
            best_cfc_df,
            best_tts_df,
            plots_dir / f"ks_statistic{tag}.pdf",
            title=f"KS Statistic — Best Model {best_fold_key}",
        )

    n_1 = int((y == 1).sum())
    n_0 = int((y == 0).sum())
    for tbl in (validation, reduction_validation):
        tbl["positives"] = n_1
        tbl["negatives"] = n_0
        tbl["class_imbalance"] = round(n_1 / n_0, 4) if n_0 > 0 else float("inf")
        if outer_cv_groups is not None:
            tbl["n_groups"] = int(pd.Series(outer_cv_groups).nunique())

    reduction_validation.to_csv(
        results_dir / f"reduction_validation{tag}.tsv", sep="\t", index=True
    )
    validation.to_csv(results_dir / f"validation{tag}.tsv", sep="\t", index=True)

    # ---------------------------------------------------------------------#
    # Best hyperparameters per outer fold                                  #
    # ---------------------------------------------------------------------#
    best_params_rows = []
    for fold_key, study in studies.items():
        if not study.trials:
            continue
        row = {
            "fold": fold_key,
            "best_score": study.best_value,
        }
        # user_attrs override params so suggest_power values appear in
        # their exponentiated form (e.g. learning_rate=0.01 not -2).
        row.update(study.best_trial.params)
        row.update(study.best_trial.user_attrs)
        best_params_rows.append(row)

    if best_params_rows:
        best_params_df = pd.DataFrame(best_params_rows).set_index("fold")
        best_params_df.to_csv(
            results_dir / f"best_params{tag}.tsv", sep="\t", index=True
        )
        print("[magic_now] Best hyperparameters per fold:")
        print(best_params_df.to_string())

    return validation, model_payload, studies
