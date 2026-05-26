# -*- coding: utf-8 -*-

###########
# Imports #
###########

import pandas as pd
import numpy as np

import pickle

from joblib import Parallel, delayed
from typing import Union, Callable, Dict, List, Optional
from os import PathLike

from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.svm import SVC

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
    plot_calibration_curve,
    fit_score_calibrator,
    pick_best_score_calibration_method,
    find_best_threshold,
    compute_calibration_variants,
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
    sw = pd.Series(
        compute_sample_weight(class_weight="balanced", y=y), index=y.index
    )
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

    fold_name = f"fold_{fold_idx}"
    return fold_name, final_pipeline, study


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
    models : dict
        A dictionary containing trained models for each fold.
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

    models = {name: model for name, model, _ in flat}
    studies = {name: study for name, _, study in flat}

    return models, studies


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
    Percentages run 10 % → 100 % in 10 % steps.  A percentage is included
    only when ``int(pct × min_n_train) ≥ 2000``.  When the smallest training
    fold has fewer than 2 000 rows, only 80 % and 90 % are tested (100 % is
    always included).  Reduced indices are computed **once per (split, pct)**
    and reused across all pipelines so reduction noise never confounds model
    comparisons.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    pipelines : dict
        Mapping ``"fold_{i}"`` → fitted Pipeline (one per outer fold).
    scoring : str or callable, default ``"MCC"``
        Metric used for early stopping during the reduction-pct sweep and
        for picking each fold's best pct. Selection runs on the **CFC**
        column (``"<metric> CFC"``); the TTS column is reported alongside
        for diagnostic comparison. Should match the metric the HPO
        optimised. Recognised string aliases: ``"MCC"``, ``"PR AUC"``,
        ``"ROC AUC"``, ``"KS"``, plus common sklearn names
        (``"balanced_accuracy"``, ``"f1"``, ...).
    random_state : int, default 42
        Drives the TTS splits, which match train()'s outer CV grid.
    cfc_random_state : int, default 2024
        Independent fixed seed for the CFC evaluation grid. Differs from
        ``random_state`` so the cross-fold competition is computed on a
        shared, neutral grid (not the same one each fold_key was trained
        against).
    outer_cv_groups : array-like, optional
        Group labels for the outer StratifiedGroupKFold (used by both
        TTS and CFC grids).
    n_jobs : int, default 16

    Returns
    -------
    validation : pd.DataFrame
        Classification report with MultiIndex as described above. Every
        metric column appears twice, suffixed ``" CFC"`` (cross-fold
        competition, concatenated across the cfc_splits grid) and
        ``" TTS"`` (true test set, the fold_key's original held-out
        outer-fold split).
    best_pipelines : dict[str, Pipeline]
        One refitted pipeline per fold_key, each at its own best CFC pct.
        Keys are ``"fold_{i}"`` in single mode, ``"fold_{i}_{alg}"`` in
        per-algorithm mode.
    best_predictions : dict[str, dict[str, pd.DataFrame]]
        ``{fold_key: {"cfc": cfc_oof_df, "tts": tts_oof_df}}``. CFC is the
        concatenation across cfc_splits at the fold's best pct; TTS is the
        single OOF DataFrame from the fold_key's original outer-fold split.
        Both DataFrames have columns ``["true", "predicted", "score"]``.
    """
    sample_weight = _balanced_sample_weight(y, pos_weight_factor=pos_weight_factor)

    # The early-stopping criterion and the final (fold, pct) ranking use
    # the same metric the HPO optimised. Falls back to MCC for callable
    # scorers or unrecognised strings. Each metric is computed twice in
    # the validation report: once on CFC (cross-fold competition; every
    # fold_key evaluated on a shared, separately-seeded 4-fold grid) and
    # once on TTS (true test set; each fold_key evaluated on the exact
    # outer-fold split it was held out from during train()).
    metric_col = scoring_to_metric_column(scoring, default="MCC")
    cfc_metric_col = f"{metric_col} CFC"
    tts_metric_col = f"{metric_col} TTS"
    lower_better = metric_is_lower_better(metric_col)
    _is_better = (
        (lambda new, cur: new <= cur) if lower_better else (lambda new, cur: new >= cur)
    )
    _idxbest = (lambda s: s.idxmin()) if lower_better else (lambda s: s.idxmax())
    _worst_init = float("inf") if lower_better else float("-inf")

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
    # of train()'s outer CV; all fold_key models compete on the same grid.
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
        """Recover the 0-indexed outer-fold position from a fold_key
        like ``"fold_3"`` or ``"fold_3_LGBM"``."""
        return int(fold_key.split("_")[1]) - 1

    # ------------------------------------------------------------------ #
    # Valid reduction percentages                                        #
    # ------------------------------------------------------------------ #
    all_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_size = 2000
    min_n_train = min(
        min(len(train_idx) for train_idx, _ in cfc_splits),
        min(len(train_idx) for train_idx, _ in tts_splits),
    )

    if min_n_train < min_size:
        valid_pcts = [0.8, 0.9, 1.0]
    else:
        valid_pcts = [p for p in all_percentages if int(p * min_n_train) >= min_size]

    # Per-fold-key pct list: kernel SVM only tests pcts where
    # pct × min_n_train ≤ max_svm_samples to avoid O(n²) blowup.
    fold_key_pcts: dict = {}
    for fk, pipe in pipelines.items():
        m = pipe.named_steps.get("model")
        if m is not None and hasattr(m, "kernel"):
            svm_pcts = [
                p for p in valid_pcts if int(p * min_n_train) <= max_svm_samples
            ]
            if not svm_pcts:
                # Dataset too large for any standard pct: one custom pct at the cap
                svm_pcts = [round(max_svm_samples / min_n_train, 4)]
            fold_key_pcts[fk] = svm_pcts
            print(
                f"[nested_crossval] {fk!r} kernel SVM — pcts: "
                + ", ".join(
                    f"{int(p * 100)}%" if p in all_percentages else f"{p:.2%}"
                    for p in svm_pcts
                )
                + f" (≤ {max_svm_samples:,} samples)"
            )
        else:
            fold_key_pcts[fk] = valid_pcts

    # Union of all pcts needed by any fold_key, sorted ascending
    loop_pcts = sorted({p for pcts in fold_key_pcts.values() for p in pcts})

    print(
        "[nested_crossval] Reduction sizes: "
        + ", ".join(
            f"{int(p * 100)}%" if p in all_percentages else f"{p:.2%}"
            for p in loop_pcts
        )
    )

    # ------------------------------------------------------------------ #
    # Incremental pct loop: parallel over active fold_keys per pct,      #
    # with early stopping after two consecutive dips below best score.   #
    # ------------------------------------------------------------------ #
    best_score_es: dict = {fk: _worst_init for fk in pipelines}
    flagged_es: dict = {fk: False for fk in pipelines}
    active_es: dict = {fk: True for fk in pipelines}
    cfc_reduced_idx: dict = {}
    tts_reduced_idx: dict = {}
    pred_store_cfc: dict = {}
    pred_store_tts: dict = {}
    reports: dict = {}

    def _reduce_for(train_index, pct, key, store):
        if (key, pct) in store:
            return
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        if pct == 1.0:
            store[(key, pct)] = X_train.index
        else:
            store[(key, pct)] = reduce_dataset(
                X=X_train,
                y=y_train,
                target_size=int(pct * len(train_index)),
                stage2_shrink=1,
                class_weight="balanced",
                random_state=random_state,
                verbose=False,
            )

    for pct in loop_pcts:
        active_keys = [
            fk for fk in pipelines if active_es[fk] and pct in fold_key_pcts[fk]
        ]
        if not active_keys:
            if not any(active_es[fk] for fk in pipelines):
                break  # every fold_key has been early-stopped
            continue  # some fold_keys still active but not at this pct

        # Lazy-compute reduced indices for this pct on both grids.
        for cfc_si, (train_index, _) in enumerate(cfc_splits):
            _reduce_for(train_index, pct, cfc_si, cfc_reduced_idx)

        # TTS only needs the splits that match an active fold_key.
        needed_tts_indices = {_tts_split_idx(fk) for fk in active_keys}
        for tts_si in needed_tts_indices:
            _reduce_for(tts_splits[tts_si][0], pct, tts_si, tts_reduced_idx)

        # Build one parallel batch for both CFC and TTS fits.
        cfc_tasks = [
            (fk, cfc_si)
            for fk in active_keys
            for cfc_si in range(n_cfc_splits)
        ]
        tts_tasks = [(fk, _tts_split_idx(fk)) for fk in active_keys]

        cfc_delayed = [
            delayed(_fit_and_predict)(
                pipelines[fk],
                X,
                y,
                sample_weight,
                cfc_splits[cfc_si][0],
                cfc_splits[cfc_si][1],
                cfc_reduced_idx[(cfc_si, pct)],
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
                tts_reduced_idx[(tts_si, pct)],
            )
            for fk, tts_si in tts_tasks
        ]

        results = Parallel(n_jobs=n_jobs)(cfc_delayed + tts_delayed)
        n_cfc = len(cfc_tasks)
        for (fk, cfc_si), df in zip(cfc_tasks, results[:n_cfc]):
            pred_store_cfc[(fk, cfc_si, pct)] = df
        for (fk, tts_si), df in zip(tts_tasks, results[n_cfc:]):
            pred_store_tts[(fk, pct)] = df

        # Aggregate per fold_key; build a combined CFC + TTS report.
        for fold_key in active_keys:
            cfc_preds = pd.concat(
                [pred_store_cfc[(fold_key, si, pct)] for si in range(n_cfc_splits)],
                axis=0,
            )
            cfc_report = classification_report(
                cfc_preds["true"],
                cfc_preds["predicted"],
                y_score=cfc_preds["score"] if "score" in cfc_preds.columns else None,
            )

            tts_preds = pred_store_tts[(fold_key, pct)]
            tts_report = classification_report(
                tts_preds["true"],
                tts_preds["predicted"],
                y_score=tts_preds["score"] if "score" in tts_preds.columns else None,
            )

            # Suffix and concatenate so each metric appears twice in the row.
            combined = pd.concat(
                [cfc_report.add_suffix(" CFC"), tts_report.add_suffix(" TTS")]
            )
            reports[(fold_key, pct)] = combined

            # Early stopping uses CFC (more data, lower variance).
            score_val = combined[cfc_metric_col]

            cmp = "<" if not lower_better else ">"
            if not flagged_es[fold_key]:
                if _is_better(score_val, best_score_es[fold_key]):
                    best_score_es[fold_key] = score_val
                else:
                    flagged_es[fold_key] = True
                    print(
                        f"[nested_crossval] {fold_key!r} dip at {int(pct * 100)}%"
                        f" ({cfc_metric_col}={score_val:.4f} {cmp} best={best_score_es[fold_key]:.4f})"
                        " — giving one more chance …"
                    )
            else:
                if _is_better(score_val, best_score_es[fold_key]):
                    flagged_es[fold_key] = False
                    best_score_es[fold_key] = score_val
                else:
                    active_es[fold_key] = False
                    flagged_es[fold_key] = False
                    print(
                        f"[nested_crossval] Early stop {fold_key!r} at {int(pct * 100)}%"
                        f" ({cfc_metric_col}={score_val:.4f} {cmp} best={best_score_es[fold_key]:.4f})"
                    )

    # ------------------------------------------------------------------ #
    # Build validation table (CFC + TTS metrics, MultiIndex (fold, pct)).#
    # ------------------------------------------------------------------ #
    index_tuples = list(reports.keys())
    validation = pd.DataFrame(
        list(reports.values()),
        index=pd.MultiIndex.from_tuples(
            index_tuples, names=["fold", "reduction"]
        ),
    )

    # ------------------------------------------------------------------ #
    # Per-fold selection: each fold_key picks its own best CFC pct,      #
    # gets refit, and returns its CFC+TTS OOF predictions. No single     #
    # cross-fold winner — all outer-fold pipelines are deployable.       #
    # ------------------------------------------------------------------ #
    best_pipelines = {}
    best_predictions = {}
    for fold_key in pipelines:
        fold_series = pd.Series(
            {
                pct: combined[cfc_metric_col]
                for (fk, pct), combined in reports.items()
                if fk == fold_key
            }
        )
        if fold_series.empty:
            continue
        best_pct = _idxbest(fold_series)
        best_score = fold_series[best_pct]
        print(
            f"[nested_crossval] {fold_key}: best reduction={int(best_pct * 100)}% "
            f"({cfc_metric_col}={best_score:.4f}) — refitting …"
        )
        best_pipelines[fold_key] = _refit_pipeline(
            pipelines[fold_key],
            X,
            y,
            sample_weight,
            pct=best_pct,
            random_state=random_state,
            max_svm_samples=max_svm_samples,
        )
        cfc_oof = pd.concat(
            [pred_store_cfc[(fold_key, si, best_pct)] for si in range(n_cfc_splits)],
            axis=0,
        )
        tts_oof = pred_store_tts[(fold_key, best_pct)]
        best_predictions[fold_key] = {"cfc": cfc_oof, "tts": tts_oof}

    return validation, best_pipelines, best_predictions


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

    pipelines, studies = train(
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

    reduction_validation, pipeline_by_fold, predictions_by_fold = nested_crossval(
        X=X,
        y=y,
        pipelines=pipelines,
        scoring=scoring,
        outer_cv_groups=outer_cv_groups,
        n_jobs=n_jobs,
        pos_weight_factor=pos_weight_factor,
        cfc_random_state=cfc_random_state,
    )

    # Keep an uncalibrated reference for the calibration-variants diagnostic
    # plot; calibrated_artifacts below wraps each pipeline in
    # CalibratedThresholdedClassifier when calibration/threshold are on.
    uncalibrated_pipeline = dict(pipeline_by_fold)

    sample_weight = _balanced_sample_weight(y, pos_weight_factor=pos_weight_factor)

    # ---------------------------------------------------------------------#
    # Optional calibration + threshold tuning per outer-fold model.        #
    # Score-level: ScoreCalibrator is fit on the CFC OOF scores already    #
    # produced by nested_crossval — zero extra model fits. The calibrator  #
    # is applied to both the CFC and TTS score arrays so the per-variant   #
    # validation rows carry CFC + TTS metric columns on equal footing.     #
    # ---------------------------------------------------------------------#
    evaluation_rows: list = []
    evaluation_index: list = []
    calibrated_artifacts: dict = {}

    if calibration is not None or optimize_threshold:
        for fold_key, fold_pipeline in pipeline_by_fold.items():
            oof = predictions_by_fold.get(fold_key)
            if oof is None:
                continue
            cfc_df = oof["cfc"]
            tts_df = oof["tts"]

            cfc_true = cfc_df["true"].to_numpy().astype(int)
            cfc_pred_raw = cfc_df["predicted"].to_numpy().astype(int)
            cfc_score_raw = cfc_df["score"].to_numpy(dtype=float)
            cfc_weights = sample_weight.loc[cfc_df.index].to_numpy()

            tts_true = tts_df["true"].to_numpy().astype(int)
            tts_pred_raw = tts_df["predicted"].to_numpy().astype(int)
            tts_score_raw = tts_df["score"].to_numpy(dtype=float)

            # ----- pick + fit calibrator on CFC OOF (zero model fits) -----
            calibrated_method = None
            if calibration == "auto":
                calibrated_method, brier = pick_best_score_calibration_method(
                    cfc_true,
                    cfc_score_raw,
                    sample_weight=cfc_weights,
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
                    cfc_true,
                    cfc_score_raw,
                    method=calibrated_method,
                    sample_weight=cfc_weights,
                )
                cfc_score_cal = calibrator.predict(cfc_score_raw)
                tts_score_cal = calibrator.predict(tts_score_raw)
                print(f"[magic_now] {fold_key}: calibrated ({calibrated_method}).")
            else:
                calibrator = None
                cfc_score_cal = cfc_score_raw
                tts_score_cal = tts_score_raw

            cfc_pred_cal_05 = (cfc_score_cal >= 0.5).astype(int)
            tts_pred_cal_05 = (tts_score_cal >= 0.5).astype(int)

            calibrated_threshold = 0.5
            if optimize_threshold:
                calibrated_threshold, threshold_score, used_metric = (
                    find_best_threshold(
                        cfc_true, cfc_score_cal, metric=scoring
                    )
                )
                print(
                    f"[magic_now] {fold_key}: threshold={calibrated_threshold:.3f} "
                    f"({used_metric}={threshold_score:.4f})."
                )

            cfc_pred_cal_thr = (cfc_score_cal >= calibrated_threshold).astype(int)
            tts_pred_cal_thr = (tts_score_cal >= calibrated_threshold).astype(int)

            variant_specs = [
                (
                    "raw model",
                    cfc_pred_raw, cfc_score_raw,
                    tts_pred_raw, tts_score_raw,
                ),
                (
                    "calibrated model",
                    cfc_pred_cal_05, cfc_score_cal,
                    tts_pred_cal_05, tts_score_cal,
                ),
            ]
            if optimize_threshold:
                variant_specs.append(
                    (
                        "calibrated model with threshold",
                        cfc_pred_cal_thr, cfc_score_cal,
                        tts_pred_cal_thr, tts_score_cal,
                    )
                )

            for label, c_pred, c_score, t_pred, t_score in variant_specs:
                cfc_report = classification_report(
                    cfc_true, c_pred, y_score=c_score
                )
                tts_report = classification_report(
                    tts_true, t_pred, y_score=t_score
                )
                combined = pd.concat(
                    [cfc_report.add_suffix(" CFC"), tts_report.add_suffix(" TTS")]
                )
                combined["label"] = label
                combined["calibration_method"] = calibrated_method
                combined["threshold"] = calibrated_threshold
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
        validation = reduction_validation.copy()
        validation["label"] = "raw model"

    # ---------------------------------------------------------------------#
    # Persist artefacts. Single pickle keyed by fold_key; values are       #
    # CalibratedThresholdedClassifier (when calibration/threshold are on)  #
    # or bare pipelines (otherwise). Both expose the sklearn API.          #
    # ---------------------------------------------------------------------#
    model_payload = calibrated_artifacts if calibrated_artifacts else pipeline_by_fold
    with open(models_dir / f"pipelines{tag}.pkl", "wb") as handle:
        pickle.dump(model_payload, handle)

    with open(
        trials_dir / f"studies{tag}.pkl",
        "wb",
    ) as handle:
        pickle.dump(studies, handle)

    # ---------------------------------------------------------------------#
    # Plots: CFC OOF feeds PR / ROC / KS (more samples per curve).         #
    # Calibration variants stay refit-based per fold_key.                  #
    # ---------------------------------------------------------------------#
    if plots:
        cfc_only = {fk: oof["cfc"] for fk, oof in predictions_by_fold.items()}
        plot_pr_curve(
            cfc_only, plots_dir / f"pr_curve{tag}.pdf", title="PR curve"
        )
        plot_roc_curve(
            cfc_only, plots_dir / f"roc_curve{tag}.pdf", title="ROC curve"
        )
        plot_ks_statistic(
            cfc_only,
            plots_dir / f"ks_statistic{tag}.pdf",
            title="KS statistic",
        )

        for fold_key, uncal_pipe in uncalibrated_pipeline.items():
            variants = compute_calibration_variants(
                uncal_pipe, X, y, sample_weight, random_state=random_state,
            )
            plot_calibration_curve(
                variants,
                plots_dir / f"calibration_curve_{fold_key}{tag}.pdf",
                title=f"Calibration plots — {fold_key}",
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
