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
    plot_pr_curve,
    plot_roc_curve,
    plot_ks_statistic,
)
from ..feature_selection import select_best_dataset_combo, select_best_feature_selector

#############
# Functions #
#############


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
    # Build final pipeline: FS steps (phase 1) + scaler + model (phase 2) #
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


def _train_fold_multi_alg(
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
    algorithms,
    n_trials_per_alg,
    n_cv_jobs,
    random_state,
    inner_cv_groups=None,
):
    """
    Two-phase optimisation for one outer fold when a list of algorithms is
    provided with ``hyperparameters="optimize"``.

    Phase 1 – Feature-selection HPO runs **once** for the fold, shared across
    all algorithms so FS is not duplicated for each model.

    Phase 2 – One independent Optuna study per algorithm, each with
    ``n_trials_per_alg`` trials on the pre-selected feature matrix.

    Returns
    -------
    list of (name, pipeline, study)
        One entry per algorithm; name = ``fold_{fold_idx}_{alg}``.
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
    # Phase 1 – feature-selection HPO (shared across all algorithms)      #
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

    # ------------------------------------------------------------------ #
    # Phase 2 – one Optuna study per algorithm                            #
    # ------------------------------------------------------------------ #
    if n_cv_jobs < 2:
        _model_jobs, _inner_jobs = n_cv_jobs, 1
    elif n_cv_jobs == 3:
        _model_jobs, _inner_jobs = 1, n_cv_jobs
    elif n_cv_jobs < 6:
        _model_jobs, _inner_jobs = n_cv_jobs, 1
    else:
        _model_jobs, _inner_jobs = n_cv_jobs // 3, 3

    # Pre-compute inner CV splits and reduced indices once; shared across all algorithms
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

    fold_results = []
    for alg in algorithms:
        sampler = TPESampler(seed=random_state, multivariate=True)
        study = create_study(
            direction="maximize",
            pruner=SuccessiveHalvingPruner(reduction_factor=2),
            sampler=sampler,
        )
        # default-argument capture avoids the closure-over-loop-variable trap
        study.optimize(
            lambda trial, _alg=alg: objective(
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
                algorithm=_alg,  # fixed to this algorithm
                hyperparameters="optimize",
                random_state=random_state,
                n_cv_jobs=_inner_jobs,
                model_jobs=_model_jobs,
                inner_cv_groups=groups_train,
                inner_splits=_p2_splits,
                inner_reduced_indices=_p2_reduced,
            ),
            n_trials=n_trials_per_alg,
            n_jobs=1,
        )

        phase2_pipeline = instantiate_pipeline(
            trial=study.best_trial,
            feature_selector=None,
            scaler=scaler,
            algorithm=alg,
            hyperparameters="optimize",
            random_state=random_state,
        )
        final_pipeline = Pipeline(
            fs_pipeline.steps
            + [s for s in phase2_pipeline.steps if s[0] in ("scaler", "model")]
        )

        fold_results.append((f"fold_{fold_idx}_{alg}", final_pipeline, study))

    return fold_results


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

    sample_weight = pd.Series(
        compute_sample_weight(class_weight="balanced", y=y),
        index=y.index,
    )

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

    per_alg_mode = isinstance(algorithm, list) and hyperparameters == "optimize"

    if per_alg_mode:
        n_trials_per_alg = max(1, n_trials // len(algorithm))
        print(
            f"[train] Per-algorithm HPO: {len(algorithm)} algorithm(s) × "
            f"{n_trials_per_alg} trial(s) each."
        )
        raw = Parallel(n_jobs=n_outer_folds)(
            delayed(_train_fold_multi_alg)(
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
                algorithms=algorithm,
                n_trials_per_alg=n_trials_per_alg,
                n_cv_jobs=n_cv_jobs,
                random_state=random_state,
                inner_cv_groups=inner_cv_groups,
            )
            for fold_idx, (train_index, test_index) in enumerate(splits, start=1)
        )
        # raw is list-of-lists; flatten into (name, model, study) triples
        flat = [entry for fold_list in raw for entry in fold_list]
    else:
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
    outer_cv_groups: Union[np.ndarray, pd.Series, list] = None,
    n_jobs: int = 16,
    algorithms: Union[list, None] = None,
    max_svm_samples: int = 10_000,
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
        Mapping key → fitted Pipeline.  Single-algorithm mode: keys are
        ``"fold_{i}"``; per-algorithm mode: ``"fold_{i}_{alg}"``.
    scoring : str or callable, default ``"MCC"``
        Metric used both for early stopping during the reduction-pct sweep
        and for picking the winning ``(fold, pct)`` per algorithm. Should
        match the metric the HPO optimised. Recognised string aliases:
        ``"MCC"``, ``"PR AUC"``, ``"ROC AUC"``, ``"KS"``, plus common
        sklearn names (``"balanced_accuracy"``, ``"f1"``, ...).
    random_state : int, default 42
    outer_cv_groups : array-like, optional
        Group labels for the outer StratifiedGroupKFold.
    n_jobs : int, default 16
    algorithms : list of str, optional
        Per-algorithm mode: one best ``(fold, pct)`` is chosen per algorithm
        and returned as an ordered list of refitted pipelines.  Validation has
        a MultiIndex ``(algorithm, fold, reduction)``.
        Single mode (``None``): one best ``(fold, pct)`` overall; validation
        has a MultiIndex ``(fold, reduction)``.

    Returns
    -------
    validation : pd.DataFrame
        Classification report with MultiIndex as described above.
    best_pipeline : Pipeline or list of Pipeline
        Single best pipeline or one per algorithm (per-algorithm mode).
    best_predictions : dict or list of dict
        Per-outer-fold model predictions ``{fold_key -> DataFrame[true,
        predicted, score]}``. Each entry concatenates the n_splits test
        folds for that model at its own best reduction pct, so the dict
        contains one row per competing model (typically 4). In per-algorithm
        mode, an ordered list aligned with ``algorithms``. Suitable input
        for ``plot_pr_curve`` / ``plot_roc_curve``.
    """
    sample_weight = pd.Series(
        compute_sample_weight(class_weight="balanced", y=y),
        index=y.index,
    )

    # The early-stopping criterion and the final (fold, pct) ranking use
    # the same metric the HPO optimised. Falls back to MCC for callable
    # scorers or unrecognised strings.
    metric_col = scoring_to_metric_column(scoring, default="MCC")

    if outer_cv_groups is not None:
        strat_kfold_outer = StratifiedGroupKFold(n_splits=4)
    else:
        strat_kfold_outer = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=random_state,
        )

    splits = list(strat_kfold_outer.split(X, y.astype("str"), groups=outer_cv_groups))
    n_splits = len(splits)

    # ------------------------------------------------------------------ #
    # Valid reduction percentages                                          #
    # ------------------------------------------------------------------ #
    all_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_size = 2000
    min_n_train = min(len(train_idx) for train_idx, _ in splits)

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
    # Incremental pct loop: parallel over active fold_keys per pct,     #
    # with early stopping after two consecutive dips below best score.  #
    # ------------------------------------------------------------------ #
    best_score_es: dict = {fk: float("-inf") for fk in pipelines}
    flagged_es: dict = {fk: False for fk in pipelines}
    active_es: dict = {fk: True for fk in pipelines}
    reduced_idx: dict = {}
    pred_store: dict = {}
    reports: dict = {}

    for pct in loop_pcts:
        active_keys = [
            fk for fk in pipelines if active_es[fk] and pct in fold_key_pcts[fk]
        ]
        if not active_keys:
            if not any(active_es[fk] for fk in pipelines):
                break  # every fold_key has been early-stopped
            continue  # some fold_keys still active but not at this pct

        # Lazy-compute reduced indices for this pct only
        for split_idx, (train_index, _) in enumerate(splits):
            if (split_idx, pct) not in reduced_idx:
                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]
                n_train = len(train_index)
                if pct == 1.0:
                    reduced_idx[(split_idx, pct)] = X_train.index
                else:
                    reduced_idx[(split_idx, pct)] = reduce_dataset(
                        X=X_train,
                        y=y_train,
                        target_size=int(pct * n_train),
                        stage2_shrink=1,
                        class_weight="balanced",
                        random_state=random_state,
                        verbose=False,
                    )

        # Parallel over active fold_keys × all splits
        task_keys_pct = [
            (fold_key, split_idx)
            for fold_key in active_keys
            for split_idx in range(n_splits)
        ]
        results_pct = Parallel(n_jobs=n_jobs)(
            delayed(_fit_and_predict)(
                pipelines[fold_key],
                X,
                y,
                sample_weight,
                splits[split_idx][0],
                splits[split_idx][1],
                reduced_idx[(split_idx, pct)],
            )
            for fold_key, split_idx in task_keys_pct
        )
        for (fold_key, split_idx), df in zip(task_keys_pct, results_pct):
            pred_store[(fold_key, split_idx, pct)] = df

        # Aggregate and apply early-stopping logic per fold_key
        for fold_key in active_keys:
            all_preds = pd.concat(
                [pred_store[(fold_key, si, pct)] for si in range(n_splits)], axis=0
            )
            report = classification_report(
                all_preds["true"],
                all_preds["predicted"],
                y_score=all_preds["score"] if "score" in all_preds.columns else None,
            )
            reports[(fold_key, pct)] = report
            score_val = report[metric_col]

            if not flagged_es[fold_key]:
                if score_val >= best_score_es[fold_key]:
                    best_score_es[fold_key] = score_val
                else:
                    flagged_es[fold_key] = True
                    print(
                        f"[nested_crossval] {fold_key!r} dip at {int(pct * 100)}%"
                        f" ({metric_col}={score_val:.4f} < best={best_score_es[fold_key]:.4f})"
                        " — giving one more chance …"
                    )
            else:
                if score_val >= best_score_es[fold_key]:
                    flagged_es[fold_key] = False
                    best_score_es[fold_key] = score_val
                else:
                    active_es[fold_key] = False
                    flagged_es[fold_key] = False
                    print(
                        f"[nested_crossval] Early stop {fold_key!r} at {int(pct * 100)}%"
                        f" ({metric_col}={score_val:.4f} < best={best_score_es[fold_key]:.4f})"
                    )

    # ------------------------------------------------------------------ #
    # Per-algorithm mode                                                 #
    # ------------------------------------------------------------------ #
    if algorithms is not None:
        index_tuples, rows = [], []
        for alg in algorithms:
            for (fold_key, pct), report in reports.items():
                if fold_key.endswith(f"_{alg}"):
                    index_tuples.append((alg, fold_key, pct))
                    rows.append(report)

        validation = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(
                index_tuples, names=["algorithm", "fold", "reduction"]
            ),
        )

        best_pipelines = []
        best_predictions = []
        for alg in algorithms:
            alg_df = validation.loc[alg]
            best_fold_key, best_pct = alg_df[metric_col].idxmax()
            best_score = alg_df.loc[(best_fold_key, best_pct), metric_col]
            print(
                f"[nested_crossval] {alg}: best fold={best_fold_key!r} "
                f"@ {int(best_pct * 100)}% ({metric_col}={best_score:.4f}) "
                f"— refitting …"
            )
            best_pipelines.append(
                _refit_pipeline(
                    pipelines[best_fold_key],
                    X,
                    y,
                    sample_weight,
                    pct=best_pct,
                    random_state=random_state,
                    max_svm_samples=max_svm_samples,
                )
            )
            # Surface every competing outer-fold model for this algorithm at
            # its own best pct, concatenated across the n_splits test folds.
            alg_predictions = {}
            for fk in alg_df.index.get_level_values("fold").unique():
                fk_best_pct = alg_df.loc[fk, metric_col].idxmax()
                alg_predictions[fk] = pd.concat(
                    [pred_store[(fk, si, fk_best_pct)] for si in range(n_splits)],
                    axis=0,
                )
            best_predictions.append(alg_predictions)

        return validation, best_pipelines, best_predictions

    # ------------------------------------------------------------------ #
    # Single-algorithm mode                                                #
    # ------------------------------------------------------------------ #
    index_tuples = list(reports.keys())
    validation = pd.DataFrame(
        list(reports.values()),
        index=pd.MultiIndex.from_tuples(index_tuples, names=["fold", "reduction"]),
    )

    best_fold, best_pct = validation[metric_col].idxmax()
    best_score = validation.loc[(best_fold, best_pct), metric_col]
    print(
        f"[nested_crossval] Best: {best_fold!r} @ {int(best_pct * 100)}% "
        f"({metric_col}={best_score:.4f}) — refitting …"
    )
    best_model = _refit_pipeline(
        pipelines[best_fold],
        X,
        y,
        sample_weight,
        pct=best_pct,
        random_state=random_state,
        max_svm_samples=max_svm_samples,
    )

    # Surface every competing outer-fold model at its own best pct,
    # concatenated across the n_splits test folds.
    best_predictions = {}
    for fk in validation.index.get_level_values("fold").unique():
        fk_best_pct = validation.loc[fk, metric_col].idxmax()
        best_predictions[fk] = pd.concat(
            [pred_store[(fk, si, fk_best_pct)] for si in range(n_splits)],
            axis=0,
        )

    return validation, best_model, best_predictions


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
):

    if not output_dir:
        raise ValueError("Output directory must be specified.")

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

    per_alg_mode = isinstance(algorithm, list) and hyperparameters == "optimize"

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
    )

    validation, pipeline, predictions = nested_crossval(
        X=X,
        y=y,
        pipelines=pipelines,
        scoring=scoring,
        outer_cv_groups=outer_cv_groups,
        n_jobs=n_jobs,
        algorithms=algorithm if per_alg_mode else None,
    )

    # ---------------------------------------------------------------------#
    # Persist pipelines                                                    #
    # Per-algorithm mode: one file per algorithm named pipelines_{alg}.pkl #
    # Single mode: one file named pipelines.pkl (unchanged)                #
    # ---------------------------------------------------------------------#
    if per_alg_mode:
        for alg, pipe in zip(algorithm, pipeline):
            with open(models_dir / f"pipelines_{alg}{tag}.pkl", "wb") as handle:
                pickle.dump(pipe, handle)
    else:
        with open(models_dir / f"pipelines{tag}.pkl", "wb") as handle:
            pickle.dump(pipeline, handle)

    with open(
        trials_dir / f"studies{tag}.pkl",
        "wb",
    ) as handle:
        pickle.dump(studies, handle)

    # ---------------------------------------------------------------------#
    # PR / ROC curve plots — one curve per outer-fold model, mean +/- std #
    # ---------------------------------------------------------------------#
    if plots:
        if per_alg_mode:
            for alg, preds in zip(algorithm, predictions):
                plot_pr_curve(
                    preds,
                    plots_dir / f"pr_curve_{alg}{tag}.pdf",
                    title=f"PR curve — {alg}",
                )
                plot_roc_curve(
                    preds,
                    plots_dir / f"roc_curve_{alg}{tag}.pdf",
                    title=f"ROC curve — {alg}",
                )
                plot_ks_statistic(
                    preds,
                    plots_dir / f"ks_statistic_{alg}{tag}.pdf",
                    title=f"KS statistic — {alg}",
                )
        else:
            plot_pr_curve(predictions, plots_dir / f"pr_curve{tag}.pdf", title="PR curve")
            plot_roc_curve(predictions, plots_dir / f"roc_curve{tag}.pdf", title="ROC curve")
            plot_ks_statistic(
                predictions, plots_dir / f"ks_statistic{tag}.pdf", title="KS statistic"
            )

    n_1 = int((y == 1).sum())
    n_0 = int((y == 0).sum())
    validation["positives"] = n_1
    validation["negatives"] = n_0
    validation["class_imbalance"] = round(n_1 / n_0, 4) if n_0 > 0 else float("inf")
    if outer_cv_groups is not None:
        validation["n_groups"] = int(pd.Series(outer_cv_groups).nunique())

    validation.to_csv(results_dir / f"validation{tag}.tsv", sep="\t", index=True)

    # ---------------------------------------------------------------------#
    # Best hyperparameters per algorithm                                   #
    # ---------------------------------------------------------------------#
    if per_alg_mode:
        best_params_rows = []
        for alg in algorithm:
            alg_studies = {k: v for k, v in studies.items() if k.endswith(f"_{alg}")}
            best_key = max(
                alg_studies,
                key=lambda k: (
                    alg_studies[k].best_value if alg_studies[k].trials else -np.inf
                ),
            )
            best_study = alg_studies[best_key]
            row = {
                "algorithm": alg,
                "fold": best_key,
                "best_score": best_study.best_value,
            }
            row.update(best_study.best_trial.params)
            best_params_rows.append(row)

        best_params_df = pd.DataFrame(best_params_rows).set_index("algorithm")
        best_params_df.to_csv(
            results_dir / f"best_params{tag}.tsv", sep="\t", index=True
        )
        print("[magic_now] Best hyperparameters per algorithm:")
        print(best_params_df.to_string())

    return validation, pipeline, studies
