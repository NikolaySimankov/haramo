###########
# Imports #
###########

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from ..utils import reduce_dataset, resolve_scorer

from optuna import create_study
from optuna.samplers import TPESampler

from ._feature_selection import instantiate_feature_selector

#############
# Functions #
#############


def _fs_trial_budget(feature_selector) -> int:
    """
    Number of trials dedicated to the feature-selection HPO phase.

    +----------------------------------------------+--------+
    | Components being optimised                   | Trials |
    +----------------------------------------------+--------+
    | Variance filter only (feature_selector=None) |      5 |
    | Variance filter + pvalue or boruta           |     10 |
    | Variance filter + pvalue + boruta (optimize) |     15 |
    +----------------------------------------------+--------+
    """
    if feature_selector is None:
        return 5
    if feature_selector in ("pvalue", "boruta"):
        return 10
    if isinstance(feature_selector, list):
        return 10 if len(feature_selector) <= 1 else 15
    if feature_selector == "optimize":
        return 15
    return 10


def _fs_objective(
    trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_selector,
    task: str,
    scoring,
    random_state: int,
    inner_cv_groups,
    n_jobs: int,
    inner_splits=None,
    inner_reduced_indices=None,
) -> float:
    """
    Inner-CV objective for phase 1 (feature-selection HPO).

    Feature-selection hyperparameters are optimised; scaler and model are
    fixed to StandardScaler + LGBM (default params) so no compute is wasted
    on the rest of the pipeline during this phase.

    All available CPUs (``n_jobs``) are forwarded to the boruta / random-forest
    estimators; trials themselves run sequentially.
    """
    # Deferred imports to avoid the circular dependency:
    # feature_selection ← classification._instantiation ← feature_selection
    from ..utils._scalers import instantiate_standard_scaler
    from ..classification._instantiation import instantiate_model

    # instantiate_feature_selector already returns Pipeline([filter, selector])
    feature_selector_pipeline = instantiate_feature_selector(
        trial,
        task=task,
        method=feature_selector,
        hyperparameters="optimize",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    # Scaler and model are fixed to defaults: no trial.suggest_* calls are made.
    scaler = instantiate_standard_scaler(trial, hyperparameters="default")

    model = instantiate_model(
        trial,
        algorithm="LGBM",
        hyperparameters="default",
        random_state=random_state,
        n_jobs=1,
    )

    pipeline = Pipeline(
        feature_selector_pipeline.steps + [("scaler", scaler), ("model", model)]
    )

    scorer = resolve_scorer(scoring)

    if inner_splits is not None:
        splits = inner_splits
    elif inner_cv_groups is not None:
        splits = list(
            StratifiedGroupKFold(n_splits=3).split(
                X_train, y_train.astype(str), groups=inner_cv_groups
            )
        )
    else:
        splits = list(
            StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state).split(
                X_train, y_train.astype(str)
            )
        )

    scores = []
    for i, (train_idx, val_idx) in enumerate(splits):
        pipe = clone(pipeline)
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        try:
            if inner_reduced_indices is not None:
                reduced_index = inner_reduced_indices[i]
            else:
                reduced_index = reduce_dataset(
                    X=X_tr,
                    y=y_tr,
                    target_size=2000,
                    stage2_shrink=1,
                    class_weight="balanced",
                    random_state=random_state,
                    verbose=False,
                )
            pipe.fit(X_tr.loc[reduced_index], y_tr.loc[reduced_index])
            scores.append(scorer(pipe, X_val, y_val))

        except Exception:
            scores.append(np.nan)

    score = float(pd.Series(scores, dtype=float).fillna(0.01).mean())

    return score


def select_best_feature_selector(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_selector,
    task: str = "classification",
    scoring: Union[str, Callable] = "AUC",
    random_state: int = 42,
    inner_cv_groups=None,
    n_jobs: int = 1,
) -> Pipeline:
    """
    Phase-1 HPO: find the best feature-selection configuration for one outer
    fold using a small trial budget and a fixed evaluation model.

    The trial budget scales with the complexity of the selector search space
    (5 / 10 / 15 trials — see ``_fs_trial_budget``).  Scaler and model are
    fixed to StandardScaler + LGBM (default hyperparameters) so that every
    trial focuses entirely on feature-selection decisions.  All available CPUs
    are forwarded to the boruta / random-forest internals; Optuna trials run
    sequentially.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (outer fold's training split).
    y_train : pd.Series
        Corresponding target vector.
    feature_selector : str, list, or None
        Same argument accepted by ``instantiate_feature_selector``.
    task : str
        ``"classification"`` or ``"regression"``.
    scoring : str or callable, default ``"AUC"``
        Scorer for the inner 3-fold CV. Internal aliases: ``"AUC"`` (PR-AUC),
        ``"PR_AUC"``, ``"ROC_AUC"``, ``"MCC"``. Any sklearn scorer string or
        callable is also accepted.
    random_state : int
    inner_cv_groups : array-like, optional
        Group labels for ``StratifiedGroupKFold`` in the inner CV.
    n_jobs : int
        Forwarded to the boruta / random-forest estimators inside the selector.

    Returns
    -------
    fs_pipeline : sklearn.pipeline.Pipeline
        A *fitted* two-step pipeline ``[("filter", ...), ("feature_selector", ...)]``
        built from the best trial's parameters and trained on the full
        ``X_train``.  Ready to be used with ``.transform()``.
    """
    n_trials = _fs_trial_budget(feature_selector)

    print(f"[FS HPO] {n_trials} trial(s), feature_selector={feature_selector!r} …")

    # Pre-compute inner CV splits and reduced indices once; reused across all FS trials
    if inner_cv_groups is not None:
        _fs_splits = list(
            StratifiedGroupKFold(n_splits=3).split(
                X_train, y_train.astype("str"), groups=inner_cv_groups
            )
        )
    else:
        _fs_splits = list(
            StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state).split(
                X_train, y_train.astype("str")
            )
        )
    _fs_reduced = [
        reduce_dataset(
            X=X_train.iloc[tr],
            y=y_train.iloc[tr],
            target_size=2000,
            stage2_shrink=1,
            class_weight="balanced",
            random_state=random_state,
            verbose=False,
        )
        for tr, _ in _fs_splits
    ]

    study = create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state, multivariate=True),
    )
    study.optimize(
        lambda trial: _fs_objective(
            trial,
            X_train,
            y_train,
            feature_selector,
            task,
            scoring,
            random_state,
            inner_cv_groups,
            n_jobs,
            inner_splits=_fs_splits,
            inner_reduced_indices=_fs_reduced,
        ),
        n_trials=n_trials,
        n_jobs=1,  # sequential: all CPUs go to boruta RF inside each trial
    )

    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None

    if best_trial is None or best_trial.value == 0.0:
        # Every trial failed or eliminated all features.  Fall back to an
        # identity transformer so _train_fold can continue with the full set.
        from sklearn.preprocessing import FunctionTransformer

        print(
            "[FS HPO] all trials failed — falling back to identity (no feature selection)"
        )
        fallback = Pipeline([("feature_selector", FunctionTransformer())])
        fallback.fit(X_train, y_train)
        return fallback

    print(f"[FS HPO] best score={best_trial.value:.4f} | params={best_trial.params}")

    # Reconstruct the FS pipeline from the best trial's params and fit it on the
    # full training fold so it is ready to transform X_train / X_test.
    # instantiate_feature_selector returns Pipeline([filter, selector]) so no
    # manual variance-filter wrapping is needed here.
    fs_pipeline = instantiate_feature_selector(
        best_trial,
        task=task,
        method=feature_selector,
        hyperparameters="optimize",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    try:
        fs_pipeline.fit(X_train, y_train)
    except ValueError:
        from sklearn.preprocessing import FunctionTransformer

        print(
            "[FS HPO] best trial params eliminated all features on full fold — falling back to identity"
        )
        fs_pipeline = Pipeline([("feature_selector", FunctionTransformer())])
        fs_pipeline.fit(X_train, y_train)

    return fs_pipeline
