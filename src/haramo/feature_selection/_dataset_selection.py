###########
# Imports #
###########

from __future__ import annotations

from itertools import combinations as _iter_combinations
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC
from ..utils import reduce_dataset, resolve_scorer

#############
# Functions #
#############


def _build_default_pipeline(task: str, random_state: int):
    """
    Build a minimal pipeline with default hyperparameters for fast dataset evaluation.

    Uses LGBM + StandardScaler with no feature selection beyond the variance
    filter.  Passing concrete strings to ``instantiate_pipeline`` bypasses every
    ``trial.suggest_*`` branch, so the dummy Optuna trial is never queried.
    """
    # Deferred import to avoid a circular dependency at module load time
    # (feature_selection ← classification ← feature_selection).
    from optuna import create_study
    from ..classification import instantiate_pipeline

    study = create_study()
    trial = study.ask()

    return instantiate_pipeline(
        trial,
        task=task,
        feature_selector=None,  # identity – no suggest call
        scaler="standard",  # concrete string – no suggest call
        algorithm="LGBM",  # concrete string – no suggest call
        hyperparameters="default",
        random_state=random_state,
        n_jobs=1,
    )


def _score_dataset_combo(
    X_combo: pd.DataFrame,
    y: pd.Series,
    scoring: Union[str, Callable],
    task: str,
    random_state: int,
    inner_cv_groups,
) -> float:
    """
    Score a single dataset combination via a lightweight 3-fold CV.

    This helper intentionally skips the ``reduce_dataset`` step used in the
    main training loop — the goal is a cheap informative signal, not a full
    simulation of training.

    Returns
    -------
    float
        Mean CV score, or ``nan`` if every fold raised an exception.
    """
    pipeline = _build_default_pipeline(task, random_state)

    scorer = resolve_scorer(scoring)

    if inner_cv_groups is not None:
        cv = StratifiedGroupKFold(n_splits=4)
        splits = list(cv.split(X_combo, y.astype(str), groups=inner_cv_groups))
    else:
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
        splits = list(cv.split(X_combo, y.astype(str)))

    scores = []
    for train_idx, test_idx in splits:
        pipe = clone(pipeline)
        X_train, X_test = X_combo.iloc[train_idx], X_combo.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        reduced_index = reduce_dataset(
            X=X_train,
            y=y_train,
            target_size=2000,
            stage2_shrink=1,
            class_weight="balanced",
            random_state=random_state,
            verbose=False,
        )

        try:
            pipe.fit(X_train.loc[reduced_index], y_train.loc[reduced_index])
            scores.append(scorer(pipe, X_test, y_test))
        except Exception:
            scores.append(np.nan)

    score = float(pd.Series(scores, dtype=float).fillna(0.01).mean())

    return score


def select_best_dataset_combo(
    datasets: Dict[str, pd.DataFrame],
    y: pd.Series,
    scoring: Union[str, Callable] = "PR AUC",
    task: str = "classification",
    random_state: int = 42,
    inner_cv_groups=None,
    n_jobs: int = 1,
    beam_width: int = 2,
) -> Tuple[str, pd.DataFrame, pd.Series]:
    """
    Beam-search greedy forward dataset selection.

    Algorithm
    ---------
    1. Score every singleton in parallel → rank, keep the top ``beam_width``
       as the current beam.
    2. For every combo in the beam, score all extensions (combo + one remaining
       dataset not already in that combo) in parallel.
    3. Rank all extension scores → keep the top ``beam_width`` as the new beam.
       Stop if the best score in the new beam does not improve on the best score
       from the previous step.
    4. Return the overall best combo seen across all steps.

    Worst-case evaluations: ``n + beam_width*(n-1) + beam_width*(n-2) + …``
    which for ``beam_width=2`` and ``n`` datasets is roughly ``2n²/2 = n²``,
    far cheaper than exhaustive ``2^n``.

    Parameters
    ----------
    datasets : dict
        Mapping ``name → DataFrame``.  All DataFrames must share the same
        index as ``y``.
    y : pd.Series
        Target vector.
    scoring : str or callable, default ``"PR AUC"``
        Internal aliases:  ``"PR AUC"``, ``"ROC AUC"``,
        ``"MCC"``. Any sklearn scorer string or callable is also accepted.
    task : str, default ``"classification"``
    random_state : int, default 42
    inner_cv_groups : array-like, optional
    n_jobs : int, default 1
        All CPUs used for parallel combo evaluation at each step.
    beam_width : int, default 2
        Number of top combos carried forward at each step.

    Returns
    -------
    best_combo_name : str
        Dataset names joined by ``" + "``.
    best_X : pd.DataFrame
        Concatenated feature matrix of the winning combination.
    scores : pd.Series
        Mean CV score for every evaluated combination (all steps),
        sorted descending.
    """
    names = list(datasets.keys())
    n = len(names)
    all_scores: dict = {}

    def _score_one(combo: tuple) -> tuple:
        X_combo = pd.concat([datasets[name] for name in combo], axis=1)
        score = _score_dataset_combo(
            X_combo, y, scoring, task, random_state, inner_cv_groups
        )
        return combo, score

    # ------------------------------------------------------------------ #
    # Step 1 – score every singleton, seed the beam                       #
    # ------------------------------------------------------------------ #
    print(f"[Dataset Selection] Step 1: scoring {n} singleton(s) …")
    results = Parallel(n_jobs=n_jobs)(delayed(_score_one)((name,)) for name in names)
    for combo, score in results:
        all_scores[" + ".join(combo)] = score

    results_sorted = sorted(
        results, key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True
    )
    beam = [combo for combo, _ in results_sorted[:beam_width]]
    best_score = results_sorted[0][1]

    print(
        f"[Dataset Selection] Top-{beam_width} singletons: "
        + ", ".join(f"{' + '.join(c)!r}" for c in beam)
        + f" | best score={best_score:.4f}"
    )

    # ------------------------------------------------------------------ #
    # Steps 2+ – beam extension                                           #
    # ------------------------------------------------------------------ #
    step = 2
    while True:
        # Build all candidate extensions across every combo in the beam.
        # Each combo is extended with every dataset not already in it.
        candidates = [
            combo + (name,) for combo in beam for name in names if name not in combo
        ]

        if not candidates:
            break

        # Deduplicate: the same set of datasets can arise from different beam
        # members (unlikely but possible with beam_width > 1).
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = frozenset(c)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        print(
            f"[Dataset Selection] Step {step}: trying {len(unique_candidates)} "
            f"extension(s) from {len(beam)} beam member(s) …"
        )
        results = Parallel(n_jobs=n_jobs)(
            delayed(_score_one)(combo) for combo in unique_candidates
        )
        for combo, score in results:
            all_scores[" + ".join(combo)] = score

        results_sorted = sorted(
            results,
            key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf,
            reverse=True,
        )
        best_candidate_score = results_sorted[0][1]

        if best_candidate_score <= best_score:
            print(
                f"[Dataset Selection] No improvement "
                f"(best extension score={best_candidate_score:.4f} "
                f"<= current best={best_score:.4f}). Stopping."
            )
            break

        beam = [combo for combo, _ in results_sorted[:beam_width]]
        best_score = best_candidate_score
        print(
            f"[Dataset Selection] Top-{beam_width} at step {step}: "
            + ", ".join(f"{' + '.join(c)!r}" for c in beam)
            + f" | best score={best_score:.4f}"
        )
        step += 1

    # Overall best combo across all steps
    best_combo_name = max(
        all_scores,
        key=lambda k: all_scores[k] if not np.isnan(all_scores[k]) else -np.inf,
    )
    best_X = pd.concat(
        [datasets[name] for name in best_combo_name.split(" + ")], axis=1
    )
    scores = pd.Series(all_scores, name="score").sort_values(ascending=False)

    return best_combo_name, best_X, scores
