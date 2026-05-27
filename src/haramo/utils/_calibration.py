###########
# Imports #
###########

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
)

from ._evaluation import fn_fp_loss

from sklearn.model_selection import StratifiedKFold, train_test_split

import matplotlib.pyplot as plt

###########
# Classes #
###########


class ThresholdedClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a fitted classifier and applies a custom decision threshold.

    ``predict_proba`` is delegated unchanged; ``predict`` returns 1 when the
    positive-class probability is ``>= threshold``, else 0.
    """

    def __init__(self, estimator, threshold=0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y, **kw):
        self.estimator.fit(X, y, **kw)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def decision_function(self, X):
        return self.estimator.decision_function(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    @property
    def classes_(self):
        return self.estimator.classes_


class ScoreCalibrator(BaseEstimator):
    """Learn a 1-D mapping ``score → calibrated_probability``.

    Operates directly on a 1-D score array (positive-class probability or
    decision_function output) rather than on a feature matrix. Intended for
    nested-CV workflows where the base model is trained separately and
    calibration is fit on its out-of-fold scores — no pipeline refits.
    """

    def __init__(self, method="isotonic"):
        self.method = method

    def fit(self, y_true, y_score, sample_weight=None):
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=float).reshape(-1)

        if self.method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            try:
                calibrator.fit(y_score, y_true, sample_weight=sample_weight)
            except TypeError:
                calibrator.fit(y_score, y_true)
        elif self.method == "sigmoid":
            calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
            X_score = y_score.reshape(-1, 1)
            try:
                calibrator.fit(X_score, y_true, sample_weight=sample_weight)
            except TypeError:
                calibrator.fit(X_score, y_true)
        else:
            raise ValueError("method must be 'isotonic' or 'sigmoid'")

        self.calibrator_ = calibrator
        return self

    def predict(self, y_score):
        y_score = np.asarray(y_score, dtype=float).reshape(-1)
        if self.method == "isotonic":
            return np.clip(
                np.asarray(self.calibrator_.predict(y_score), dtype=float), 0.0, 1.0
            )
        if self.method == "sigmoid":
            proba = self.calibrator_.predict_proba(y_score.reshape(-1, 1))
            return np.clip(np.asarray(proba[:, 1], dtype=float), 0.0, 1.0)
        raise ValueError("method must be 'isotonic' or 'sigmoid'")

    def predict_proba(self, y_score):
        calibrated = self.predict(y_score)
        return np.column_stack([1.0 - calibrated, calibrated])


class CalibratedThresholdedClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible deployment wrapper: ``pipeline → calibrator → threshold``.

    Pickled artefact produced by ``magic_now`` when calibration and/or
    threshold tuning are enabled. Applies the same decision logic at
    inference as the per-fold magic_now diagnostics:

    1. ``pipeline`` produces raw positive-class scores (``predict_proba[:, 1]``
       or ``decision_function`` fallback).
    2. Optional ``calibrator`` (a :class:`ScoreCalibrator`) maps raw → calibrated.
    3. ``predict`` thresholds the calibrated probability.

    ``calibrator`` / ``threshold`` are typically fit on out-of-fold predictions
    from ``nested_crossval`` and should NOT be re-derived on the training set.
    ``fit(X, y)`` only refits the underlying pipeline.
    """

    def __init__(self, pipeline, calibrator=None, threshold=0.5):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.threshold = threshold

    def fit(self, X, y, **kw):
        self.pipeline.fit(X, y, **kw)
        return self

    def _raw_positive_score(self, X):
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.ravel()
        if hasattr(self.pipeline, "decision_function"):
            return self.pipeline.decision_function(X)
        raise ValueError(
            "Underlying pipeline has neither predict_proba nor decision_function"
        )

    def predict_proba(self, X):
        raw = self._raw_positive_score(X)
        if self.calibrator is not None:
            cal = self.calibrator.predict(raw)
        else:
            cal = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    @property
    def classes_(self):
        return self.pipeline.classes_


#############
# Functions #
#############


def calibrate_pipeline(pipeline, X, y, sample_weight=None, method="isotonic", cv=3):
    """Wrap an unfitted pipeline in CalibratedClassifierCV and fit it.

    ``method`` must be ``"isotonic"`` or ``"sigmoid"``. Returns the fitted
    calibrator, which exposes the same sklearn interface as the original
    pipeline.
    """
    base = clone(pipeline)
    cal = CalibratedClassifierCV(base, cv=cv, method=method)
    try:
        cal.fit(X, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        cal.fit(X, y)
    return cal


def pick_best_calibration_method(
    pipeline, X, y, sample_weight=None, random_state=42, holdout_frac=0.2
):
    """Pick ``"isotonic"`` vs ``"sigmoid"`` by Brier on a single stratified
    holdout.

    Cheap heuristic: one 80/20 stratified split, two
    ``CalibratedClassifierCV`` fits (cv=2 each, ~4 model fits total). The
    chosen method is then refit on full data via :func:`calibrate_pipeline`.
    """
    split_kwargs = dict(test_size=holdout_frac, stratify=y, random_state=random_state)
    if sample_weight is not None:
        X_tr, X_te, y_tr, y_te, sw_tr, _ = train_test_split(
            X, y, sample_weight, **split_kwargs
        )
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, **split_kwargs)
        sw_tr = None

    best, best_brier = "isotonic", float("inf")
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(clone(pipeline), cv=2, method=method)
        try:
            cal.fit(X_tr, y_tr, sample_weight=sw_tr)
        except (TypeError, ValueError):
            cal.fit(X_tr, y_tr)
        proba = cal.predict_proba(X_te)[:, 1]
        b = brier_score_loss(y_te, proba)
        if b < best_brier:
            best, best_brier = method, b
    return best, best_brier


def fit_score_calibrator(y_true, y_score, method="isotonic", sample_weight=None):
    """Fit a :class:`ScoreCalibrator` on out-of-fold scores."""
    return ScoreCalibrator(method=method).fit(
        y_true=y_true, y_score=y_score, sample_weight=sample_weight
    )


def pick_best_score_calibration_method(
    y_true, y_score, sample_weight=None, random_state=42, cv=4
):
    """Pick ``isotonic`` vs ``sigmoid`` by Brier on stratified k-fold splits
    of the OOF score array.

    Operates entirely on the 1-D ``(y_true, y_score)`` arrays produced by
    ``nested_crossval`` — **no model refits**. Returns ``(best_method,
    best_brier)`` where ``best_brier`` is the mean of held-out Brier across
    the k folds.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    brier_scores = {"isotonic": [], "sigmoid": []}

    for train_idx, test_idx in kf.split(y_score, y_true):
        y_tr, y_te = y_true[train_idx], y_true[test_idx]
        s_tr, s_te = y_score[train_idx], y_score[test_idx]
        sw_tr = sample_weight[train_idx] if sample_weight is not None else None
        for method in ("isotonic", "sigmoid"):
            calibrator = fit_score_calibrator(
                y_tr, s_tr, method=method, sample_weight=sw_tr
            )
            calibrated = calibrator.predict(s_te)
            brier_scores[method].append(brier_score_loss(y_te, calibrated))

    best_method, best_brier = "isotonic", float("inf")
    for method, scores in brier_scores.items():
        mean_brier = float(np.mean(scores))
        if mean_brier < best_brier:
            best_method, best_brier = method, mean_brier
    return best_method, best_brier


_THRESHOLD_METRIC_FNS = {
    "MCC": matthews_corrcoef,
    "F1-score": f1_score,
    "Bal. Acc.": balanced_accuracy_score,
    "Kappa": cohen_kappa_score,
    "FNFP": fn_fp_loss,
}


def find_best_threshold(y_true, y_score, metric="FNFP", n_thresholds=50):
    """Scan thresholds in ``[0, 1]``; return ``(best_threshold, best_value, metric_used)``.

    ``metric`` is any identifier accepted by ``scoring_to_metric_column``.
    For higher-is-better metrics (MCC, F1, ...) the threshold maximising the
    metric is returned; for lower-is-better losses (FNFP loss) the minimising
    threshold is returned. **Anything else** — threshold-free metrics (PR AUC,
    ROC AUC, KS, Brier), callables, or unrecognised strings — falls back to
    FNFP loss with a printed note. The third return value is the name of the
    metric actually used.
    """
    from ._evaluation import scoring_to_metric_column, metric_is_lower_better

    _UNRECOGNISED = object()
    resolved = scoring_to_metric_column(metric, default=_UNRECOGNISED)

    if resolved is _UNRECOGNISED:
        print(
            f"[threshold] scoring {metric!r} is not recognised for "
            "threshold tuning; falling back to FNFP."
        )
        metric_col = "FNFP"
    elif resolved not in _THRESHOLD_METRIC_FNS:
        print(
            f"[threshold] {resolved!r} is threshold-free; "
            "falling back to FNFP loss for threshold scan."
        )
        metric_col = "FNFP"
    else:
        metric_col = resolved

    fn = _THRESHOLD_METRIC_FNS[metric_col]
    lower_better = metric_is_lower_better(metric_col)

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.linspace(0.2, 0.8, n_thresholds)
    scores = np.array([fn(y_true, (y_score >= t).astype(int)) for t in thresholds])
    best_idx = int(np.argmin(scores) if lower_better else np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx]), metric_col


def _proba_for_display(estimator, X):
    """Return positive-class probabilities for an estimator.

    Falls back to a min-max-scaled ``decision_function`` (mirrors the
    ``NaivelyCalibratedLinearSVC`` pattern) when ``predict_proba`` is missing,
    so the uncalibrated curve still has something to plot for Ridge/SGD.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        df = estimator.decision_function(X)
        denom = df.max() - df.min()
        if denom <= 0:
            return np.full_like(df, 0.5, dtype=float)
        return np.clip((df - df.min()) / denom, 0.0, 1.0)
    raise ValueError("Estimator has neither predict_proba nor decision_function")


def compute_calibration_variants(
    pipeline, X, y, sample_weight=None, random_state=42, test_size=0.2
):
    """Fit pipeline + isotonic + sigmoid variants on a single 80/20 holdout.

    Returns a dict mapping variant name to ``{"y_true": array, "y_score":
    array}``, suitable for :func:`plot_calibration_curve`. Cheap diagnostic
    — ~5 fits total (1 uncalibrated + 2 × cv=2 calibrated).
    """
    split_kwargs = dict(test_size=test_size, stratify=y, random_state=random_state)
    if sample_weight is not None:
        X_tr, X_te, y_tr, y_te, sw_tr, _ = train_test_split(
            X, y, sample_weight, **split_kwargs
        )
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, **split_kwargs)
        sw_tr = None

    variants = {}

    uncal = clone(pipeline)
    try:
        uncal.fit(X_tr, y_tr, model__sample_weight=sw_tr)
    except (TypeError, ValueError):
        uncal.fit(X_tr, y_tr)
    variants["No calibration"] = {
        "y_true": np.asarray(y_te).astype(int),
        "y_score": _proba_for_display(uncal, X_te),
    }

    for method, label in (("isotonic", "Isotonic"), ("sigmoid", "Sigmoid")):
        cal = CalibratedClassifierCV(clone(pipeline), cv=2, method=method)
        try:
            cal.fit(X_tr, y_tr, sample_weight=sw_tr)
        except (TypeError, ValueError):
            cal.fit(X_tr, y_tr)
        variants[label] = {
            "y_true": np.asarray(y_te).astype(int),
            "y_score": cal.predict_proba(X_te)[:, 1],
        }

    return variants


def plot_calibration_curve(variants, output_path, title=None, n_bins=5):
    """Side-by-side calibration plot for {uncalibrated, isotonic, sigmoid}.

    Parameters
    ----------
    variants : dict[str, dict]
        Mapping ``variant_name -> {"y_true": array, "y_score": array}``.
        One curve per variant; the Brier score is computed on each variant's
        scores and displayed in the legend.
    output_path : path-like
    title : str, optional
    n_bins : int, default 5
        Number of equal-width bins across ``[0, 1]``. Combined with the
        ``"uniform"`` strategy below, this exposes confidence pile-ups
        (large empty bins between rare points) rather than collapsing
        well-separated models into 3–4 quantile-clustered points.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {
        "No calibration": "tab:red",
        "Isotonic": "tab:green",
        "Sigmoid": "tab:blue",
    }

    for name, data in variants.items():
        y_true = np.asarray(data["y_true"]).astype(int)
        y_score = np.asarray(data["y_score"], dtype=float)
        if y_score.size == 0:
            continue
        # Clip to [0,1] defensively; calibrated outputs are already there,
        # the uncalibrated min-max-scaled decision_function path is too.
        y_score = np.clip(y_score, 0.0, 1.0)
        prob_true, prob_pred = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy="uniform"
        )
        brier = brier_score_loss(y_true, y_score)
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            lw=1.5,
            color=colors.get(name),
            label=f"{name} ({brier:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        alpha=0.5,
        label="Perfectly calibrated",
    )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Fraction of positives",
        xlim=(-0.01, 1.01),
        ylim=(-0.01, 1.01),
        title=title or "Calibration plots (Brier in legend)",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
