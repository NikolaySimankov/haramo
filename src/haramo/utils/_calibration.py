###########
# Imports #
###########

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


###########
# Classes #
###########


class ThresholdedClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a fitted classifier and applies a custom decision threshold.

    ``predict_proba`` is delegated unchanged; ``predict`` returns 1 when the
    positive-class probability is ``>= threshold``, else 0. Used by
    ``magic_now`` when ``optimize_threshold=True`` so the saved model behaves
    as a sklearn classifier with the tuned cutoff baked in.
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


# Label-based metrics suitable for threshold tuning. Threshold-free metrics
# (PR AUC, ROC AUC, KS, Brier) are NOT in this map — passing them falls
# back to MCC with a printed note.
_THRESHOLD_METRIC_FNS = {
    "MCC": matthews_corrcoef,
    "F1-score": f1_score,
    "Bal. Acc.": balanced_accuracy_score,
    "Kappa": cohen_kappa_score,
}


def find_best_threshold(y_true, y_score, metric="MCC", n_thresholds=101):
    """Scan thresholds in ``[0, 1]``; return ``(best_threshold, best_value)``.

    ``metric`` is any identifier accepted by ``scoring_to_metric_column``.
    Threshold-free metrics (PR AUC, ROC AUC, KS, Brier) silently fall back
    to MCC.
    """
    from ._evaluation import scoring_to_metric_column

    metric_col = scoring_to_metric_column(metric, default="MCC")
    if metric_col not in _THRESHOLD_METRIC_FNS:
        print(
            f"[threshold] {metric_col!r} is threshold-free; "
            "falling back to MCC for threshold scan."
        )
        metric_col = "MCC"
    fn = _THRESHOLD_METRIC_FNS[metric_col]

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    scores = np.array([fn(y_true, (y_score >= t).astype(int)) for t in thresholds])
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


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
    raise ValueError(
        "Estimator has neither predict_proba nor decision_function"
    )


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


def plot_calibration_curve(variants, output_path, title=None, n_bins=10):
    """Side-by-side calibration plot for {uncalibrated, isotonic, sigmoid}.

    Parameters
    ----------
    variants : dict[str, dict]
        Mapping ``variant_name -> {"y_true": array, "y_score": array}``.
        One curve per variant; the Brier score is computed on each variant's
        scores and displayed in the legend.
    output_path : path-like
    title : str, optional
    n_bins : int, default 10
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"No calibration": "tab:red", "Isotonic": "tab:green",
              "Sigmoid": "tab:blue"}

    for name, data in variants.items():
        y_true = np.asarray(data["y_true"]).astype(int)
        y_score = np.asarray(data["y_score"], dtype=float)
        if y_score.size == 0:
            continue
        # Clip to [0,1] defensively; calibrated outputs are already there,
        # the uncalibrated min-max-scaled decision_function path is too.
        y_score = np.clip(y_score, 0.0, 1.0)
        prob_true, prob_pred = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy="quantile"
        )
        brier = brier_score_loss(y_true, y_score)
        ax.plot(
            prob_pred, prob_true, marker="o", lw=1.5,
            color=colors.get(name),
            label=f"{name} ({brier:.3f})",
        )

    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5,
        label="Perfectly calibrated",
    )
    ax.set(
        xlabel="Mean predicted probability", ylabel="Fraction of positives",
        xlim=(-0.01, 1.01), ylim=(-0.01, 1.01),
        title=title or "Calibration plots (Brier in legend)",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
