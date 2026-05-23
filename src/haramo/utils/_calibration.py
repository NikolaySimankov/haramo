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


def plot_calibration_curve(predictions_by_model, output_path, title=None, n_bins=10):
    """Reliability diagram per outer-fold model + mean curve with std band.

    Parameters
    ----------
    predictions_by_model : dict[str, pd.DataFrame]
        Mapping ``model_key -> DataFrame`` with columns
        ``["true", "predicted", "score"]``. Score values must be probabilities
        in ``[0, 1]``; models whose score column lies outside that range
        (e.g. decision_function fallback) are silently skipped.
    output_path : path-like
    title : str, optional
    n_bins : int, default 10
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    grid = np.linspace(0, 1, n_bins)
    per_model = []

    for model_key, df in sorted(predictions_by_model.items()):
        y_true = df["true"].to_numpy().astype(int)
        y_score = df["score"].to_numpy(dtype=float)
        if y_score.size == 0 or y_score.min() < 0 or y_score.max() > 1:
            continue
        prob_true, prob_pred = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy="quantile"
        )
        per_model.append((prob_pred, prob_true))
        ax.plot(
            prob_pred, prob_true, marker="o", alpha=0.4, lw=1,
            label=f"{model_key}",
        )

    if per_model:
        interp = np.asarray([np.interp(grid, x, y) for x, y in per_model])
        mean_c, std_c = interp.mean(axis=0), interp.std(axis=0)
        ax.plot(grid, mean_c, color="b", lw=2, label="Mean")
        ax.fill_between(
            grid,
            np.clip(mean_c - std_c, 0, 1),
            np.clip(mean_c + std_c, 0, 1),
            color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.",
        )

    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5,
        label="Perfectly calibrated",
    )
    ax.set(
        xlabel="Mean predicted probability", ylabel="Fraction of positives",
        xlim=(-0.01, 1.01), ylim=(-0.01, 1.01),
        title=title or "Calibration curve (outer-fold models)",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
