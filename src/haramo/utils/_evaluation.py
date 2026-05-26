###########
# Imports #
###########

import pandas as pd
import numpy as np

from scipy.stats import (
    pointbiserialr,
    pearsonr,
    kendalltau,
    spearmanr,
)

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    auc,
    make_scorer,
    get_scorer,
)

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

#############
# Functions #
#############


def pearson_scorer(X, y):
    """
    Compute the Pearson correlation coefficient and p-value for each feature in X with respect to y.

    Args:
        X (numpy.ndarray): A 2D array where each column represents a feature.
        y (array-like): An array of target values.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - scores: The Pearson correlation coefficient for each feature.
            - pvalue: The p-value for the hypothesis test for each feature.
    """

    # Calculate Pearson correlation coefficients for each feature with respect to y
    scores = X.apply(lambda x: pearsonr(x, y).statistic)

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = X.apply(lambda x: pearsonr(x, y).pvalue)

    # Return the scores and p-values as a tuple
    return (scores, pvalue)


def kendall_scorer(X, y):
    """
    Compute the Kendall Tau correlation coefficient and p-value for each feature in X with respect to y.

    Args:
        X (numpy.ndarray): A 2D array where each column represents a feature.
        y (array-like): An array of target values.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - scores: The Kendall Tau correlation coefficient for each feature.
            - pvalue: The p-value for the hypothesis test for each feature.
    """

    # Calculate Kendall Tau correlation coefficients for each feature with respect to y
    scores = X.apply(lambda x: kendalltau(x, y).statistic)

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = X.apply(lambda x: kendalltau(x, y).pvalue)

    # Return the scores and p-values as a tuple
    return (scores, pvalue)


def spearman_scorer(X, y):
    """
    Compute the Spearman rank-order correlation coefficient and p-value for each feature in X with respect to y.

    Args:
        X (numpy.ndarray): A 2D array where each column represents a feature.
        y (array-like): An array of target values.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - scores: The Spearman rank-order correlation coefficient for each feature.
            - pvalue: The p-value for the hypothesis test for each feature.
    """

    # Calculate Spearman rank-order correlation coefficients for each feature with respect to y
    scores = X.apply(lambda x: spearmanr(x, y).statistic)

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = X.apply(lambda x: spearmanr(x, y).pvalue)

    # Return the scores and p-values as a tuple
    return (scores, pvalue)


def biserial_scorer(X, y):
    """
    Compute the Point-Biserial correlation coefficient and p-value for each feature in X with respect to y.

    Args:
        X (numpy.ndarray): A 2D array where each column represents a feature.
        y (array-like): An array of binary target values.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - scores: The Point-Biserial correlation coefficient for each feature.
            - pvalue: The p-value for the hypothesis test for each feature.
    """

    # Calculate Point-Biserial rank-order correlation coefficients for each feature with respect to y
    scores = X.apply(lambda x: pointbiserialr(x, y).statistic)

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = X.apply(lambda x: pointbiserialr(x, y).pvalue)

    # Return the scores and p-values as a tuple
    return (scores, pvalue)


mcc_scorer = make_scorer(matthews_corrcoef)

# PR-AUC (Average Precision) is better adapted to imbalanced binary than
# ROC-AUC because it focuses on the positive class. The response_method
# list tries predict_proba first, then falls back to decision_function so
# classifiers without predict_proba (e.g. RidgeClassifier) still score.
pr_auc_scorer = make_scorer(
    average_precision_score,
    response_method=["predict_proba", "decision_function"],
)

roc_auc_scorer = make_scorer(
    roc_auc_score,
    response_method=["predict_proba", "decision_function"],
)


def decile_table_ks(y_true, y_score, n_deciles=10):
    """Build a decile table for the KS statistic.

    Sorts instances by score (descending) and splits them into ``n_deciles``
    equal-sized buckets. For each cumulative bucket, computes:

    - ``cum_positive_rate``: share of all positives captured so far
    - ``cum_negative_rate``: share of all negatives captured so far
    - ``KS``: ``|cum_positive_rate - cum_negative_rate|``

    The maximum of ``KS`` is the Kolmogorov-Smirnov statistic.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]

    n = len(y_sorted)
    total_pos = max(int(y_true.sum()), 1)
    total_neg = max(int(n - y_true.sum()), 1)
    edges = np.linspace(0, n, n_deciles + 1, dtype=int)

    rows = []
    cum_pos = 0
    cum_neg = 0
    for i in range(n_deciles):
        bucket = y_sorted[edges[i] : edges[i + 1]]
        cum_pos += int(bucket.sum())
        cum_neg += int(len(bucket) - bucket.sum())
        cpr = cum_pos / total_pos
        cnr = cum_neg / total_neg
        rows.append(
            {
                "decile": i + 1,
                "cum_positive_rate": cpr,
                "cum_negative_rate": cnr,
                "KS": abs(cpr - cnr),
            }
        )
    return pd.DataFrame(rows)


def ks_statistic_score(y_true, y_score):
    """Compute the KS statistic from binary labels and positive-class scores.

    Returns a value in ``[0, 1]``; higher is better separation.
    """
    return float(decile_table_ks(y_true, y_score, n_deciles=10)["KS"].max())


ks_scorer = make_scorer(
    ks_statistic_score,
    response_method=["predict_proba", "decision_function"],
)


# Brier score loss is a proper scoring rule: lower is better. The Optuna
# objective treats sklearn scorers as "higher is better", so make_scorer
# with greater_is_better=False internally negates the value so Optuna
# minimises it correctly. response_method is restricted to predict_proba
# because decision_function outputs are not probabilities in [0, 1].
brier_scorer = make_scorer(
    brier_score_loss,
    response_method="predict_proba",
    greater_is_better=False,
)


def fn_fp_loss(y_true, y_pred, fp_weight=1.5):
    """Asymmetric label-based loss: ``(1 - sens)^2 + fp_weight * (1 - sel)^2``.

    ``sens`` (sensitivity / true-positive rate) and ``sel`` (selectivity /
    true-negative rate) are recall scores on the positive and negative class
    respectively. Both error terms are squared so larger gaps are penalised
    super-linearly; ``fp_weight`` (default 1.5) makes false positives more
    costly than false negatives. Lower is better.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sens = recall_score(y_true, y_pred, average="binary", zero_division=0)
    sel = recall_score(
        y_true, y_pred, pos_label=0, average="binary", zero_division=0
    )
    return float((1.0 - sens) ** 2 + fp_weight * (1.0 - sel) ** 2)


# Label-based, so no response_method needed. greater_is_better=False so
# Optuna minimises it via internal sign flip.
fn_fp_loss_scorer = make_scorer(fn_fp_loss, greater_is_better=False)


_INTERNAL_SCORERS = {
    "MCC": mcc_scorer,
    "PR AUC": pr_auc_scorer,
    "ROC AUC": roc_auc_scorer,
    "KS": ks_scorer,
    "Brier": brier_scorer,
    "FNFP Loss": fn_fp_loss_scorer,
}


def resolve_scorer(scoring):
    """Resolve a scoring argument to a sklearn-compatible scorer callable.

    Accepts an internal alias ("MCC", "PR AUC", "ROC AUC"), any
    string supported by sklearn's ``get_scorer``, or a callable
    (returned as-is).
    """
    if callable(scoring):
        return scoring
    if scoring in _INTERNAL_SCORERS:
        return _INTERNAL_SCORERS[scoring]
    return get_scorer(scoring)


# Maps any accepted scoring identifier to the matching column in
# ``classification_report``. Used by ``nested_crossval`` to rank
# folds/reductions by the same metric the HPO optimised.
_SCORING_TO_COLUMN = {
    "MCC": "MCC",
    "PR AUC": "PR AUC",
    "ROC AUC": "ROC AUC",
    "KS": "KS",
    "Brier": "Brier",
    "FNFP Loss": "FNFP Loss",
    "balanced_accuracy": "Bal. Acc.",
    "f1": "F1-score",
    "precision": "Precision",
    "recall": "Sensitivity",
    "average_precision": "PR AUC",
    "roc_auc": "ROC AUC",
    "neg_brier_score": "Brier",
}

# Columns in classification_report where lower values are better. Used by
# the ranking/early-stopping logic in nested_crossval so that loss-style
# metrics (Brier, FNFP Loss) pick the minimum rather than the maximum.
_LOWER_IS_BETTER_COLUMNS = {"Brier", "FNFP Loss"}


def scoring_to_metric_column(scoring, default="MCC"):
    """Map a ``scoring`` argument to the matching ``classification_report``
    column name. Falls back to ``default`` for callables or unknown strings.
    """
    if callable(scoring):
        return default
    return _SCORING_TO_COLUMN.get(scoring, default)


def metric_is_lower_better(metric_col: str) -> bool:
    """Return True when the named report column is a loss (lower=better)."""
    return metric_col in _LOWER_IS_BETTER_COLUMNS


def classification_report(true_value, predicted_value, y_score=None):
    """Compute classification metrics for a binary problem.

    Parameters
    ----------
    true_value : array-like
        Ground-truth labels.
    predicted_value : array-like
        Predicted labels.
    y_score : array-like, optional
        Positive-class score (probability or decision function). Required
        for PR AUC and ROC AUC; when omitted those entries are NaN so the
        resulting Series shape stays stable.
    """
    table = {}

    table["MCC"] = matthews_corrcoef(
        true_value,
        predicted_value,
    )

    table["F1-score"] = f1_score(
        true_value,
        predicted_value,
    )

    table["Kappa"] = cohen_kappa_score(
        true_value,
        predicted_value,
    )

    table["Bal. Acc."] = balanced_accuracy_score(
        true_value,
        predicted_value,
    )

    table["Precision"] = precision_score(
        true_value,
        predicted_value,
        average="binary",
    )

    table["Sensitivity"] = recall_score(
        true_value,
        predicted_value,
        average="binary",
    )

    table["Selectivity"] = recall_score(
        true_value,
        predicted_value,
        pos_label=0,
        average="binary",
    )

    table["FNFP Loss"] = fn_fp_loss(
        true_value,
        predicted_value,
        )

    if y_score is not None and not pd.isna(np.asarray(y_score, dtype=float)).all():
        table["PR AUC"] = average_precision_score(true_value, y_score)
        table["ROC AUC"] = roc_auc_score(true_value, y_score)
        table["KS"] = ks_statistic_score(true_value, y_score)
        # Brier requires probabilities in [0, 1]; only compute when the
        # score column looks like one (decision_function outputs don't).
        score_arr = np.asarray(y_score, dtype=float)
        if score_arr.size and score_arr.min() >= 0.0 and score_arr.max() <= 1.0:
            table["Brier"] = brier_score_loss(true_value, y_score)
        else:
            table["Brier"] = float("nan")
    else:
        table["PR AUC"] = float("nan")
        table["ROC AUC"] = float("nan")
        table["KS"] = float("nan")
        table["Brier"] = float("nan")

    return pd.Series(table)


def plot_confusion_matrix(true_values, predicted_values, title):
    """
    Plot and save a stacked 5 fold Cross-Validation confusion matrix as a heatmap.

    Args:
        true_values (array-like): Array of true values.
        predicted_values (array-like): Array of predicted values.
        title (str): Title for the plot.

    Returns:
        None
    """

    # Define the path for saving the plots
    path = Path(".")
    plots = path / "plots"
    plots.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

    # Get unique labels from true values and convert to float type
    labels = pd.Series(np.unique(true_values)).astype("float")

    # Calculate confusion matrix without normalization
    cm = confusion_matrix(true_values, predicted_values, labels=labels, normalize=None)
    cm = cm[~np.all(cm == 0, axis=1)]  # Remove rows that are all zeros
    cm_sum = np.sum(
        cm, axis=1, keepdims=True
    )  # Sum of each row for percentage calculation
    cm_perc = cm / cm_sum.astype(float) * 100  # Convert to percentages

    # Initialize an empty array for annotations
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]  # Count value
            p = cm_perc[i, j]  # Percentage value
            if i == j:
                s = cm_sum[i][0]  # Sum of the row
                annot[i, j] = (
                    f"{p:.1f}%\n{c:d}/{s:d}"  # Annotate with percentage and count/sum for diagonal
                )
            elif c == 0:
                annot[i, j] = ""  # Leave empty if count is zero
            else:
                annot[i, j] = (
                    f"{p:.1f}%\n{c:d}"  # Annotate with percentage and count for off-diagonal
                )

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    sns.heatmap(cm_perc, annot=annot, fmt="", ax=ax, vmin=0, vmax=100, cmap="Blues")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("True value")

    # Set title and labels
    title = f"{title}_stackedCV_confusion_matrix"
    ax.set_title(title)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot as a PDF file
    plt.savefig(plots / f"{title}.pdf")
    # plt.show()  # Uncomment to show the plot
    plt.close()  # Close the plot to free memory


def _aggregate_curves(curves, grid):
    """Interpolate per-fold curves onto a shared grid; return (mean, std).

    Each entry in ``curves`` is a tuple ``(x, y)`` where x is monotonically
    increasing — callers are responsible for the order-flip when needed
    (e.g. PR curves return recall decreasing).
    """
    interp = [np.interp(grid, x, y) for x, y in curves]
    interp = np.asarray(interp)
    return interp.mean(axis=0), interp.std(axis=0)


def plot_pr_curve(predictions_by_model, output_path, title=None, mark_cutoff=0.5):
    """Plot one PR curve per competing outer-fold model + mean +/- std band.

    Parameters
    ----------
    predictions_by_model : dict[str, pd.DataFrame]
        Mapping ``model_key -> DataFrame`` with columns
        ``["true", "predicted", "score"]``. Each entry is the model's
        out-of-fold predictions concatenated across its CV test splits.
    output_path : path-like
        Destination PDF path.
    title : str, optional
    mark_cutoff : float, default 0.5
        Probability threshold to highlight as the default decision point;
        the average ``(recall, precision)`` at this threshold across models
        is plotted as a marker. Pass ``None`` to skip the marker.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    aps, per_model = [], []
    recall_grid = np.linspace(0, 1, 100)

    for model_key, df in sorted(predictions_by_model.items()):
        y_true = df["true"].to_numpy()
        y_score = df["score"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)
        # precision_recall_curve returns recall decreasing; flip for np.interp
        per_model.append((recall[::-1], precision[::-1]))
        ax.plot(
            recall, precision, alpha=0.4, lw=1,
            label=f"{model_key} (AP={ap:.2f})",
        )

    mean_p, std_p = _aggregate_curves(per_model, recall_grid)
    ax.plot(
        recall_grid, mean_p, color="b", lw=2,
        label=fr"Mean PR (AP = {np.mean(aps):.2f} $\pm$ {np.std(aps):.2f})",
    )
    ax.fill_between(
        recall_grid,
        np.clip(mean_p - std_p, 0, 1),
        np.clip(mean_p + std_p, 0, 1),
        color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.",
    )

    if mark_cutoff is not None:
        rs, ps = [], []
        for df in predictions_by_model.values():
            y_true = df["true"].to_numpy()
            y_score = df["score"].to_numpy()
            y_pred_cut = (y_score >= mark_cutoff).astype(int)
            rs.append(recall_score(y_true, y_pred_cut, zero_division=0))
            ps.append(precision_score(y_true, y_pred_cut, zero_division=0))
        ax.plot(
            np.mean(rs), np.mean(ps), marker="o", markersize=10,
            color="tab:blue",
            label=f"Default cut-off @ p={mark_cutoff}",
        )

    ax.set(
        xlabel="Recall", ylabel="Precision",
        xlim=(-0.01, 1.01), ylim=(-0.01, 1.01),
        title=title or "Precision-Recall curve (outer-fold models)",
    )
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_ks_statistic(predictions_by_model, output_path, title=None, n_deciles=10):
    """Plot per-model KS curves (cum-positive vs cum-negative) + mean band.

    For each competing outer-fold model, builds a decile table and plots its
    cumulative positive- and negative-class rates against decile rank. A bold
    mean curve and ``+/- 1`` std band are overlaid across models. The KS
    statistic (max gap between the two mean curves) is annotated.

    Parameters
    ----------
    predictions_by_model : dict[str, pd.DataFrame]
        Mapping ``model_key -> DataFrame`` with columns
        ``["true", "predicted", "score"]``.
    output_path : path-like
    title : str, optional
    n_deciles : int, default 10
        Number of buckets to split the score-ranked instances into.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    deciles = np.arange(0, n_deciles + 1)

    pos_curves, neg_curves, ks_maxes = [], [], []

    for model_key, df in sorted(predictions_by_model.items()):
        dt = decile_table_ks(
            df["true"].to_numpy(), df["score"].to_numpy(), n_deciles=n_deciles
        )
        cum_pos = np.append(0.0, dt["cum_positive_rate"].values)
        cum_neg = np.append(0.0, dt["cum_negative_rate"].values)
        pos_curves.append(cum_pos)
        neg_curves.append(cum_neg)
        ks_max = float(dt["KS"].max())
        ks_maxes.append(ks_max)

        ax.plot(deciles, cum_pos, alpha=0.35, color="tab:green", lw=1)
        ax.plot(deciles, cum_neg, alpha=0.35, color="tab:red", lw=1)

    pos_arr = np.asarray(pos_curves)
    neg_arr = np.asarray(neg_curves)
    mean_pos, std_pos = pos_arr.mean(axis=0), pos_arr.std(axis=0)
    mean_neg, std_neg = neg_arr.mean(axis=0), neg_arr.std(axis=0)

    ax.plot(deciles, mean_pos, marker="o", color="tab:green", lw=2,
            label="Mean cum. positives")
    ax.plot(deciles, mean_neg, marker="o", color="tab:red", lw=2,
            label="Mean cum. negatives")
    ax.fill_between(deciles, np.clip(mean_pos - std_pos, 0, 1),
                    np.clip(mean_pos + std_pos, 0, 1),
                    color="tab:green", alpha=0.15)
    ax.fill_between(deciles, np.clip(mean_neg - std_neg, 0, 1),
                    np.clip(mean_neg + std_neg, 0, 1),
                    color="tab:red", alpha=0.15)

    diffs = np.abs(mean_pos - mean_neg)
    ks_decile_mean = int(np.argmax(diffs))
    mean_ks = float(np.mean(ks_maxes))
    std_ks = float(np.std(ks_maxes))
    ax.plot(
        [ks_decile_mean, ks_decile_mean],
        [mean_neg[ks_decile_mean], mean_pos[ks_decile_mean]],
        linestyle="--", marker="o", color="tab:blue", lw=2,
        label=fr"KS = {mean_ks:.3f} $\pm$ {std_ks:.3f} @ decile {ks_decile_mean}",
    )

    ax.set(
        xlabel="Deciles", ylabel="Cumulative rate",
        xlim=(0, n_deciles), ylim=(0, 1),
        title=title or "KS statistic (outer-fold models)",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_roc_curve(predictions_by_model, output_path, title=None):
    """Plot one ROC curve per competing outer-fold model + mean +/- std band.

    Parameters
    ----------
    predictions_by_model : dict[str, pd.DataFrame]
        Mapping ``model_key -> DataFrame`` with columns
        ``["true", "predicted", "score"]``. Each entry is the model's
        out-of-fold predictions concatenated across its CV test splits.
    output_path : path-like
    title : str, optional
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    aucs, per_model = [], []
    fpr_grid = np.linspace(0, 1, 100)

    for model_key, df in sorted(predictions_by_model.items()):
        y_true = df["true"].to_numpy()
        y_score = df["score"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc_val = auc(fpr, tpr)
        aucs.append(roc_auc_val)
        per_model.append((fpr, tpr))
        ax.plot(
            fpr, tpr, alpha=0.4, lw=1,
            label=f"{model_key} (AUC={roc_auc_val:.2f})",
        )

    mean_tpr, std_tpr = _aggregate_curves(per_model, fpr_grid)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    ax.plot(
        fpr_grid, mean_tpr, color="b", lw=2,
        label=fr"Mean ROC (AUC = {np.mean(aucs):.2f} $\pm$ {np.std(aucs):.2f})",
    )
    ax.fill_between(
        fpr_grid,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.",
    )
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Chance",
    )

    ax.set(
        xlabel="False Positive Rate", ylabel="True Positive Rate",
        xlim=(-0.01, 1.01), ylim=(-0.01, 1.01),
        title=title or "ROC curve (outer-fold models)",
    )
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
