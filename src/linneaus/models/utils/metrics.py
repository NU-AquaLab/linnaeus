"""
Enhanced evaluation metrics for AS classification models.

This module provides functions for computing, comparing, visualising, and
persisting per-label and aggregate classification metrics.  It operates on
raw ``pandas.DataFrame`` objects (binary ground-truth vs. binary predictions)
and is intended as a lightweight complement to the higher-level
:class:`~linnaeus.models.evaluation.ClassificationEvaluator`.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    jaccard_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def get_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_score: Optional[pd.DataFrame] = None,
    print_results: bool = True,
    sortby: str = "Precision",
) -> Dict[str, Any]:
    """
    Compute comprehensive per-label and macro-averaged classification metrics.

    Parameters
    ----------
    y_true : pd.DataFrame
        Binary ground-truth labels (one column per tag).
    y_pred : pd.DataFrame
        Binary predicted labels (one column per tag).
    y_score : pd.DataFrame, optional
        Predicted scores / probabilities used to compute PR AUC.
    print_results : bool, default True
        Whether to log the results to the console.
    sortby : str, default ``"Precision"``
        Column name to sort the per-label metrics table by.

    Returns
    -------
    dict
        Dictionary containing:

        - ``label_metrics`` -- :class:`pd.DataFrame` with per-label scores.
        - ``accuracy`` -- Exact-match accuracy.
        - ``macro_precision``, ``macro_recall``, ``macro_f1``,
          ``macro_jaccard`` -- Macro-averaged scores.
        - ``mean_pr_auc`` -- Mean PR AUC (``None`` when *y_score* is not
          provided).
    """
    y_true = y_true.reindex(columns=y_pred.columns)
    if y_score is not None:
        y_score = y_score.reindex(columns=y_pred.columns)

    accuracy: float = accuracy_score(y_true, y_pred)
    precision: np.ndarray = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    recall: np.ndarray = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1: np.ndarray = f1_score(y_true, y_pred, average=None, zero_division=0)
    jaccard: np.ndarray = jaccard_score(y_true, y_pred, average=None, zero_division=0)

    pr_auc: List[Optional[float]]
    if y_score is not None:
        pr_auc = []
        for i in range(y_true.shape[1]):
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true.iloc[:, i], y_score.iloc[:, i]
            )
            pr_auc.append(auc(recall_curve, precision_curve))
    else:
        pr_auc = [None] * y_true.shape[1]

    macro_precision: float = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_recall: float = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1: float = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_jaccard: float = jaccard_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics_df = pd.DataFrame(
        {
            "Label": y_true.columns,
            "Accuracy": [
                accuracy_score(y_true.iloc[:, i], y_pred.iloc[:, i])
                for i in range(y_true.shape[1])
            ],
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Jaccard Score": jaccard,
            "PR AUC": pr_auc,
        }
    )
    metrics_df = metrics_df.sort_values(by=sortby, ascending=True)

    mean_pr_auc: Optional[str] = None
    if pr_auc[0] is not None:
        mean_pr_auc = f"{sum(p for p in pr_auc if p is not None) / len(pr_auc):.3f}"

    if print_results:
        logger.info("Accuracy: %s", accuracy)
        logger.info("Metrics per label:\n%s", metrics_df.to_string())
        logger.info(
            "Means  Accuracy: %.3f, Precision: %.3f, Recall: %.3f, "
            "F1 Score: %.3f, Jaccard Score: %.3f, PR AUC: %s",
            metrics_df["Accuracy"].mean(),
            macro_precision,
            macro_recall,
            macro_f1,
            macro_jaccard,
            mean_pr_auc,
        )

    return {
        "label_metrics": metrics_df,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_jaccard": macro_jaccard,
        "mean_pr_auc": mean_pr_auc,
    }


def compare_model_metrics(
    model_1_metrics: Dict[str, Any],
    model_2_metrics: Dict[str, Any],
    model_1_name: str,
    model_2_name: str,
    metric_to_compare: str = "All",
) -> None:
    """
    Plot side-by-side bar charts comparing per-label metrics of two models.

    Parameters
    ----------
    model_1_metrics : dict
        Metrics dictionary (as returned by :func:`get_metrics`) for the first
        model.
    model_2_metrics : dict
        Metrics dictionary for the second model.
    model_1_name : str
        Display name for the first model.
    model_2_name : str
        Display name for the second model.
    metric_to_compare : str, default ``"All"``
        A single metric name (e.g. ``"Precision"``) or ``"All"`` to compare
        Accuracy, Precision, Recall, F1 Score, and Jaccard Score.
    """
    import matplotlib.pyplot as plt  # lazy import

    model_1_df: pd.DataFrame = model_1_metrics["label_metrics"]
    model_2_df: pd.DataFrame = model_2_metrics["label_metrics"]
    merged_df = model_1_df.merge(
        model_2_df,
        on="Label",
        suffixes=(f"_{model_1_name}", f"_{model_2_name}"),
    )

    metrics: List[str]
    if metric_to_compare == "All":
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Jaccard Score"]
    else:
        metrics = [metric_to_compare]

    labels = merged_df["Label"]

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        plt.bar(
            labels,
            merged_df[f"{metric}_{model_1_name}"],
            width=0.4,
            label=model_1_name,
            align="center",
            color="skyblue",
            alpha=0.9,
        )
        plt.bar(
            labels,
            merged_df[f"{metric}_{model_2_name}"],
            width=0.4,
            label=model_2_name,
            align="edge",
            color="orange",
            alpha=0.7,
        )
        plt.xlabel("Categories")
        plt.ylabel(metric)
        plt.title(f"Comparison of {metric} Across Categories")
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()


def get_global_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only the aggregate (non-per-label) metrics from a results dictionary.

    Parameters
    ----------
    metrics : dict
        Full metrics dictionary as returned by :func:`get_metrics`.

    Returns
    -------
    dict
        Dictionary containing only the scalar / aggregate metrics.
    """
    return {key: value for key, value in metrics.items() if key != "label_metrics"}


def save_metrics_for_debugging(
    metrics: Dict[str, Any],
    y_preds: pd.DataFrame,
    output_dir: Union[str, Path],
    prefix: str,
) -> None:
    """
    Persist metrics and predictions to disk for offline debugging.

    Writes a human-readable text file with summary and per-label metrics, and a
    CSV file with the raw predictions.

    Parameters
    ----------
    metrics : dict
        Full metrics dictionary as returned by :func:`get_metrics`.
    y_preds : pd.DataFrame
        Predicted labels DataFrame.
    output_dir : str or Path
        Directory in which to write the output files.
    prefix : str
        Filename prefix used to distinguish different runs
        (e.g. ``"baseline"``, ``"finetuned_v2"``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"metrics_{prefix}.txt"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("===== SUMMARY METRICS =====\n")
        for key, value in metrics.items():
            if key == "label_metrics":
                continue
            if isinstance(value, np.generic):
                value = value.item()
            f.write(f"{key}: {value}\n")

        f.write("\n===== LABEL METRICS (DataFrame) =====\n")
        label_metrics = metrics.get("label_metrics")
        if isinstance(label_metrics, pd.DataFrame):
            f.write(label_metrics.to_string(index=False))
        else:
            raise ValueError("'label_metrics' key must contain a pandas DataFrame")

    preds_path = output_dir / f"y_preds_{prefix}.csv"
    y_preds.to_csv(preds_path, index=True)

    logger.info("Saved all metrics to: %s", filepath)
    logger.info("Saved predictions to: %s", preds_path)
