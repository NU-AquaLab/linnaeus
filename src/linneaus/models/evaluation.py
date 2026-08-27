"""
Evaluation module for AS classification models.

This module provides comprehensive evaluation metrics and analysis tools
for assessing the performance of AS classification models.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from .schemas import ASClassification, ClassificationTags

console = Console()


class ClassificationEvaluator:
    """
    Evaluator for AS classification model performance.

    This class provides comprehensive evaluation metrics including accuracy,
    precision, recall, F1-score, and specialized multi-label metrics.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.tag_names = [tag.value for tag in ClassificationTags]

    def evaluate_predictions(
        self,
        predictions: List[ASClassification],
        ground_truth: pd.DataFrame,
        include_pr_auc: bool = True,
        print_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions against ground truth labels.

        Parameters
        ----------
        predictions : List[ASClassification]
            Model predictions.
        ground_truth : pd.DataFrame
            Ground truth labels with ASN as index and binary tag columns.
        include_pr_auc : bool, default True
            Whether to compute PR AUC (requires confidence scores).
        print_results : bool, default True
            Whether to print evaluation results.

        Returns
        -------
        Dict[str, Any]
            Comprehensive evaluation metrics.
        """
        # Convert predictions to binary format
        y_pred, y_true = self._prepare_evaluation_data(predictions, ground_truth)

        if y_pred.empty or y_true.empty:
            raise ValueError(
                "No matching ASNs found between predictions and ground truth"
            )

        # Ensure same order
        common_asns = y_pred.index.intersection(y_true.index)
        y_pred = y_pred.loc[common_asns]
        y_true = y_true.loc[common_asns]

        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred, include_pr_auc, predictions)

        if print_results:
            self._print_evaluation_results(metrics)

        return metrics

    def evaluate_from_files(
        self,
        predictions_path: Path,
        ground_truth_path: Path,
        include_pr_auc: bool = True,
        print_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate predictions from files.

        Parameters
        ----------
        predictions_path : Path
            Path to predictions file (JSON or CSV).
        ground_truth_path : Path
            Path to ground truth CSV file.
        include_pr_auc : bool, default True
            Whether to compute PR AUC.
        print_results : bool, default True
            Whether to print results.

        Returns
        -------
        Dict[str, Any]
            Evaluation metrics.
        """
        # Load predictions
        if predictions_path.suffix.lower() == ".json":
            predictions = self._load_predictions_from_json(predictions_path)
        else:
            predictions = self._load_predictions_from_csv(predictions_path)

        # Load ground truth
        ground_truth = pd.read_csv(ground_truth_path, index_col="asn")

        return self.evaluate_predictions(
            predictions, ground_truth, include_pr_auc, print_results
        )

    def compare_models(
        self,
        model_results: Dict[str, List[ASClassification]],
        ground_truth: pd.DataFrame,
        print_comparison: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models' performance.

        Parameters
        ----------
        model_results : Dict[str, List[ASClassification]]
            Dictionary mapping model names to their predictions.
        ground_truth : pd.DataFrame
            Ground truth labels.
        print_comparison : bool, default True
            Whether to print comparison table.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Evaluation metrics for each model.
        """
        all_metrics = {}

        for model_name, predictions in model_results.items():
            console.print(f"\n[blue]Evaluating {model_name}...[/blue]")
            metrics = self.evaluate_predictions(
                predictions, ground_truth, include_pr_auc=False, print_results=False
            )
            all_metrics[model_name] = metrics

        if print_comparison:
            self._print_model_comparison(all_metrics)

        return all_metrics

    def _prepare_evaluation_data(
        self, predictions: List[ASClassification], ground_truth: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for evaluation."""
        # Convert predictions to binary DataFrame
        pred_data = {}
        for pred in predictions:
            pred_data[pred.asn] = {tag.value: 0 for tag in ClassificationTags}
            for tag in pred.tags:
                pred_data[pred.asn][tag.value] = 1

        y_pred = pd.DataFrame.from_dict(pred_data, orient="index")

        # Ensure all tag columns exist in both DataFrames
        for tag in self.tag_names:
            if tag not in y_pred.columns:
                y_pred[tag] = 0
            if tag not in ground_truth.columns:
                ground_truth[tag] = 0

        # Select only common columns
        common_tags = list(set(y_pred.columns) & set(ground_truth.columns))
        y_pred = y_pred[common_tags]

        # Filter ground truth to match prediction tags
        y_true = ground_truth[common_tags]

        return y_pred.fillna(0).astype(int), y_true.fillna(0).astype(int)

    def _compute_metrics(
        self,
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
        include_pr_auc: bool,
        predictions: List[ASClassification],
    ) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        metrics = {}

        # Overall metrics
        metrics["overall_accuracy"] = accuracy_score(y_true, y_pred)
        metrics["macro_precision"] = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["macro_recall"] = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["macro_jaccard"] = jaccard_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        # Micro averages
        metrics["micro_precision"] = precision_score(
            y_true, y_pred, average="micro", zero_division=0
        )
        metrics["micro_recall"] = recall_score(
            y_true, y_pred, average="micro", zero_division=0
        )
        metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

        # Per-label metrics
        precision_per_label = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_label = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
        jaccard_per_label = jaccard_score(y_true, y_pred, average=None, zero_division=0)

        # Accuracy per label (since it's multi-label)
        accuracy_per_label = []
        for i in range(y_true.shape[1]):
            accuracy_per_label.append(
                accuracy_score(y_true.iloc[:, i], y_pred.iloc[:, i])
            )

        # PR AUC if confidence scores available
        pr_auc_per_label = None
        if include_pr_auc:
            pr_auc_per_label = self._compute_pr_auc(y_true, predictions)

        # Create per-label metrics DataFrame
        label_metrics_data = {
            "label": y_true.columns.tolist(),
            "accuracy": accuracy_per_label,
            "precision": precision_per_label.tolist(),
            "recall": recall_per_label.tolist(),
            "f1_score": f1_per_label.tolist(),
            "jaccard_score": jaccard_per_label.tolist(),
        }

        if pr_auc_per_label is not None:
            label_metrics_data["pr_auc"] = pr_auc_per_label

        metrics["label_metrics"] = pd.DataFrame(label_metrics_data)

        # Support (number of true instances per label)
        metrics["support"] = y_true.sum().to_dict()

        # Confusion matrices
        metrics["confusion_matrices"] = self._compute_confusion_matrices(y_true, y_pred)

        return metrics

    def _compute_pr_auc(
        self, y_true: pd.DataFrame, predictions: List[ASClassification]
    ) -> List[Optional[float]]:
        """Compute PR AUC for each label if confidence scores available."""
        # Create confidence scores DataFrame
        confidence_data = {}

        for pred in predictions:
            if pred.confidence_scores:
                confidence_data[pred.asn] = pred.confidence_scores
            else:
                # Use binary predictions as confidence (not ideal)
                confidence_data[pred.asn] = {tag.value: 1.0 for tag in pred.tags}

        if not confidence_data:
            return [None] * len(y_true.columns)

        confidence_df = pd.DataFrame.from_dict(confidence_data, orient="index").fillna(
            0.0
        )

        pr_aucs = []
        for col in y_true.columns:
            if col in confidence_df.columns:
                # Get common ASNs
                common_asns = y_true.index.intersection(confidence_df.index)

                if len(common_asns) > 0:
                    y_true_col = y_true.loc[common_asns, col]
                    y_score_col = confidence_df.loc[common_asns, col]

                    if y_true_col.sum() > 0:  # At least one positive example
                        precision_curve, recall_curve, _ = precision_recall_curve(
                            y_true_col, y_score_col
                        )
                        pr_aucs.append(auc(recall_curve, precision_curve))
                    else:
                        pr_aucs.append(None)
                else:
                    pr_aucs.append(None)
            else:
                pr_aucs.append(None)

        return pr_aucs

    def _compute_confusion_matrices(
        self, y_true: pd.DataFrame, y_pred: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Compute confusion matrix for each label."""
        confusion_matrices = {}

        for i, label in enumerate(y_true.columns):
            cm = confusion_matrix(y_true.iloc[:, i], y_pred.iloc[:, i])
            confusion_matrices[label] = cm

        return confusion_matrices

    def _print_evaluation_results(self, metrics: Dict[str, Any]) -> None:
        """Print formatted evaluation results."""
        console.print("\n[bold blue]Classification Evaluation Results[/bold blue]")

        # Overall metrics
        console.print(f"\n[bold]Overall Metrics:[/bold]")
        console.print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        console.print(f"Macro Precision:  {metrics['macro_precision']:.4f}")
        console.print(f"Macro Recall:     {metrics['macro_recall']:.4f}")
        console.print(f"Macro F1:         {metrics['macro_f1']:.4f}")
        console.print(f"Macro Jaccard:    {metrics['macro_jaccard']:.4f}")

        console.print(f"\nMicro Precision:  {metrics['micro_precision']:.4f}")
        console.print(f"Micro Recall:     {metrics['micro_recall']:.4f}")
        console.print(f"Micro F1:         {metrics['micro_f1']:.4f}")

        # Per-label metrics table
        console.print("\n[bold]Per-Label Metrics:[/bold]")

        table = Table(title="Classification Metrics by Label")
        table.add_column("Label", style="cyan", width=30)
        table.add_column("Accuracy", justify="right", style="green")
        table.add_column("Precision", justify="right", style="green")
        table.add_column("Recall", justify="right", style="green")
        table.add_column("F1-Score", justify="right", style="green")
        table.add_column("Jaccard", justify="right", style="green")
        table.add_column("Support", justify="right", style="yellow")

        if "pr_auc" in metrics["label_metrics"].columns:
            table.add_column("PR AUC", justify="right", style="magenta")

        # Sort by F1 score descending
        sorted_metrics = metrics["label_metrics"].sort_values(
            "f1_score", ascending=False
        )

        for _, row in sorted_metrics.iterrows():
            label = row["label"]
            support = metrics["support"].get(label, 0)

            row_data = [
                label[:28] + "..." if len(label) > 30 else label,
                f"{row['accuracy']:.3f}",
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['f1_score']:.3f}",
                f"{row['jaccard_score']:.3f}",
                str(support),
            ]

            if "pr_auc" in row and pd.notna(row["pr_auc"]):
                row_data.append(f"{row['pr_auc']:.3f}")
            elif "pr_auc" in metrics["label_metrics"].columns:
                row_data.append("N/A")

            table.add_row(*row_data)

        console.print(table)

        # Summary statistics
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"Mean Accuracy:    {sorted_metrics['accuracy'].mean():.4f}")
        console.print(f"Mean Precision:   {sorted_metrics['precision'].mean():.4f}")
        console.print(f"Mean Recall:      {sorted_metrics['recall'].mean():.4f}")
        console.print(f"Mean F1-Score:    {sorted_metrics['f1_score'].mean():.4f}")
        console.print(f"Mean Jaccard:     {sorted_metrics['jaccard_score'].mean():.4f}")

        if "pr_auc" in sorted_metrics.columns:
            valid_pr_aucs = sorted_metrics["pr_auc"].dropna()
            if len(valid_pr_aucs) > 0:
                console.print(f"Mean PR AUC:      {valid_pr_aucs.mean():.4f}")

    def _print_model_comparison(self, all_metrics: Dict[str, Dict[str, Any]]) -> None:
        """Print comparison table for multiple models."""
        console.print("\n[bold blue]Model Comparison[/bold blue]")

        table = Table(title="Model Performance Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Overall Acc", justify="right", style="green")
        table.add_column("Macro F1", justify="right", style="green")
        table.add_column("Micro F1", justify="right", style="green")
        table.add_column("Macro Prec", justify="right", style="yellow")
        table.add_column("Macro Rec", justify="right", style="yellow")
        table.add_column("Jaccard", justify="right", style="magenta")

        # Sort models by macro F1 score
        sorted_models = sorted(
            all_metrics.items(), key=lambda x: x[1]["macro_f1"], reverse=True
        )

        for model_name, metrics in sorted_models:
            table.add_row(
                model_name,
                f"{metrics['overall_accuracy']:.4f}",
                f"{metrics['macro_f1']:.4f}",
                f"{metrics['micro_f1']:.4f}",
                f"{metrics['macro_precision']:.4f}",
                f"{metrics['macro_recall']:.4f}",
                f"{metrics['macro_jaccard']:.4f}",
            )

        console.print(table)

    def _load_predictions_from_json(self, file_path: Path) -> List[ASClassification]:
        """Load predictions from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both single result and batch result formats
        if isinstance(data, dict) and "classifications" in data:
            # BatchClassificationResponse format
            classifications_data = data["classifications"]
        elif isinstance(data, list):
            # List of classifications
            classifications_data = data
        else:
            raise ValueError("Unsupported JSON format")

        predictions = []
        for item in classifications_data:
            # Convert tag strings back to enum values
            tags = [ClassificationTags(tag) for tag in item["tags"]]

            prediction = ASClassification(
                asn=item["asn"],
                organization_name=item["organization_name"],
                tags=tags,
                confidence_scores=item.get("confidence_scores"),
                model_used=item["model_used"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
            )
            predictions.append(prediction)

        return predictions

    def _load_predictions_from_csv(self, file_path: Path) -> List[ASClassification]:
        """Load predictions from CSV file."""
        df = pd.read_csv(file_path)

        predictions = []
        for _, row in df.iterrows():
            # Parse tags from comma-separated string
            tag_strings = [tag.strip() for tag in row["tags"].split(",")]
            tags = [ClassificationTags(tag) for tag in tag_strings]

            # Extract confidence scores if available
            confidence_scores = None
            confidence_cols = [
                col for col in df.columns if col.startswith("confidence_")
            ]
            if confidence_cols:
                confidence_scores = {}
                for col in confidence_cols:
                    tag_name = col.replace("confidence_", "")
                    if pd.notna(row[col]):
                        confidence_scores[tag_name] = float(row[col])

            prediction = ASClassification(
                asn=int(row["asn"]),
                organization_name=row["organization_name"],
                tags=tags,
                confidence_scores=confidence_scores,
                model_used=row["model_used"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            predictions.append(prediction)

        return predictions


def export_evaluation_report(
    metrics: Dict[str, Any], output_path: Path, format: str = "json"
) -> None:
    """
    Export evaluation report to file.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Evaluation metrics to export.
    output_path : Path
        Output file path.
    format : str, default "json"
        Output format: "json", "csv", or "excel".
    """
    if format == "json":
        # Convert non-serializable objects
        export_data = {
            "overall_metrics": {
                "overall_accuracy": metrics["overall_accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "macro_jaccard": metrics["macro_jaccard"],
                "micro_precision": metrics["micro_precision"],
                "micro_recall": metrics["micro_recall"],
                "micro_f1": metrics["micro_f1"],
            },
            "label_metrics": metrics["label_metrics"].to_dict("records"),
            "support": metrics["support"],
            "evaluation_timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

    elif format in ["csv", "excel"]:
        # Export label metrics
        if format == "csv":
            metrics["label_metrics"].to_csv(output_path, index=False)
        else:
            metrics["label_metrics"].to_excel(output_path, index=False)

    console.print(f"[green]✓ Evaluation report exported to {output_path}[/green]")
