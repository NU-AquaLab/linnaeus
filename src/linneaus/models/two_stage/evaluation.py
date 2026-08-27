"""
Hierarchical evaluation framework for two-stage classification pipeline.

This module provides specialized evaluation metrics for hierarchical classification
including stage-specific metrics, hierarchical consistency, and comprehensive
performance analysis.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..hierarchical_schemas import CategoryMapping, TopLevelCategory
from ..two_stage_schemas import TwoStageASClassification

logger = logging.getLogger(__name__)


class HierarchicalEvaluator:
    """
    Comprehensive evaluator for two-stage hierarchical classification.

    This evaluator provides specialized metrics for hierarchical classification
    including stage-specific performance, consistency validation, and detailed
    error analysis.
    """

    def __init__(self, category_mappings: Optional[List[CategoryMapping]] = None):
        """
        Initialize hierarchical evaluator.

        Parameters
        ----------
        category_mappings : Optional[List[CategoryMapping]]
            Category mappings for validation. Uses default if None.
        """
        self.category_mappings = category_mappings or CategoryMapping.get_all_mappings()
        self.category_map = {
            mapping.category.value: mapping for mapping in self.category_mappings
        }

        # All possible categories and subcategories
        self.top_level_categories = [cat.value for cat in TopLevelCategory]
        self.all_subcategories = set()
        for mapping in self.category_mappings:
            self.all_subcategories.update(mapping.subcategories)

        logger.info(
            f"Initialized HierarchicalEvaluator with {len(self.top_level_categories)} categories"
        )

    def evaluate(
        self,
        predictions: List[TwoStageASClassification],
        ground_truth: pd.DataFrame,
        include_detailed_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of two-stage predictions.

        Parameters
        ----------
        predictions : List[TwoStageASClassification]
            Two-stage predictions to evaluate.
        ground_truth : pd.DataFrame
            Ground truth with ASN, TopLevelCategory, SubCategory columns.
        include_detailed_analysis : bool
            Whether to include detailed error analysis.

        Returns
        -------
        Dict[str, Any]
            Comprehensive evaluation results.
        """
        logger.info(f"Evaluating {len(predictions)} two-stage predictions")

        # Prepare data for evaluation
        pred_data, true_data = self._prepare_evaluation_data(predictions, ground_truth)

        # Stage 1 evaluation (top-level categories)
        stage1_metrics = self._evaluate_stage1(pred_data, true_data)

        # Stage 2 evaluation (sublevel categories)
        stage2_metrics = self._evaluate_stage2(pred_data, true_data)

        # Hierarchical consistency evaluation
        consistency_metrics = self._evaluate_hierarchical_consistency(
            pred_data, true_data
        )

        # Overall system evaluation
        system_metrics = self._evaluate_overall_system(pred_data, true_data)

        # Detailed analysis
        detailed_analysis = {}
        if include_detailed_analysis:
            detailed_analysis = self._detailed_error_analysis(pred_data, true_data)

        # Combine results
        results = {
            "stage1": stage1_metrics,
            "stage2": stage2_metrics,
            "consistency": consistency_metrics,
            "system": system_metrics,
            "detailed_analysis": detailed_analysis,
            "metadata": {
                "num_predictions": len(predictions),
                "num_ground_truth": len(ground_truth),
                "evaluation_timestamp": pd.Timestamp.now().isoformat(),
            },
        }

        logger.info("Evaluation completed successfully")
        return results

    def _prepare_evaluation_data(
        self, predictions: List[TwoStageASClassification], ground_truth: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare prediction and ground truth data for evaluation.

        Parameters
        ----------
        predictions : List[TwoStageASClassification]
            Predictions to evaluate.
        ground_truth : pd.DataFrame
            Ground truth data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Processed prediction and ground truth dataframes.
        """
        # Convert predictions to DataFrame
        pred_records = []
        for pred in predictions:
            # Stage 1 predictions
            stage1_categories = [p.category.value for p in pred.stage1_predictions]
            stage1_confidences = [p.confidence.value for p in pred.stage1_predictions]

            # Stage 2 predictions
            stage2_data = {}
            for s2_pred in pred.stage2_predictions:
                category = s2_pred.parent_category.value
                if category not in stage2_data:
                    stage2_data[category] = []
                stage2_data[category].append(
                    {
                        "subcategory": s2_pred.subcategory,
                        "confidence": s2_pred.confidence.value,
                    }
                )

            pred_record = {
                "asn": pred.asn,
                "stage1_categories": stage1_categories,
                "stage1_confidences": stage1_confidences,
                "stage2_predictions": stage2_data,
                "overall_confidence": pred.overall_confidence,
            }
            pred_records.append(pred_record)

        pred_df = pd.DataFrame(pred_records)

        # Process ground truth
        true_df = ground_truth.copy()

        # Find common ASNs
        common_asns = set(pred_df["asn"]) & set(true_df["ASN"])
        pred_df = pred_df[pred_df["asn"].isin(common_asns)]
        true_df = true_df[true_df["ASN"].isin(common_asns)]

        logger.info(f"Prepared evaluation data: {len(common_asns)} common ASNs")
        return pred_df, true_df

    def _evaluate_stage1(
        self, pred_data: pd.DataFrame, true_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate Stage 1 (top-level) predictions.

        Parameters
        ----------
        pred_data : pd.DataFrame
            Prediction data.
        true_data : pd.DataFrame
            Ground truth data.

        Returns
        -------
        Dict[str, Any]
            Stage 1 evaluation metrics.
        """
        # Create binary matrices for multi-label evaluation
        y_true_matrix = np.zeros((len(pred_data), len(self.top_level_categories)))
        y_pred_matrix = np.zeros((len(pred_data), len(self.top_level_categories)))

        # Map ASNs to indices
        asn_to_idx = {asn: i for i, asn in enumerate(pred_data["asn"])}

        # Fill true labels matrix
        for _, row in true_data.iterrows():
            if row["ASN"] in asn_to_idx:
                idx = asn_to_idx[row["ASN"]]
                category = row["TopLevelCategory"]
                if category in self.top_level_categories:
                    cat_idx = self.top_level_categories.index(category)
                    y_true_matrix[idx, cat_idx] = 1

        # Fill predicted labels matrix
        for i, row in pred_data.iterrows():
            for category in row["stage1_categories"]:
                if category in self.top_level_categories:
                    cat_idx = self.top_level_categories.index(category)
                    y_pred_matrix[i, cat_idx] = 1

        # Calculate metrics
        metrics = {}

        # Overall metrics
        metrics["accuracy"] = accuracy_score(y_true_matrix, y_pred_matrix)
        metrics["precision_macro"] = precision_score(
            y_true_matrix, y_pred_matrix, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            y_true_matrix, y_pred_matrix, average="macro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(
            y_true_matrix, y_pred_matrix, average="macro", zero_division=0
        )

        # Per-category metrics
        metrics["per_category"] = {}
        for i, category in enumerate(self.top_level_categories):
            cat_true = y_true_matrix[:, i]
            cat_pred = y_pred_matrix[:, i]

            if (
                cat_true.sum() > 0 or cat_pred.sum() > 0
            ):  # Only calculate if category appears
                metrics["per_category"][category] = {
                    "precision": precision_score(cat_true, cat_pred, zero_division=0),
                    "recall": recall_score(cat_true, cat_pred, zero_division=0),
                    "f1": f1_score(cat_true, cat_pred, zero_division=0),
                    "support": int(cat_true.sum()),
                }

        # Exact match accuracy (all categories correct)
        exact_matches = np.all(y_true_matrix == y_pred_matrix, axis=1)
        metrics["exact_match_accuracy"] = exact_matches.mean()

        # Subset accuracy (predicted categories are subset of true categories)
        subset_matches = np.all(y_pred_matrix <= y_true_matrix, axis=1)
        metrics["subset_accuracy"] = subset_matches.mean()

        logger.info(
            f"Stage 1 metrics: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_macro']:.3f}"
        )
        return metrics

    def _evaluate_stage2(
        self, pred_data: pd.DataFrame, true_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate Stage 2 (sublevel) predictions.

        Parameters
        ----------
        pred_data : pd.DataFrame
            Prediction data.
        true_data : pd.DataFrame
            Ground truth data.

        Returns
        -------
        Dict[str, Any]
            Stage 2 evaluation metrics.
        """
        metrics = {}

        # Group ground truth by ASN and category
        true_sublevel = {}
        for _, row in true_data.iterrows():
            asn = row["ASN"]
            category = row["TopLevelCategory"]
            subcategory = row["SubCategory"]

            if pd.notna(subcategory):
                if asn not in true_sublevel:
                    true_sublevel[asn] = {}
                if category not in true_sublevel[asn]:
                    true_sublevel[asn][category] = set()
                true_sublevel[asn][category].add(subcategory)

        # Collect predictions and ground truth for each category
        category_metrics = {}
        all_correct = []
        all_predicted = []

        for category in self.top_level_categories:
            if category not in self.category_map:
                continue

            set(self.category_map[category].subcategories)

            cat_correct = []
            cat_predicted = []

            for _, pred_row in pred_data.iterrows():
                asn = pred_row["asn"]

                # Get true subcategories for this ASN and category
                true_subcats = true_sublevel.get(asn, {}).get(category, set())

                # Get predicted subcategories
                pred_subcats = set()
                if category in pred_row["stage2_predictions"]:
                    for s2_pred in pred_row["stage2_predictions"][category]:
                        pred_subcats.add(s2_pred["subcategory"])

                # Only evaluate if there are true subcategories for this category
                if true_subcats:
                    # Calculate accuracy for this prediction
                    correct = len(true_subcats & pred_subcats) / max(
                        len(true_subcats | pred_subcats), 1
                    )
                    cat_correct.append(correct)
                    cat_predicted.append(len(pred_subcats) > 0)

                    all_correct.append(correct)
                    all_predicted.append(len(pred_subcats) > 0)

            if cat_correct:
                category_metrics[category] = {
                    "accuracy": np.mean(cat_correct),
                    "coverage": np.mean(cat_predicted),
                    "samples": len(cat_correct),
                }

        # Overall Stage 2 metrics
        if all_correct:
            metrics["overall_accuracy"] = np.mean(all_correct)
            metrics["coverage"] = np.mean(all_predicted)
            metrics["total_samples"] = len(all_correct)
        else:
            metrics["overall_accuracy"] = 0.0
            metrics["coverage"] = 0.0
            metrics["total_samples"] = 0

        metrics["per_category"] = category_metrics

        logger.info(
            f"Stage 2 metrics: Accuracy={metrics.get('overall_accuracy', 0):.3f}"
        )
        return metrics

    def _evaluate_hierarchical_consistency(
        self, pred_data: pd.DataFrame, true_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate hierarchical consistency of predictions.

        Parameters
        ----------
        pred_data : pd.DataFrame
            Prediction data.
        true_data : pd.DataFrame
            Ground truth data.

        Returns
        -------
        Dict[str, Any]
            Consistency metrics.
        """
        consistency_violations = []
        total_stage2_predictions = 0

        for _, pred_row in pred_data.iterrows():
            asn = pred_row["asn"]
            stage1_categories = set(pred_row["stage1_categories"])

            # Check each Stage 2 prediction
            for category, s2_preds in pred_row["stage2_predictions"].items():
                for s2_pred in s2_preds:
                    total_stage2_predictions += 1

                    # Check if Stage 2 category matches a Stage 1 prediction
                    if category not in stage1_categories:
                        consistency_violations.append(
                            {
                                "asn": asn,
                                "violation_type": "stage2_without_stage1",
                                "category": category,
                                "subcategory": s2_pred["subcategory"],
                            }
                        )

                    # Check if subcategory is valid for category
                    if category in self.category_map:
                        valid_subcats = self.category_map[category].subcategories
                        if s2_pred["subcategory"] not in valid_subcats:
                            consistency_violations.append(
                                {
                                    "asn": asn,
                                    "violation_type": "invalid_subcategory",
                                    "category": category,
                                    "subcategory": s2_pred["subcategory"],
                                }
                            )

        # Calculate consistency metrics
        consistency_rate = 1.0 - (
            len(consistency_violations) / max(total_stage2_predictions, 1)
        )

        violation_types = {}
        for violation in consistency_violations:
            v_type = violation["violation_type"]
            violation_types[v_type] = violation_types.get(v_type, 0) + 1

        metrics = {
            "consistency_rate": consistency_rate,
            "total_violations": len(consistency_violations),
            "total_stage2_predictions": total_stage2_predictions,
            "violation_types": violation_types,
            "violations_sample": consistency_violations[
                :10
            ],  # Sample violations for analysis
        }

        logger.info(
            f"Consistency metrics: Rate={consistency_rate:.3f}, Violations={len(consistency_violations)}"
        )
        return metrics

    def _evaluate_overall_system(
        self, pred_data: pd.DataFrame, true_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate overall system performance.

        Parameters
        ----------
        pred_data : pd.DataFrame
            Prediction data.
        true_data : pd.DataFrame
            Ground truth data.

        Returns
        -------
        Dict[str, Any]
            Overall system metrics.
        """
        exact_matches = 0
        partial_matches = 0
        total_predictions = len(pred_data)

        # Create ground truth hierarchical tags by ASN
        true_hierarchical = {}
        for _, row in true_data.iterrows():
            asn = row["ASN"]
            if asn not in true_hierarchical:
                true_hierarchical[asn] = set()

            category = row["TopLevelCategory"]
            subcategory = row["SubCategory"]

            if pd.notna(subcategory):
                true_hierarchical[asn].add(f"{category}:{subcategory}")
            else:
                true_hierarchical[asn].add(category)

        # Compare predictions with ground truth
        for _, pred_row in pred_data.iterrows():
            asn = pred_row["asn"]
            true_tags = true_hierarchical.get(asn, set())

            # Generate predicted hierarchical tags
            pred_tags = set()

            # Add Stage 1 categories
            for category in pred_row["stage1_categories"]:
                if category not in pred_row["stage2_predictions"]:
                    # No Stage 2 predictions for this category
                    pred_tags.add(category)
                else:
                    # Has Stage 2 predictions
                    for s2_pred in pred_row["stage2_predictions"][category]:
                        pred_tags.add(f"{category}:{s2_pred['subcategory']}")

            # Check for exact match
            if pred_tags == true_tags:
                exact_matches += 1
                partial_matches += 1
            elif pred_tags & true_tags:  # Any overlap
                partial_matches += 1

        exact_match_rate = exact_matches / max(total_predictions, 1)
        partial_match_rate = partial_matches / max(total_predictions, 1)

        metrics = {
            "exact_match_rate": exact_match_rate,
            "partial_match_rate": partial_match_rate,
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "total_predictions": total_predictions,
        }

        logger.info(
            f"System metrics: Exact={exact_match_rate:.3f}, Partial={partial_match_rate:.3f}"
        )
        return metrics

    def _detailed_error_analysis(
        self, pred_data: pd.DataFrame, true_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Perform detailed error analysis.

        Parameters
        ----------
        pred_data : pd.DataFrame
            Prediction data.
        true_data : pd.DataFrame
            Ground truth data.

        Returns
        -------
        Dict[str, Any]
            Detailed error analysis.
        """
        analysis = {
            "common_errors": {},
            "category_confusion": {},
            "confidence_analysis": {},
            "error_patterns": [],
        }

        # Analyze common prediction errors
        category_errors = {}
        confidence_by_accuracy = {"correct": [], "incorrect": []}

        # Create ground truth lookup
        true_by_asn = {}
        for _, row in true_data.iterrows():
            asn = row["ASN"]
            if asn not in true_by_asn:
                true_by_asn[asn] = {"categories": set(), "subcategories": {}}

            category = row["TopLevelCategory"]
            subcategory = row["SubCategory"]

            true_by_asn[asn]["categories"].add(category)
            if pd.notna(subcategory):
                if category not in true_by_asn[asn]["subcategories"]:
                    true_by_asn[asn]["subcategories"][category] = set()
                true_by_asn[asn]["subcategories"][category].add(subcategory)

        # Analyze each prediction
        for _, pred_row in pred_data.iterrows():
            asn = pred_row["asn"]
            true_data_asn = true_by_asn.get(
                asn, {"categories": set(), "subcategories": {}}
            )

            pred_categories = set(pred_row["stage1_categories"])
            true_categories = true_data_asn["categories"]

            # Stage 1 analysis
            stage1_correct = pred_categories == true_categories

            if not stage1_correct:
                # Record category errors
                missed = true_categories - pred_categories
                extra = pred_categories - true_categories

                for cat in missed:
                    error_key = f"missed_{cat}"
                    category_errors[error_key] = category_errors.get(error_key, 0) + 1

                for cat in extra:
                    error_key = f"extra_{cat}"
                    category_errors[error_key] = category_errors.get(error_key, 0) + 1

            # Confidence analysis
            avg_confidence = np.mean(pred_row["stage1_confidences"])
            if stage1_correct:
                confidence_by_accuracy["correct"].append(avg_confidence)
            else:
                confidence_by_accuracy["incorrect"].append(avg_confidence)

        # Compile results
        analysis["common_errors"] = dict(
            sorted(category_errors.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        if confidence_by_accuracy["correct"]:
            analysis["confidence_analysis"]["avg_confidence_correct"] = np.mean(
                confidence_by_accuracy["correct"]
            )
        if confidence_by_accuracy["incorrect"]:
            analysis["confidence_analysis"]["avg_confidence_incorrect"] = np.mean(
                confidence_by_accuracy["incorrect"]
            )

        return analysis

    def create_performance_report(
        self, evaluation_results: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """
        Create a comprehensive performance report.

        Parameters
        ----------
        evaluation_results : Dict[str, Any]
            Results from evaluate() method.
        output_path : Optional[str]
            Path to save the report.

        Returns
        -------
        str
            Performance report as text.
        """
        report_lines = [
            "Two-Stage Hierarchical Classification Performance Report",
            "=" * 55,
            "",
        ]

        # Metadata
        metadata = evaluation_results.get("metadata", {})
        report_lines.extend(
            [
                f"Generated: {metadata.get('evaluation_timestamp', 'Unknown')}",
                f"Predictions evaluated: {metadata.get('num_predictions', 'Unknown')}",
                f"Ground truth samples: {metadata.get('num_ground_truth', 'Unknown')}",
                "",
            ]
        )

        # Stage 1 Results
        stage1 = evaluation_results.get("stage1", {})
        report_lines.extend(
            [
                "STAGE 1 (Top-Level Categories) Performance:",
                "-" * 45,
                f"Overall Accuracy: {stage1.get('accuracy', 0):.4f}",
                f"Macro F1-Score: {stage1.get('f1_macro', 0):.4f}",
                f"Macro Precision: {stage1.get('precision_macro', 0):.4f}",
                f"Macro Recall: {stage1.get('recall_macro', 0):.4f}",
                f"Exact Match Accuracy: {stage1.get('exact_match_accuracy', 0):.4f}",
                f"Subset Accuracy: {stage1.get('subset_accuracy', 0):.4f}",
                "",
            ]
        )

        # Top categories by performance
        if "per_category" in stage1:
            sorted_cats = sorted(
                stage1["per_category"].items(),
                key=lambda x: x[1].get("f1", 0),
                reverse=True,
            )

            report_lines.extend(
                [
                    "Top Performing Categories (by F1-Score):",
                    *[
                        f"  {cat}: {metrics.get('f1', 0):.4f} (Support: {metrics.get('support', 0)})"
                        for cat, metrics in sorted_cats[:5]
                    ],
                    "",
                ]
            )

        # Stage 2 Results
        stage2 = evaluation_results.get("stage2", {})
        report_lines.extend(
            [
                "STAGE 2 (Sublevel Categories) Performance:",
                "-" * 45,
                f"Overall Accuracy: {stage2.get('overall_accuracy', 0):.4f}",
                f"Coverage Rate: {stage2.get('coverage', 0):.4f}",
                f"Total Samples: {stage2.get('total_samples', 0)}",
                "",
            ]
        )

        # Consistency Results
        consistency = evaluation_results.get("consistency", {})
        report_lines.extend(
            [
                "HIERARCHICAL CONSISTENCY:",
                "-" * 30,
                f"Consistency Rate: {consistency.get('consistency_rate', 0):.4f}",
                f"Total Violations: {consistency.get('total_violations', 0)}",
                f"Stage 2 Predictions: {consistency.get('total_stage2_predictions', 0)}",
                "",
            ]
        )

        # Overall System Results
        system = evaluation_results.get("system", {})
        report_lines.extend(
            [
                "OVERALL SYSTEM PERFORMANCE:",
                "-" * 35,
                f"Exact Match Rate: {system.get('exact_match_rate', 0):.4f}",
                f"Partial Match Rate: {system.get('partial_match_rate', 0):.4f}",
                f"Exact Matches: {system.get('exact_matches', 0)}",
                f"Partial Matches: {system.get('partial_matches', 0)}",
                "",
            ]
        )

        # Error Analysis
        if "detailed_analysis" in evaluation_results:
            analysis = evaluation_results["detailed_analysis"]
            if "common_errors" in analysis and analysis["common_errors"]:
                report_lines.extend(
                    [
                        "COMMON ERRORS:",
                        "-" * 15,
                        *[
                            f"  {error}: {count}"
                            for error, count in list(analysis["common_errors"].items())[
                                :5
                            ]
                        ],
                        "",
                    ]
                )

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Performance report saved to {output_path}")

        return report_text


# Export main classes
__all__ = ["HierarchicalEvaluator"]
