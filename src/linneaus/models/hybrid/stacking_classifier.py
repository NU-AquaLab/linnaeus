"""
Hybrid stacking classifier combining SVM and LLM approaches.

This module implements the core hybrid classification system that
combines predictions from traditional ML (SVM) and modern LLM models
using a stacking meta-learner approach.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from ...data.access import DataAccessLayer
from ..svm.svm_models import SVMClassifier
from ..unified_schemas import HierarchicalTags, TopLevelTags, UnifiedASClassification
from ..utils.cross_validation import ASNStratifiedKFold
from .hierarchical_tagger import ModernHierarchicalTagger

logger = logging.getLogger(__name__)


class HybridASClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid classifier combining SVM and LLM approaches for AS classification.

    This classifier uses a stacking approach where:
    1. SVM models provide traditional ML predictions based on engineered features
    2. LLM models provide contextual predictions based on organization descriptions
    3. A meta-learner combines predictions optimally for final classification

    Supports both flat and hierarchical classification approaches.
    """

    def __init__(
        self,
        approach: str = "hybrid",
        svm_weight: float = 0.4,
        llm_weight: float = 0.6,
        use_hierarchical_svm: bool = True,
        meta_learner: Optional[BaseEstimator] = None,
        svm_params: Optional[Dict] = None,
        llm_model_id: Optional[str] = None,
        data_access: Optional[DataAccessLayer] = None,
    ):
        """
        Initialize hybrid classifier.

        Parameters
        ----------
        approach : str
            Classification approach: 'flat', 'hierarchical', or 'hybrid'.
        svm_weight : float
            Weight for SVM predictions in ensemble (if not using meta-learner).
        llm_weight : float
            Weight for LLM predictions in ensemble (if not using meta-learner).
        use_hierarchical_svm : bool
            Whether to use hierarchical SVM approach.
        meta_learner : BaseEstimator, optional
            Meta-learner for stacking. Defaults to LogisticRegression.
        svm_params : Dict, optional
            Parameters for SVM classifier.
        llm_model_id : str, optional
            LLM model identifier for inference.
        data_access : DataAccessLayer, optional
            Data access layer for feature extraction.
        """
        self.approach = approach
        self.svm_weight = svm_weight
        self.llm_weight = llm_weight
        self.use_hierarchical_svm = use_hierarchical_svm
        self.meta_learner = meta_learner or LogisticRegression(
            max_iter=1000, random_state=42
        )
        self.svm_params = svm_params or {}
        self.llm_model_id = llm_model_id
        self.data_access = data_access or DataAccessLayer()

        # Component classifiers
        if use_hierarchical_svm:
            self.svm_classifier = ModernHierarchicalTagger(data_access=data_access)
        else:
            self.svm_classifier = SVMClassifier(approach=approach, **self.svm_params)

        # TODO: Update to use HierarchicalBatchInferenceProcessor
        # For now, disable LLM component to avoid import issues
        self.llm_classifier = None
        if llm_model_id:
            logger.warning(
                "LLM integration temporarily disabled due to interface changes"
            )

        # Stacking classifier
        self.stacking_classifier: Optional[StackingClassifier] = None

        # Fitted attributes
        self.classes_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "HybridASClassifier":
        """
        Fit the hybrid classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing ASN and/or features.
        y : pd.DataFrame
            Label DataFrame with tag columns.

        Returns
        -------
        self
            Fitted hybrid classifier.
        """
        logger.info(f"Fitting hybrid classifier with approach: {self.approach}")

        # Store classes
        self.classes_ = [col for col in y.columns if col != "asn"]

        # Prepare base estimators
        estimators = []

        # Add SVM estimator
        logger.info("Training SVM component")
        self.svm_classifier.fit(X, y)
        estimators.append(("svm", SVMWrapper(self.svm_classifier)))

        # Add LLM estimator if available
        if self.llm_classifier:
            logger.info("Training LLM component")
            # For LLM, we need to create a wrapper that handles async operations
            estimators.append(
                ("llm", LLMWrapper(self.llm_classifier, self.data_access))
            )

        # Create stacking classifier
        if len(estimators) > 1:
            logger.info("Training stacking meta-learner")
            # Use ASN-aware cross-validation for stacking
            cv_splitter = ASNStratifiedKFold(
                n_splits=3,  # TODO: Make this configurable
                shuffle=True,
                random_state=42,
                stratify=True,
            )

            self.stacking_classifier = StackingClassifier(
                estimators=estimators,
                final_estimator=self.meta_learner,
                cv=cv_splitter,  # Use ASN-aware cross-validation
                n_jobs=-1,
                passthrough=False,  # Don't pass original features to meta-learner
            )

            # Fit stacking classifier
            self.stacking_classifier.fit(X, y.drop("asn", axis=1, errors="ignore"))
        else:
            logger.info("Only one component available, using single classifier")

        self.is_fitted_ = True
        logger.info("Hybrid classifier training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict labels using hybrid approach.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if not self.is_fitted_:
            raise ValueError("Classifier not fitted. Call fit() first.")

        logger.info(f"Making hybrid predictions for {len(X)} samples")

        if self.stacking_classifier:
            # Use stacking classifier
            predictions = self.stacking_classifier.predict(X)
        else:
            # Use single classifier or weighted ensemble
            predictions = self._ensemble_predict(X)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using hybrid approach.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        if not self.is_fitted_:
            raise ValueError("Classifier not fitted. Call fit() first.")

        if self.stacking_classifier:
            # Use stacking classifier probabilities
            probabilities = self.stacking_classifier.predict_proba(X)

            # Handle multi-output case
            if isinstance(probabilities, list):
                # For multi-output, get probability of positive class for each output
                prob_matrix = np.zeros((X.shape[0], len(self.classes_)))
                for i, prob_array in enumerate(probabilities):
                    if prob_array.shape[1] > 1:
                        prob_matrix[:, i] = prob_array[
                            :, 1
                        ]  # Probability of positive class
                    else:
                        prob_matrix[:, i] = prob_array[:, 0]
                return prob_matrix
            else:
                return probabilities
        else:
            # Use weighted ensemble probabilities
            return self._ensemble_predict_proba(X)

    def predict_unified(self, X: pd.DataFrame) -> List[UnifiedASClassification]:
        """
        Predict and return results in unified format.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing ASN.

        Returns
        -------
        List[UnifiedASClassification]
            List of unified classification results.
        """
        predictions = self.predict(X)
        probabilities = None

        try:
            probabilities = self.predict_proba(X)
        except Exception as e:
            logger.warning(f"Could not get probabilities: {e}")

        results = []

        for i, (idx, row) in enumerate(X.iterrows()):
            # Extract ASN
            asn = int(row.get("asn", idx))

            # Get organization data for name
            org_data = self.data_access.get_organization_data(asn)
            org_name = org_data.name if org_data else f"ASN{asn}"

            # Process predictions
            pred_row = predictions[i]

            # Get predicted tags
            top_level_tags = []
            hierarchical_tags = []
            confidence_scores = {}

            for j, class_name in enumerate(self.classes_):
                if pred_row[j] > 0:  # Positive prediction
                    # Check if hierarchical tag
                    hierarchical_tag = None
                    for tag in HierarchicalTags:
                        if tag.value == class_name:
                            hierarchical_tag = tag
                            break

                    if hierarchical_tag:
                        hierarchical_tags.append(hierarchical_tag)
                    else:
                        # Check if top-level tag
                        top_level_tag = None
                        for tag in TopLevelTags:
                            if tag.value == class_name:
                                top_level_tag = tag
                                break
                        if top_level_tag:
                            top_level_tags.append(top_level_tag)

                # Add confidence score if available
                if probabilities is not None:
                    confidence_scores[class_name] = float(probabilities[i, j])

            # Determine approach for this result
            effective_approach = self.approach
            if effective_approach == "hierarchical" and not hierarchical_tags:
                effective_approach = "flat"

            # Create unified result
            result = UnifiedASClassification(
                asn=asn,
                organization_name=org_name,
                top_level_tags=top_level_tags,
                hierarchical_tags=hierarchical_tags,
                top_level_confidence=(
                    confidence_scores if probabilities is not None else None
                ),
                model_used=f"Hybrid({self.approach})",
                classification_approach=effective_approach,
            )

            results.append(result)

        return results

    def _ensemble_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction using weighted combination."""
        predictions = []
        weights = []

        # SVM predictions
        svm_pred = self.svm_classifier.predict(X)
        predictions.append(svm_pred)
        weights.append(self.svm_weight)

        # LLM predictions if available
        if self.llm_classifier:
            try:
                llm_pred = self._get_llm_predictions(X)
                predictions.append(llm_pred)
                weights.append(self.llm_weight)
            except Exception as e:
                logger.warning(f"LLM prediction failed: {e}")

        # Weighted ensemble
        if len(predictions) > 1:
            weights = np.array(weights) / np.sum(weights)  # Normalize weights
            ensemble_pred = np.zeros_like(predictions[0])

            for pred, weight in zip(predictions, weights):
                ensemble_pred += weight * pred

            # Threshold at 0.5 for binary predictions
            return (ensemble_pred > 0.5).astype(int)
        else:
            return predictions[0]

    def _ensemble_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble probability prediction using weighted combination."""
        probabilities = []
        weights = []

        # SVM probabilities
        try:
            svm_proba = self.svm_classifier.predict_proba(X)
            probabilities.append(svm_proba)
            weights.append(self.svm_weight)
        except Exception as e:
            logger.warning(f"SVM probability prediction failed: {e}")

        # LLM probabilities if available
        if self.llm_classifier:
            try:
                llm_proba = self._get_llm_probabilities(X)
                probabilities.append(llm_proba)
                weights.append(self.llm_weight)
            except Exception as e:
                logger.warning(f"LLM probability prediction failed: {e}")

        # Weighted ensemble
        if len(probabilities) > 1:
            weights = np.array(weights) / np.sum(weights)  # Normalize weights
            ensemble_proba = np.zeros_like(probabilities[0])

            for proba, weight in zip(probabilities, weights):
                ensemble_proba += weight * proba

            return ensemble_proba
        elif len(probabilities) == 1:
            return probabilities[0]
        else:
            # Fallback to predictions as probabilities
            return self.predict(X).astype(float)

    def _get_llm_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get LLM predictions synchronously."""
        # This would need to be implemented based on the actual LLM interface
        # For now, return zeros as placeholder
        logger.warning("LLM prediction not implemented yet")
        return np.zeros((len(X), len(self.classes_)))

    def _get_llm_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Get LLM probabilities synchronously."""
        # This would need to be implemented based on the actual LLM interface
        # For now, return uniform probabilities as placeholder
        logger.warning("LLM probability prediction not implemented yet")
        return np.full((len(X), len(self.classes_)), 0.5)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """
        Evaluate hybrid classifier performance.

        Parameters
        ----------
        X : pd.DataFrame
            Test features.
        y : pd.DataFrame
            True labels.

        Returns
        -------
        Dict
            Evaluation metrics including component and ensemble results.
        """
        results = {"hybrid": {}, "components": {}}

        # Hybrid predictions
        y_pred_hybrid = self.predict(X)
        y_true = y.drop("asn", axis=1, errors="ignore")

        # Overall hybrid performance
        for i, class_name in enumerate(self.classes_):
            report = classification_report(
                y_true.iloc[:, i],
                y_pred_hybrid[:, i],
                output_dict=True,
                zero_division=0,
            )
            results["hybrid"][class_name] = report

        # Component performance
        if hasattr(self.svm_classifier, "predict"):
            y_pred_svm = self.svm_classifier.predict(X)
            for i, class_name in enumerate(self.classes_):
                report = classification_report(
                    y_true.iloc[:, i],
                    y_pred_svm[:, i] if y_pred_svm.ndim > 1 else [y_pred_svm[i]],
                    output_dict=True,
                    zero_division=0,
                )
                if "svm" not in results["components"]:
                    results["components"]["svm"] = {}
                results["components"]["svm"][class_name] = report

        return results

    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance from component classifiers."""
        importance = {}

        # SVM feature importance
        if hasattr(self.svm_classifier, "get_feature_importance"):
            importance["svm"] = self.svm_classifier.get_feature_importance()

        # Meta-learner importance (if available)
        if self.stacking_classifier and hasattr(
            self.stacking_classifier.final_estimator_, "coef_"
        ):
            coef = self.stacking_classifier.final_estimator_.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)  # Average across classes
            else:
                coef = np.abs(coef)

            estimator_names = [name for name, _ in self.stacking_classifier.estimators_]
            meta_importance = pd.DataFrame(
                {"component": estimator_names, "importance": coef}
            ).sort_values("importance", ascending=False)

            importance["meta_learner"] = meta_importance

        return importance


class SVMWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to make hierarchical SVM compatible with sklearn stacking."""

    def __init__(self, svm_classifier):
        self.svm_classifier = svm_classifier

    def fit(self, X, y):
        # SVM is already fitted
        return self

    def predict(self, X):
        pred = self.svm_classifier.predict(X)
        if isinstance(pred, pd.DataFrame):
            return pred.values
        return pred

    def predict_proba(self, X):
        if hasattr(self.svm_classifier, "predict_proba"):
            return self.svm_classifier.predict_proba(X)
        else:
            # Fallback: use predictions as probabilities
            pred = self.predict(X)
            return pred.astype(float)


class LLMWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to make LLM classifier compatible with sklearn stacking."""

    def __init__(self, llm_classifier, data_access):
        self.llm_classifier = llm_classifier
        self.data_access = data_access

    def fit(self, X, y):
        # LLM doesn't need fitting in the traditional sense
        return self

    def predict(self, X):
        # This would need async handling - placeholder implementation
        logger.warning("LLM wrapper predict not fully implemented")
        return np.zeros((len(X), 1))  # Placeholder

    def predict_proba(self, X):
        # This would need async handling - placeholder implementation
        logger.warning("LLM wrapper predict_proba not fully implemented")
        return np.full((len(X), 1), 0.5)  # Placeholder
