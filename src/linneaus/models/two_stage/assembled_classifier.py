"""
Assembled top-level classifier for Stage 1 of two-stage hierarchical pipeline.

This module implements the assembled classifier that combines LLM and SVM
approaches using a stacking meta-learner to predict top-level categories.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from ...data.access import DataAccessLayer
from ..hierarchical_schemas import TopLevelCategory
from ..llm.inference import HierarchicalBatchInferenceProcessor
from ..svm.svm_models import SVMClassifier
from ..two_stage_schemas import (
    ConfidenceScore,
    ModelType,
    PipelineConfig,
    TopLevelPrediction,
)
from ..utils.cross_validation import ASNStratifiedKFold
from ..utils.data_adapters import UnifiedDataAdapter

logger = logging.getLogger(__name__)


class LLMWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for LLM classifier.

    This wrapper makes the HierarchicalBatchInferenceProcessor compatible
    with scikit-learn's StackingClassifier.
    """

    def __init__(
        self,
        llm_processor: HierarchicalBatchInferenceProcessor,
        data_adapter: UnifiedDataAdapter,
    ):
        """
        Initialize LLM wrapper.

        Parameters
        ----------
        llm_processor : HierarchicalBatchInferenceProcessor
            The LLM inference processor.
        data_adapter : UnifiedDataAdapter
            Data adapter for format conversion.
        """
        self.llm_processor = llm_processor
        self.data_adapter = data_adapter
        self.classes_ = [cat.value for cat in TopLevelCategory]
        self.n_classes_ = len(self.classes_)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "LLMWrapper":
        """
        Fit method (no-op for LLM as it's pre-trained).

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.
        y : pd.DataFrame
            Target labels.

        Returns
        -------
        self : LLMWrapper
            Returns self.
        """
        logger.info(
            "LLM wrapper fit() called - no training needed for pre-trained model"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict top-level categories using LLM.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe containing ASN.

        Returns
        -------
        np.ndarray
            Binary predictions for each top-level category.
        """
        try:
            # Use data adapter to prepare LLM input
            organizations = self.data_adapter.prepare_for_llm(X)

            # Create input DataFrame for LLM
            llm_input = self.data_adapter.llm_adapter.create_llm_input(organizations)

            # Process batch through LLM
            results = self.llm_processor.process_batch(llm_input.to_dict("records"))

            # Convert to binary format
            predictions = np.zeros((len(X), self.n_classes_))

            for i, result in enumerate(results):
                for classification in result.get("classifications", []):
                    category = classification.get("category")
                    if category in self.classes_:
                        cat_idx = self.classes_.index(category)
                        predictions[i, cat_idx] = 1.0

            return predictions

        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            # Return zeros as fallback
            return np.zeros((len(X), self.n_classes_))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using LLM.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.

        Returns
        -------
        np.ndarray
            Probability predictions for each class.
        """
        # For now, use hard predictions as probabilities
        # TODO: Extract actual probabilities from LLM logprobs
        predictions = self.predict(X)

        # Convert binary predictions to probabilities
        # Add small epsilon for numerical stability
        epsilon = 1e-10
        proba = predictions + epsilon
        proba = proba / proba.sum(axis=1, keepdims=True)

        return proba


class SVMWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for SVM classifier.

    Ensures consistent interface with the LLM wrapper.
    """

    def __init__(self, svm_classifier: SVMClassifier, data_adapter: UnifiedDataAdapter):
        """
        Initialize SVM wrapper.

        Parameters
        ----------
        svm_classifier : SVMClassifier
            The SVM classifier instance.
        data_adapter : UnifiedDataAdapter
            Data adapter for format conversion.
        """
        self.svm_classifier = svm_classifier
        self.data_adapter = data_adapter
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "SVMWrapper":
        """
        Fit the SVM classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.
        y : pd.DataFrame
            Target labels.

        Returns
        -------
        self : SVMWrapper
            Returns self.
        """
        # Prepare features using data adapter
        features = self.data_adapter.prepare_for_svm(X)
        self.svm_classifier.fit(features, y)
        self.classes_ = self.svm_classifier.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using SVM.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        features = self.data_adapter.prepare_for_svm(X)
        return self.svm_classifier.predict(features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using SVM.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.

        Returns
        -------
        np.ndarray
            Probability predictions.
        """
        features = self.data_adapter.prepare_for_svm(X)
        if hasattr(self.svm_classifier, "predict_proba"):
            return self.svm_classifier.predict_proba(features)
        else:
            # Fallback: use predictions as probabilities
            predictions = self.svm_classifier.predict(features)
            return predictions.astype(float)


class AssembledTopLevelClassifier(BaseEstimator, ClassifierMixin):
    """
    Assembled classifier for Stage 1 top-level category prediction.

    Combines LLM and SVM approaches using a stacking meta-learner to predict
    top-level categories with confidence scores and model attribution.
    """

    def __init__(
        self,
        llm_processor: Optional[HierarchicalBatchInferenceProcessor] = None,
        svm_params: Optional[Dict[str, Any]] = None,
        meta_learner: Optional[BaseEstimator] = None,
        svm_weight: float = 0.4,
        llm_weight: float = 0.6,
        data_access: Optional[DataAccessLayer] = None,
        config: Optional[PipelineConfig] = None,
        parallel_batch_size: int = 10,
    ):
        """
        Initialize assembled top-level classifier.

        Parameters
        ----------
        llm_processor : Optional[HierarchicalBatchInferenceProcessor]
            LLM inference processor for top-level classification.
        svm_params : Optional[Dict[str, Any]]
            Parameters for SVM classifier.
        meta_learner : Optional[BaseEstimator]
            Meta-learner for stacking. Defaults to LogisticRegression.
        svm_weight : float
            Weight for SVM predictions in ensemble fallback.
        llm_weight : float
            Weight for LLM predictions in ensemble fallback.
        data_access : Optional[DataAccessLayer]
            Data access layer for feature extraction.
        config : Optional[PipelineConfig]
            Pipeline configuration.
        parallel_batch_size : int
            Batch size for parallel processing.
        """
        self.llm_processor = llm_processor
        self.svm_params = svm_params or {}
        self.meta_learner = meta_learner or LogisticRegression(
            max_iter=1000, random_state=42
        )
        self.svm_weight = svm_weight
        self.llm_weight = llm_weight
        self.data_access = data_access or DataAccessLayer()
        self.config = config
        self.parallel_batch_size = parallel_batch_size

        # Data adapter for format conversion
        self.data_adapter = UnifiedDataAdapter()

        # Component classifiers
        self.svm_classifier = SVMClassifier(approach="hierarchical", **self.svm_params)
        self.llm_wrapper = None
        self.svm_wrapper = None
        self.stacking_classifier = None

        # Fitted attributes
        self.classes_ = [cat.value for cat in TopLevelCategory]
        self.is_fitted_ = False

        logger.info("Initialized AssembledTopLevelClassifier")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "AssembledTopLevelClassifier":
        """
        Fit the assembled classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe containing ASN and/or features.
        y : pd.DataFrame
            Target labels with top-level category columns.

        Returns
        -------
        self : AssembledTopLevelClassifier
            Fitted classifier.
        """
        logger.info("Fitting AssembledTopLevelClassifier")
        start_time = time.time()

        try:
            # Create component wrappers
            estimators = []

            # Add SVM estimator
            logger.info("Training SVM component")
            self.svm_wrapper = SVMWrapper(self.svm_classifier, self.data_adapter)
            self.svm_wrapper.fit(X, y)
            estimators.append(("svm", self.svm_wrapper))

            # Add LLM estimator if available
            if self.llm_processor:
                logger.info("Setting up LLM component")
                self.llm_wrapper = LLMWrapper(self.llm_processor, self.data_adapter)
                self.llm_wrapper.fit(X, y)  # No-op for LLM
                estimators.append(("llm", self.llm_wrapper))
            else:
                logger.warning("No LLM processor provided - using SVM-only approach")

            # Create stacking classifier if multiple estimators
            if len(estimators) > 1:
                logger.info("Training stacking meta-learner")

                # Get CV folds from config or use default
                cv_folds = 3  # Default
                if self.config and hasattr(self.config, "two_stage_pipeline"):
                    stage1_config = getattr(
                        self.config.two_stage_pipeline, "stage1", None
                    )
                    if stage1_config:
                        meta_learner_config = getattr(
                            stage1_config, "meta_learner_params", None
                        )
                        if meta_learner_config:
                            cv_folds = getattr(meta_learner_config, "cv_folds", 3)

                # Use ASN-aware cross-validation
                cv_splitter = ASNStratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=42, stratify=True
                )

                self.stacking_classifier = StackingClassifier(
                    estimators=estimators,
                    final_estimator=self.meta_learner,
                    cv=cv_splitter,  # Use ASN-aware cross-validation
                    n_jobs=-1,
                    passthrough=False,
                )

                # Fit stacking classifier
                y_binary = y.drop("asn", axis=1, errors="ignore")
                self.stacking_classifier.fit(X, y_binary)
            else:
                logger.info("Single component classifier - no stacking needed")

            self.is_fitted_ = True
            fit_time = time.time() - start_time
            logger.info(f"AssembledTopLevelClassifier fitted in {fit_time:.2f}s")

        except Exception as e:
            logger.error(f"Error fitting AssembledTopLevelClassifier: {e}")
            raise

        return self

    def predict(self, X: pd.DataFrame) -> List[TopLevelPrediction]:
        """
        Predict top-level categories with confidence scores.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.

        Returns
        -------
        List[TopLevelPrediction]
            List of top-level predictions with confidence scores.
        """
        if not self.is_fitted_:
            raise ValueError("Classifier not fitted. Call fit() first.")

        logger.info(f"Making predictions for {len(X)} samples")
        start_time = time.time()

        try:
            if self.stacking_classifier:
                # Use stacking classifier
                predictions = self.stacking_classifier.predict(X)
                probabilities = self.stacking_classifier.predict_proba(X)
            else:
                # Use single classifier (fallback)
                predictions, probabilities = self._ensemble_predict(X)

            # Convert to TopLevelPrediction objects
            results = []
            for i in range(len(X)):
                X.iloc[i].get("asn", i)

                # Get predicted categories (multi-label)
                sample_predictions = []
                for j, class_name in enumerate(self.classes_):
                    if predictions[i, j] > 0.5:  # Threshold for positive prediction

                        # Calculate confidence score
                        if isinstance(probabilities, list):
                            # Multi-output case
                            confidence_value = (
                                float(probabilities[j][i, 1])
                                if probabilities[j].shape[1] > 1
                                else float(probabilities[j][i, 0])
                            )
                        else:
                            confidence_value = float(probabilities[i, j])

                        # Get source scores if available
                        source_scores = None
                        if hasattr(self.stacking_classifier, "estimators_"):
                            source_scores = {}
                            for name, estimator in self.stacking_classifier.estimators_:
                                try:
                                    if hasattr(estimator, "predict_proba"):
                                        est_proba = estimator.predict_proba(X.iloc[[i]])
                                        if isinstance(est_proba, np.ndarray):
                                            source_scores[name] = float(
                                                est_proba[0, j]
                                                if est_proba.shape[1] > j
                                                else 0.0
                                            )
                                except Exception as e:
                                    logger.warning(
                                        f"Could not get {name} probability: {e}"
                                    )

                        confidence = ConfidenceScore(
                            value=confidence_value,
                            model_type=ModelType.ASSEMBLED,
                            source_scores=source_scores,
                        )

                        # Get feature importance if available
                        feature_importance = None
                        if hasattr(self.svm_wrapper, "svm_classifier") and hasattr(
                            self.svm_wrapper.svm_classifier, "get_feature_importance"
                        ):
                            try:
                                importance_df = (
                                    self.svm_wrapper.svm_classifier.get_feature_importance()
                                )
                                if not importance_df.empty:
                                    feature_importance = (
                                        importance_df.to_dict("records")[0]
                                        if len(importance_df) > 0
                                        else None
                                    )
                            except Exception as e:
                                logger.warning(f"Could not get feature importance: {e}")

                        prediction = TopLevelPrediction(
                            category=TopLevelCategory(class_name),
                            confidence=confidence,
                            feature_importance=feature_importance,
                        )

                        sample_predictions.append(prediction)

                # Ensure at least one prediction
                if not sample_predictions:
                    # Add most confident prediction as fallback
                    if isinstance(probabilities, list):
                        max_prob_idx = np.argmax(
                            [prob[i].max() for prob in probabilities]
                        )
                        confidence_value = float(probabilities[max_prob_idx][i].max())
                    else:
                        max_prob_idx = np.argmax(probabilities[i])
                        confidence_value = float(probabilities[i, max_prob_idx])

                    confidence = ConfidenceScore(
                        value=confidence_value, model_type=ModelType.ASSEMBLED
                    )

                    fallback_prediction = TopLevelPrediction(
                        category=TopLevelCategory(self.classes_[max_prob_idx]),
                        confidence=confidence,
                    )
                    sample_predictions = [fallback_prediction]

                results.extend(sample_predictions)

            prediction_time = time.time() - start_time
            logger.info(
                f"Generated {len(results)} predictions in {prediction_time:.2f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def _ensemble_predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble prediction using weighted combination (fallback method).

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Predictions and probabilities.
        """
        predictions = []
        probabilities = []
        weights = []

        # SVM predictions
        if self.svm_wrapper:
            try:
                svm_pred = self.svm_wrapper.predict(X)
                svm_proba = self.svm_wrapper.predict_proba(X)
                predictions.append(svm_pred)
                probabilities.append(svm_proba)
                weights.append(self.svm_weight)
            except Exception as e:
                logger.warning(f"SVM prediction failed: {e}")

        # LLM predictions
        if self.llm_wrapper:
            try:
                llm_pred = self.llm_wrapper.predict(X)
                llm_proba = self.llm_wrapper.predict_proba(X)
                predictions.append(llm_pred)
                probabilities.append(llm_proba)
                weights.append(self.llm_weight)
            except Exception as e:
                logger.warning(f"LLM prediction failed: {e}")

        if not predictions:
            raise ValueError("No component classifiers available for prediction")

        # Weighted ensemble
        weights = np.array(weights) / np.sum(weights)  # Normalize

        ensemble_pred = np.zeros_like(predictions[0])
        ensemble_proba = np.zeros_like(probabilities[0])

        for pred, proba, weight in zip(predictions, probabilities, weights):
            ensemble_pred += weight * pred
            ensemble_proba += weight * proba

        # Threshold predictions
        ensemble_pred = (ensemble_pred > 0.5).astype(int)

        return ensemble_pred, ensemble_proba

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the assembled classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Test features.
        y : pd.DataFrame
            True labels.

        Returns
        -------
        Dict[str, Any]
            Evaluation metrics.
        """
        if not self.is_fitted_:
            raise ValueError("Classifier not fitted. Call fit() first.")

        logger.info("Evaluating AssembledTopLevelClassifier")

        # Get predictions
        predictions = self.predict(X)

        # Convert predictions to binary format for evaluation
        y_pred = np.zeros((len(X), len(self.classes_)))
        pred_idx = 0

        for i in range(len(X)):
            # Find predictions for this sample
            while (
                pred_idx < len(predictions)
                and predictions[pred_idx].category.value in self.classes_
            ):
                cat_idx = self.classes_.index(predictions[pred_idx].category.value)
                y_pred[i, cat_idx] = 1.0
                pred_idx += 1

                # Check if this was the last prediction for this sample
                if pred_idx >= len(predictions) or (i < len(X) - 1):
                    break

        y_true = y.drop("asn", axis=1, errors="ignore").values

        # Calculate metrics
        results = {}
        for i, class_name in enumerate(self.classes_):
            report = classification_report(
                y_true[:, i], y_pred[:, i], output_dict=True, zero_division=0
            )
            results[class_name] = report

        return results


# Export the main class
__all__ = ["AssembledTopLevelClassifier", "LLMWrapper", "SVMWrapper"]
