"""
Tests for the assembled top-level classifier (Stage 1).
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from linnaeus.models.two_stage.assembled_classifier import (
    AssembledTopLevelClassifier,
    LLMWrapper,
    SVMWrapper,
)
from linnaeus.models.two_stage_schemas import (
    ConfidenceScore,
    ModelType,
    TopLevelPrediction,
)


class TestAssembledTopLevelClassifier:
    """Test the assembled classifier for Stage 1."""

    @pytest.fixture
    def mock_llm_processor(self):
        """Mock LLM processor."""
        mock_processor = Mock()
        # Mock process_batch to return predictions (used by LLMWrapper.predict)
        mock_processor.process_batch.return_value = [
            {"classifications": [{"category": "Transit"}]},
            {"classifications": [{"category": "ContentProvider"}]},
        ]
        return mock_processor

    @pytest.fixture
    def mock_data_access(self):
        """Mock data access layer."""
        mock_data = Mock()
        # Mock get_organization_data
        mock_data.get_organization_data.return_value = Mock(
            name="Test Organization", description="Test description"
        )
        return mock_data

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        X = pd.DataFrame(
            {"asn": [174, 15169, 32934], "name": ["Cogent", "Google", "Facebook"]}
        )
        y = pd.DataFrame(
            {
                "asn": [174, 15169, 32934],
                "Transit": [1, 0, 0],
                "ContentProvider": [0, 1, 1],
                "Enterprise": [1, 1, 1],
            }
        )
        return X, y

    @pytest.fixture
    def classifier(self, mock_llm_processor, mock_data_access):
        """Create test classifier instance."""
        return AssembledTopLevelClassifier(
            llm_processor=mock_llm_processor,
            svm_params={
                "svm_params": {"kernel": "rbf", "C": 1.0, "probability": True},
                "random_state": 42,
            },
            svm_weight=0.4,
            llm_weight=0.6,
            data_access=mock_data_access,
        )

    def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.svm_weight == 0.4
        assert classifier.llm_weight == 0.6
        assert isinstance(classifier.meta_learner, LogisticRegression)
        assert classifier.svm_params is not None

    def test_weights_validation(self, mock_llm_processor, mock_data_access):
        """Test weight validation."""
        # Weights that don't sum to 1.0 should still work but log warning
        classifier = AssembledTopLevelClassifier(
            llm_processor=mock_llm_processor,
            svm_weight=0.3,
            llm_weight=0.5,  # Sum = 0.8
            data_access=mock_data_access,
            svm_params={},  # No extra SVM params
        )
        assert classifier.svm_weight == 0.3
        assert classifier.llm_weight == 0.5

    def test_fit_method(self, classifier, sample_training_data):
        """Test the fit method creates SVM wrapper and sets is_fitted_."""
        X, y = sample_training_data

        # Mock SVMWrapper and LLMWrapper so we don't need real training
        mock_svm_wrapper = Mock(spec=SVMWrapper)
        mock_svm_wrapper.fit.return_value = mock_svm_wrapper

        mock_llm_wrapper = Mock(spec=LLMWrapper)
        mock_llm_wrapper.fit.return_value = mock_llm_wrapper

        mock_stacking = Mock()

        with (
            patch(
                "linnaeus.models.two_stage.assembled_classifier.SVMWrapper",
                return_value=mock_svm_wrapper,
            ),
            patch(
                "linnaeus.models.two_stage.assembled_classifier.LLMWrapper",
                return_value=mock_llm_wrapper,
            ),
            patch(
                "linnaeus.models.two_stage.assembled_classifier.StackingClassifier",
                return_value=mock_stacking,
            ),
        ):
            fitted_classifier = classifier.fit(X, y)

            assert fitted_classifier is classifier
            assert classifier.is_fitted_
            assert classifier.svm_wrapper is mock_svm_wrapper
            mock_svm_wrapper.fit.assert_called_once()

    def test_predict_method(self, classifier):
        """Test the predict method returns TopLevelPrediction objects."""
        # Set up fitted state
        classifier.is_fitted_ = True

        # Mock svm_wrapper for ensemble prediction (no stacking)
        mock_svm = Mock()
        n_classes = len(classifier.classes_)
        # Predictions: one sample, values > 0.5 mean positive prediction
        mock_svm.predict.return_value = np.zeros((2, n_classes))
        mock_svm.predict.return_value[0, 1] = 1.0  # Transit for first sample
        mock_svm.predict.return_value[1, 4] = 1.0  # ContentProvider for second
        mock_svm.predict_proba.return_value = np.full((2, n_classes), 0.1)
        mock_svm.predict_proba.return_value[0, 1] = 0.9
        mock_svm.predict_proba.return_value[1, 4] = 0.85
        classifier.svm_wrapper = mock_svm
        classifier.stacking_classifier = None  # Use ensemble fallback

        # Test data
        X = pd.DataFrame({"asn": [174, 15169], "name": ["Cogent", "Google"]})

        predictions = classifier.predict(X)

        assert len(predictions) >= 2
        assert all(isinstance(pred, TopLevelPrediction) for pred in predictions)
        assert all(0 <= pred.confidence.value <= 1 for pred in predictions)

    def test_predict_not_fitted_error(self, classifier):
        """Test error when predicting with unfitted classifier."""
        X = pd.DataFrame({"asn": [174], "name": ["Test"]})

        with pytest.raises(ValueError, match="not fitted"):
            classifier.predict(X)

    def test_fit_creates_svm_wrapper(self, classifier, sample_training_data):
        """Test that fit creates SVMWrapper and trains it."""
        X, y = sample_training_data

        # Mock SVMWrapper and LLMWrapper to avoid real training
        mock_svm_wrapper = Mock(spec=SVMWrapper)
        mock_svm_wrapper.fit.return_value = mock_svm_wrapper

        mock_llm_wrapper = Mock(spec=LLMWrapper)
        mock_llm_wrapper.fit.return_value = mock_llm_wrapper

        mock_stacking = Mock()

        with (
            patch(
                "linnaeus.models.two_stage.assembled_classifier.SVMWrapper",
                return_value=mock_svm_wrapper,
            ) as mock_svm_class,
            patch(
                "linnaeus.models.two_stage.assembled_classifier.LLMWrapper",
                return_value=mock_llm_wrapper,
            ),
            patch(
                "linnaeus.models.two_stage.assembled_classifier.StackingClassifier",
                return_value=mock_stacking,
            ),
        ):
            classifier.fit(X, y)

            # SVMWrapper was created with svm_classifier and data_adapter
            mock_svm_class.assert_called_once_with(
                classifier.svm_classifier, classifier.data_adapter
            )
            # SVMWrapper was fitted
            mock_svm_wrapper.fit.assert_called_once_with(X, y)

    def test_fit_with_llm_creates_stacking(self, classifier, sample_training_data):
        """Test that fit with LLM processor creates stacking classifier."""
        X, y = sample_training_data

        # Mock SVMWrapper
        mock_svm_wrapper = Mock(spec=SVMWrapper)
        mock_svm_wrapper.fit.return_value = mock_svm_wrapper

        # Mock LLMWrapper
        mock_llm_wrapper = Mock(spec=LLMWrapper)
        mock_llm_wrapper.fit.return_value = mock_llm_wrapper

        # Mock StackingClassifier
        mock_stacking = Mock()

        with (
            patch(
                "linnaeus.models.two_stage.assembled_classifier.SVMWrapper",
                return_value=mock_svm_wrapper,
            ),
            patch(
                "linnaeus.models.two_stage.assembled_classifier.LLMWrapper",
                return_value=mock_llm_wrapper,
            ),
            patch(
                "linnaeus.models.two_stage.assembled_classifier.StackingClassifier",
                return_value=mock_stacking,
            ),
        ):
            classifier.fit(X, y)

            assert classifier.is_fitted_
            assert classifier.svm_wrapper is mock_svm_wrapper
            assert classifier.llm_wrapper is mock_llm_wrapper
            assert classifier.stacking_classifier is mock_stacking
            mock_stacking.fit.assert_called_once()

    def test_fit_without_llm_no_stacking(self, mock_data_access, sample_training_data):
        """Test that fit without LLM processor does not create stacking."""
        X, y = sample_training_data

        classifier = AssembledTopLevelClassifier(
            llm_processor=None, svm_params={}, data_access=mock_data_access  # No LLM
        )

        # Mock SVMWrapper
        mock_svm_wrapper = Mock(spec=SVMWrapper)
        mock_svm_wrapper.fit.return_value = mock_svm_wrapper

        with patch(
            "linnaeus.models.two_stage.assembled_classifier.SVMWrapper",
            return_value=mock_svm_wrapper,
        ):
            classifier.fit(X, y)

            assert classifier.is_fitted_
            assert (
                classifier.stacking_classifier is None
            )  # No stacking with single estimator
            assert classifier.llm_wrapper is None

    def test_meta_learner_default(self, classifier):
        """Test meta-learner defaults to LogisticRegression."""
        assert classifier.meta_learner is not None
        assert isinstance(classifier.meta_learner, LogisticRegression)
        assert hasattr(classifier.meta_learner, "predict_proba")

    def test_confidence_score_in_predictions(self, classifier):
        """Test that predictions include proper confidence scores."""
        classifier.is_fitted_ = True
        classifier.stacking_classifier = None

        # Mock svm_wrapper for ensemble predict
        n_classes = len(classifier.classes_)
        mock_svm = Mock()
        mock_svm.predict.return_value = np.zeros((1, n_classes))
        mock_svm.predict.return_value[0, 1] = 1.0  # Transit
        mock_svm.predict_proba.return_value = np.full((1, n_classes), 0.05)
        mock_svm.predict_proba.return_value[0, 1] = 0.85
        classifier.svm_wrapper = mock_svm

        X = pd.DataFrame({"asn": [174], "name": ["Test"]})
        predictions = classifier.predict(X)

        assert len(predictions) >= 1
        for pred in predictions:
            assert isinstance(pred.confidence, ConfidenceScore)
            assert pred.confidence.model_type == ModelType.ASSEMBLED
            assert 0 <= pred.confidence.value <= 1

    def test_ensemble_predict_fallback(self, classifier):
        """Test _ensemble_predict when no stacking classifier."""
        classifier.is_fitted_ = True
        classifier.stacking_classifier = None

        n_classes = len(classifier.classes_)

        # Mock svm_wrapper
        mock_svm = Mock()
        mock_svm.predict.return_value = np.zeros((1, n_classes))
        mock_svm.predict.return_value[0, 1] = 1.0  # Transit
        mock_svm.predict_proba.return_value = np.full((1, n_classes), 0.05)
        mock_svm.predict_proba.return_value[0, 1] = 0.9
        classifier.svm_wrapper = mock_svm

        # Mock llm_wrapper
        mock_llm = Mock()
        mock_llm.predict.return_value = np.zeros((1, n_classes))
        mock_llm.predict.return_value[0, 1] = 1.0  # Transit
        mock_llm.predict_proba.return_value = np.full((1, n_classes), 0.05)
        mock_llm.predict_proba.return_value[0, 1] = 0.8
        classifier.llm_wrapper = mock_llm

        X = pd.DataFrame({"asn": [174]})
        predictions, probabilities = classifier._ensemble_predict(X)

        # Should be weighted average
        assert predictions.shape == (1, n_classes)
        assert probabilities.shape == (1, n_classes)
        # Transit prediction should be positive (weighted avg of 1.0 and 1.0 > 0.5)
        assert predictions[0, 1] == 1

    def test_error_handling_in_prediction(self, classifier):
        """Test error handling during prediction."""
        classifier.is_fitted_ = True
        classifier.stacking_classifier = None

        # Mock svm_wrapper to raise exception
        mock_svm = Mock()
        mock_svm.predict.side_effect = Exception("SVM prediction failed")
        mock_svm.predict_proba.side_effect = Exception("SVM prediction failed")
        classifier.svm_wrapper = mock_svm
        classifier.llm_wrapper = None  # No LLM fallback

        X = pd.DataFrame({"asn": [174], "name": ["Test"]})

        with pytest.raises((Exception, ValueError)):
            classifier.predict(X)

    def test_model_parameters_validation(self):
        """Test model parameter validation."""
        # AssembledTopLevelClassifier accepts meta_learner as BaseEstimator
        # or None (defaults to LogisticRegression). Passing an invalid string
        # would fail at fit time, not at init time.
        classifier = AssembledTopLevelClassifier(
            meta_learner=LogisticRegression(), svm_params={}
        )
        assert classifier.meta_learner is not None

    def test_sklearn_compatibility(self, classifier):
        """Test sklearn interface compatibility."""
        # Should have sklearn methods from BaseEstimator/ClassifierMixin
        assert hasattr(classifier, "fit")
        assert hasattr(classifier, "predict")
        assert hasattr(classifier, "get_params")
        assert hasattr(classifier, "set_params")

        # get_params(deep=False) returns only the top-level constructor parameters
        # (deep=True would fail because Mock objects are not iterable)
        params = classifier.get_params(deep=False)
        assert "svm_weight" in params
        assert "llm_weight" in params
        assert params["svm_weight"] == 0.4
        assert params["llm_weight"] == 0.6

    def test_classes_attribute(self, classifier):
        """Test that classes_ attribute is set on initialization."""
        assert hasattr(classifier, "classes_")
        assert isinstance(classifier.classes_, list)
        assert "Transit" in classifier.classes_
        assert "Access" in classifier.classes_
        assert "ContentProvider" in classifier.classes_

    def test_is_fitted_initially_false(self, classifier):
        """Test that is_fitted_ is False before fit."""
        assert classifier.is_fitted_ is False
