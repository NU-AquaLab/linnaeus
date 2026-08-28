"""
Tests for hybrid classification models.
"""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from linnaeus.models.hybrid.stacking_classifier import (
    HybridASClassifier,
    LLMWrapper,
    SVMWrapper,
)
from linnaeus.models.unified_schemas import HierarchicalTags, UnifiedASClassification


class TestHybridASClassifier:
    """Test hybrid stacking classifier."""

    @pytest.fixture
    def mock_data_access(self, sample_organization_data):
        """Mock data access layer."""
        mock_access = Mock()
        mock_access.get_organization_data.return_value = sample_organization_data
        return mock_access

    @pytest.fixture
    def mock_svm_classifier(self):
        """Mock SVM classifier."""
        mock_svm = Mock()

        # Mock fit method
        mock_svm.fit.return_value = mock_svm

        # Mock predict method to return binary matrix
        mock_predictions = np.array([[1, 0, 0], [0, 1, 1]])  # 2 samples, 3 classes
        mock_svm.predict.return_value = mock_predictions

        # Mock predict_proba method
        mock_probabilities = np.array([[0.8, 0.2, 0.1], [0.1, 0.9, 0.7]])
        mock_svm.predict_proba.return_value = mock_probabilities

        return mock_svm

    @pytest.fixture
    def mock_llm_classifier(self):
        """Mock LLM classifier."""
        mock_llm = Mock()

        # Mock async methods
        mock_llm.classify_batch = AsyncMock()

        return mock_llm

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        X = pd.DataFrame({"asn": [174, 15169, 32934]})

        y = pd.DataFrame(
            {
                "asn": [174, 15169, 32934],
                "Transit": [1, 0, 0],
                "Content Provider": [0, 1, 1],
                "Enterprise": [1, 1, 1],
            }
        )

        return X, y

    @pytest.fixture
    def hybrid_classifier(self, mock_data_access):
        """Create hybrid classifier instance."""
        return HybridASClassifier(
            approach="hybrid",
            svm_weight=0.4,
            llm_weight=0.6,
            data_access=mock_data_access,
        )

    def test_initialization(self, hybrid_classifier):
        """Test hybrid classifier initialization."""
        assert hybrid_classifier.approach == "hybrid"
        assert hybrid_classifier.svm_weight == 0.4
        assert hybrid_classifier.llm_weight == 0.6
        assert hybrid_classifier.use_hierarchical_svm is True
        assert hybrid_classifier.is_fitted_ is False

    def test_different_approaches(self, mock_data_access):
        """Test different classification approaches."""
        # Test flat approach
        flat_classifier = HybridASClassifier(
            approach="flat", use_hierarchical_svm=False, data_access=mock_data_access
        )
        assert flat_classifier.approach == "flat"
        assert flat_classifier.use_hierarchical_svm is False

        # Test hierarchical approach
        hierarchical_classifier = HybridASClassifier(
            approach="hierarchical",
            use_hierarchical_svm=True,
            data_access=mock_data_access,
        )
        assert hierarchical_classifier.approach == "hierarchical"
        assert hierarchical_classifier.use_hierarchical_svm is True

    def test_fit_svm_only(
        self, mock_data_access, mock_svm_classifier, sample_training_data
    ):
        """Test fitting with SVM classifier only."""
        X, y = sample_training_data

        classifier = HybridASClassifier(
            approach="hybrid", llm_model_id=None, data_access=mock_data_access  # No LLM
        )

        # Mock the SVM classifier
        with patch.object(classifier, "svm_classifier", mock_svm_classifier):
            result = classifier.fit(X, y)

            # Should return self
            assert result is classifier
            assert classifier.is_fitted_ is True

            # Should set classes
            assert classifier.classes_ is not None
            assert len(classifier.classes_) == 3  # Excluding 'asn' column

    def test_fit_with_stacking(
        self, mock_data_access, mock_svm_classifier, sample_training_data
    ):
        """Test fitting with stacking classifier.

        Note: LLM integration is currently disabled in HybridASClassifier,
        so stacking is not created even when llm_model_id is provided.
        The classifier falls back to single-component mode.
        """
        X, y = sample_training_data

        classifier = HybridASClassifier(
            approach="hybrid", llm_model_id="test-model", data_access=mock_data_access
        )

        # Mock both classifiers
        with patch.object(classifier, "svm_classifier", mock_svm_classifier):
            result = classifier.fit(X, y)

            assert result is classifier
            assert classifier.is_fitted_ is True
            # LLM is currently disabled, so stacking is not created.
            # Only SVM is available, so it falls back to single classifier mode.
            assert classifier.stacking_classifier is None

    def test_predict_not_fitted(self, hybrid_classifier):
        """Test prediction before fitting."""
        X = pd.DataFrame({"asn": [174]})

        with pytest.raises(ValueError, match="not fitted"):
            hybrid_classifier.predict(X)

    def test_predict_with_stacking(
        self, mock_data_access, mock_svm_classifier, sample_training_data
    ):
        """Test prediction with stacking classifier."""
        X, y = sample_training_data
        X_test = pd.DataFrame({"asn": [174, 15169]})

        classifier = HybridASClassifier(
            approach="hybrid", llm_model_id="test-model", data_access=mock_data_access
        )

        # Mock stacking classifier
        mock_stacking_instance = Mock()
        mock_predictions = np.array([[1, 0, 0], [0, 1, 1]])
        mock_stacking_instance.predict.return_value = mock_predictions

        with (
            patch.object(classifier, "svm_classifier", mock_svm_classifier),
            patch.object(classifier, "stacking_classifier", mock_stacking_instance),
        ):
            classifier.classes_ = ["Transit", "Content Provider", "Enterprise"]
            classifier.is_fitted_ = True

            predictions = classifier.predict(X_test)

            assert predictions.shape == (2, 3)
            assert np.array_equal(predictions, mock_predictions)

    def test_predict_proba_with_stacking(
        self, mock_data_access, mock_svm_classifier, sample_training_data
    ):
        """Test probability prediction with stacking classifier."""
        X, y = sample_training_data
        X_test = pd.DataFrame({"asn": [174]})

        classifier = HybridASClassifier(
            approach="hybrid", llm_model_id="test-model", data_access=mock_data_access
        )

        # Mock stacking classifier probability prediction
        mock_stacking_instance = Mock()
        mock_probabilities = [
            np.array([[0.8, 0.2]]),  # Transit: prob of class 1
            np.array([[0.3, 0.7]]),  # Content Provider: prob of class 1
            np.array([[0.6, 0.4]]),  # Enterprise: prob of class 1
        ]
        mock_stacking_instance.predict_proba.return_value = mock_probabilities

        with (
            patch.object(classifier, "svm_classifier", mock_svm_classifier),
            patch.object(classifier, "stacking_classifier", mock_stacking_instance),
        ):
            classifier.classes_ = ["Transit", "Content Provider", "Enterprise"]
            classifier.is_fitted_ = True

            probabilities = classifier.predict_proba(X_test)

            assert probabilities.shape == (1, 3)
            # Should extract probability of positive class for each output
            expected_probs = np.array([[0.2, 0.7, 0.4]])  # Prob of class 1 for each
            assert np.array_equal(probabilities, expected_probs)

    def test_predict_unified(
        self, mock_data_access, mock_svm_classifier, sample_training_data
    ):
        """Test unified prediction interface."""
        X, y = sample_training_data
        X_test = pd.DataFrame({"asn": [174, 15169]})

        classifier = HybridASClassifier(approach="hybrid", data_access=mock_data_access)

        # Mock organization data
        mock_org_data = Mock()
        mock_org_data.name = "Test Organization"
        mock_data_access.get_organization_data.return_value = mock_org_data

        with patch.object(classifier, "svm_classifier", mock_svm_classifier):
            classifier.classes_ = ["Transit", "Content Provider", "Enterprise"]
            classifier.is_fitted_ = True

            # Mock predict and predict_proba
            mock_predictions = np.array([[1, 0, 1], [0, 1, 0]])
            mock_probabilities = np.array([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3]])

            with (
                patch.object(classifier, "predict", return_value=mock_predictions),
                patch.object(
                    classifier, "predict_proba", return_value=mock_probabilities
                ),
            ):
                results = classifier.predict_unified(X_test)

                assert len(results) == 2
                assert all(isinstance(r, UnifiedASClassification) for r in results)

                # Check first result
                result1 = results[0]
                assert result1.asn == 174
                assert result1.organization_name == "Test Organization"
                assert result1.classification_approach == "hybrid"
                assert result1.top_level_confidence is not None

    def test_ensemble_predict(self, mock_data_access, mock_svm_classifier):
        """Test ensemble prediction without stacking."""
        X_test = pd.DataFrame({"asn": [174]})

        classifier = HybridASClassifier(
            approach="hybrid",
            svm_weight=0.6,
            llm_weight=0.4,
            data_access=mock_data_access,
        )

        # Mock SVM predictions
        svm_pred = np.array([[0.8, 0.2, 0.9]])

        with (
            patch.object(classifier, "svm_classifier", mock_svm_classifier),
            patch.object(
                classifier,
                "_get_llm_predictions",
                return_value=np.array([[0.3, 0.9, 0.1]]),
            ),
        ):
            classifier.classes_ = ["Transit", "Content Provider", "Enterprise"]
            classifier.is_fitted_ = True
            classifier.llm_classifier = Mock()  # Enable LLM
            mock_svm_classifier.predict.return_value = svm_pred

            predictions = classifier._ensemble_predict(X_test)

            # Should be weighted combination
            expected = 0.6 * svm_pred + 0.4 * np.array([[0.3, 0.9, 0.1]])
            expected_binary = (expected > 0.5).astype(int)

            assert np.array_equal(predictions, expected_binary)

    def test_evaluation(
        self, mock_data_access, mock_svm_classifier, sample_training_data
    ):
        """Test model evaluation."""
        X, y = sample_training_data

        classifier = HybridASClassifier(approach="hybrid", data_access=mock_data_access)

        with patch.object(classifier, "svm_classifier", mock_svm_classifier):
            classifier.classes_ = ["Transit", "Content Provider", "Enterprise"]
            classifier.is_fitted_ = True

            # Mock predictions - must match the 3 samples in X
            mock_predictions = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])

            # Also mock SVM predict to return matching shape (3 samples)
            mock_svm_predictions = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
            mock_svm_classifier.predict.return_value = mock_svm_predictions

            with patch.object(classifier, "predict", return_value=mock_predictions):
                results = classifier.evaluate(X, y)

                assert "hybrid" in results
                assert "components" in results
                assert len(results["hybrid"]) == 3  # One result per class

    def test_feature_importance(self, mock_data_access, mock_svm_classifier):
        """Test feature importance extraction."""
        classifier = HybridASClassifier(approach="hybrid", data_access=mock_data_access)

        # Mock SVM with feature importance
        mock_svm_classifier.get_feature_importance.return_value = pd.DataFrame(
            {"feature": ["feat1", "feat2"], "importance": [0.8, 0.6]}
        )

        # Mock stacking classifier with coefficients
        mock_stacking = Mock()
        mock_meta_estimator = Mock()
        mock_meta_estimator.coef_ = np.array([0.7, 0.3])
        mock_stacking.final_estimator_ = mock_meta_estimator
        mock_stacking.estimators_ = [("svm", Mock()), ("llm", Mock())]

        with (
            patch.object(classifier, "svm_classifier", mock_svm_classifier),
            patch.object(classifier, "stacking_classifier", mock_stacking),
        ):
            importance = classifier.get_feature_importance()

            assert "svm" in importance
            assert "meta_learner" in importance
            assert isinstance(importance["meta_learner"], pd.DataFrame)


class TestSVMWrapper:
    """Test SVM wrapper for sklearn stacking compatibility."""

    @pytest.fixture
    def mock_svm_classifier(self):
        """Mock SVM classifier."""
        mock_svm = Mock()
        mock_svm.predict.return_value = pd.DataFrame([[1, 0], [0, 1]])
        mock_svm.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        return mock_svm

    def test_initialization(self, mock_svm_classifier):
        """Test SVM wrapper initialization."""
        wrapper = SVMWrapper(mock_svm_classifier)
        assert wrapper.svm_classifier is mock_svm_classifier

    def test_fit(self, mock_svm_classifier):
        """Test wrapper fit method."""
        wrapper = SVMWrapper(mock_svm_classifier)

        X = np.array([[1], [2]])
        y = np.array([0, 1])

        result = wrapper.fit(X, y)
        assert result is wrapper

    def test_predict(self, mock_svm_classifier):
        """Test wrapper predict method."""
        wrapper = SVMWrapper(mock_svm_classifier)

        X = np.array([[1], [2]])
        predictions = wrapper.predict(X)

        # Should convert DataFrame to numpy array
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (2, 2)

    def test_predict_proba(self, mock_svm_classifier):
        """Test wrapper predict_proba method."""
        wrapper = SVMWrapper(mock_svm_classifier)

        X = np.array([[1], [2]])
        probabilities = wrapper.predict_proba(X)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (2, 2)

    def test_predict_proba_fallback(self, mock_svm_classifier):
        """Test predict_proba fallback when not available."""
        # Remove predict_proba method
        del mock_svm_classifier.predict_proba

        wrapper = SVMWrapper(mock_svm_classifier)

        X = np.array([[1], [2]])
        probabilities = wrapper.predict_proba(X)

        # Should fallback to predictions as probabilities
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.dtype == float


class TestLLMWrapper:
    """Test LLM wrapper for sklearn stacking compatibility."""

    @pytest.fixture
    def mock_llm_classifier(self):
        """Mock LLM classifier."""
        return Mock()

    @pytest.fixture
    def mock_data_access(self):
        """Mock data access."""
        return Mock()

    def test_initialization(self, mock_llm_classifier, mock_data_access):
        """Test LLM wrapper initialization."""
        wrapper = LLMWrapper(mock_llm_classifier, mock_data_access)
        assert wrapper.llm_classifier is mock_llm_classifier
        assert wrapper.data_access is mock_data_access

    def test_fit(self, mock_llm_classifier, mock_data_access):
        """Test wrapper fit method."""
        wrapper = LLMWrapper(mock_llm_classifier, mock_data_access)

        X = np.array([[1], [2]])
        y = np.array([0, 1])

        result = wrapper.fit(X, y)
        assert result is wrapper

    def test_predict_placeholder(self, mock_llm_classifier, mock_data_access):
        """Test wrapper predict placeholder."""
        wrapper = LLMWrapper(mock_llm_classifier, mock_data_access)

        X = np.array([[1], [2]])
        predictions = wrapper.predict(X)

        # Currently returns placeholder zeros
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (2, 1)
        assert np.all(predictions == 0)

    def test_predict_proba_placeholder(self, mock_llm_classifier, mock_data_access):
        """Test wrapper predict_proba placeholder."""
        wrapper = LLMWrapper(mock_llm_classifier, mock_data_access)

        X = np.array([[1], [2]])
        probabilities = wrapper.predict_proba(X)

        # Currently returns placeholder 0.5 values
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (2, 1)
        assert np.all(probabilities == 0.5)


class TestHybridIntegration:
    """Test integration between hybrid model components."""

    @pytest.fixture
    def complete_mock_setup(self, sample_organization_data):
        """Set up complete mock environment."""
        mock_data_access = Mock()
        mock_data_access.get_organization_data.return_value = sample_organization_data

        mock_svm = Mock()
        mock_svm.fit.return_value = mock_svm
        mock_svm.predict.return_value = np.array([[1, 0], [0, 1]])
        mock_svm.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

        return mock_data_access, mock_svm

    def test_end_to_end_hybrid_workflow(self, complete_mock_setup):
        """Test complete hybrid classification workflow."""
        mock_data_access, mock_svm = complete_mock_setup

        # Create training data
        X_train = pd.DataFrame({"asn": [174, 15169]})
        y_train = pd.DataFrame(
            {"asn": [174, 15169], "Transit": [1, 0], "Content Provider": [0, 1]}
        )

        # Create test data
        X_test = pd.DataFrame({"asn": [32934]})

        # Initialize classifier
        classifier = HybridASClassifier(
            approach="hybrid",
            svm_weight=0.7,
            llm_weight=0.3,
            data_access=mock_data_access,
        )

        with patch.object(classifier, "svm_classifier", mock_svm):
            # Training phase
            classifier.fit(X_train, y_train)
            assert classifier.is_fitted_ is True

            # Override SVM predict for test data (1 sample, 2 classes)
            mock_svm.predict.return_value = np.array([[1, 0]])

            # Prediction phase
            predictions = classifier.predict(X_test)
            assert predictions.shape == (1, 2)

            # Unified prediction
            unified_results = classifier.predict_unified(X_test)
            assert len(unified_results) == 1
            assert isinstance(unified_results[0], UnifiedASClassification)

    def test_different_approach_consistency(self, complete_mock_setup):
        """Test consistency across different approaches."""
        mock_data_access, mock_svm = complete_mock_setup

        X = pd.DataFrame({"asn": [174]})
        y = pd.DataFrame({"asn": [174], "Transit": [1]})

        approaches = ["flat", "hierarchical", "hybrid"]
        results = {}

        for approach in approaches:
            classifier = HybridASClassifier(
                approach=approach, data_access=mock_data_access
            )

            # Set mock SVM to return 1 sample prediction
            mock_svm.predict.return_value = np.array([[1, 0]])
            mock_svm.predict_proba.return_value = np.array([[0.8, 0.2]])

            with patch.object(classifier, "svm_classifier", mock_svm):
                classifier.fit(X, y)
                unified_results = classifier.predict_unified(X)
                results[approach] = unified_results[0]

        # All approaches should produce valid results
        for approach, result in results.items():
            assert result.asn == 174
            # Note: 'hierarchical' approach may fall back to 'flat' when no
            # hierarchical tags are predicted (this is correct behavior)
            if approach == "hierarchical":
                assert result.classification_approach in ("hierarchical", "flat")
            else:
                assert result.classification_approach == approach
            assert isinstance(result, UnifiedASClassification)

    def test_tag_system_integration(self, complete_mock_setup):
        """Test integration with unified tag system."""
        mock_data_access, mock_svm = complete_mock_setup

        classifier = HybridASClassifier(
            approach="hierarchical", data_access=mock_data_access
        )

        X = pd.DataFrame({"asn": [15169]})

        # Mock hierarchical tags in classes
        classifier.classes_ = [tag.value for tag in list(HierarchicalTags)[:5]]
        classifier.is_fitted_ = True

        # Mock predictions with hierarchical tags
        mock_predictions = np.array([[1, 0, 1, 0, 0]])  # Predict first and third tags

        with (
            patch.object(classifier, "predict", return_value=mock_predictions),
            patch.object(
                classifier, "predict_proba", return_value=mock_predictions.astype(float)
            ),
        ):
            results = classifier.predict_unified(X)

            assert len(results) == 1
            result = results[0]

            # Should have hierarchical tags
            assert len(result.hierarchical_tags) == 2  # Two predicted tags
            assert all(
                isinstance(tag, HierarchicalTags) for tag in result.hierarchical_tags
            )
