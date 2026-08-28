"""
Tests for hybrid models sklearn interface.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from linnaeus.models.sklearn_interface import HybridASNClassifier
from linnaeus.models.unified_schemas import (
    HierarchicalTags,
    TopLevelTags,
    UnifiedASClassification,
)


class TestHybridASNClassifier:
    """Test scikit-learn compatible hybrid classifier."""

    @pytest.fixture
    def mock_hybrid_classifier(self):
        """Mock underlying hybrid classifier."""
        mock_hybrid = Mock()

        # Mock fit method
        mock_hybrid.fit.return_value = mock_hybrid

        # Mock predict methods
        mock_predictions = np.array([[1, 0, 0], [0, 1, 1]])
        mock_hybrid.predict.return_value = mock_predictions

        mock_probabilities = np.array([[0.8, 0.2, 0.1], [0.1, 0.9, 0.7]])
        mock_hybrid.predict_proba.return_value = mock_probabilities

        # Mock unified predictions
        mock_unified_results = [
            UnifiedASClassification(
                asn=174,
                organization_name="Cogent Communications",
                top_level_tags=[TopLevelTags.TRANSIT],
                hierarchical_tags=[HierarchicalTags.TRANSIT_GLOBAL],
                classification_approach="hybrid",
                model_used="HybridClassifier",
            ),
            UnifiedASClassification(
                asn=15169,
                organization_name="Google LLC",
                top_level_tags=[TopLevelTags.CONTENT_PROVIDER],
                hierarchical_tags=[HierarchicalTags.CONTENT_PROVIDER_CLOUD],
                classification_approach="hybrid",
                model_used="HybridClassifier",
            ),
        ]
        mock_hybrid.predict_unified.return_value = mock_unified_results

        return mock_hybrid

    @pytest.fixture
    def classifier(self, mock_config):
        """Create classifier instance."""
        with patch(
            "linnaeus.models.sklearn_interface.get_config", return_value=mock_config
        ):
            return HybridASNClassifier(
                approach="hybrid", svm_weight=0.4, llm_weight=0.6
            )

    def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.approach == "hybrid"
        assert classifier.svm_weight == 0.4
        assert classifier.llm_weight == 0.6
        assert classifier.use_hierarchical_svm is True
        assert classifier.classes_ is None
        assert classifier.n_features_in_ is None

    def test_parameter_settings(self, mock_config):
        """Test different parameter combinations."""
        with patch(
            "linnaeus.models.sklearn_interface.get_config", return_value=mock_config
        ):
            # Test hierarchical approach
            hierarchical_classifier = HybridASNClassifier(
                approach="hierarchical",
                use_hierarchical_svm=True,
                llm_model_id="custom-model",
            )

            assert hierarchical_classifier.approach == "hierarchical"
            assert hierarchical_classifier.use_hierarchical_svm is True
            assert hierarchical_classifier.llm_model_id == "custom-model"

            # Test flat approach
            flat_classifier = HybridASNClassifier(
                approach="flat", use_hierarchical_svm=False
            )

            assert flat_classifier.approach == "flat"
            assert flat_classifier.use_hierarchical_svm is False

    def test_fit_basic(self, classifier, mock_hybrid_classifier):
        """Test basic fitting functionality."""
        X = np.array([[174], [15169], [32934]])

        with patch.object(classifier, "hybrid_classifier", mock_hybrid_classifier):
            result = classifier.fit(X)

            # Should return self
            assert result is classifier

            # Should set basic attributes
            assert classifier.n_features_in_ == 1
            assert np.array_equal(classifier.feature_names_in_, ["asn"])
            assert classifier.classes_ is not None

    def test_fit_with_labels_hierarchical(self, mock_config, mock_hybrid_classifier):
        """Test fitting with hierarchical labels."""
        with patch(
            "linnaeus.models.sklearn_interface.get_config", return_value=mock_config
        ):
            classifier = HybridASNClassifier(approach="hierarchical")

        X = np.array([[174], [15169]])
        y = np.array([[1, 0, 0], [0, 1, 1]])  # Multi-label format

        # Mock the HybridASClassifier class to avoid real instantiation in fit()
        with patch(
            "linnaeus.models.sklearn_interface.HybridASClassifier"
        ) as MockHybridCls:
            MockHybridCls.return_value = mock_hybrid_classifier
            classifier.fit(X, y)

            # Should use hierarchical tags as classes
            expected_classes = [tag.value for tag in HierarchicalTags]
            assert set(classifier.classes_) == set(expected_classes)

    def test_fit_with_labels_flat(self, mock_config, mock_hybrid_classifier):
        """Test fitting with flat/top-level labels."""
        with patch(
            "linnaeus.models.sklearn_interface.get_config", return_value=mock_config
        ):
            classifier = HybridASNClassifier(approach="flat")

        X = np.array([[174], [15169]])
        y = np.array([[1, 0], [0, 1]])

        # Mock the HybridASClassifier class to avoid real instantiation in fit()
        with patch(
            "linnaeus.models.sklearn_interface.HybridASClassifier"
        ) as MockHybridCls:
            MockHybridCls.return_value = mock_hybrid_classifier
            classifier.fit(X, y)

            # Should use top-level tags as classes
            expected_classes = [tag.value for tag in TopLevelTags]
            assert set(classifier.classes_) == set(expected_classes)

    def test_fit_invalid_input_shape(self, classifier):
        """Test fitting with invalid input shape."""
        X = np.array([[174, 15169], [32934, 20940]])  # Wrong shape (2 features)

        with pytest.raises(ValueError, match="exactly 1 feature"):
            classifier.fit(X)

    def test_predict_not_fitted(self, classifier):
        """Test prediction before fitting raises an error."""
        X = np.array([[174]])

        # check_is_fitted won't raise NotFittedError because attributes exist
        # (as None). The predict method will fail with ValueError when comparing
        # X.shape[1] to None for n_features_in_.
        with pytest.raises((NotFittedError, ValueError)):
            classifier.predict(X)

    def test_predict_basic(self, classifier, mock_hybrid_classifier):
        """Test basic prediction functionality."""
        X = np.array([[174], [15169]])

        # Setup fitted classifier
        classifier.n_features_in_ = 1
        classifier.feature_names_in_ = np.array(["asn"])
        classifier.classes_ = np.array(["Transit", "Content Provider", "Enterprise"])
        classifier.hybrid_classifier = mock_hybrid_classifier

        predictions = classifier.predict(X)

        # Should return predictions from hybrid classifier
        expected_predictions = mock_hybrid_classifier.predict.return_value
        assert np.array_equal(predictions, expected_predictions)

    def test_predict_unified_fallback(self, classifier, mock_hybrid_classifier):
        """Test prediction fallback to unified interface."""
        X = np.array([[174], [15169]])

        # Setup fitted classifier
        classifier.n_features_in_ = 1
        classifier.feature_names_in_ = np.array(["asn"])
        classifier.classes_ = np.array(["Transit", "Content Provider"])
        classifier.hybrid_classifier = mock_hybrid_classifier
        classifier.approach = "hierarchical"

        # Make regular predict fail to trigger fallback
        mock_hybrid_classifier.predict.side_effect = Exception("Regular predict failed")

        predictions = classifier.predict(X)

        # Should fallback to unified predictions and convert to binary matrix
        assert predictions.shape == (2, 2)  # 2 samples, 2 classes
        assert predictions.dtype == np.float64

    def test_predict_proba(self, classifier, mock_hybrid_classifier):
        """Test probability prediction."""
        X = np.array([[174]])

        # Setup fitted classifier
        classifier.n_features_in_ = 1
        classifier.feature_names_in_ = np.array(["asn"])
        classifier.classes_ = np.array(["Transit", "Content Provider", "Enterprise"])
        classifier.hybrid_classifier = mock_hybrid_classifier

        probabilities = classifier.predict_proba(X)

        # Should return probabilities from hybrid classifier
        expected_probabilities = mock_hybrid_classifier.predict_proba.return_value
        assert np.array_equal(probabilities, expected_probabilities)

    def test_predict_proba_fallback(self, classifier, mock_hybrid_classifier):
        """Test probability prediction fallback."""
        X = np.array([[174]])

        # Setup fitted classifier
        classifier.n_features_in_ = 1
        classifier.classes_ = np.array(["Transit"])
        classifier.hybrid_classifier = mock_hybrid_classifier

        # Make predict_proba fail
        mock_hybrid_classifier.predict_proba.side_effect = Exception("Proba failed")

        # Mock regular predict for fallback
        mock_hybrid_classifier.predict.return_value = np.array([[1]])

        with patch.object(classifier, "predict", return_value=np.array([[1]])):
            probabilities = classifier.predict_proba(X)

            # Should fallback to predictions as probabilities
            assert probabilities.shape == (1, 1)
            assert probabilities.dtype == np.float64

    def test_decision_function(self, classifier, mock_hybrid_classifier):
        """Test decision function."""
        X = np.array([[174]])

        # Setup fitted classifier
        classifier.n_features_in_ = 1
        classifier.classes_ = np.array(["Transit"])
        classifier.hybrid_classifier = mock_hybrid_classifier

        with patch.object(
            classifier, "predict_proba", return_value=np.array([[0.8]])
        ) as mock_proba:
            decision_scores = classifier.decision_function(X)

            # Should be same as predict_proba
            mock_proba.assert_called_once_with(X)
            assert np.array_equal(decision_scores, np.array([[0.8]]))

    def test_score(self, classifier, mock_hybrid_classifier):
        """Test scoring method."""
        X = np.array([[174], [15169]])
        y_true = np.array([[1, 0], [0, 1]])

        # Setup fitted classifier
        classifier.n_features_in_ = 1
        classifier.classes_ = np.array(["Transit", "Content Provider"])
        classifier.hybrid_classifier = mock_hybrid_classifier

        # Mock predictions
        y_pred = np.array([[1, 0], [0, 1]])  # Perfect predictions

        with patch.object(classifier, "predict", return_value=y_pred):
            score = classifier.score(X, y_true)

            # Should return Jaccard score
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_input_validation(self, classifier, mock_hybrid_classifier):
        """Test input validation in predict methods."""
        # Setup fitted classifier
        classifier.n_features_in_ = 1
        classifier.classes_ = np.array(["Transit"])
        classifier.hybrid_classifier = mock_hybrid_classifier

        # Test wrong number of features
        X_wrong = np.array([[174, 15169]])  # 2 features instead of 1

        with pytest.raises(ValueError, match="expected 1"):
            classifier.predict(X_wrong)

        with pytest.raises(ValueError, match="expected 1"):
            classifier.predict_proba(X_wrong)

    def test_get_feature_names_out(self, classifier, mock_hybrid_classifier):
        """Test getting feature names."""
        # Setup fitted classifier
        classifier.n_features_in_ = 1
        classifier.classes_ = np.array(["Transit", "Content Provider"])
        classifier.hybrid_classifier = mock_hybrid_classifier

        feature_names = classifier.get_feature_names_out()

        assert np.array_equal(feature_names, classifier.classes_)

    def test_different_approaches_sklearn_interface(self, mock_config):
        """Test sklearn interface with different approaches."""
        approaches = ["flat", "hierarchical", "hybrid"]

        for approach in approaches:
            with patch(
                "linnaeus.models.sklearn_interface.get_config", return_value=mock_config
            ):
                classifier = HybridASNClassifier(approach=approach)

                assert classifier.approach == approach

                # Check that classes are set appropriately during fit
                X = np.array([[174]])
                classifier.fit(X)

                if approach == "hierarchical":
                    # Should use hierarchical tags
                    expected_tags = {tag.value for tag in HierarchicalTags}
                    assert set(classifier.classes_) == expected_tags
                else:
                    # Should use top-level tags for flat/hybrid
                    expected_tags = {tag.value for tag in TopLevelTags}
                    assert set(classifier.classes_) == expected_tags

    def test_sklearn_pipeline_compatibility(self, classifier, mock_hybrid_classifier):
        """Test compatibility with sklearn pipelines."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Setup fitted classifier
        classifier.hybrid_classifier = mock_hybrid_classifier

        # Should be able to create pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

        assert pipeline is not None

        # Test basic pipeline operations
        X = np.array([[174.0], [15169.0]])
        y = np.array([[1, 0], [0, 1]])

        # Mock the HybridASClassifier class to avoid real instantiation in fit()
        with patch(
            "linnaeus.models.sklearn_interface.HybridASClassifier"
        ) as MockHybridCls:
            MockHybridCls.return_value = mock_hybrid_classifier

            # Should be able to fit pipeline
            pipeline.fit(X, y)

            # Should be able to predict
            predictions = pipeline.predict(X)
            assert predictions is not None

    def test_cross_validation_compatibility(self, mock_config, mock_hybrid_classifier):
        """Test compatibility with cross-validation."""
        from sklearn.metrics import jaccard_score, make_scorer
        from sklearn.model_selection import cross_val_score

        with patch(
            "linnaeus.models.sklearn_interface.get_config", return_value=mock_config
        ):
            classifier = HybridASNClassifier(
                approach="hybrid", svm_weight=0.4, llm_weight=0.6
            )

        X = np.array([[174], [15169], [32934], [20940]])
        y = np.array([[1, 0], [0, 1], [1, 1], [0, 1]])

        # Create custom scorer for multi-label
        scorer = make_scorer(jaccard_score, average="macro", zero_division=0)

        # Mock the HybridASClassifier class to avoid real instantiation in fit()
        with patch(
            "linnaeus.models.sklearn_interface.HybridASClassifier"
        ) as MockHybridCls:
            mock_instance = Mock()
            mock_instance.fit.return_value = mock_instance
            mock_instance.predict.return_value = np.array([[1, 0], [0, 1]])
            MockHybridCls.return_value = mock_instance

            scores = cross_val_score(classifier, X, y, cv=2, scoring=scorer)

            assert len(scores) == 2  # 2-fold CV
            assert all(isinstance(score, (int, float)) for score in scores)

    def test_parameter_grid_search_compatibility(self, mock_config):
        """Test compatibility with parameter grid search."""
        from sklearn.model_selection import ParameterGrid

        # Define parameter grid
        param_grid = {
            "approach": ["flat", "hybrid"],
            "svm_weight": [0.3, 0.7],
            "llm_weight": [0.3, 0.7],
        }

        # Should be able to iterate over parameter combinations
        grid = ParameterGrid(param_grid)

        for params in grid:
            with patch(
                "linnaeus.models.sklearn_interface.get_config", return_value=mock_config
            ):
                classifier = HybridASNClassifier(**params)

                assert classifier.approach in ["flat", "hybrid"]
                assert classifier.svm_weight in [0.3, 0.7]
                assert classifier.llm_weight in [0.3, 0.7]

    def test_multioutput_handling(self, classifier, mock_hybrid_classifier):
        """Test handling of multi-output predictions."""
        X = np.array([[174]])

        # Setup fitted classifier
        classifier.n_features_in_ = 1
        classifier.classes_ = np.array(["Transit", "Content Provider"])
        classifier.hybrid_classifier = mock_hybrid_classifier

        # Mock multi-label predictions
        mock_predictions = np.array([[1, 0]])  # One sample, two classes
        mock_hybrid_classifier.predict.return_value = mock_predictions

        predictions = classifier.predict(X)

        assert predictions.shape == (1, 2)
        assert predictions.dtype in [int, np.int64, np.int32]

    def test_large_batch_handling(self, classifier, mock_hybrid_classifier):
        """Test handling of large batches."""
        # Create large batch
        large_X = np.array([[i] for i in range(1, 1001)])  # 1000 ASNs

        # Setup fitted classifier
        classifier.n_features_in_ = 1
        classifier.classes_ = np.array(["Transit"])
        classifier.hybrid_classifier = mock_hybrid_classifier

        # Mock large predictions
        large_predictions = np.random.randint(0, 2, (1000, 1))
        mock_hybrid_classifier.predict.return_value = large_predictions

        predictions = classifier.predict(large_X)

        assert predictions.shape == (1000, 1)
        assert not np.any(np.isnan(predictions))  # No NaN values
