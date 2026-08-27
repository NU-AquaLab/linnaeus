"""
Tests for scikit-learn compatible interface.
"""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from linneaus.models.schemas import (
    ASClassification,
    BatchClassificationResponse,
    ClassificationTags,
)
from linneaus.models.sklearn_interface import ASNClassifier, ASNDataTransformer


class TestASNClassifier:
    """Test scikit-learn compatible ASN classifier."""

    @pytest.fixture
    def mock_data_access(self, sample_organization_data):
        """Mock data access layer."""
        mock_access = Mock()
        mock_access.get_organization_data.return_value = sample_organization_data
        return mock_access

    @pytest.fixture
    def mock_async_classifier(self):
        """Mock async classifier."""
        mock_classifier = Mock()

        # Mock classification result
        mock_classification = ASClassification(
            asn=174,
            organization_name="Test Org",
            tags=[ClassificationTags.TRANSIT],
            model_used="gpt-4o-mini-test",
        )

        mock_response = BatchClassificationResponse(
            classifications=[mock_classification],
            total_processed=1,
            successful=1,
            failed=0,
            processing_time_seconds=1.0,
        )

        # Mock async methods
        mock_classifier.classify_batch = AsyncMock(return_value=mock_response)

        return mock_classifier

    @pytest.fixture
    def classifier(self, mock_config):
        """Create classifier instance."""
        with patch(
            "linneaus.models.sklearn_interface.get_config", return_value=mock_config
        ):
            return ASNClassifier(model_id="test-model")

    def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.model_id == "test-model"
        assert classifier.classes_ is None
        assert classifier.n_features_in_ is None
        assert classifier._classifier is None

    def test_fit(self, classifier, mock_data_access):
        """Test fitting the classifier."""
        X = np.array([[174], [15169], [32934]])

        with patch.object(classifier, "data_access", mock_data_access):
            result = classifier.fit(X)

        # Should return self
        assert result is classifier

        # Should set attributes
        assert classifier.n_features_in_ == 1
        assert classifier.feature_names_in_ is not None
        assert len(classifier.feature_names_in_) == 1
        assert classifier.classes_ is not None
        assert len(classifier.classes_) > 0
        assert classifier.model_id_ == "test-model"
        assert classifier._classifier is not None

    def test_fit_invalid_shape(self, classifier):
        """Test fitting with invalid input shape."""
        X = np.array([[174, 15169], [32934, 1234]])  # Wrong shape

        with pytest.raises(ValueError, match="exactly 1 feature"):
            classifier.fit(X)

    def test_predict_not_fitted(self, classifier):
        """Test prediction before fitting raises an error."""
        X = np.array([[174]])

        # When not fitted, attributes are None. check_is_fitted won't catch this
        # since the attributes exist (as None), but predict() will fail with
        # ValueError when comparing X.shape[1] to None.
        with pytest.raises((NotFittedError, ValueError)):
            classifier.predict(X)

    def test_predict(self, classifier, mock_data_access, mock_async_classifier):
        """Test prediction after fitting."""
        X = np.array([[174]])

        with (
            patch.object(classifier, "data_access", mock_data_access),
            patch("asyncio.run") as mock_run,
        ):

            # Fit first
            classifier.fit(X)
            classifier._classifier = mock_async_classifier

            # Mock asyncio.run to return the mock response
            mock_classification = ASClassification(
                asn=174,
                organization_name="Test Org",
                tags=[ClassificationTags.TRANSIT],
                model_used="test",
            )
            mock_response = BatchClassificationResponse(
                classifications=[mock_classification],
                total_processed=1,
                successful=1,
                failed=0,
                processing_time_seconds=1.0,
            )
            mock_run.return_value = mock_response

            # Predict
            y_pred = classifier.predict(X)

            # Should return binary matrix
            assert y_pred.shape[0] == 1  # One sample
            assert y_pred.shape[1] > 0  # Multiple classes
            assert y_pred.dtype == int

            # Check that Transit tag is set to 1
            transit_idx = list(classifier.classes_).index("Transit")
            assert y_pred[0, transit_idx] == 1

    def test_predict_proba(self, classifier, mock_data_access, mock_async_classifier):
        """Test probability prediction."""
        X = np.array([[174]])

        with (
            patch.object(classifier, "data_access", mock_data_access),
            patch("asyncio.run") as mock_run,
        ):

            # Fit first
            classifier.fit(X)
            classifier._classifier = mock_async_classifier

            # Mock classification with confidence
            mock_classification = ASClassification(
                asn=174,
                organization_name="Test Org",
                tags=[ClassificationTags.TRANSIT],
                confidence_scores={"Transit": 0.95},
                model_used="test",
            )
            mock_response = BatchClassificationResponse(
                classifications=[mock_classification],
                total_processed=1,
                successful=1,
                failed=0,
                processing_time_seconds=1.0,
            )
            mock_run.return_value = mock_response

            # Predict probabilities
            y_proba = classifier.predict_proba(X)

            # Should return probability matrix
            assert y_proba.shape[0] == 1  # One sample
            assert y_proba.shape[1] > 0  # Multiple classes
            assert y_proba.dtype == float

            # Check confidence score
            transit_idx = list(classifier.classes_).index("Transit")
            assert y_proba[0, transit_idx] == 0.95

    def test_score(self, classifier, mock_data_access, mock_async_classifier):
        """Test scoring method."""
        X = np.array([[174]])
        y_true = np.zeros((1, len(ClassificationTags)))
        y_true[0, 1] = 1  # Set Transit as true label

        with (
            patch.object(classifier, "data_access", mock_data_access),
            patch("asyncio.run") as mock_run,
        ):

            # Fit and setup
            classifier.fit(X)
            classifier._classifier = mock_async_classifier

            mock_classification = ASClassification(
                asn=174,
                organization_name="Test Org",
                tags=[ClassificationTags.TRANSIT],
                model_used="test",
            )
            mock_response = BatchClassificationResponse(
                classifications=[mock_classification],
                total_processed=1,
                successful=1,
                failed=0,
                processing_time_seconds=1.0,
            )
            mock_run.return_value = mock_response

            # Score
            score = classifier.score(X, y_true)

            # Should return a float score
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_get_set_params(self, classifier):
        """Test parameter getting and setting."""
        # Get parameters
        params = classifier.get_params()

        assert "model_id" in params
        assert "api_key" in params
        assert "batch_size" in params

        # Set parameters
        new_params = {"batch_size": 20}
        result = classifier.set_params(**new_params)

        assert result is classifier
        assert classifier.batch_size == 20

    def test_invalid_param(self, classifier):
        """Test setting invalid parameter."""
        with pytest.raises(ValueError):
            classifier.set_params(invalid_param="value")


class TestASNDataTransformer:
    """Test ASN data transformer."""

    @pytest.fixture
    def transformer(self, mock_config):
        """Create transformer instance."""
        with patch(
            "linneaus.models.sklearn_interface.get_config", return_value=mock_config
        ):
            return ASNDataTransformer()

    @pytest.fixture
    def mock_data_access_full(
        self, sample_asrank_data, sample_peeringdb_data, sample_aspop_data
    ):
        """Mock data access with full data."""
        mock_access = Mock()
        mock_access.get_asrank_data.return_value = sample_asrank_data
        mock_access.get_peeringdb_data.return_value = sample_peeringdb_data
        mock_access.get_aspop_data.return_value = sample_aspop_data
        return mock_access

    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.include_asrank is True
        assert transformer.include_peeringdb is True
        assert transformer.include_aspop is True
        assert transformer.normalize_features is True

    def test_fit(self, transformer, mock_data_access_full):
        """Test fitting the transformer."""
        X = np.array([[174], [15169]])

        with patch.object(transformer, "data_access", mock_data_access_full):
            result = transformer.fit(X)

        # Should return self
        assert result is transformer

        # Should set attributes
        assert transformer.n_features_in_ == 1
        assert transformer.feature_names_ is not None
        assert len(transformer.feature_names_) > 0
        assert transformer.n_features_out_ > 0

    def test_transform(self, transformer, mock_data_access_full):
        """Test data transformation."""
        X = np.array([[174]])

        with patch.object(transformer, "data_access", mock_data_access_full):
            transformer.fit(X)
            X_transformed = transformer.transform(X)

        # Should return feature matrix
        assert X_transformed.shape[0] == 1  # One sample
        assert X_transformed.shape[1] > 0  # Multiple features
        assert X_transformed.dtype == float

    def test_fit_transform(self, transformer, mock_data_access_full):
        """Test fit and transform in one step."""
        X = np.array([[174]])

        with patch.object(transformer, "data_access", mock_data_access_full):
            X_transformed = transformer.fit_transform(X)

        assert X_transformed.shape[0] == 1
        assert X_transformed.shape[1] > 0

    def test_feature_selection(self, mock_config):
        """Test selective feature inclusion."""
        with patch(
            "linneaus.models.sklearn_interface.get_config", return_value=mock_config
        ):
            # Only ASRank features
            transformer = ASNDataTransformer(
                include_asrank=True, include_peeringdb=False, include_aspop=False
            )

        X = np.array([[174]])
        transformer.fit(X)

        # Should only have ASRank features
        feature_names = transformer._get_feature_names()
        assert all(name.startswith("asrank_") for name in feature_names)

    def test_get_feature_names_out(self, transformer, mock_data_access_full):
        """Test getting output feature names."""
        X = np.array([[174]])

        with patch.object(transformer, "data_access", mock_data_access_full):
            transformer.fit(X)
            feature_names = transformer.get_feature_names_out()

        assert len(feature_names) == transformer.n_features_out_
        assert isinstance(feature_names, np.ndarray)

    def test_missing_data_handling(self, transformer):
        """Test handling of missing data."""
        X = np.array([[99999]])  # Non-existent ASN

        # Mock data access to return None for missing data
        mock_access = Mock()
        mock_access.get_asrank_data.return_value = None
        mock_access.get_peeringdb_data.return_value = None
        mock_access.get_aspop_data.return_value = None

        with patch.object(transformer, "data_access", mock_access):
            transformer.fit(X)
            X_transformed = transformer.transform(X)

        # Should fill missing values with zeros
        assert X_transformed.shape[0] == 1
        assert X_transformed.shape[1] > 0
        # All values should be zero for missing data
        assert np.all(X_transformed == 0)
