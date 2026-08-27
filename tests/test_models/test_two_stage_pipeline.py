"""
Tests for the two-stage hierarchical classification pipeline.
"""

from unittest.mock import Mock, patch

import pytest

from linneaus.models.two_stage.pipeline import TwoStageHierarchicalPipeline
from linneaus.models.two_stage_schemas import (
    BatchTwoStageClassificationResponse,
    ConfidenceScore,
    ModelType,
    PipelineConfig,
    SublevelPrediction,
    TopLevelPrediction,
    TwoStageASClassification,
)


class TestTwoStageHierarchicalPipeline:
    """Test the main two-stage pipeline."""

    @pytest.fixture
    def mock_stage1_classifier(self):
        """Mock Stage 1 classifier."""
        mock_classifier = Mock()
        # Mock predict method to return TopLevelPrediction objects
        mock_prediction = TopLevelPrediction(
            category="Transit",
            confidence=ConfidenceScore(value=0.85, model_type=ModelType.ASSEMBLED),
        )
        mock_classifier.predict.return_value = [mock_prediction]
        return mock_classifier

    @pytest.fixture
    def mock_stage2_classifier(self):
        """Mock Stage 2 classifier."""
        mock_classifier = Mock()
        # Mock predict method to return SublevelPrediction objects
        mock_prediction = SublevelPrediction(
            parent_category="Transit",
            subcategory="Global",
            confidence=ConfidenceScore(value=0.78, model_type=ModelType.LLM),
        )
        mock_classifier.predict.return_value = [mock_prediction]
        mock_classifier.category_models = {"Transit": "test-model"}
        return mock_classifier

    @pytest.fixture
    def mock_config(self):
        """Mock pipeline configuration."""
        config = PipelineConfig(
            stage1={
                "svm_weight": 0.4,
                "llm_weight": 0.6,
                "meta_learner": "LogisticRegression",
                "confidence_weight": 0.6,
            },
            stage2={
                "confidence_weight": 0.4,
                "category_models": {"Transit": "test-model"},
            },
            prompts={
                "stage1_system": "Test stage 1 prompt",
                "stage2_transit": "Test transit prompt",
            },
        )
        return config

    @pytest.fixture
    def pipeline(self, mock_stage1_classifier, mock_stage2_classifier, mock_config):
        """Create test pipeline instance."""
        with patch("linneaus.models.two_stage.pipeline.DataAccessLayer"):
            pipeline = TwoStageHierarchicalPipeline(
                stage1_classifier=mock_stage1_classifier,
                stage2_classifier=mock_stage2_classifier,
                config=mock_config,
                enable_monitoring=False,  # Disable for testing
            )
            return pipeline

    def test_predict_single_success(self, pipeline):
        """Test successful single prediction."""
        # Mock data access
        pipeline.data_access.get_organization_data = Mock(
            return_value=Mock(name="Test Org", description="Test description")
        )

        result = pipeline.predict_single(asn=174, organization_name="Test Organization")

        assert isinstance(result, TwoStageASClassification)
        assert result.asn == 174
        assert result.organization_name == "Test Organization"
        assert len(result.stage1_predictions) == 1
        assert len(result.stage2_predictions) == 1
        assert result.overall_confidence > 0

    def test_predict_single_with_timing(self, pipeline):
        """Test single prediction with timing enabled."""
        pipeline.data_access.get_organization_data = Mock(
            return_value=Mock(name="Test Org", description="Test description")
        )

        result = pipeline.predict_single(
            asn=174, organization_name="Test Organization", include_timing=True
        )

        assert result.processing_time_seconds is not None
        assert result.processing_time_seconds >= 0

    def test_predict_batch_sequential(self, pipeline):
        """Test batch prediction in sequential mode."""
        organizations = [{"asn": 174, "name": "Org 1"}, {"asn": 15169, "name": "Org 2"}]

        # Create valid stage1 predictions (validator requires at least one)
        stage1_pred = TopLevelPrediction(
            category="Transit",
            confidence=ConfidenceScore(value=0.85, model_type=ModelType.ASSEMBLED),
        )

        # Mock individual predictions
        with patch.object(pipeline, "predict_single") as mock_predict:
            mock_predict.side_effect = [
                TwoStageASClassification(
                    asn=174,
                    organization_name="Org 1",
                    stage1_predictions=[stage1_pred],
                    stage2_predictions=[],
                    overall_confidence=0.8,
                ),
                TwoStageASClassification(
                    asn=15169,
                    organization_name="Org 2",
                    stage1_predictions=[stage1_pred],
                    stage2_predictions=[],
                    overall_confidence=0.7,
                ),
            ]

            result = pipeline.predict_batch(organizations=organizations, parallel=False)

            assert isinstance(result, BatchTwoStageClassificationResponse)
            assert result.batch_size == 2
            assert result.success_count == 2
            assert result.error_count == 0
            assert len(result.results) == 2

    def test_predict_batch_parallel(self, pipeline):
        """Test batch prediction in parallel mode."""
        organizations = [{"asn": 174, "name": "Org 1"}, {"asn": 15169, "name": "Org 2"}]

        # Create valid stage1 predictions (validator requires at least one)
        stage1_pred = TopLevelPrediction(
            category="Transit",
            confidence=ConfidenceScore(value=0.85, model_type=ModelType.ASSEMBLED),
        )

        # Mock individual predictions
        with patch.object(pipeline, "predict_single") as mock_predict:
            mock_predict.side_effect = [
                TwoStageASClassification(
                    asn=174,
                    organization_name="Org 1",
                    stage1_predictions=[stage1_pred],
                    stage2_predictions=[],
                    overall_confidence=0.8,
                ),
                TwoStageASClassification(
                    asn=15169,
                    organization_name="Org 2",
                    stage1_predictions=[stage1_pred],
                    stage2_predictions=[],
                    overall_confidence=0.7,
                ),
            ]

            result = pipeline.predict_batch(organizations=organizations, parallel=True)

            assert isinstance(result, BatchTwoStageClassificationResponse)
            assert result.success_count == 2
            assert result.error_count == 0

    def test_error_handling(self, pipeline):
        """Test error handling in predictions."""
        # Mock stage 1 to raise an exception
        pipeline.stage1_classifier.predict.side_effect = Exception("Stage 1 failed")

        with pytest.raises(Exception):
            pipeline.predict_single(asn=174, organization_name="Test Org")

    def test_fallback_prediction(self, pipeline):
        """Test fallback behavior when stage 2 fails."""
        # Mock stage 2 to fail
        pipeline.stage2_classifier.predict.side_effect = Exception("Stage 2 failed")

        result = pipeline.predict_single(asn=174, organization_name="Test Org")

        # Should still return result with stage 1 predictions only
        assert isinstance(result, TwoStageASClassification)
        assert len(result.stage1_predictions) == 1
        assert len(result.stage2_predictions) == 0  # Stage 2 failed

    def test_confidence_calculation(self, pipeline):
        """Test overall confidence calculation."""
        result = pipeline.predict_single(asn=174, organization_name="Test Org")

        # Should combine stage 1 and stage 2 confidences
        assert 0 <= result.overall_confidence <= 1

    def test_performance_stats_tracking(self, pipeline):
        """Test performance statistics tracking."""
        # Enable monitoring so stats are tracked
        pipeline.enable_monitoring = True

        initial_stats = pipeline.get_performance_summary()
        assert initial_stats["total_predictions"] == 0

        pipeline.predict_single(asn=174, organization_name="Test Org")

        updated_stats = pipeline.get_performance_summary()
        assert updated_stats["total_predictions"] == 1
        assert updated_stats["success_rate"] == 1.0

    def test_reset_performance_stats(self, pipeline):
        """Test resetting performance statistics."""
        # Enable monitoring so stats are tracked
        pipeline.enable_monitoring = True

        pipeline.predict_single(asn=174, organization_name="Test Org")

        stats_before = pipeline.get_performance_summary()
        assert stats_before["total_predictions"] > 0

        pipeline.reset_performance_stats()

        stats_after = pipeline.get_performance_summary()
        assert stats_after["total_predictions"] == 0

    def test_organization_data_fallback(self, pipeline):
        """Test fallback when organization data is not available."""
        # Mock data access to return None
        pipeline.data_access.get_organization_data = Mock(return_value=None)

        result = pipeline.predict_single(asn=999999, organization_name=None)

        # Should use fallback organization data
        assert result.organization_name == "AS999999"

    def test_batch_error_handling(self, pipeline):
        """Test error handling in batch processing."""
        organizations = [
            {"asn": 174, "name": "Good Org"},
            {"asn": 999999, "name": "Bad Org"},  # This will cause an error
        ]

        # Create valid stage1 prediction
        stage1_pred = TopLevelPrediction(
            category="Transit",
            confidence=ConfidenceScore(value=0.85, model_type=ModelType.ASSEMBLED),
        )

        # Mock predict_single to fail for second org
        def side_effect(asn, name, timing=False):
            if asn == 999999:
                raise Exception("Failed prediction")
            return TwoStageASClassification(
                asn=asn,
                organization_name=name,
                stage1_predictions=[stage1_pred],
                stage2_predictions=[],
                overall_confidence=0.8,
            )

        with patch.object(pipeline, "predict_single", side_effect=side_effect):
            result = pipeline.predict_batch(organizations=organizations)

            assert result.success_count == 1
            assert result.error_count == 1
            assert len(result.results) == 1  # Only successful predictions

    def test_model_versions_tracking(self, pipeline):
        """Test that model versions are tracked in results."""
        result = pipeline.predict_single(asn=174, organization_name="Test Org")

        assert hasattr(result, "model_versions")
        assert "pipeline_version" in result.model_versions
        assert "stage1_model" in result.model_versions
        assert "stage2_models" in result.model_versions
