"""
Tests for the hierarchical sublevel classifier (Stage 2).
"""

from unittest.mock import Mock, patch

import pytest
from openai import OpenAI

from linnaeus.models.two_stage.sublevel_classifier import HierarchicalSublevelClassifier
from linnaeus.models.two_stage_schemas import (
    ConfidenceScore,
    ModelType,
    SublevelPrediction,
    TopLevelPrediction,
)


class TestHierarchicalSublevelClassifier:
    """Test the sublevel classifier for Stage 2."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        mock_client = Mock(spec=OpenAI)
        return mock_client

    @pytest.fixture
    def category_models(self):
        """Sample category models mapping."""
        return {
            "Transit": "ft:gpt-4o-mini:transit-model",
            "ContentProvider": "ft:gpt-4o-mini:content-model",
            "Government": "ft:gpt-4o-mini:gov-model",
        }

    @pytest.fixture
    def category_prompts(self):
        """Sample category prompts."""
        return {
            "Transit": "Classify transit networks into subcategories: Global, Regional, Domestic",
            "ContentProvider": "Classify content providers: CDN, Cloud, Hosting",
            "Government": "Classify government networks: Federal, State, Local",
        }

    @pytest.fixture
    def classifier(self, mock_openai_client, category_models, category_prompts):
        """Create test classifier instance with mocked processor initialization."""
        with patch(
            "linnaeus.models.two_stage.sublevel_classifier.HierarchicalBatchInferenceProcessor"
        ) as mock_processor_class:
            # Each category gets its own mock processor
            pass

            def create_processor(*args, **kwargs):
                proc = Mock()
                proc.process_batch.return_value = []
                return proc

            mock_processor_class.side_effect = create_processor

            classifier = HierarchicalSublevelClassifier(
                openai_client=mock_openai_client,
                category_models=category_models,
                category_prompts=category_prompts,
                temperature=0.1,
            )
            return classifier

    @pytest.fixture
    def sample_organizations(self):
        """Sample organization data."""
        return [
            {
                "asn": 174,
                "name": "Cogent Communications",
                "description": "Global IP transit provider",
            },
            {
                "asn": 15169,
                "name": "Google LLC",
                "description": "Technology company providing cloud services",
            },
        ]

    @pytest.fixture
    def sample_stage1_predictions(self):
        """Sample Stage 1 predictions (flat list, one per org)."""
        return [
            TopLevelPrediction(
                category="Transit",
                confidence=ConfidenceScore(value=0.9, model_type=ModelType.ASSEMBLED),
            ),
            TopLevelPrediction(
                category="ContentProvider",
                confidence=ConfidenceScore(value=0.85, model_type=ModelType.ASSEMBLED),
            ),
        ]

    def test_initialization(self, classifier, category_models):
        """Test classifier initialization."""
        assert classifier.category_models == category_models
        assert classifier.temperature == 0.1
        assert "Transit" in classifier.category_models
        assert len(classifier.processors) > 0

    def test_predict_with_valid_categories(
        self, classifier, sample_organizations, sample_stage1_predictions
    ):
        """Test prediction with valid categories."""
        # Mock the processor for Transit to return results with valid subcategory
        transit_processor = classifier.processors.get("Transit")
        if transit_processor:
            transit_processor.process_batch.return_value = [
                {
                    "responses": [
                        {
                            "classifications": [
                                {"subcategory": "Global", "confidence": 0.85}
                            ]
                        }
                    ]
                }
            ]

        content_processor = classifier.processors.get("ContentProvider")
        if content_processor:
            content_processor.process_batch.return_value = [
                {
                    "responses": [
                        {
                            "classifications": [
                                {"subcategory": "CDN", "confidence": 0.78}
                            ]
                        }
                    ]
                }
            ]

        predictions = classifier.predict(
            sample_organizations, sample_stage1_predictions
        )

        # Should return a list of SublevelPrediction objects
        assert isinstance(predictions, list)
        # Valid predictions depend on subcategory validation
        for pred in predictions:
            assert isinstance(pred, SublevelPrediction)

    def test_predict_with_empty_inputs(self, classifier):
        """Test prediction with empty inputs."""
        predictions = classifier.predict([], [])
        assert predictions == []

    def test_group_by_category(
        self, classifier, sample_organizations, sample_stage1_predictions
    ):
        """Test grouping organizations by their predicted category."""
        grouped = classifier._group_by_category(
            sample_organizations, sample_stage1_predictions
        )

        assert isinstance(grouped, dict)
        # Should have Transit and ContentProvider groups
        assert "Transit" in grouped
        assert "ContentProvider" in grouped

    def test_predict_category_calls_processor(self, classifier):
        """Test that _predict_category calls the correct processor."""
        org_data = [
            {
                "asn": 174,
                "name": "Cogent Communications",
                "description": "Global IP transit provider",
                "stage1_confidence": 0.9,
                "predicted_category": "Transit",
            }
        ]

        # Mock processor to return results with valid subcategory
        transit_processor = classifier.processors.get("Transit")
        if transit_processor:
            transit_processor.process_batch.return_value = [
                {
                    "responses": [
                        {
                            "classifications": [
                                {"subcategory": "Global", "confidence": 0.85}
                            ]
                        }
                    ]
                }
            ]

        predictions = classifier._predict_category("Transit", org_data)

        assert isinstance(predictions, list)
        if transit_processor:
            transit_processor.process_batch.assert_called()

    def test_predict_category_api_error(self, classifier):
        """Test category processing with API error."""
        # Mock processor to raise exception
        transit_processor = classifier.processors.get("Transit")
        if transit_processor:
            transit_processor.process_batch.side_effect = Exception("API Error")

        org_data = [
            {
                "asn": 174,
                "name": "Test Org",
                "description": "Test",
                "stage1_confidence": 0.9,
                "predicted_category": "Transit",
            }
        ]

        # Should raise the exception (error handling is in predict(), not _predict_category())
        with pytest.raises(Exception, match="API Error"):
            classifier._predict_category("Transit", org_data)

    def test_is_valid_subcategory(self, classifier):
        """Test subcategory validation."""
        # Transit has subcategories: Global, Regional, Domestic
        assert classifier._is_valid_subcategory("Transit", "Global") is True
        assert classifier._is_valid_subcategory("Transit", "Regional") is True
        assert classifier._is_valid_subcategory("Transit", "Invalid") is False

    def test_create_fallback_predictions(self, classifier):
        """Test fallback prediction creation when LLM fails."""
        org_data = [
            {"asn": 174, "name": "Test Org", "description": "Test"},
            {"asn": 15169, "name": "Test Org 2", "description": "Test 2"},
        ]

        fallbacks = classifier._create_fallback_predictions("Transit", org_data)

        assert len(fallbacks) == 2
        for fb in fallbacks:
            assert isinstance(fb, SublevelPrediction)
            assert fb.confidence.value == 0.1  # Low confidence for fallback
            assert fb.confidence.model_type == ModelType.LLM
            assert fb.parent_category.value == "Transit"

    def test_get_available_categories(self, classifier):
        """Test getting available categories with processors."""
        available = classifier.get_available_categories()

        assert isinstance(available, set)
        # Should have processors for categories from category_models
        assert "Transit" in available

    def test_default_category_prompt(self, classifier):
        """Test default prompt generation for categories."""
        prompt = classifier._get_default_category_prompt("Transit")

        assert "Transit" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_default_category_prompt_unknown(self, classifier):
        """Test default prompt for unknown category."""
        prompt = classifier._get_default_category_prompt("UnknownCategory")

        # Should still return a string prompt, even if category is unknown
        assert isinstance(prompt, str)
        assert "UnknownCategory" in prompt

    def test_add_category_processor(self, classifier):
        """Test adding a new category processor."""
        with patch(
            "linnaeus.models.two_stage.sublevel_classifier.HierarchicalBatchInferenceProcessor"
        ) as mock_processor_class:
            mock_proc = Mock()
            mock_processor_class.return_value = mock_proc

            classifier.add_category_processor(
                "NewCategory", "ft:gpt-4o-mini:new-model", system_prompt="Test prompt"
            )

            assert "NewCategory" in classifier.processors
            assert "NewCategory" in classifier.category_models
            assert (
                classifier.category_models["NewCategory"] == "ft:gpt-4o-mini:new-model"
            )

    def test_remove_category_processor(self, classifier):
        """Test removing a category processor."""
        len(classifier.processors)

        classifier.remove_category_processor("Transit")

        assert "Transit" not in classifier.processors
        assert "Transit" not in classifier.category_models

    def test_predict_skips_unknown_categories(self, classifier, sample_organizations):
        """Test that predict skips categories without processors."""
        # Create a prediction for a category that has no processor
        unknown_predictions = [
            TopLevelPrediction(
                category="Access",  # Valid enum value but may not have processor
                confidence=ConfidenceScore(value=0.8, model_type=ModelType.ASSEMBLED),
            )
        ]

        # Remove the Access processor if it exists
        classifier.processors.pop("Access", None)

        # Should still work, just skip the unknown category
        predictions = classifier.predict([sample_organizations[0]], unknown_predictions)
        assert isinstance(predictions, list)

    def test_batch_processing(
        self, classifier, sample_organizations, sample_stage1_predictions
    ):
        """Test batch processing of multiple organizations."""
        # Set up processors to return results
        for category_name, processor in classifier.processors.items():
            processor.process_batch.return_value = [
                {
                    "responses": [
                        {
                            "classifications": [
                                {"subcategory": "Global", "confidence": 0.8}
                            ]
                        }
                    ]
                }
            ]

        predictions = classifier.predict(
            sample_organizations, sample_stage1_predictions
        )

        assert isinstance(predictions, list)

    def test_category_mappings_loaded(self, classifier):
        """Test that category mappings are loaded from hierarchical schemas."""
        assert hasattr(classifier, "category_mappings")
        assert isinstance(classifier.category_mappings, dict)
        # Should have Transit mapping with subcategories
        if "Transit" in classifier.category_mappings:
            transit_mapping = classifier.category_mappings["Transit"]
            assert hasattr(transit_mapping, "subcategories")
            assert "Global" in transit_mapping.subcategories

    def test_client_attribute(self, classifier, mock_openai_client):
        """Test that the OpenAI client is stored as 'client' attribute."""
        assert classifier.client is mock_openai_client

    def test_temperature_setting(self, mock_openai_client, category_models):
        """Test custom temperature setting."""
        with patch(
            "linnaeus.models.two_stage.sublevel_classifier.HierarchicalBatchInferenceProcessor"
        ):
            classifier = HierarchicalSublevelClassifier(
                openai_client=mock_openai_client,
                category_models=category_models,
                temperature=0.5,
            )
            assert classifier.temperature == 0.5

    def test_batch_size_setting(self, mock_openai_client, category_models):
        """Test custom batch size setting."""
        with patch(
            "linnaeus.models.two_stage.sublevel_classifier.HierarchicalBatchInferenceProcessor"
        ):
            classifier = HierarchicalSublevelClassifier(
                openai_client=mock_openai_client,
                category_models=category_models,
                batch_size=20,
            )
            assert classifier.batch_size == 20
