"""
Tests for unified schema system and tag hierarchy.
"""

from datetime import datetime
from typing import List

import pytest

from linnaeus.models.unified_schemas import (
    BatchUnifiedClassificationResponse,
    ClassificationRequest,
    HierarchicalTags,
    TagHierarchy,
    TopLevelTags,
    UnifiedASClassification,
)


class TestTagEnums:
    """Test tag enumeration classes."""

    def test_top_level_tags(self):
        """Test TopLevelTags enum."""
        # Test that all expected categories exist
        expected_tags = {
            "Access",
            "Transit",
            "Mobile",
            "Satellite",
            "Content Provider",
            "Educational Research",
            "Government",
            "Internet Exchange Point",
            "DNS",
            "Energy & Utility",
            "Enterprise",
            "Financial",
            "Law Enforcement",
            "Health",
            "Cooperatives",
            "TV/Radio and Cultural Amenities",
            "Transportation",
            "Virtual Private Networks",
            "Personal",
            "Community",
        }

        actual_tags = {tag.value for tag in TopLevelTags}
        assert actual_tags == expected_tags

    def test_hierarchical_tags(self):
        """Test HierarchicalTags enum."""
        hierarchical_values = {tag.value for tag in HierarchicalTags}

        # Should contain specific subcategories (not top-level values like "Access")
        assert "Access Large ISP" in hierarchical_values
        assert "Access Small ISP" in hierarchical_values
        assert "Transit Global" in hierarchical_values
        assert "Transit Regional" in hierarchical_values
        assert "Transit Domestic" in hierarchical_values
        assert "ContentProvider Cloud" in hierarchical_values
        assert "ContentProvider Hosting" in hierarchical_values
        assert "ContentProvider CDN" in hierarchical_values

        # Single-level tags that map directly should also exist
        assert "Mobile" in hierarchical_values
        assert "Satellite" in hierarchical_values
        assert "Internet Exchange Point" in hierarchical_values

    def test_tag_hierarchy_mapping(self):
        """Test TagHierarchy mapping between systems."""
        # Test specific mappings
        isp_mapping = TagHierarchy.get_top_level_for_hierarchical(
            HierarchicalTags.ACCESS_LARGE_ISP
        )
        assert isp_mapping == TopLevelTags.ACCESS

        cloud_mapping = TagHierarchy.get_top_level_for_hierarchical(
            HierarchicalTags.CONTENT_PROVIDER_CLOUD
        )
        assert cloud_mapping == TopLevelTags.CONTENT_PROVIDER

        transit_mapping = TagHierarchy.get_top_level_for_hierarchical(
            HierarchicalTags.TRANSIT_GLOBAL
        )
        assert transit_mapping == TopLevelTags.TRANSIT

        # Test reverse mapping
        access_hierarchicals = TagHierarchy.get_hierarchical_for_top_level(
            TopLevelTags.ACCESS
        )
        assert HierarchicalTags.ACCESS_LARGE_ISP in access_hierarchicals
        assert HierarchicalTags.ACCESS_SMALL_ISP in access_hierarchicals

    def test_tag_hierarchy_coverage(self):
        """Test that all hierarchical tags have top-level mappings."""
        for hierarchical_tag in HierarchicalTags:
            top_level = TagHierarchy.get_top_level_for_hierarchical(hierarchical_tag)
            assert (
                top_level is not None
            ), f"No top-level mapping for {hierarchical_tag.value}"


class TestUnifiedASClassification:
    """Test unified classification result schema."""

    def test_basic_creation(self):
        """Test basic classification result creation."""
        result = UnifiedASClassification(
            asn=174,
            organization_name="Cogent Communications",
            top_level_tags=[TopLevelTags.TRANSIT],
            hierarchical_tags=[HierarchicalTags.TRANSIT_GLOBAL],
            model_used="test-model",
            classification_approach="hybrid",
        )

        assert result.asn == 174
        assert result.organization_name == "Cogent Communications"
        assert result.top_level_tags == [TopLevelTags.TRANSIT]
        assert result.hierarchical_tags == [HierarchicalTags.TRANSIT_GLOBAL]
        assert result.classification_approach == "hybrid"
        assert isinstance(result.timestamp, datetime)

    def test_multi_tag_classification(self):
        """Test classification with multiple tags."""
        result = UnifiedASClassification(
            asn=15169,
            organization_name="Google LLC",
            top_level_tags=[TopLevelTags.CONTENT_PROVIDER, TopLevelTags.ENTERPRISE],
            hierarchical_tags=[
                HierarchicalTags.CONTENT_PROVIDER_CLOUD,
                HierarchicalTags.ENTERPRISE_TECHNOLOGY,
                HierarchicalTags.ENTERPRISE_ECOMMERCE,
            ],
            model_used="test-model",
            classification_approach="hierarchical",
            top_level_confidence={"Content Provider": 0.95, "Enterprise": 0.80},
        )

        assert len(result.top_level_tags) == 2
        assert len(result.hierarchical_tags) == 3
        assert result.top_level_confidence["Content Provider"] == 0.95

    def test_empty_tags(self):
        """Test classification with no predicted tags."""
        result = UnifiedASClassification(
            asn=99999,
            organization_name="Unknown Organization",
            top_level_tags=[],
            hierarchical_tags=[],
            model_used="test-model",
            classification_approach="flat",
        )

        assert len(result.top_level_tags) == 0
        assert len(result.hierarchical_tags) == 0

    def test_confidence_scores(self):
        """Test confidence score handling."""
        confidence_scores = {
            "Content Provider": 0.92,
            "Enterprise": 0.78,
            "Educational Research": 0.45,
        }

        result = UnifiedASClassification(
            asn=32934,
            organization_name="Facebook",
            top_level_tags=[TopLevelTags.CONTENT_PROVIDER],
            hierarchical_tags=[HierarchicalTags.ENTERPRISE_ENTERTAINMENT],
            model_used="test-model",
            classification_approach="hybrid",
            top_level_confidence=confidence_scores,
        )

        assert result.top_level_confidence == confidence_scores
        assert result.top_level_confidence["Content Provider"] == 0.92

    def test_serialization(self):
        """Test JSON serialization compatibility."""
        result = UnifiedASClassification(
            asn=174,
            organization_name="Cogent Communications",
            top_level_tags=[TopLevelTags.TRANSIT],
            hierarchical_tags=[HierarchicalTags.TRANSIT_GLOBAL],
            model_used="test-model",
            classification_approach="hybrid",
        )

        # Test that it can be converted to dict
        result_dict = result.model_dump()
        assert result_dict["asn"] == 174
        assert result_dict["organization_name"] == "Cogent Communications"
        assert "top_level_tags" in result_dict
        assert "hierarchical_tags" in result_dict

    def test_model_metadata(self):
        """Test model metadata fields."""
        result = UnifiedASClassification(
            asn=15169,
            organization_name="Google",
            top_level_tags=[TopLevelTags.CONTENT_PROVIDER],
            hierarchical_tags=[HierarchicalTags.CONTENT_PROVIDER_CLOUD],
            model_used="HybridClassifier-v1.0",
            classification_approach="hybrid",
        )

        assert result.model_used == "HybridClassifier-v1.0"
        assert result.classification_approach == "hybrid"
        assert isinstance(result.timestamp, datetime)

    def test_asn_validation(self):
        """Test that ASN must be a positive integer."""
        with pytest.raises(ValueError, match="ASN must be a positive integer"):
            UnifiedASClassification(
                asn=0,
                organization_name="Invalid",
                top_level_tags=[],
                hierarchical_tags=[],
                model_used="test-model",
                classification_approach="flat",
            )

        with pytest.raises(ValueError, match="ASN must be a positive integer"):
            UnifiedASClassification(
                asn=-1,
                organization_name="Invalid",
                top_level_tags=[],
                hierarchical_tags=[],
                model_used="test-model",
                classification_approach="flat",
            )

    def test_sync_tag_systems_derives_top_level(self):
        """Test that the model validator derives top-level tags from hierarchical tags."""
        result = UnifiedASClassification(
            asn=174,
            organization_name="Cogent Communications",
            hierarchical_tags=[
                HierarchicalTags.TRANSIT_GLOBAL,
                HierarchicalTags.TRANSIT_REGIONAL,
            ],
            model_used="test-model",
            classification_approach="hybrid",
        )

        # top_level_tags should be auto-derived from hierarchical_tags
        assert TopLevelTags.TRANSIT in result.top_level_tags

    def test_sync_tag_systems_requires_hierarchical_tags(self):
        """Test that hierarchical approach requires hierarchical_tags when only top-level provided."""
        with pytest.raises(
            ValueError,
            match="Hierarchical classification approach requires hierarchical_tags",
        ):
            UnifiedASClassification(
                asn=174,
                organization_name="Cogent Communications",
                top_level_tags=[TopLevelTags.TRANSIT],
                hierarchical_tags=[],
                model_used="test-model",
                classification_approach="hierarchical",
            )

    def test_get_all_tags(self):
        """Test get_all_tags returns all tag string values without duplicates."""
        result = UnifiedASClassification(
            asn=174,
            organization_name="Cogent Communications",
            top_level_tags=[TopLevelTags.TRANSIT],
            hierarchical_tags=[HierarchicalTags.TRANSIT_GLOBAL],
            model_used="test-model",
            classification_approach="hybrid",
        )

        all_tags = result.get_all_tags()
        assert "Transit" in all_tags
        assert "Transit Global" in all_tags


class TestBatchUnifiedClassificationResponse:
    """Test batch response schema."""

    def create_sample_classifications(self) -> List[UnifiedASClassification]:
        """Create sample classification results."""
        return [
            UnifiedASClassification(
                asn=174,
                organization_name="Cogent Communications",
                top_level_tags=[TopLevelTags.TRANSIT],
                hierarchical_tags=[HierarchicalTags.TRANSIT_GLOBAL],
                model_used="test-model",
                classification_approach="hybrid",
            ),
            UnifiedASClassification(
                asn=15169,
                organization_name="Google LLC",
                top_level_tags=[TopLevelTags.CONTENT_PROVIDER],
                hierarchical_tags=[HierarchicalTags.CONTENT_PROVIDER_CLOUD],
                model_used="test-model",
                classification_approach="hybrid",
            ),
        ]

    def test_batch_response_creation(self):
        """Test batch response creation."""
        classifications = self.create_sample_classifications()

        response = BatchUnifiedClassificationResponse(
            classifications=classifications,
            total_processed=2,
            successful=2,
            failed=0,
            processing_time_seconds=2.5,
            approach_used="hybrid",
        )

        assert len(response.classifications) == 2
        assert response.total_processed == 2
        assert response.successful == 2
        assert response.failed == 0
        assert response.processing_time_seconds == 2.5
        assert response.approach_used == "hybrid"

    def test_batch_metrics_calculation(self):
        """Test batch response metrics."""
        classifications = self.create_sample_classifications()

        response = BatchUnifiedClassificationResponse(
            classifications=classifications,
            total_processed=3,  # One failed
            successful=2,
            failed=1,
            processing_time_seconds=1.8,
            approach_used="hybrid",
        )

        # Calculate success rate
        success_rate = response.successful / response.total_processed
        assert success_rate == pytest.approx(0.667, abs=0.01)

        # Calculate processing rate
        processing_rate = response.total_processed / response.processing_time_seconds
        assert processing_rate == pytest.approx(1.67, abs=0.01)

    def test_empty_batch_response(self):
        """Test handling of empty batch."""
        response = BatchUnifiedClassificationResponse(
            classifications=[],
            total_processed=0,
            successful=0,
            failed=0,
            processing_time_seconds=0.0,
            approach_used="hybrid",
        )

        assert len(response.classifications) == 0
        assert response.total_processed == 0

    def test_batch_id_optional(self):
        """Test that batch_id is optional."""
        classifications = self.create_sample_classifications()

        # Without batch_id
        response = BatchUnifiedClassificationResponse(
            classifications=classifications,
            total_processed=2,
            successful=2,
            failed=0,
            processing_time_seconds=1.0,
            approach_used="hybrid",
        )
        assert response.batch_id is None

        # With batch_id
        response_with_id = BatchUnifiedClassificationResponse(
            classifications=classifications,
            batch_id="batch-001",
            total_processed=2,
            successful=2,
            failed=0,
            processing_time_seconds=1.0,
            approach_used="hybrid",
        )
        assert response_with_id.batch_id == "batch-001"


class TestClassificationRequest:
    """Test classification request schema."""

    def test_basic_request(self):
        """Test basic classification request."""
        request = ClassificationRequest(
            organizations=[174, 15169, 32934],
            approach="hybrid",
            include_confidence=True,
        )

        assert request.organizations == [174, 15169, 32934]
        assert request.approach == "hybrid"
        assert request.include_confidence is True

    def test_request_with_options(self):
        """Test request with additional options."""
        request = ClassificationRequest(
            organizations=[174],
            approach="hierarchical",
            include_confidence=False,
            batch_size=5,
            model="gpt-4o-mini-finetuned",
        )

        assert request.approach == "hierarchical"
        assert request.batch_size == 5
        assert request.model == "gpt-4o-mini-finetuned"
        assert request.include_confidence is False

    def test_approach_validation(self):
        """Test approach parameter validation."""
        # Valid approaches should work
        valid_approaches = ["flat", "hierarchical", "hybrid"]

        for approach in valid_approaches:
            request = ClassificationRequest(organizations=[174], approach=approach)
            assert request.approach == approach

    def test_invalid_approach_rejected(self):
        """Test that invalid approaches are rejected."""
        invalid_approaches = ["svm-only", "llm-only", "random", ""]

        for approach in invalid_approaches:
            with pytest.raises(ValueError):
                ClassificationRequest(organizations=[174], approach=approach)

    def test_default_approach(self):
        """Test that the default approach is hybrid."""
        request = ClassificationRequest(organizations=[174])
        assert request.approach == "hybrid"

    def test_organizations_list_validation(self):
        """Test organizations list validation."""
        # Single organization should work
        request = ClassificationRequest(organizations=[174], approach="flat")
        assert request.organizations == [174]

        # Multiple organizations should work
        request = ClassificationRequest(
            organizations=[174, 15169, 32934, 20940], approach="hierarchical"
        )
        assert len(request.organizations) == 4

    def test_empty_organizations_rejected(self):
        """Test that empty organizations list is rejected."""
        with pytest.raises(
            ValueError, match="At least one organization must be provided"
        ):
            ClassificationRequest(organizations=[], approach="hybrid")

    def test_optional_fields_default_to_none(self):
        """Test that optional fields default to None or their defaults."""
        request = ClassificationRequest(organizations=[174])

        assert request.model is None
        assert request.batch_size is None
        assert request.include_confidence is False


class TestTagHierarchyIntegration:
    """Test integration between different tag systems."""

    def test_hierarchical_to_flat_conversion(self):
        """Test conversion from hierarchical to flat tags."""
        hierarchical_result = UnifiedASClassification(
            asn=15169,
            organization_name="Google LLC",
            top_level_tags=[],
            hierarchical_tags=[
                HierarchicalTags.CONTENT_PROVIDER_CLOUD,
                HierarchicalTags.ENTERPRISE_TECHNOLOGY,
                HierarchicalTags.ENTERPRISE_ECOMMERCE,
            ],
            model_used="test-model",
            classification_approach="hybrid",
        )

        # The model validator should have auto-derived top_level_tags
        assert TopLevelTags.CONTENT_PROVIDER in hierarchical_result.top_level_tags
        assert TopLevelTags.ENTERPRISE in hierarchical_result.top_level_tags

        # Also test manual conversion via TagHierarchy
        top_level_tags = set()
        for h_tag in hierarchical_result.hierarchical_tags:
            top_level = TagHierarchy.get_top_level_for_hierarchical(h_tag)
            if top_level:
                top_level_tags.add(top_level)

        assert TopLevelTags.CONTENT_PROVIDER in top_level_tags
        assert TopLevelTags.ENTERPRISE in top_level_tags

    def test_flat_to_hierarchical_expansion(self):
        """Test expansion from flat to hierarchical tags."""
        flat_result = UnifiedASClassification(
            asn=174,
            organization_name="Cogent Communications",
            top_level_tags=[TopLevelTags.TRANSIT],
            hierarchical_tags=[],
            model_used="test-model",
            classification_approach="flat",
        )

        # Expand top-level to hierarchical
        hierarchical_tags = []
        for tl_tag in flat_result.top_level_tags:
            h_tags = TagHierarchy.get_hierarchical_for_top_level(tl_tag)
            hierarchical_tags.extend(h_tags)

        # Should include transit-related hierarchical tags
        assert HierarchicalTags.TRANSIT_GLOBAL in hierarchical_tags
        assert HierarchicalTags.TRANSIT_REGIONAL in hierarchical_tags
        assert HierarchicalTags.TRANSIT_DOMESTIC in hierarchical_tags

    def test_consistency_across_approaches(self):
        """Test that tag mappings are consistent."""
        # For each top-level tag, check that its hierarchical mappings
        # correctly map back to the original top-level tag
        for top_level_tag in TopLevelTags:
            hierarchical_tags = TagHierarchy.get_hierarchical_for_top_level(
                top_level_tag
            )

            for hierarchical_tag in hierarchical_tags:
                mapped_back = TagHierarchy.get_top_level_for_hierarchical(
                    hierarchical_tag
                )
                assert mapped_back == top_level_tag, (
                    f"Inconsistent mapping: {hierarchical_tag.value} maps to "
                    f"{mapped_back.value if mapped_back else None} instead of {top_level_tag.value}"
                )

    def test_hierarchical_to_flat_batch_conversion(self):
        """Test the batch conversion utility on TagHierarchy."""
        hierarchical_tags = [
            HierarchicalTags.ACCESS_LARGE_ISP,
            HierarchicalTags.ACCESS_SMALL_ISP,
            HierarchicalTags.CONTENT_PROVIDER_CDN,
        ]

        flat_tags = TagHierarchy.hierarchical_to_flat(hierarchical_tags)

        assert TopLevelTags.ACCESS in flat_tags
        assert TopLevelTags.CONTENT_PROVIDER in flat_tags
        # Two access tags should collapse to one top-level Access tag
        assert len(flat_tags) == 2

    def test_flat_to_hierarchical_batch_expansion(self):
        """Test the batch expansion utility on TagHierarchy."""
        flat_tags = [TopLevelTags.DNS]

        hierarchical_tags = TagHierarchy.flat_to_hierarchical(flat_tags)

        assert HierarchicalTags.DNS_ROOTS in hierarchical_tags
        assert HierarchicalTags.DNS_CCTLD in hierarchical_tags
        assert HierarchicalTags.DNS_ANS in hierarchical_tags
