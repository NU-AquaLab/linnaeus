"""
Tests for data schemas and models.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from linneaus.models.schemas import (
    ASClassification,
    ASPOPData,
    ASRankData,
    BatchClassificationResponse,
    ClassificationRequest,
    ClassificationTags,
    DataDownloadRequest,
    OrganizationData,
    PeeringDBData,
)


class TestClassificationTags:
    """Test classification tags enum."""

    def test_all_tags_exist(self):
        """Test that all expected tags exist."""
        expected_tags = [
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
            "Finance",
            "Law Enforcement",
            "Health",
            "Cooperatives",
            "TV/Radio and Cultural Amenities",
            "Transportation",
            "Virtual Private Networks",
            "Personal",
            "Community",
        ]

        enum_values = [tag.value for tag in ClassificationTags]

        for expected_tag in expected_tags:
            assert expected_tag in enum_values

    def test_tag_creation(self):
        """Test creating tags from values."""
        tag = ClassificationTags("Access")
        assert tag == ClassificationTags.ACCESS
        assert tag.value == "Access"


class TestASClassification:
    """Test AS classification model."""

    def test_valid_classification(self):
        """Test creating valid classification."""
        classification = ASClassification(
            asn=174,
            organization_name="Test Org",
            tags=[ClassificationTags.TRANSIT, ClassificationTags.ENTERPRISE],
            model_used="gpt-4o-mini",
        )

        assert classification.asn == 174
        assert classification.organization_name == "Test Org"
        assert len(classification.tags) == 2
        assert ClassificationTags.TRANSIT in classification.tags
        assert classification.model_used == "gpt-4o-mini"
        assert isinstance(classification.timestamp, datetime)

    def test_invalid_asn(self):
        """Test validation of invalid ASN."""
        with pytest.raises(ValidationError):
            ASClassification(
                asn=0,  # Invalid ASN
                organization_name="Test Org",
                tags=[ClassificationTags.ACCESS],
                model_used="gpt-4o-mini",
            )

        with pytest.raises(ValidationError):
            ASClassification(
                asn=-1,  # Invalid ASN
                organization_name="Test Org",
                tags=[ClassificationTags.ACCESS],
                model_used="gpt-4o-mini",
            )

    def test_with_confidence_scores(self):
        """Test classification with confidence scores."""
        classification = ASClassification(
            asn=174,
            organization_name="Test Org",
            tags=[ClassificationTags.TRANSIT],
            confidence_scores={"Transit": 0.95},
            model_used="gpt-4o-mini",
        )

        assert classification.confidence_scores == {"Transit": 0.95}


class TestBatchClassificationResponse:
    """Test batch classification response."""

    def test_valid_batch_response(self, sample_classification):
        """Test creating valid batch response."""
        response = BatchClassificationResponse(
            classifications=[sample_classification],
            batch_id="test_batch_123",
            total_processed=1,
            successful=1,
            failed=0,
            processing_time_seconds=1.5,
        )

        assert len(response.classifications) == 1
        assert response.batch_id == "test_batch_123"
        assert response.total_processed == 1
        assert response.successful == 1
        assert response.failed == 0
        assert response.processing_time_seconds == 1.5


class TestASRankData:
    """Test ASRank data model."""

    def test_valid_asrank_data(self):
        """Test creating valid ASRank data."""
        data = ASRankData(
            asn=174,
            asn_name="COGENT-174",
            rank=10,
            organization_name="Cogent Communications",
            clique_member=True,
            seen=True,
            longitude=-77.0365,
            latitude=38.8951,
            country_iso="US",
            country_name="United States",
        )

        assert data.asn == 174
        assert data.asn_name == "COGENT-174"
        assert data.rank == 10
        assert data.organization_name == "Cogent Communications"
        assert data.clique_member is True
        assert data.country_iso == "US"

    def test_optional_fields(self):
        """Test ASRank data with optional fields."""
        data = ASRankData(asn=174)

        assert data.asn == 174
        assert data.asn_name is None
        assert data.rank is None
        assert data.longitude is None


class TestPeeringDBData:
    """Test PeeringDB data model."""

    def test_valid_peeringdb_data(self):
        """Test creating valid PeeringDB data."""
        data = PeeringDBData(
            asn=174,
            name="Cogent Communications",
            organization="Cogent Communications, Inc.",
            website="https://www.cogentco.com",
            info_unicast=True,
            info_ipv6=True,
        )

        assert data.asn == 174
        assert data.name == "Cogent Communications"
        assert data.organization == "Cogent Communications, Inc."
        assert data.website == "https://www.cogentco.com"
        assert data.info_unicast is True
        assert data.info_ipv6 is True


class TestASPOPData:
    """Test ASPOP data model."""

    def test_valid_aspop_data(self):
        """Test creating valid ASPOP data."""
        data = ASPOPData(
            asn=174,
            name="COGENT-174",
            country="US",
            rir="ARIN",
            customer_cone_asns=12000,
            customer_cone_prefixes=45000,
        )

        assert data.asn == 174
        assert data.name == "COGENT-174"
        assert data.country == "US"
        assert data.rir == "ARIN"
        assert data.customer_cone_asns == 12000


class TestOrganizationData:
    """Test organization data model."""

    def test_organization_data_creation(
        self, sample_asrank_data, sample_peeringdb_data, sample_aspop_data
    ):
        """Test creating organization data."""
        org_data = OrganizationData(
            asn=174,
            asrank=sample_asrank_data,
            peeringdb=sample_peeringdb_data,
            aspop=sample_aspop_data,
        )

        assert org_data.asn == 174
        assert org_data.asrank == sample_asrank_data
        assert org_data.peeringdb == sample_peeringdb_data
        assert org_data.aspop == sample_aspop_data

    def test_name_extraction_priority(self):
        """Test name extraction priority."""
        # Test ASRank priority
        asrank_data = ASRankData(asn=174, organization_name="ASRank Name")
        peeringdb_data = PeeringDBData(asn=174, organization="PeeringDB Name")

        org_data = OrganizationData(
            asn=174, asrank=asrank_data, peeringdb=peeringdb_data
        )

        # Should use ASRank name (highest priority)
        assert org_data.name == "ASRank Name"

    def test_name_fallback(self):
        """Test name fallback when ASRank unavailable."""
        peeringdb_data = PeeringDBData(asn=174, organization="PeeringDB Name")
        aspop_data = ASPOPData(asn=174, name="ASPOP Name")

        org_data = OrganizationData(asn=174, peeringdb=peeringdb_data, aspop=aspop_data)

        # Should use PeeringDB name (second priority)
        assert org_data.name == "PeeringDB Name"


class TestClassificationRequest:
    """Test classification request model."""

    def test_valid_request(self, sample_organization_data):
        """Test creating valid classification request."""
        request = ClassificationRequest(
            organizations=[sample_organization_data],
            model="gpt-4o-mini",
            batch_size=10,
            include_confidence=True,
        )

        assert len(request.organizations) == 1
        assert request.model == "gpt-4o-mini"
        assert request.batch_size == 10
        assert request.include_confidence is True

    def test_empty_organizations(self):
        """Test validation of empty organizations list."""
        with pytest.raises(ValidationError):
            ClassificationRequest(organizations=[])


class TestDataDownloadRequest:
    """Test data download request model."""

    def test_valid_request(self):
        """Test creating valid download request."""
        request = DataDownloadRequest(
            sources=["peeringdb", "asrank"], date="2024-01-01", force_refresh=True
        )

        assert request.sources == ["peeringdb", "asrank"]
        assert request.date == "2024-01-01"
        assert request.force_refresh is True

    def test_invalid_source(self):
        """Test validation of invalid data source."""
        with pytest.raises(ValidationError):
            DataDownloadRequest(sources=["invalid_source"])

    def test_invalid_date_format(self):
        """Test validation of invalid date format."""
        with pytest.raises(ValidationError):
            DataDownloadRequest(date="01-01-2024")  # Wrong format

    def test_default_sources(self):
        """Test default sources."""
        request = DataDownloadRequest()
        assert "peeringdb" in request.sources
        assert "asrank" in request.sources
        assert "aspop" in request.sources
