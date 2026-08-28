"""
Tests for data access layer.
"""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from linnaeus.data.access import DataAccessLayer
from linnaeus.models.schemas import (
    ASPOPData,
    ASRankData,
    OrganizationData,
    PeeringDBData,
)


class TestDataAccessLayer:
    """Test data access layer functionality."""

    @pytest.fixture
    def data_layer(self, mock_config, temp_dir):
        """Create a data access layer with mocked config."""
        with patch("linnaeus.data.access.get_config", return_value=mock_config):
            return DataAccessLayer(temp_dir / "processed")

    @pytest.fixture
    def sample_asrank_csv(self, temp_dir):
        """Create sample ASRank CSV data."""
        csv_data = """asn,asnName,rank,orgId,orgName,cliqueMember,seen,longitude,latitude,country_iso,country_name
174,COGENT-174,10,COGENT,Cogent Communications,true,true,-77.0365,38.8951,US,United States
15169,GOOGLE,1,GOOGLE,Google LLC,false,true,-122.0838,37.4220,US,United States"""

        csv_file = temp_dir / "processed" / "as_rank_features.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_file, "w") as f:
            f.write(csv_data)

        return csv_file

    @pytest.fixture
    def sample_peeringdb_json(self, temp_dir):
        """Create sample PeeringDB JSON data."""
        json_data = {
            "174": {
                "name": "Cogent Communications",
                "organization": "Cogent Communications, Inc.",
                "website": "https://www.cogentco.com",
                "info_type": "NSP",
                "info_unicast": True,
            },
            "15169": {
                "name": "Google",
                "organization": "Google LLC",
                "website": "https://www.google.com",
                "info_type": "Content",
                "info_unicast": True,
            },
        }

        json_file = temp_dir / "processed" / "peeringdb_net.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        return json_file

    @pytest.fixture
    def sample_aspop_json(self, temp_dir):
        """Create sample ASPOP JSON data."""
        json_data = {
            "174": {
                "name": "COGENT-174",
                "country": "US",
                "rir": "ARIN",
                "customer_cone_asns": 12000,
            },
            "15169": {
                "name": "GOOGLE",
                "country": "US",
                "rir": "ARIN",
                "customer_cone_asns": 500,
            },
        }

        json_file = temp_dir / "processed" / "aspop_processed.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        return json_file

    def test_get_asrank_data_success(self, data_layer, sample_asrank_csv):
        """Test successful ASRank data retrieval."""
        asrank_data = data_layer.get_asrank_data(174)

        assert asrank_data is not None
        assert isinstance(asrank_data, ASRankData)
        assert asrank_data.asn == 174
        assert asrank_data.asn_name == "COGENT-174"
        assert asrank_data.organization_name == "Cogent Communications"
        assert asrank_data.rank == 10

    def test_get_asrank_data_not_found(self, data_layer, sample_asrank_csv):
        """Test ASRank data retrieval for non-existent ASN."""
        asrank_data = data_layer.get_asrank_data(99999)
        assert asrank_data is None

    def test_get_peeringdb_data_success(self, data_layer, sample_peeringdb_json):
        """Test successful PeeringDB data retrieval."""
        peeringdb_data = data_layer.get_peeringdb_data(174)

        assert peeringdb_data is not None
        assert isinstance(peeringdb_data, PeeringDBData)
        assert peeringdb_data.asn == 174
        assert peeringdb_data.organization == "Cogent Communications, Inc."
        assert peeringdb_data.website == "https://www.cogentco.com"

    def test_get_aspop_data_success(self, data_layer, sample_aspop_json):
        """Test successful ASPOP data retrieval."""
        aspop_data = data_layer.get_aspop_data(174)

        assert aspop_data is not None
        assert isinstance(aspop_data, ASPOPData)
        assert aspop_data.asn == 174
        assert aspop_data.name == "COGENT-174"
        assert aspop_data.country == "US"

    def test_get_organization_data_combined(
        self, data_layer, sample_asrank_csv, sample_peeringdb_json, sample_aspop_json
    ):
        """Test combined organization data retrieval."""
        org_data = data_layer.get_organization_data(174)

        assert org_data is not None
        assert isinstance(org_data, OrganizationData)
        assert org_data.asn == 174
        assert org_data.asrank is not None
        assert org_data.peeringdb is not None
        assert org_data.aspop is not None

        # Test derived fields
        assert org_data.name == "Cogent Communications"  # From ASRank
        assert org_data.country == "US"  # From ASRank
        assert org_data.website == "https://www.cogentco.com"  # From PeeringDB

    def test_get_organization_name_priority(
        self, data_layer, sample_asrank_csv, sample_peeringdb_json, sample_aspop_json
    ):
        """Test organization name priority fallback."""
        # Test with ASRank data (highest priority)
        name = data_layer.get_organization_name(174)
        assert name == "Cogent Communications"

        # Test fallback to PeeringDB when ASRank unavailable
        name = data_layer.get_organization_name(99999)
        assert name is None  # No data for this ASN

    def test_get_bulk_data(
        self, data_layer, sample_asrank_csv, sample_peeringdb_json, sample_aspop_json
    ):
        """Test bulk data retrieval."""
        asn_list = [174, 15169, 99999]  # Include non-existent ASN
        results = data_layer.get_bulk_data(asn_list)

        # Should return 2 results (174 and 15169), excluding 99999
        assert len(results) == 2
        assert all(isinstance(org, OrganizationData) for org in results)
        assert {org.asn for org in results} == {174, 15169}

    def test_get_complete_dataset(
        self, data_layer, sample_asrank_csv, sample_peeringdb_json, sample_aspop_json
    ):
        """Test complete dataset generation."""
        df = data_layer.get_complete_dataset()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two ASNs in test data
        assert df.index.name == "asn"
        assert 174 in df.index
        assert 15169 in df.index

        # Check prefixed columns exist
        asrank_cols = [col for col in df.columns if col.startswith("asrank_")]
        peeringdb_cols = [col for col in df.columns if col.startswith("peeringdb_")]
        aspop_cols = [col for col in df.columns if col.startswith("aspop_")]

        assert len(asrank_cols) > 0
        assert len(peeringdb_cols) > 0
        assert len(aspop_cols) > 0

    def test_clear_cache(self, data_layer, sample_asrank_csv):
        """Test cache clearing functionality."""
        # Load data to populate cache
        data_layer.get_asrank_data(174)
        assert data_layer._asrank_cache is not None

        # Clear cache
        data_layer.clear_cache()
        assert data_layer._asrank_cache is None
        assert data_layer._peeringdb_cache is None
        assert data_layer._aspop_cache is None

    def test_missing_files(self, data_layer):
        """Test behavior when data files are missing."""
        # Should handle missing files gracefully
        asrank_data = data_layer.get_asrank_data(174)
        assert asrank_data is None

        peeringdb_data = data_layer.get_peeringdb_data(174)
        assert peeringdb_data is None

        aspop_data = data_layer.get_aspop_data(174)
        assert aspop_data is None
