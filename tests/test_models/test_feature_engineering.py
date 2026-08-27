"""
Tests for SVM feature engineering.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from linneaus.models.schemas import ASRankData, OrganizationData, PeeringDBData
from linneaus.models.svm.feature_engineering import ASNFeatureEngineer


class TestASNFeatureEngineer:
    """Test ASN feature engineering for SVM models."""

    @pytest.fixture
    def mock_data_access(
        self, sample_asrank_data, sample_peeringdb_data, sample_aspop_data
    ):
        """Mock data access layer with comprehensive data."""
        mock_access = Mock()
        org_data = OrganizationData(
            asn=174,
            asrank=sample_asrank_data,
            peeringdb=sample_peeringdb_data,
            aspop=sample_aspop_data,
        )
        mock_access.get_organization_data.return_value = org_data
        return mock_access

    @pytest.fixture
    def feature_engineer(self, mock_data_access):
        """Create feature engineer instance with mocked data access."""
        return ASNFeatureEngineer(data_access=mock_data_access)

    def test_initialization(self, feature_engineer):
        """Test feature engineer initialization."""
        assert feature_engineer.data_access is not None
        assert feature_engineer.scaler is not None
        assert feature_engineer.imputer is not None
        assert feature_engineer.feature_selector is not None
        assert feature_engineer.feature_names_ == []

    def test_initialization_default_data_access(self, mock_config):
        """Test initialization with default data access."""
        with patch("linneaus.models.svm.feature_engineering.DataAccessLayer"):
            engineer = ASNFeatureEngineer()
            assert engineer.data_access is not None

    def test_extract_asrank_features(self, feature_engineer, sample_asrank_data):
        """Test ASRank feature extraction."""
        features = feature_engineer._extract_asrank_features(sample_asrank_data)

        # Check that all expected features are present
        expected_features = {
            "asrank_rank",
            "asrank_longitude",
            "asrank_latitude",
            "asrank_cone_asns",
            "asrank_cone_prefixes",
            "asrank_cone_addresses",
            "asrank_degree_provider",
            "asrank_degree_peer",
            "asrank_degree_customer",
            "asrank_degree_total",
            "asrank_degree_transit",
            "asrank_degree_sibling",
            "asrank_announcing_prefixes",
            "asrank_announcing_addresses",
            "asrank_clique_member",
            "asrank_seen",
            # Derived features
            "asrank_cone_density",
            "asrank_peer_customer_ratio",
            "asrank_provider_customer_ratio",
            "asrank_transit_ratio",
            "asrank_announce_efficiency",
            "asrank_cone_size_log",
            "asrank_address_size_log",
            "asrank_prefix_size_log",
        }

        assert set(features.keys()) >= expected_features

        # Check specific values
        assert features["asrank_rank"] == 10
        assert features["asrank_cone_asns"] == 15000
        assert features["asrank_clique_member"] == 1  # Boolean converted to int
        assert features["asrank_longitude"] == -77.0365

    def test_extract_peeringdb_features(self, feature_engineer, sample_peeringdb_data):
        """Test PeeringDB feature extraction."""
        features = feature_engineer._extract_peeringdb_features(sample_peeringdb_data)

        # Check boolean features
        assert features["peeringdb_has_website"] == 1
        assert features["peeringdb_has_looking_glass"] == 1
        assert features["peeringdb_info_unicast"] == 1
        assert features["peeringdb_info_multicast"] == 0
        assert features["peeringdb_info_ipv6"] == 1

        # Check categorical features (lowercase one-hot encoded)
        assert features["peeringdb_type_nsp"] == 1
        assert features["peeringdb_policy_open"] == 1
        assert features["peeringdb_scope_global"] == 1

    def test_extract_aspop_features(self, feature_engineer, sample_aspop_data):
        """Test ASPOP feature extraction."""
        features = feature_engineer._extract_aspop_features(sample_aspop_data)

        # Check numerical features
        assert features["aspop_customer_cone_asns"] == 12000
        assert features["aspop_customer_cone_prefixes"] == 45000
        assert features["aspop_customer_cone_addresses"] == 900000

        # Check categorical features (RIR, lowercase)
        assert features["aspop_rir_arin"] == 1
        assert features["aspop_rir_ripe"] == 0
        assert features["aspop_rir_apnic"] == 0
        assert features["aspop_rir_afrinic"] == 0
        assert features["aspop_rir_lacnic"] == 0

    def test_derive_network_features(self, feature_engineer, sample_asrank_data):
        """Test derived network topology features."""
        features = feature_engineer._extract_asrank_features(sample_asrank_data)

        # cone_density = cone_addresses / cone_prefixes
        assert features["asrank_cone_density"] == pytest.approx(
            sample_asrank_data.cone_addresses / sample_asrank_data.cone_prefixes,
            rel=0.01,
        )
        # peer_customer_ratio = degree_peer / degree_customer
        assert features["asrank_peer_customer_ratio"] == pytest.approx(
            sample_asrank_data.degree_peer / sample_asrank_data.degree_customer,
            rel=0.01,
        )
        # provider_customer_ratio = degree_provider / degree_customer
        assert features["asrank_provider_customer_ratio"] == pytest.approx(
            sample_asrank_data.degree_provider / sample_asrank_data.degree_customer,
            rel=0.01,
        )
        # announce_efficiency = announcing_addresses / announcing_prefixes
        assert features["asrank_announce_efficiency"] == pytest.approx(
            sample_asrank_data.announcing_addresses
            / sample_asrank_data.announcing_prefixes,
            rel=0.01,
        )
        # Log-based size features
        assert features["asrank_cone_size_log"] == pytest.approx(
            np.log1p(sample_asrank_data.cone_asns), rel=0.01
        )

    def test_extract_features_integration(self, feature_engineer):
        """Test complete extract_features pipeline."""
        features_df = feature_engineer.extract_features([174])

        # Check output is a DataFrame with ASN as index
        assert isinstance(features_df, pd.DataFrame)
        assert 174 in features_df.index
        assert features_df.shape[0] == 1
        assert features_df.shape[1] > 0

        # Check that feature names are stored
        assert len(feature_engineer.feature_names_) > 0
        assert feature_engineer.feature_names_ == list(features_df.columns)

        # Check that features include all source prefixes
        col_names = list(features_df.columns)
        assert any("asrank_" in c for c in col_names)
        assert any("peeringdb_" in c for c in col_names)
        assert any("aspop_" in c for c in col_names)
        assert any("org_" in c for c in col_names)

    def test_extract_features_multiple_asns(self, feature_engineer):
        """Test extracting features for multiple ASNs."""
        features_df = feature_engineer.extract_features([174, 15169])

        # All mock calls return the same org data, so both rows should exist
        assert features_df.shape[0] == 2
        assert features_df.shape[1] > 0

    def test_missing_asn_handling(self, mock_config):
        """Test handling of ASNs with no data."""
        mock_access = Mock()
        mock_access.get_organization_data.return_value = None

        engineer = ASNFeatureEngineer(data_access=mock_access)
        features_df = engineer.extract_features([99999, 99998])

        # Should return feature matrix with minimal/default values
        assert features_df.shape[0] == 2
        assert features_df.shape[1] > 0

        # Values should all be finite (no NaN or inf from minimal features)
        assert np.all(np.isfinite(features_df.values))

    def test_missing_data_creates_zero_features(self, mock_config):
        """Test that missing data produces zero-filled features."""
        mock_access = Mock()
        mock_access.get_organization_data.return_value = None

        engineer = ASNFeatureEngineer(data_access=mock_access)
        features_df = engineer.extract_features([99999])

        # All features should be 0 for missing data (minimal features)
        assert (features_df.values == 0).all()

    def test_exception_handling_in_extract(self, mock_config):
        """Test that exceptions during extraction are handled gracefully."""
        mock_access = Mock()
        mock_access.get_organization_data.side_effect = Exception("Network error")

        engineer = ASNFeatureEngineer(data_access=mock_access)
        features_df = engineer.extract_features([174])

        # Should still return a DataFrame with minimal features
        assert features_df.shape[0] == 1
        assert features_df.shape[1] > 0

    def test_categorical_encoding(self, feature_engineer):
        """Test one-hot encoding of categorical features."""
        pdb_data_nsp = PeeringDBData(
            asn=174, name="Test", info_type="NSP", policy_general="Open"
        )
        pdb_data_content = PeeringDBData(
            asn=15169, name="Test", info_type="Content", policy_general="Selective"
        )

        features_nsp = feature_engineer._extract_peeringdb_features(pdb_data_nsp)
        features_content = feature_engineer._extract_peeringdb_features(
            pdb_data_content
        )

        # Check one-hot encoding (lowercase)
        assert features_nsp.get("peeringdb_type_nsp", 0) == 1
        assert features_nsp.get("peeringdb_type_content", 0) == 0
        assert features_content.get("peeringdb_type_nsp", 0) == 0
        assert features_content.get("peeringdb_type_content", 0) == 1

        # Check policy encoding (lowercase)
        assert features_nsp.get("peeringdb_policy_open", 0) == 1
        assert features_nsp.get("peeringdb_policy_selective", 0) == 0
        assert features_content.get("peeringdb_policy_open", 0) == 0
        assert features_content.get("peeringdb_policy_selective", 0) == 1

    def test_edge_cases_zero_values(self, feature_engineer):
        """Test edge cases with zero values in ASRank data."""
        extreme_asrank = ASRankData(
            asn=1,
            asn_name="Test",
            rank=1,
            organization_name="Test",
            cone_asns=0,
            cone_prefixes=0,
            cone_addresses=0,
            degree_total=0,
            degree_provider=0,
            degree_peer=0,
            degree_customer=0,
            degree_transit=0,
            degree_sibling=0,
            announcing_prefixes=0,
            announcing_addresses=0,
        )

        features = feature_engineer._extract_asrank_features(extreme_asrank)

        # Should handle division by zero gracefully via _safe_divide
        assert np.isfinite(features["asrank_cone_density"])
        assert features["asrank_cone_density"] == 0.0
        assert np.isfinite(features["asrank_peer_customer_ratio"])
        assert features["asrank_peer_customer_ratio"] == 0.0
        assert np.isfinite(features["asrank_provider_customer_ratio"])
        assert np.isfinite(features["asrank_transit_ratio"])
        assert np.isfinite(features["asrank_announce_efficiency"])

    def test_edge_cases_large_values(self, feature_engineer):
        """Test edge cases with very large values."""
        large_asrank = ASRankData(
            asn=2,
            asn_name="Test",
            rank=100000,
            organization_name="Test",
            cone_asns=1000000,
            cone_prefixes=10000000,
            cone_addresses=1000000000,
            degree_total=10000,
            degree_provider=1000,
            degree_peer=5000,
            degree_customer=4000,
            degree_transit=5000,
            degree_sibling=0,
            announcing_prefixes=100000,
            announcing_addresses=50000000,
        )

        features = feature_engineer._extract_asrank_features(large_asrank)

        # Should handle large values without overflow
        assert all(np.isfinite(v) for v in features.values())

    def test_batch_processing(self, feature_engineer):
        """Test batch processing of multiple ASNs."""
        asns = list(range(1, 101))  # 100 ASNs
        features_df = feature_engineer.extract_features(asns)

        # Should handle batch processing
        assert features_df.shape[0] == 100
        assert features_df.shape[1] > 0

        # Should not contain NaN values
        assert not features_df.isnull().any().any()

    def test_feature_statistics(self, feature_engineer):
        """Test that we can compute feature statistics on output."""
        features_df = feature_engineer.extract_features([174])

        # Compute basic statistics
        means = features_df.mean()
        features_df.std()

        # Should be able to compute statistics without errors
        assert len(means) == features_df.shape[1]
        assert all(np.isfinite(means))

    def test_preprocess_features(self, feature_engineer):
        """Test the preprocess_features method with fit=True."""
        features_df = feature_engineer.extract_features([174])

        preprocessed = feature_engineer.preprocess_features(features_df, fit=True)

        # Should return a DataFrame with the same shape
        assert isinstance(preprocessed, pd.DataFrame)
        assert preprocessed.shape == features_df.shape

        # All values should be finite
        assert np.all(np.isfinite(preprocessed.values))

    def test_preprocess_features_transform_only(self, feature_engineer):
        """Test preprocess_features with fit=False after fitting."""
        features_df = feature_engineer.extract_features([174])

        # First fit
        feature_engineer.preprocess_features(features_df, fit=True)

        # Then transform-only
        preprocessed = feature_engineer.preprocess_features(features_df, fit=False)

        assert isinstance(preprocessed, pd.DataFrame)
        assert preprocessed.shape == features_df.shape

    def test_safe_divide(self, feature_engineer):
        """Test the _safe_divide helper method."""
        assert feature_engineer._safe_divide(10, 2) == 5.0
        assert feature_engineer._safe_divide(10, 0) == 0.0
        assert feature_engineer._safe_divide(0, 0) == 0.0
        assert feature_engineer._safe_divide(0, 5) == 0.0

    def test_create_minimal_features(self, feature_engineer):
        """Test that _create_minimal_features returns all-zero features."""
        features = feature_engineer._create_minimal_features(99999)

        # All values should be zero
        assert all(v == 0 for v in features.values())

        # Should contain features from all sources
        keys = list(features.keys())
        assert any(k.startswith("asrank_") for k in keys)
        assert any(k.startswith("peeringdb_") for k in keys)
        assert any(k.startswith("aspop_") for k in keys)
        assert any(k.startswith("org_") for k in keys)

    def test_feature_names_stored_after_extract(self, feature_engineer):
        """Test that feature_names_ is populated after extract_features."""
        assert feature_engineer.feature_names_ == []

        features_df = feature_engineer.extract_features([174])

        assert len(feature_engineer.feature_names_) > 0
        assert feature_engineer.feature_names_ == list(features_df.columns)

    def test_extract_features_returns_asn_index(self, feature_engineer):
        """Test that extract_features returns DataFrame with ASN as index."""
        features_df = feature_engineer.extract_features([174])

        assert features_df.index.name == "asn"
        assert 174 in features_df.index
