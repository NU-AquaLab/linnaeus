"""
Feature engineering for SVM-based AS classification.

This module provides comprehensive feature extraction and preprocessing
for traditional machine learning approaches to AS classification.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from ...data.access import DataAccessLayer
from ..schemas import OrganizationData

logger = logging.getLogger(__name__)


class ASNFeatureEngineer:
    """
    Feature engineering pipeline for AS classification.

    Extracts and processes features from multiple data sources:
    - ASRank topology and ranking features
    - PeeringDB network characteristics
    - ASPOP population and cone statistics
    - Geographic and organizational metadata
    """

    def __init__(self, data_access: Optional[DataAccessLayer] = None):
        """Initialize feature engineer."""
        self.data_access = data_access or DataAccessLayer()
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_selector = SelectKBest(score_func=f_classif)
        self.feature_names_ = []

    def extract_features(self, asns: List[int]) -> pd.DataFrame:
        """
        Extract comprehensive features for a list of ASNs.

        Parameters
        ----------
        asns : List[int]
            List of Autonomous System Numbers.

        Returns
        -------
        pd.DataFrame
            Feature matrix with ASN as index and feature columns.
        """
        logger.info(f"Extracting features for {len(asns)} ASNs")

        features_list = []

        for asn in asns:
            try:
                org_data = self.data_access.get_organization_data(asn)
                if org_data:
                    features = self._extract_single_asn_features(org_data)
                else:
                    # Create minimal features for missing data
                    features = self._create_minimal_features(asn)

                features["asn"] = asn
                features_list.append(features)

            except Exception as e:
                logger.warning(f"Failed to extract features for ASN {asn}: {e}")
                features = self._create_minimal_features(asn)
                features["asn"] = asn
                features_list.append(features)

        features_df = pd.DataFrame(features_list)
        features_df.set_index("asn", inplace=True)

        # Store feature names
        self.feature_names_ = list(features_df.columns)

        logger.info(f"Extracted {len(self.feature_names_)} features")
        return features_df

    def _extract_single_asn_features(self, org_data: OrganizationData) -> Dict:
        """Extract features for a single ASN."""
        features = {}

        # ASRank features
        if org_data.asrank:
            features.update(self._extract_asrank_features(org_data.asrank))

        # PeeringDB features
        if org_data.peeringdb:
            features.update(self._extract_peeringdb_features(org_data.peeringdb))

        # ASPOP features
        if org_data.aspop:
            features.update(self._extract_aspop_features(org_data.aspop))

        # Organizational features
        features.update(self._extract_org_features(org_data))

        return features

    def _extract_asrank_features(self, asrank_data) -> Dict:
        """Extract features from ASRank data."""
        features = {
            # Ranking features
            "asrank_rank": asrank_data.rank or 0,
            "asrank_clique_member": 1 if asrank_data.clique_member else 0,
            "asrank_seen": 1 if asrank_data.seen else 0,
            # Geographic features
            "asrank_longitude": asrank_data.longitude or 0.0,
            "asrank_latitude": asrank_data.latitude or 0.0,
            # Customer cone features
            "asrank_cone_asns": asrank_data.cone_asns or 0,
            "asrank_cone_prefixes": asrank_data.cone_prefixes or 0,
            "asrank_cone_addresses": asrank_data.cone_addresses or 0,
            # Degree features (network topology)
            "asrank_degree_provider": asrank_data.degree_provider or 0,
            "asrank_degree_peer": asrank_data.degree_peer or 0,
            "asrank_degree_customer": asrank_data.degree_customer or 0,
            "asrank_degree_total": asrank_data.degree_total or 0,
            "asrank_degree_transit": asrank_data.degree_transit or 0,
            "asrank_degree_sibling": asrank_data.degree_sibling or 0,
            # Announcing features
            "asrank_announcing_prefixes": asrank_data.announcing_prefixes or 0,
            "asrank_announcing_addresses": asrank_data.announcing_addresses or 0,
        }

        # Derived features
        features.update(
            {
                # Ratios and derived metrics
                "asrank_cone_density": self._safe_divide(
                    features["asrank_cone_addresses"], features["asrank_cone_prefixes"]
                ),
                "asrank_peer_customer_ratio": self._safe_divide(
                    features["asrank_degree_peer"], features["asrank_degree_customer"]
                ),
                "asrank_provider_customer_ratio": self._safe_divide(
                    features["asrank_degree_provider"],
                    features["asrank_degree_customer"],
                ),
                "asrank_transit_ratio": self._safe_divide(
                    features["asrank_degree_transit"], features["asrank_degree_total"]
                ),
                "asrank_announce_efficiency": self._safe_divide(
                    features["asrank_announcing_addresses"],
                    features["asrank_announcing_prefixes"],
                ),
                # Size categories (log-based)
                "asrank_cone_size_log": np.log1p(features["asrank_cone_asns"]),
                "asrank_address_size_log": np.log1p(features["asrank_cone_addresses"]),
                "asrank_prefix_size_log": np.log1p(
                    features["asrank_announcing_prefixes"]
                ),
            }
        )

        return features

    def _extract_peeringdb_features(self, peeringdb_data) -> Dict:
        """Extract features from PeeringDB data."""
        features = {
            # Network type and scope
            "peeringdb_has_website": 1 if peeringdb_data.website else 0,
            "peeringdb_has_looking_glass": 1 if peeringdb_data.looking_glass else 0,
            "peeringdb_has_route_server": 1 if peeringdb_data.route_server else 0,
            "peeringdb_has_irr_as_set": 1 if peeringdb_data.irr_as_set else 0,
            # Technical capabilities
            "peeringdb_info_unicast": 1 if peeringdb_data.info_unicast else 0,
            "peeringdb_info_multicast": 1 if peeringdb_data.info_multicast else 0,
            "peeringdb_info_ipv6": 1 if peeringdb_data.info_ipv6 else 0,
            # Policy information
            "peeringdb_has_policy_url": 1 if peeringdb_data.policy_url else 0,
        }

        # Categorical features (one-hot encoded)
        if peeringdb_data.info_type:
            features[
                f'peeringdb_type_{peeringdb_data.info_type.lower().replace(" ", "_")}'
            ] = 1

        if peeringdb_data.info_scope:
            features[
                f'peeringdb_scope_{peeringdb_data.info_scope.lower().replace(" ", "_")}'
            ] = 1

        if peeringdb_data.policy_general:
            features[
                f'peeringdb_policy_{peeringdb_data.policy_general.lower().replace(" ", "_")}'
            ] = 1

        return features

    def _extract_aspop_features(self, aspop_data) -> Dict:
        """Extract features from ASPOP data."""
        return {
            "aspop_customer_cone_asns": aspop_data.customer_cone_asns or 0,
            "aspop_customer_cone_prefixes": aspop_data.customer_cone_prefixes or 0,
            "aspop_customer_cone_addresses": aspop_data.customer_cone_addresses or 0,
            "aspop_customer_cone_unique": aspop_data.customer_cone_unique or 0,
            # RIR information
            "aspop_rir_afrinic": 1 if aspop_data.rir == "AFRINIC" else 0,
            "aspop_rir_apnic": 1 if aspop_data.rir == "APNIC" else 0,
            "aspop_rir_arin": 1 if aspop_data.rir == "ARIN" else 0,
            "aspop_rir_lacnic": 1 if aspop_data.rir == "LACNIC" else 0,
            "aspop_rir_ripe": 1 if aspop_data.rir == "RIPE NCC" else 0,
        }

    def _extract_org_features(self, org_data: OrganizationData) -> Dict:
        """Extract organizational metadata features."""
        features = {
            "org_has_name": 1 if org_data.name else 0,
            "org_has_website": 1 if org_data.website else 0,
            "org_has_country": 1 if org_data.country else 0,
        }

        # Name-based features
        if org_data.name:
            name_lower = org_data.name.lower()
            features.update(
                {
                    "org_name_has_isp": 1 if "isp" in name_lower else 0,
                    "org_name_has_telecom": (
                        1
                        if any(
                            term in name_lower
                            for term in ["telecom", "telco", "telecommunications"]
                        )
                        else 0
                    ),
                    "org_name_has_university": (
                        1
                        if any(
                            term in name_lower
                            for term in ["university", "college", "edu"]
                        )
                        else 0
                    ),
                    "org_name_has_government": (
                        1
                        if any(
                            term in name_lower
                            for term in ["gov", "government", "municipal"]
                        )
                        else 0
                    ),
                    "org_name_has_network": 1 if "network" in name_lower else 0,
                    "org_name_has_cloud": 1 if "cloud" in name_lower else 0,
                    "org_name_has_hosting": 1 if "hosting" in name_lower else 0,
                    "org_name_length": len(org_data.name),
                }
            )

        # Country-based features (basic geographic categories)
        if org_data.country:
            country_code = org_data.country.upper()
            # Major economic regions
            features.update(
                {
                    "org_region_north_america": (
                        1 if country_code in ["US", "CA", "MX"] else 0
                    ),
                    "org_region_europe": (
                        1
                        if country_code in ["DE", "FR", "GB", "IT", "ES", "NL", "PL"]
                        else 0
                    ),
                    "org_region_asia_pacific": (
                        1 if country_code in ["CN", "JP", "KR", "IN", "AU", "SG"] else 0
                    ),
                    "org_region_south_america": (
                        1 if country_code in ["BR", "AR", "CL", "CO", "PE"] else 0
                    ),
                }
            )

        return features

    def _create_minimal_features(self, asn: int) -> Dict:
        """Create minimal feature set for missing ASN data."""
        # Create zero-filled feature dictionary with all expected features
        features = {}

        # ASRank features (all zeros)
        asrank_features = [
            "rank",
            "clique_member",
            "seen",
            "longitude",
            "latitude",
            "cone_asns",
            "cone_prefixes",
            "cone_addresses",
            "degree_provider",
            "degree_peer",
            "degree_customer",
            "degree_total",
            "degree_transit",
            "degree_sibling",
            "announcing_prefixes",
            "announcing_addresses",
        ]

        for feature in asrank_features:
            features[f"asrank_{feature}"] = 0

        # Add derived ASRank features
        features.update(
            {
                "asrank_cone_density": 0,
                "asrank_peer_customer_ratio": 0,
                "asrank_provider_customer_ratio": 0,
                "asrank_transit_ratio": 0,
                "asrank_announce_efficiency": 0,
                "asrank_cone_size_log": 0,
                "asrank_address_size_log": 0,
                "asrank_prefix_size_log": 0,
            }
        )

        # PeeringDB features (all zeros)
        peeringdb_features = [
            "has_website",
            "has_looking_glass",
            "has_route_server",
            "has_irr_as_set",
            "info_unicast",
            "info_multicast",
            "info_ipv6",
            "has_policy_url",
        ]

        for feature in peeringdb_features:
            features[f"peeringdb_{feature}"] = 0

        # ASPOP features (all zeros)
        aspop_features = [
            "customer_cone_asns",
            "customer_cone_prefixes",
            "customer_cone_addresses",
            "customer_cone_unique",
        ]

        for feature in aspop_features:
            features[f"aspop_{feature}"] = 0

        # RIR features (all zeros)
        for rir in ["afrinic", "apnic", "arin", "lacnic", "ripe"]:
            features[f"aspop_rir_{rir}"] = 0

        # Organizational features (all zeros)
        org_features = [
            "has_name",
            "has_website",
            "has_country",
            "name_has_isp",
            "name_has_telecom",
            "name_has_university",
            "name_has_government",
            "name_has_network",
            "name_has_cloud",
            "name_has_hosting",
            "name_length",
        ]

        for feature in org_features:
            features[f"org_{feature}"] = 0

        # Regional features (all zeros)
        for region in ["north_america", "europe", "asia_pacific", "south_america"]:
            features[f"org_region_{region}"] = 0

        return features

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safely divide two numbers, returning 0 if denominator is 0."""
        return numerator / denominator if denominator != 0 else 0.0

    def preprocess_features(
        self, features_df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess features with imputation, scaling, and selection.

        Parameters
        ----------
        features_df : pd.DataFrame
            Raw feature matrix.
        fit : bool
            Whether to fit the preprocessing pipeline.

        Returns
        -------
        pd.DataFrame
            Preprocessed feature matrix.
        """
        # Separate ASN column if present
        asns = None
        if "asn" in features_df.columns:
            asns = features_df["asn"]
            features_df = features_df.drop("asn", axis=1)

        # Impute missing values
        if fit:
            features_imputed = pd.DataFrame(
                self.imputer.fit_transform(features_df),
                index=features_df.index,
                columns=features_df.columns,
            )
        else:
            features_imputed = pd.DataFrame(
                self.imputer.transform(features_df),
                index=features_df.index,
                columns=features_df.columns,
            )

        # Scale features
        if fit:
            features_scaled = pd.DataFrame(
                self.scaler.fit_transform(features_imputed),
                index=features_imputed.index,
                columns=features_imputed.columns,
            )
        else:
            features_scaled = pd.DataFrame(
                self.scaler.transform(features_imputed),
                index=features_imputed.index,
                columns=features_imputed.columns,
            )

        # Re-add ASN column if it was present
        if asns is not None:
            features_scaled["asn"] = asns

        return features_scaled

    def select_features(
        self, X: pd.DataFrame, y: pd.DataFrame, k: int = 50
    ) -> pd.DataFrame:
        """
        Select top k features based on univariate statistical tests.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.DataFrame
            Target labels.
        k : int
            Number of top features to select.

        Returns
        -------
        pd.DataFrame
            Selected features.
        """
        # Remove ASN column for feature selection
        X_features = X.drop("asn", axis=1, errors="ignore")

        # Perform feature selection
        self.feature_selector.set_params(k=min(k, X_features.shape[1]))
        X_selected = self.feature_selector.fit_transform(X_features, y)

        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        selected_features = X_features.columns[selected_mask]

        # Create DataFrame with selected features
        X_selected_df = pd.DataFrame(
            X_selected, index=X.index, columns=selected_features
        )

        # Re-add ASN column if present
        if "asn" in X.columns:
            X_selected_df["asn"] = X["asn"]

        logger.info(
            f"Selected {len(selected_features)} features from {len(X_features.columns)}"
        )
        return X_selected_df

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from selection."""
        if hasattr(self.feature_selector, "scores_"):
            feature_names = self.feature_names_
            if "asn" in feature_names:
                feature_names = [f for f in feature_names if f != "asn"]

            importance_df = pd.DataFrame(
                {"feature": feature_names, "score": self.feature_selector.scores_}
            ).sort_values("score", ascending=False)

            return importance_df
        else:
            logger.warning("Feature selector has not been fitted yet")
            return pd.DataFrame()
