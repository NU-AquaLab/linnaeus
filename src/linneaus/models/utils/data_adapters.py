"""
Data adapters for converting between different classifier data formats.

This module provides adapters to handle the conversion between the unified
data format and the specific formats expected by SVM and LLM classifiers.
"""

import logging
from typing import Dict, List, Union

import pandas as pd

from ...data.access import DataAccessLayer
from ..schemas import OrganizationData
from ..svm.feature_engineering import ASNFeatureEngineer

logger = logging.getLogger(__name__)


class SVMDataAdapter:
    """
    Adapter for converting unified data format to SVM feature format.

    This adapter handles the conversion from raw ASN data or organization
    data to the feature matrix expected by SVM classifiers.
    """

    def __init__(
        self,
        include_asrank: bool = True,
        include_peeringdb: bool = True,
        include_aspop: bool = True,
        include_ipinfo: bool = True,
        normalize_features: bool = True,
    ):
        """
        Initialize SVM data adapter.

        Parameters
        ----------
        include_asrank : bool
            Whether to include ASRank features.
        include_peeringdb : bool
            Whether to include PeeringDB features.
        include_aspop : bool
            Whether to include ASPOP features.
        include_ipinfo : bool
            Whether to include IPinfo features.
        normalize_features : bool
            Whether to normalize the features.
        """
        self.include_asrank = include_asrank
        self.include_peeringdb = include_peeringdb
        self.include_aspop = include_aspop
        self.include_ipinfo = include_ipinfo
        self.normalize_features = normalize_features
        self.feature_engineer = ASNFeatureEngineer()
        self.data_access = DataAccessLayer()

    def transform(
        self, data: Union[pd.DataFrame, List[int], List[OrganizationData]]
    ) -> pd.DataFrame:
        """
        Transform input data to SVM feature format.

        Parameters
        ----------
        data : DataFrame, list of ASNs, or list of OrganizationData
            Input data to transform.

        Returns
        -------
        pd.DataFrame
            Feature matrix suitable for SVM classification.
        """
        # Convert to ASN list if needed
        if isinstance(data, pd.DataFrame):
            if "asn" in data.columns:
                asns = data["asn"].tolist()
            elif data.index.name == "asn":
                asns = data.index.tolist()
            else:
                raise ValueError("DataFrame must have ASN information")
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], OrganizationData):
                asns = [org.asn for org in data]
            else:
                asns = data
        else:
            raise ValueError("Invalid data format")

        # Create DataFrame with ASNs
        asn_df = pd.DataFrame({"asn": asns})

        # Extract features
        logger.info(f"Extracting features for {len(asns)} ASNs")
        features = self.feature_engineer.transform(asn_df)

        return features

    def fit_transform(
        self, data: Union[pd.DataFrame, List[int], List[OrganizationData]]
    ) -> pd.DataFrame:
        """
        Fit the feature engineer and transform data.

        Parameters
        ----------
        data : DataFrame, list of ASNs, or list of OrganizationData
            Input data to fit and transform.

        Returns
        -------
        pd.DataFrame
            Feature matrix suitable for SVM classification.
        """
        # Convert to ASN DataFrame
        if isinstance(data, pd.DataFrame):
            asn_df = data
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], OrganizationData):
                asns = [org.asn for org in data]
            else:
                asns = data
            asn_df = pd.DataFrame({"asn": asns})
        else:
            raise ValueError("Invalid data format")

        # Fit and transform
        logger.info(f"Fitting feature engineer on {len(asn_df)} ASNs")
        features = self.feature_engineer.fit_transform(asn_df)

        return features


class LLMDataAdapter:
    """
    Adapter for converting unified data to LLM-compatible format.

    This adapter ensures that data passed to LLM classifiers includes
    all necessary organization information (name, description, website).
    """

    def __init__(self):
        """Initialize LLM data adapter."""
        self.data_access = DataAccessLayer()

    def transform(
        self, data: Union[pd.DataFrame, List[int], List[OrganizationData]]
    ) -> List[OrganizationData]:
        """
        Transform input data to OrganizationData format.

        Parameters
        ----------
        data : DataFrame, list of ASNs, or list of OrganizationData
            Input data to transform.

        Returns
        -------
        List[OrganizationData]
            List of organization data objects with complete information.
        """
        # If already OrganizationData, ensure completeness
        if (
            isinstance(data, list)
            and len(data) > 0
            and isinstance(data[0], OrganizationData)
        ):
            return self._ensure_complete_org_data(data)

        # Extract ASNs
        if isinstance(data, pd.DataFrame):
            if "asn" in data.columns:
                asns = data["asn"].tolist()
            elif data.index.name == "asn":
                asns = data.index.tolist()
            else:
                raise ValueError("DataFrame must have ASN information")
        elif isinstance(data, list):
            asns = data
        else:
            raise ValueError("Invalid data format")

        # Fetch organization data
        logger.info(f"Fetching organization data for {len(asns)} ASNs")
        organizations = []

        for asn in asns:
            org_data = self.data_access.get_organization_data(asn)
            if org_data:
                organizations.append(org_data)
            else:
                # Create minimal organization data
                org_name = self.data_access.get_organization_name(asn)
                organizations.append(
                    OrganizationData(
                        asn=asn,
                        name=org_name or f"AS{asn}",
                        description=None,
                        website=None,
                    )
                )

        return organizations

    def _ensure_complete_org_data(
        self, organizations: List[OrganizationData]
    ) -> List[OrganizationData]:
        """
        Ensure organization data has all required fields.

        Parameters
        ----------
        organizations : List[OrganizationData]
            List of organization data objects.

        Returns
        -------
        List[OrganizationData]
            Organizations with complete information.
        """
        complete_orgs = []

        for org in organizations:
            # Check if we need to fetch additional data
            if not org.name or org.name == f"AS{org.asn}":
                # Try to get better data
                better_org = self.data_access.get_organization_data(org.asn)
                if better_org:
                    complete_orgs.append(better_org)
                else:
                    complete_orgs.append(org)
            else:
                complete_orgs.append(org)

        return complete_orgs

    def create_llm_input(self, organizations: List[OrganizationData]) -> pd.DataFrame:
        """
        Create DataFrame suitable for LLM batch processing.

        Parameters
        ----------
        organizations : List[OrganizationData]
            List of organization data objects.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns expected by LLM inference.
        """
        data = []
        for org in organizations:
            # Create description from available data
            description_parts = []

            if org.description:
                description_parts.append(org.description)

            if org.ipinfo and org.ipinfo.org:
                description_parts.append(f"IPinfo: {org.ipinfo.org}")

            if org.peeringdb and org.peeringdb.info_type:
                description_parts.append(f"Type: {org.peeringdb.info_type}")

            description = " | ".join(description_parts) if description_parts else ""

            data.append(
                {
                    "asn": org.asn,
                    "name": org.name or f"AS{org.asn}",
                    "description": description,
                    "website": org.website or "",
                    "country": getattr(org.ipinfo, "country", "") if org.ipinfo else "",
                }
            )

        return pd.DataFrame(data)


class UnifiedDataAdapter:
    """
    Unified adapter that handles conversion between all data formats.

    This adapter provides a single interface for data conversion across
    the entire two-stage pipeline.
    """

    def __init__(self):
        """Initialize unified data adapter."""
        self.svm_adapter = SVMDataAdapter()
        self.llm_adapter = LLMDataAdapter()
        self.data_access = DataAccessLayer()

    def prepare_for_svm(
        self, data: Union[pd.DataFrame, List[int], List[OrganizationData]]
    ) -> pd.DataFrame:
        """
        Prepare data for SVM classification.

        Parameters
        ----------
        data : Various formats
            Input data to prepare.

        Returns
        -------
        pd.DataFrame
            Feature matrix for SVM.
        """
        return self.svm_adapter.transform(data)

    def prepare_for_llm(
        self, data: Union[pd.DataFrame, List[int], List[OrganizationData]]
    ) -> List[OrganizationData]:
        """
        Prepare data for LLM classification.

        Parameters
        ----------
        data : Various formats
            Input data to prepare.

        Returns
        -------
        List[OrganizationData]
            Organization data for LLM.
        """
        return self.llm_adapter.transform(data)

    def prepare_for_stacking(
        self, data: Union[pd.DataFrame, List[int], List[OrganizationData]]
    ) -> Dict[str, Union[pd.DataFrame, List[OrganizationData]]]:
        """
        Prepare data for stacking classifier.

        Parameters
        ----------
        data : Various formats
            Input data to prepare.

        Returns
        -------
        dict
            Dictionary with 'svm' and 'llm' prepared data.
        """
        return {"svm": self.prepare_for_svm(data), "llm": self.prepare_for_llm(data)}

    def extract_asns(
        self, data: Union[pd.DataFrame, List[int], List[OrganizationData]]
    ) -> List[int]:
        """
        Extract ASN list from various data formats.

        Parameters
        ----------
        data : Various formats
            Input data.

        Returns
        -------
        List[int]
            List of ASNs.
        """
        if isinstance(data, pd.DataFrame):
            if "asn" in data.columns:
                return data["asn"].tolist()
            elif data.index.name == "asn":
                return data.index.tolist()
            else:
                raise ValueError("DataFrame must have ASN information")
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], OrganizationData):
                return [org.asn for org in data]
            else:
                return data
        else:
            raise ValueError("Invalid data format")
