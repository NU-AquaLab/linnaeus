"""
Data access layer for Linnaeus.

This module provides a unified interface for accessing data from multiple sources
including ASRank, PeeringDB, ASPOP, IPinfo, and other Internet infrastructure datasets.
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import get_config
from ..models.schemas import (
    ASPOPData,
    ASRankData,
    IPinfoData,
    OrganizationData,
    PeeringDBData,
)


class DataAccessLayer:
    """
    Unified data access layer for all AS organization data sources.

    This class provides methods to retrieve data for individual ASNs or
    bulk datasets from various sources including ASRank, PeeringDB, ASPOP,
    and other Internet infrastructure datasets.
    """

    def __init__(
        self, data_dir: Optional[Path] = None, enable_auto_download: bool = False
    ):
        """
        Initialize the data access layer.

        Parameters
        ----------
        data_dir : Path, optional
            Base directory for data files. If None, uses config default.
        enable_auto_download : bool
            Whether to automatically download missing data when requested.
        """
        self.config = get_config()
        self.data_dir = data_dir or self.config.data.processed_data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.enable_auto_download = enable_auto_download

        # Cached data
        self._asrank_cache: Optional[Dict] = None
        self._peeringdb_cache: Optional[Dict] = None
        self._aspop_cache: Optional[Dict] = None
        self._ipinfo_cache: Optional[Dict] = None

        # Downloaders (lazy loaded)
        self._ipinfo_downloader = None

    def get_asrank_data(self, asn: int) -> Optional[ASRankData]:
        """
        Get ASRank data for a specific ASN.

        Parameters
        ----------
        asn : int
            Autonomous System Number.

        Returns
        -------
        ASRankData or None
            ASRank data for the ASN if available, otherwise None.
        """
        if self._asrank_cache is None:
            self._load_asrank_cache()

        if self._asrank_cache and str(asn) in self._asrank_cache:
            raw_data = self._asrank_cache[str(asn)]
            return self._parse_asrank_data(raw_data)

        return None

    def get_peeringdb_data(self, asn: int) -> Optional[PeeringDBData]:
        """
        Get PeeringDB network data for a specific ASN.

        Parameters
        ----------
        asn : int
            Autonomous System Number.

        Returns
        -------
        PeeringDBData or None
            PeeringDB data for the ASN if available, otherwise None.
        """
        if self._peeringdb_cache is None:
            self._load_peeringdb_cache()

        if self._peeringdb_cache and str(asn) in self._peeringdb_cache:
            raw_data = self._peeringdb_cache[str(asn)]
            return PeeringDBData(asn=asn, **raw_data)

        return None

    def get_aspop_data(self, asn: int) -> Optional[ASPOPData]:
        """
        Get APNIC AS Population data for a specific ASN.

        Parameters
        ----------
        asn : int
            Autonomous System Number.

        Returns
        -------
        ASPOPData or None
            ASPOP data for the ASN if available, otherwise None.
        """
        if self._aspop_cache is None:
            self._load_aspop_cache()

        if self._aspop_cache and str(asn) in self._aspop_cache:
            raw_data = self._aspop_cache[str(asn)]
            return ASPOPData(asn=asn, **raw_data)

        return None

    def get_ipinfo_data(
        self, asn: int, auto_download: Optional[bool] = None
    ) -> Optional[IPinfoData]:
        """
        Get IPinfo data for a specific ASN.

        Parameters
        ----------
        asn : int
            Autonomous System Number.
        auto_download : bool, optional
            Override the instance auto_download setting for this request.

        Returns
        -------
        IPinfoData or None
            IPinfo data for the ASN if available, otherwise None.
        """
        if self._ipinfo_cache is None:
            self._load_ipinfo_cache()

        # Check if data exists in cache
        if self._ipinfo_cache and str(asn) in self._ipinfo_cache:
            raw_data = self._ipinfo_cache[str(asn)]
            return IPinfoData(asn=asn, **raw_data)

        # If not found and auto-download is enabled, try to fetch it
        should_download = (
            auto_download if auto_download is not None else self.enable_auto_download
        )
        if should_download:
            try:
                # Download data for this specific ASN
                downloaded_data = self._download_ipinfo_asn(asn)
                if downloaded_data:
                    # Add to cache
                    if self._ipinfo_cache is None:
                        self._ipinfo_cache = {}
                    self._ipinfo_cache[str(asn)] = downloaded_data

                    # Return the data
                    return IPinfoData(asn=asn, **downloaded_data)
            except Exception as e:
                # Log error but don't fail - just return None
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to auto-download IPinfo data for ASN {asn}: {e}"
                )

        return None

    def get_organization_data(self, asn: int) -> Optional[OrganizationData]:
        """
        Get combined organization data for a specific ASN from all sources.

        Parameters
        ----------
        asn : int
            Autonomous System Number.

        Returns
        -------
        OrganizationData or None
            Combined organization data if any source has data, otherwise None.
        """
        asrank_data = self.get_asrank_data(asn)
        peeringdb_data = self.get_peeringdb_data(asn)
        aspop_data = self.get_aspop_data(asn)
        ipinfo_data = self.get_ipinfo_data(asn)

        # Return None if no data found from any source
        if not any([asrank_data, peeringdb_data, aspop_data, ipinfo_data]):
            return None

        return OrganizationData(
            asn=asn,
            asrank=asrank_data,
            peeringdb=peeringdb_data,
            aspop=aspop_data,
            ipinfo=ipinfo_data,
        )

    def get_organization_name(self, asn: int) -> Optional[str]:
        """
        Get organization name for a specific ASN with fallback priority.

        Priority order: ASRank -> PeeringDB -> IPinfo -> ASPOP

        Parameters
        ----------
        asn : int
            Autonomous System Number.

        Returns
        -------
        str or None
            Organization name if found, otherwise None.
        """
        # Try ASRank first
        asrank_data = self.get_asrank_data(asn)
        if asrank_data and asrank_data.organization_name:
            return asrank_data.organization_name

        # Try PeeringDB second
        peeringdb_data = self.get_peeringdb_data(asn)
        if peeringdb_data and peeringdb_data.organization:
            return peeringdb_data.organization

        # Try IPinfo third - enhanced fallback strategy
        ipinfo_data = self.get_ipinfo_data(asn)
        if ipinfo_data:
            # Prefer 'name' field if available (clean organization name)
            if ipinfo_data.name:
                return ipinfo_data.name
            # Fall back to 'org' field if name is not available
            elif ipinfo_data.org:
                # Clean up org field (remove AS prefix if present)
                org_name = ipinfo_data.org
                if org_name.startswith(f"AS{asn} "):
                    org_name = org_name[len(f"AS{asn} ") :]
                elif org_name.startswith("AS") and " " in org_name:
                    # Handle cases like "AS1234 Company Name"
                    parts = org_name.split(" ", 1)
                    if len(parts) > 1 and parts[0].startswith("AS"):
                        org_name = parts[1]
                return org_name

        # Try ASPOP last
        aspop_data = self.get_aspop_data(asn)
        if aspop_data and aspop_data.name:
            return aspop_data.name

        return None

    def get_bulk_data(self, asn_list: List[int]) -> List[OrganizationData]:
        """
        Get organization data for multiple ASNs efficiently.

        Parameters
        ----------
        asn_list : List[int]
            List of Autonomous System Numbers.

        Returns
        -------
        List[OrganizationData]
            List of organization data objects. Missing ASNs are excluded.
        """
        results = []
        for asn in asn_list:
            org_data = self.get_organization_data(asn)
            if org_data:
                results.append(org_data)

        return results

    def get_complete_dataset(self) -> pd.DataFrame:
        """
        Load and combine all available data sources into a single DataFrame.

        This method creates a comprehensive dataset with all ASNs and their
        associated data from all sources, with prefixed column names.

        Returns
        -------
        pd.DataFrame
            Combined dataset with all available data.
        """
        # Load all data sources
        if self._asrank_cache is None:
            self._load_asrank_cache()
        if self._peeringdb_cache is None:
            self._load_peeringdb_cache()
        if self._aspop_cache is None:
            self._load_aspop_cache()
        if self._ipinfo_cache is None:
            self._load_ipinfo_cache()

        # Get all unique ASNs
        all_asns = set()
        if self._asrank_cache:
            all_asns.update(int(asn) for asn in self._asrank_cache.keys())
        if self._peeringdb_cache:
            all_asns.update(int(asn) for asn in self._peeringdb_cache.keys())
        if self._aspop_cache:
            all_asns.update(int(asn) for asn in self._aspop_cache.keys())
        if self._ipinfo_cache:
            all_asns.update(int(asn) for asn in self._ipinfo_cache.keys())

        # Build combined dataset
        combined_data = {}
        for asn in all_asns:
            combined_data[asn] = {}

            # Add ASRank data with prefix
            asrank_data = self.get_asrank_data(asn)
            if asrank_data:
                for field, value in asrank_data.model_dump().items():
                    if field != "asn":
                        combined_data[asn][f"asrank_{field}"] = value

            # Add PeeringDB data with prefix
            peeringdb_data = self.get_peeringdb_data(asn)
            if peeringdb_data:
                for field, value in peeringdb_data.model_dump().items():
                    if field != "asn":
                        combined_data[asn][f"peeringdb_{field}"] = value

            # Add ASPOP data with prefix
            aspop_data = self.get_aspop_data(asn)
            if aspop_data:
                for field, value in aspop_data.model_dump().items():
                    if field != "asn":
                        combined_data[asn][f"aspop_{field}"] = value

            # Add IPinfo data with prefix
            ipinfo_data = self.get_ipinfo_data(asn)
            if ipinfo_data:
                for field, value in ipinfo_data.model_dump().items():
                    if field != "asn":
                        combined_data[asn][f"ipinfo_{field}"] = value

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(combined_data, orient="index")
        df.index.name = "asn"
        return df

    def _load_asrank_cache(self) -> None:
        """Load ASRank data cache from CSV file."""
        asrank_file = self.data_dir / "as_rank_features.csv"
        if asrank_file.exists():
            self._asrank_cache = {}
            with open(asrank_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    asn = row.get("asn", row.get("ASN", ""))
                    if asn:
                        self._asrank_cache[asn] = row

    def _load_peeringdb_cache(self) -> None:
        """Load PeeringDB data cache from JSON file."""
        peeringdb_file = self.data_dir / "peeringdb_net.json"
        if peeringdb_file.exists():
            with open(peeringdb_file, "r", encoding="utf-8") as f:
                self._peeringdb_cache = json.load(f)

    def _load_aspop_cache(self) -> None:
        """Load ASPOP data cache from JSON file."""
        aspop_file = self.data_dir / "aspop_processed.json"
        if aspop_file.exists():
            with open(aspop_file, "r", encoding="utf-8") as f:
                self._aspop_cache = json.load(f)

    def _load_ipinfo_cache(self) -> None:
        """Load IPinfo data cache from JSON file."""
        ipinfo_file = self.data_dir / "ipinfo_processed.json"
        if ipinfo_file.exists():
            with open(ipinfo_file, "r", encoding="utf-8") as f:
                self._ipinfo_cache = json.load(f)

    def _parse_asrank_data(self, raw_data: Dict) -> ASRankData:
        """
        Parse raw ASRank data into structured ASRankData model.

        Parameters
        ----------
        raw_data : Dict
            Raw data from ASRank CSV.

        Returns
        -------
        ASRankData
            Parsed and validated ASRank data.
        """
        # Map CSV columns to our data model
        return ASRankData(
            asn=int(raw_data.get("asn", raw_data.get("ASN", 0))),
            asn_name=raw_data.get("asnName"),
            rank=int(raw_data.get("rank", 0)) if raw_data.get("rank") else None,
            organization_id=raw_data.get("orgId"),
            organization_name=raw_data.get("orgName"),
            clique_member=raw_data.get("cliqueMember") == "true",
            seen=raw_data.get("seen") == "true",
            longitude=(
                float(raw_data.get("longitude", 0))
                if raw_data.get("longitude")
                else None
            ),
            latitude=(
                float(raw_data.get("latitude", 0)) if raw_data.get("latitude") else None
            ),
            country_iso=raw_data.get("country_iso"),
            country_name=raw_data.get("country_name"),
            cone_asns=(
                int(raw_data.get("cone_numberAsns", 0))
                if raw_data.get("cone_numberAsns")
                else None
            ),
            cone_prefixes=(
                int(raw_data.get("cone_numberPrefixes", 0))
                if raw_data.get("cone_numberPrefixes")
                else None
            ),
            cone_addresses=(
                int(raw_data.get("cone_numberAddresses", 0))
                if raw_data.get("cone_numberAddresses")
                else None
            ),
            degree_provider=(
                int(raw_data.get("degree_provider", 0))
                if raw_data.get("degree_provider")
                else None
            ),
            degree_peer=(
                int(raw_data.get("degree_peer", 0))
                if raw_data.get("degree_peer")
                else None
            ),
            degree_customer=(
                int(raw_data.get("degree_customer", 0))
                if raw_data.get("degree_customer")
                else None
            ),
            degree_total=(
                int(raw_data.get("degree_total", 0))
                if raw_data.get("degree_total")
                else None
            ),
            degree_transit=(
                int(raw_data.get("degree_transit", 0))
                if raw_data.get("degree_transit")
                else None
            ),
            degree_sibling=(
                int(raw_data.get("degree_sibling", 0))
                if raw_data.get("degree_sibling")
                else None
            ),
            announcing_prefixes=(
                int(raw_data.get("announcing_numberPrefixes", 0))
                if raw_data.get("announcing_numberPrefixes")
                else None
            ),
            announcing_addresses=(
                int(raw_data.get("announcing_numberAddresses", 0))
                if raw_data.get("announcing_numberAddresses")
                else None
            ),
        )

    def clear_cache(self) -> None:
        """Clear all cached data to force reload on next access."""
        self._asrank_cache = None
        self._peeringdb_cache = None
        self._aspop_cache = None
        self._ipinfo_cache = None

    def _get_ipinfo_downloader(self):
        """Get or create IPinfo downloader instance."""
        if self._ipinfo_downloader is None:
            from .downloaders import IPinfoDownloader

            # Use raw data directory for downloads
            raw_data_dir = self.config.data.raw_data_dir
            self._ipinfo_downloader = IPinfoDownloader(raw_data_dir)
        return self._ipinfo_downloader

    def _download_ipinfo_asn(self, asn: int) -> Optional[Dict]:
        """
        Download IPinfo data for a specific ASN.

        Parameters
        ----------
        asn : int
            Autonomous System Number to download.

        Returns
        -------
        Optional[Dict]
            Downloaded data or None if failed.
        """
        try:
            downloader = self._get_ipinfo_downloader()

            # Download data synchronously
            raw_data = asyncio.run(downloader.download_asn(asn))

            if raw_data and "error" not in raw_data:
                # Process the raw data to match our expected format
                from .processors import DataProcessor

                processor = DataProcessor(
                    input_dir=self.config.data.raw_data_dir, output_dir=self.data_dir
                )

                # Use the _process_ipinfo_record method from processors
                processed_data = processor._process_ipinfo_record(raw_data)
                return processed_data

        except Exception as e:
            import logging

            logging.getLogger(__name__).error(
                f"Failed to download IPinfo data for ASN {asn}: {e}"
            )

        return None

    def validate_all_cached_data(self) -> Dict[str, Any]:
        """Force validation of all currently cached data sources."""
        results = {}
        validator = self._get_validator()

        # Check each data source
        sources_files = {
            "asrank": self.data_dir / "as_rank_features.csv",
            "peeringdb": self.data_dir / "peeringdb_net.json",
            "aspop": self.data_dir / "aspop_processed.json",
            "ipinfo": self.data_dir / "ipinfo_processed.json",
        }

        for source, file_path in sources_files.items():
            if file_path.exists():
                try:
                    if source == "asrank":
                        result = validator.validate_asrank_data(file_path)
                    elif source == "peeringdb":
                        result = validator.validate_peeringdb_data(file_path)
                    elif source == "aspop":
                        result = validator.validate_aspop_data(file_path)
                    elif source == "ipinfo":
                        result = validator.validate_ipinfo_data(file_path)

                    results[source] = result
                    self._validation_results[source] = result

                except Exception as e:
                    import logging

                    logging.getLogger(__name__).error(
                        f"Failed to validate {source}: {e}"
                    )

        return results
