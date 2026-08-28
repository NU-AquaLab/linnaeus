"""
Data alignment utilities for loading clean, normalized AS classification data.

Provides:
- ``ColumnNormalizer``: static helpers to normalize ASN values
  and category column names.
- ``AlignedDataset``: read-only accessor for a release snapshot inside
  ``data/released/<YYYYMM>/``, produced by ``scripts/migrate_data.py``.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Current release tag – bump when a new data snapshot is committed.
# ---------------------------------------------------------------------------
CURRENT_RELEASE = "202506"

# ---------------------------------------------------------------------------
# Canonical column lists – single source of truth for the 20 top-level and
# 48 sub-level category names used throughout the project.
# ---------------------------------------------------------------------------

CANONICAL_TOPLEVEL: List[str] = [
    "Access",
    "Transit",
    "Mobile",
    "Satellite",
    "ContentProvider",
    "EducationalResearch",
    "Government",
    "IXP",
    "DNS",
    "EnergyAndUtility",
    "Enterprise",
    "Finance",
    "LawEnforcement",
    "Health",
    "Cooperatives",
    "TvRadioCulturalAmenities",
    "Transportation",
    "VPNs",
    "Personal",
    "Community",
]

CANONICAL_SUBLEVEL: List[str] = [
    # Access
    "Access_LargeISP",
    "Access_SmallISP",
    # Transit
    "Transit_Global",
    "Transit_Regional",
    "Transit_Domestic",
    # Single-level categories
    "Mobile",
    "Satellite",
    # ContentProvider
    "ContentProvider_Cloud",
    "ContentProvider_Hosting",
    "ContentProvider_CDN",
    # EducationalResearch
    "EducationalResearch_University",
    "EducationalResearch_AcademicBackbone",
    "EducationalResearch_Schools",
    "EducationalResearch_ResearchInstitutes",
    # Government
    "Government_Executive",
    "Government_Legislative",
    "Government_Judiciary",
    "Government_FederalNational",
    "Government_StateProvince",
    "Government_CityCountyMunicipality",
    "Government_Regulators",
    "Government_Agencies",
    # IXP
    "IXP",
    # DNS
    "DNS_Roots",
    "DNS_ccTLD",
    "DNS_ANS",
    # EnergyAndUtility
    "EnergyAndUtility",
    # Enterprise
    "Enterprise_Ecommerce",
    "Enterprise_Entertainment",
    "Enterprise_IndustrialManufacturing",
    "Enterprise_Technology",
    # Finance
    "Finance_Bank",
    "Finance_CentralBanks",
    "Finance_CreditUnion",
    "Finance_Entities",
    "Finance_StockExchanges",
    # LawEnforcement
    "LawEnforcement",
    # Health
    "Health_Insurances",
    "Health_Hospitals",
    # Cooperatives
    "Cooperatives",
    # TvRadioCulturalAmenities
    "TvRadioCulturalAmenities",
    # Transportation
    "Transportation_Trains",
    "Transportation_Ships",
    "Transportation_TransitAuthority",
    "Transportation_Airports",
    # VPNs / Personal / Community
    "VPNs",
    "Personal",
    "Community",
]

# Legacy column name → canonical column name
LEGACY_TO_CANONICAL: Dict[str, str] = {
    # TopLevel rename
    "Financial": "Finance",
    # SubLevel renames
    "Financial_Bank": "Finance_Bank",
    "Financial_CentralBanks": "Finance_CentralBanks",
    "Financial_CreditUnion": "Finance_CreditUnion",
    "Financial_Entities": "Finance_Entities",
    "Financial_StockExchanges": "Finance_StockExchanges",
    "Enterprise_E_commerce": "Enterprise_Ecommerce",
    "Enterprise_Industrial_Manufacturing": "Enterprise_IndustrialManufacturing",
    "Government_National": "Government_FederalNational",
}

# Columns present in legacy data that should be dropped entirely
COLUMNS_TO_DROP = frozenset({"Government_SOE"})


class ColumnNormalizer:
    """Static utilities for normalizing ASN values and category column names."""

    @staticmethod
    def normalize_asn(value: Union[int, float, str]) -> int:
        """Convert an ASN value to a plain integer.

        Handles ``"AS7922"``, ``"7922"``, ``7922``, and ``7922.0``.
        """
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value).strip()
        if s.upper().startswith("AS"):
            s = s[2:]
        return int(s)

    @staticmethod
    def normalize_asn_column(df: pd.DataFrame, col: str = "asn") -> pd.DataFrame:
        """Normalize the ASN column of *df* in-place and return *df*.

        * Renames ``"ASN"`` → *col* if needed.
        * Strips ``"AS"`` prefixes and casts to ``int``.
        """
        df = df.copy()
        if "ASN" in df.columns and col not in df.columns:
            df = df.rename(columns={"ASN": col})
        if col in df.columns:
            df[col] = df[col].apply(ColumnNormalizer.normalize_asn)
        return df

    @staticmethod
    def normalize_category_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename legacy category columns and drop obsolete ones."""
        df = df.copy()
        cols_to_drop = [c for c in df.columns if c in COLUMNS_TO_DROP]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        rename_map = {k: v for k, v in LEGACY_TO_CANONICAL.items() if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    @staticmethod
    def ensure_columns(
        df: pd.DataFrame, canonical: List[str], fill: float = 0.0
    ) -> pd.DataFrame:
        """Add any missing canonical columns (filled with *fill*) and reorder."""
        df = df.copy()
        for col in canonical:
            if col not in df.columns:
                df[col] = fill
        # Keep asn first if present, then canonical cols, then any extras
        non_cat = [c for c in df.columns if c not in canonical]
        return df[non_cat + [c for c in canonical if c in df.columns]]


class AlignedDataset:
    """Read-only accessor for a release snapshot under ``data/released/<YYYYMM>/``.

    Expected layout::

        data/released/202506/
        ├── labels/         toplevel.parquet, sublevel.parquet, hierarchical.parquet
        ├── predictions/
        │   ├── toplevel/   baseline.parquet, finetuned.parquet, svc.parquet
        │   └── sublevel/   baseline.parquet, finetuned.parquet
        ├── splits/         assignments.parquet
        ├── metrics/        *.txt
        └── metadata.json

    Users can also point at ``data/local/`` for user-generated data or at a
    future release like ``data/released/202609/``.
    """

    def __init__(self, data_dir: Union[str, Path] = None) -> None:
        if data_dir is None:
            data_dir = Path("data") / "released" / CURRENT_RELEASE
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def load_labels(self, level: str = "toplevel") -> pd.DataFrame:
        """Load label data from ``data/labels/``.

        Parameters
        ----------
        level : str
            One of ``"toplevel"``, ``"sublevel"``, or ``"hierarchical"``.
        """
        path = self.data_dir / "labels" / f"{level}.parquet"
        logger.info("Loading labels from %s", path)
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def load_features(self, source: str = "all") -> pd.DataFrame:
        """Load feature data.

        Parameters
        ----------
        source : str
            ``"ipinfo"``, ``"peeringdb"``, ``"llm_input"``, or ``"all"``
            (merges on ``asn``).
        """
        feat_dir = self.data_dir / "features"
        if source != "all":
            path = feat_dir / f"{source}.parquet"
            logger.info("Loading features from %s", path)
            return pd.read_parquet(path)

        dfs = []
        for p in sorted(feat_dir.glob("*.parquet")):
            dfs.append(pd.read_parquet(p))
        if not dfs:
            return pd.DataFrame()
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="asn", how="outer", suffixes=("", "_dup"))
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            if dup_cols:
                merged = merged.drop(columns=dup_cols)
        return merged

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def load_predictions(
        self, level: str = "toplevel", model: str = "finetuned"
    ) -> pd.DataFrame:
        """Load prediction data.

        Parameters
        ----------
        level : str
            ``"toplevel"`` or ``"sublevel"``.
        model : str
            ``"baseline"``, ``"finetuned"``, ``"svc"``, or ``"complete"``.
        """
        path = self.data_dir / "predictions" / level / f"{model}.parquet"
        logger.info("Loading predictions from %s", path)
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def load_splits(self) -> pd.DataFrame:
        """Load split assignments (``asn``, ``split``)."""
        path = self.data_dir / "splits" / "assignments.parquet"
        logger.info("Loading splits from %s", path)
        return pd.read_parquet(path)

    def get_split(
        self,
        split_name: str,
        level: str = "toplevel",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(labels, predictions)`` for a named split.

        Parameters
        ----------
        split_name : str
            ``"train"``, ``"val"``, or ``"test"``.
        level : str
            ``"toplevel"`` or ``"sublevel"``.
        """
        splits = self.load_splits()
        asns = splits.loc[splits["split"] == split_name, "asn"]
        labels = self.load_labels(level)
        labels_split = labels[labels["asn"].isin(asns)]
        return labels_split, asns

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Run basic integrity checks on the aligned data directory.

        Returns a list of error strings (empty means all OK).
        """
        errors: List[str] = []

        # Check expected directories (features is optional – may live in data/local/)
        for subdir in ("labels", "predictions", "splits"):
            d = self.data_dir / subdir
            if not d.is_dir():
                errors.append(f"Missing directory: {d}")

        # Check metadata
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            errors.append(f"Missing metadata file: {meta_path}")

        # Validate labels if present
        labels_dir = self.data_dir / "labels"
        if labels_dir.is_dir():
            for name in ("toplevel", "sublevel", "hierarchical"):
                p = labels_dir / f"{name}.parquet"
                if not p.exists():
                    errors.append(f"Missing label file: {p}")
                    continue
                df = pd.read_parquet(p)
                if "asn" not in df.columns:
                    errors.append(f"{p.name}: missing 'asn' column")
                elif not pd.api.types.is_integer_dtype(df["asn"]):
                    errors.append(f"{p.name}: 'asn' column is not integer")

            # TopLevel column check
            tl_path = labels_dir / "toplevel.parquet"
            if tl_path.exists():
                tl = pd.read_parquet(tl_path)
                cat_cols = [c for c in tl.columns if c != "asn"]
                missing = set(CANONICAL_TOPLEVEL) - set(cat_cols)
                if missing:
                    errors.append(
                        f"toplevel.parquet missing columns: {sorted(missing)}"
                    )

        # Validate optional benchmark labels (warn, don't error if missing)
        for bench in ("asdb", "isic"):
            bp = labels_dir / f"{bench}.parquet" if labels_dir.is_dir() else None
            if bp is not None and bp.exists():
                bdf = pd.read_parquet(bp)
                if "asn" not in bdf.columns:
                    errors.append(f"{bp.name}: missing 'asn' column")
                elif not pd.api.types.is_integer_dtype(bdf["asn"]):
                    errors.append(f"{bp.name}: 'asn' column is not integer")

        # Validate splits
        splits_path = self.data_dir / "splits" / "assignments.parquet"
        if splits_path.exists():
            sp = pd.read_parquet(splits_path)
            if "split" not in sp.columns:
                errors.append("assignments.parquet: missing 'split' column")
            else:
                expected_splits = {"train", "val", "test"}
                actual = set(sp["split"].unique())
                if not expected_splits.issubset(actual):
                    errors.append(
                        "assignments.parquet: expected splits"
                        f" {expected_splits}, got {actual}"
                    )

        return errors

    # ------------------------------------------------------------------
    # Taxonomy definitions
    # ------------------------------------------------------------------

    def load_category_definitions(self, taxonomy: str = "linnaeus") -> dict:
        """Load category definitions from the release dir, fallback to package resource.

        Parameters
        ----------
        taxonomy : str
            One of ``"linnaeus"``, ``"asdb"``, or ``"isic"``.
        """
        from ..models.llm.schema_generation import load_tags_descriptions

        release_path = self.data_dir / f"{taxonomy}_definitions.json"
        if release_path.exists():
            return load_tags_descriptions(path=release_path)
        return load_tags_descriptions()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def load_metadata(self) -> dict:
        """Load ``metadata.json``."""
        path = self.data_dir / "metadata.json"
        with open(path) as f:
            return json.load(f)
