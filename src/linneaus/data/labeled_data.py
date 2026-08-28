"""
Labeled data management module for AS classification.

This module handles loading, validation, and conversion of labeled AS data
from the legacy 46-column format to the new hierarchical format with
TopLevelCategory and SubCategory columns.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class LabeledDataManager:
    """Manages labeled AS classification data with hierarchical schema.

    Converts from legacy 46-column format to new format with:
    - ASN: Autonomous System Number
    - ASName: Organization name
    - TopLevelCategory: Main category (Access, Government, etc.)
    - SubCategory: Subcategory (LargeISP, Legislative, etc.) or null
    """

    def __init__(self, data_dir: Path):
        """
        Initialize the hierarchical labeled data manager.

        Parameters
        ----------
        data_dir : Path
            Root data directory containing labeled data.
        """
        self.data_dir = Path(data_dir)
        self.labeled_dir = self.data_dir / "labeled"
        self.consolidated_dir = self.labeled_dir / "consolidated"
        self.splits_dir = self.labeled_dir / "splits"

        # Ensure directories exist
        self.consolidated_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # Define the hierarchical structure mapping
        self.category_mappings = self._create_category_mappings()

        # Define top-level categories
        self.top_level_categories = [
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

    def _create_category_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Create mapping from legacy column names to hierarchical categories.

        Returns
        -------
        Dict[str, Dict[str, str]]
            Mapping from legacy column names to
            {"category": str, "subcategory": str|None}
        """
        # Mapping from legacy column names to hierarchical structure
        mapping = {
            # Access tags
            "Access_LargeISP": {"category": "Access", "subcategory": "LargeISP"},
            "Access_SmallISP": {"category": "Access", "subcategory": "SmallISP"},
            # Transit tags
            "Transit_Global": {"category": "Transit", "subcategory": "Global"},
            "Transit_Regional": {"category": "Transit", "subcategory": "Regional"},
            "Transit_Domestic": {"category": "Transit", "subcategory": "Domestic"},
            # Mobile (single level)
            "Mobile": {"category": "Mobile", "subcategory": None},
            # Satellite (single level)
            "Satellite": {"category": "Satellite", "subcategory": None},
            # Content Provider tags
            "ContentProvider_Cloud": {
                "category": "ContentProvider",
                "subcategory": "Cloud",
            },
            "ContentProvider_Hosting": {
                "category": "ContentProvider",
                "subcategory": "Hosting",
            },
            "ContentProvider_CDN": {
                "category": "ContentProvider",
                "subcategory": "CDN",
            },
            # Educational/Research tags
            "EducationalResearch_University": {
                "category": "EducationalResearch",
                "subcategory": "University",
            },
            "EducationalResearch_AcademicBackbone": {
                "category": "EducationalResearch",
                "subcategory": "AcademicBackbone",
            },
            "EducationalResearch_Schools": {
                "category": "EducationalResearch",
                "subcategory": "Schools",
            },
            "EducationalResearch_ResearchInstitutes": {
                "category": "EducationalResearch",
                "subcategory": "ResearchInstitutes",
            },
            # Government tags
            "Government_Executive": {
                "category": "Government",
                "subcategory": "Executive",
            },
            "Government_Legislative": {
                "category": "Government",
                "subcategory": "Legislative",
            },
            "Government_Judiciary": {
                "category": "Government",
                "subcategory": "Judiciary",
            },
            "Government_National": {
                "category": "Government",
                "subcategory": "FederalNational",
            },
            "Government_StateProvince": {
                "category": "Government",
                "subcategory": "StateProvince",
            },
            "Government_CityCountyMunicipality": {
                "category": "Government",
                "subcategory": "CityCountyMunicipality",
            },
            "Government_Regulators": {
                "category": "Government",
                "subcategory": "Regulators",
            },
            "Government_Agencies": {
                "category": "Government",
                "subcategory": "Agencies",
            },
            # IXP (single level)
            "IXP": {"category": "IXP", "subcategory": None},
            # DNS tags
            "DNS_Roots": {"category": "DNS", "subcategory": "Roots"},
            "DNS_ccTLD": {"category": "DNS", "subcategory": "ccTLD"},
            "DNS_ANS": {"category": "DNS", "subcategory": "ANS"},
            # Energy and Utility (single level)
            "EnergyAndUtility": {"category": "EnergyAndUtility", "subcategory": None},
            # Enterprise tags
            "Enterprise_E_commerce": {
                "category": "Enterprise",
                "subcategory": "Ecommerce",
            },
            "Enterprise_Entertainment": {
                "category": "Enterprise",
                "subcategory": "Entertainment",
            },
            "Enterprise_Industrial_Manufacturing": {
                "category": "Enterprise",
                "subcategory": "IndustrialManufacturing",
            },
            "Enterprise_Technology": {
                "category": "Enterprise",
                "subcategory": "Technology",
            },
            # Financial tags
            "Financial_Bank": {"category": "Finance", "subcategory": "Bank"},
            "Financial_CentralBanks": {
                "category": "Finance",
                "subcategory": "CentralBanks",
            },
            "Financial_CreditUnion": {
                "category": "Finance",
                "subcategory": "CreditUnion",
            },
            "Financial_Entities": {"category": "Finance", "subcategory": "Entities"},
            "Financial_StockExchanges": {
                "category": "Finance",
                "subcategory": "StockExchanges",
            },
            # Law Enforcement (single level)
            "LawEnforcement": {"category": "LawEnforcement", "subcategory": None},
            # Health tags
            "Health_Insurances": {"category": "Health", "subcategory": "Insurances"},
            "Health_Hospitals": {"category": "Health", "subcategory": "Hospitals"},
            # Cooperatives (single level)
            "Cooperatives": {"category": "Cooperatives", "subcategory": None},
            # TV/Radio/Cultural (single level)
            "TvRadioCulturalAmenities": {
                "category": "TvRadioCulturalAmenities",
                "subcategory": None,
            },
            # Transportation tags
            "Transportation_Trains": {
                "category": "Transportation",
                "subcategory": "Trains",
            },
            "Transportation_Ships": {
                "category": "Transportation",
                "subcategory": "Ships",
            },
            "Transportation_TransitAuthority": {
                "category": "Transportation",
                "subcategory": "TransitAuthority",
            },
            "Transportation_Airports": {
                "category": "Transportation",
                "subcategory": "Airports",
            },
            # Personal (single level)
            "Personal": {"category": "Personal", "subcategory": None},
            # VPNs (single level) - not in original data but included for completeness
            "VPNs": {"category": "VPNs", "subcategory": None},
            # Community (single level) - not in original data
            "Community": {"category": "Community", "subcategory": None},
        }

        return mapping

    def load_legacy_labels(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load legacy 46-column labeled data from CSV.

        Parameters
        ----------
        csv_path : Optional[Path]
            Path to the CSV file. If None, tries the new ``data/labels/``
            location first, then falls back to the legacy
            ``data/labeled_data/`` path.

        Returns
        -------
        pd.DataFrame
            DataFrame with ASN and legacy tag columns.
        """
        if csv_path is None:
            from linnaeus.data.alignment import CURRENT_RELEASE

            released_path = (
                self.data_dir / "released" / CURRENT_RELEASE / "labels" / "sublevel.csv"
            )
            flat_path = self.data_dir / "labels" / "sublevel.csv"
            legacy_path = self.data_dir / "labeled_data" / "labeled_data.csv"
            if released_path.exists():
                csv_path = released_path
            elif flat_path.exists():
                csv_path = flat_path
            else:
                csv_path = legacy_path

        logger.info(f"Loading legacy labels from {csv_path}")
        df = pd.read_csv(csv_path)

        # Clean column names (remove any whitespace)
        df.columns = df.columns.str.strip()

        # Released label files use lowercase 'asn'; legacy files use 'ASN'
        if "ASN" not in df.columns and "asn" in df.columns:
            df = df.rename(columns={"asn": "ASN"})

        # Set ASN as index but keep ASName for hierarchical format
        df.set_index("ASN", inplace=True)

        # Convert all tag values to int (handle empty strings)
        tag_columns = [col for col in df.columns if col not in ["ASName / Org."]]
        for col in tag_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        return df

    def convert_to_hierarchical_format(self, legacy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert legacy 46-column format to new hierarchical format.

        Parameters
        ----------
        legacy_df : pd.DataFrame
            DataFrame with legacy tag columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with hierarchical format
            (ASN, ASName, TopLevelCategory, SubCategory).
        """
        logger.info("Converting legacy format to hierarchical format")

        hierarchical_records = []

        for asn, row in legacy_df.iterrows():
            # Get organization name
            asn_name = row.get("ASName / Org.", f"AS{asn}")

            # Process each tag column
            for col_name, value in row.items():
                if col_name == "ASName / Org." or value != 1:
                    continue

                # Map legacy column to hierarchical format
                if col_name in self.category_mappings:
                    mapping = self.category_mappings[col_name]
                    hierarchical_records.append(
                        {
                            "ASN": asn,
                            "ASName": asn_name,
                            "TopLevelCategory": mapping["category"],
                            "SubCategory": mapping["subcategory"],
                        }
                    )
                else:
                    logger.warning(f"Unknown legacy column: {col_name}")

        hierarchical_df = pd.DataFrame(hierarchical_records)

        # Remove duplicate records (same ASN, category, subcategory)
        hierarchical_df.drop_duplicates(
            subset=["ASN", "TopLevelCategory", "SubCategory"], inplace=True
        )

        logger.info(
            "Converted to %d hierarchical records from %d ASNs",
            len(hierarchical_df),
            len(legacy_df),
        )

        return hierarchical_df

    def create_train_val_test_splits(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits for hierarchical data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with hierarchical labels.
        train_size : float
            Proportion of data for training.
        val_size : float
            Proportion of data for validation.
        test_size : float
            Proportion of data for testing.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Train, validation, and test DataFrames.
        """
        assert (
            abs(train_size + val_size + test_size - 1.0) < 1e-6
        ), "Split sizes must sum to 1.0"

        logger.info(
            f"Creating splits: train={train_size}, val={val_size}, test={test_size}"
        )

        # Get unique ASNs for splitting
        unique_asns = df["ASN"].unique()

        # First split: train+val vs test
        train_val_asns, test_asns = train_test_split(
            unique_asns, test_size=test_size, random_state=random_state
        )

        # Second split: train vs val
        train_asns, val_asns = train_test_split(
            train_val_asns,
            test_size=val_size / (train_size + val_size),
            random_state=random_state,
        )

        # Create splits based on ASN membership
        train_df = df[df["ASN"].isin(train_asns)]
        val_df = df[df["ASN"].isin(val_asns)]
        test_df = df[df["ASN"].isin(test_asns)]

        logger.info(
            f"Split sizes - Train: {len(train_df)} records ({len(train_asns)} ASNs), "
            f"Val: {len(val_df)} records ({len(val_asns)} ASNs), "
            f"Test: {len(test_df)} records ({len(test_asns)} ASNs)"
        )

        return train_df, val_df, test_df

    def save_to_parquet(self, df: pd.DataFrame, path: Path) -> None:
        """
        Save DataFrame to Parquet format.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save.
        path : Path
            Output path for Parquet file.
        """
        logger.info(f"Saving to Parquet: {path}")
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

    def save_metadata(self, hierarchical_df: pd.DataFrame) -> None:
        """
        Save metadata about the labeled data.

        Parameters
        ----------
        hierarchical_df : pd.DataFrame
            Hierarchical format DataFrame.
        """
        metadata_path = self.labeled_dir / "metadata.json"
        logger.info(f"Saving metadata to {metadata_path}")

        # Calculate statistics
        unique_asns = hierarchical_df["ASN"].unique()
        category_counts = hierarchical_df["TopLevelCategory"].value_counts().to_dict()
        subcategory_counts = (
            hierarchical_df.dropna(subset=["SubCategory"])["SubCategory"]
            .value_counts()
            .to_dict()
        )

        metadata = {
            "schema_version": "hierarchical_v2",
            "total_asns": len(unique_asns),
            "total_records": len(hierarchical_df),
            "top_level_categories": self.top_level_categories,
            "category_counts": category_counts,
            "subcategory_counts": subcategory_counts,
            "conversion_date": pd.Timestamp.now().isoformat(),
            "legacy_columns_mapped": len(self.category_mappings),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def process_and_save_all(
        self,
        legacy_csv_path: Optional[Path] = None,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> Dict[str, Path]:
        """
        Process legacy labels and save all hierarchical outputs.

        Parameters
        ----------
        legacy_csv_path : Optional[Path]
            Path to legacy CSV file.
        train_size : float
            Proportion of data for training.
        val_size : float
            Proportion of data for validation.
        test_size : float
            Proportion of data for testing.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Path]
            Paths to saved files.
        """
        # Load legacy labels
        legacy_df = self.load_legacy_labels(legacy_csv_path)

        # Convert to hierarchical format
        hierarchical_df = self.convert_to_hierarchical_format(legacy_df)

        # Save consolidated data
        data_path = self.consolidated_dir / "labeled_data.parquet"
        self.save_to_parquet(hierarchical_df, data_path)

        # Create splits
        train_df, val_df, test_df = self.create_train_val_test_splits(
            hierarchical_df, train_size, val_size, test_size, random_state
        )

        # Save splits
        train_path = self.splits_dir / "train.parquet"
        val_path = self.splits_dir / "validation.parquet"
        test_path = self.splits_dir / "test.parquet"

        self.save_to_parquet(train_df, train_path)
        self.save_to_parquet(val_df, val_path)
        self.save_to_parquet(test_df, test_path)

        # Save metadata
        self.save_metadata(hierarchical_df)

        return {
            "data": data_path,
            "train": train_path,
            "validation": val_path,
            "test": test_path,
            "metadata": self.labeled_dir / "metadata.json",
        }

    def load_data(self) -> pd.DataFrame:
        """
        Load labeled data from Parquet.

        Returns
        -------
        pd.DataFrame
            DataFrame with hierarchical labels
            (ASN, ASName, TopLevelCategory, SubCategory).
        """
        path = self.consolidated_dir / "labeled_data.parquet"
        logger.info(f"Loading labels from {path}")
        return pd.read_parquet(path)

    def load_hierarchical_data(self) -> pd.DataFrame:
        """
        Load hierarchical labeled data from Parquet.

        Alias for load_data for backward compatibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with hierarchical labels.
        """
        return self.load_data()

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train/validation/test splits.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Train, validation, and test DataFrames.
        """
        train_df = pd.read_parquet(self.splits_dir / "train.parquet")
        val_df = pd.read_parquet(self.splits_dir / "validation.parquet")
        test_df = pd.read_parquet(self.splits_dir / "test.parquet")

        return train_df, val_df, test_df

    def load_hierarchical_splits(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load hierarchical train/validation/test splits.

        Alias for load_splits for backward compatibility.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Train, validation, and test DataFrames.
        """
        return self.load_splits()

    def validate_data(self, df: pd.DataFrame) -> List[str]:
        """
        Validate data format and consistency.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with data to validate.

        Returns
        -------
        List[str]
            List of validation errors, empty if all valid.
        """
        errors = []

        # Check required columns
        required_columns = ["ASN", "ASName", "TopLevelCategory", "SubCategory"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return errors

        # Check for null ASNs
        null_asns = df["ASN"].isnull().sum()
        if null_asns > 0:
            errors.append(f"Found {null_asns} records with null ASNs")

        # Check for invalid categories
        invalid_categories = df[~df["TopLevelCategory"].isin(self.top_level_categories)]
        if len(invalid_categories) > 0:
            errors.append(
                f"Found {len(invalid_categories)} records with invalid categories"
            )

        # Check category-subcategory combinations
        for _, row in df.iterrows():
            category = row["TopLevelCategory"]
            subcategory = row["SubCategory"]

            # Find expected subcategory for this category
            expected_subcats = []
            for legacy_col, mapping in self.category_mappings.items():
                if mapping["category"] == category:
                    if mapping["subcategory"] is not None:
                        expected_subcats.append(mapping["subcategory"])

            # If category should have subcategories but doesn't, or vice versa
            if expected_subcats and pd.isna(subcategory):
                errors.append(
                    f"Category {category} for ASN {row['ASN']}"
                    " should have subcategory"
                )
            elif not expected_subcats and not pd.isna(subcategory):
                errors.append(
                    f"Category {category} for ASN {row['ASN']}"
                    " should not have subcategory"
                )
            elif expected_subcats and subcategory not in expected_subcats:
                errors.append(
                    f"Invalid subcategory {subcategory}"
                    f" for category {category}"
                    f" (ASN {row['ASN']})"
                )

        return errors


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Process labeled AS data with hierarchical format"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Root data directory"
    )
    parser.add_argument("--input-csv", type=Path, help="Input legacy CSV file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Process data
    manager = LabeledDataManager(args.data_dir)
    paths = manager.process_and_save_all(args.input_csv)

    print("\nProcessing complete! Files saved to:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Validate hierarchical data
    hierarchical_df = manager.load_hierarchical_data()
    errors = manager.validate_hierarchical_data(hierarchical_df)

    if errors:
        print("\nValidation errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nData validation passed!")
