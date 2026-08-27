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
    """Manages labeled AS classification data."""

    def __init__(self, data_dir: Path):
        """
        Initialize the labeled data manager.

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

        # Define the 20 consolidated tags (Energy and Utility merged)
        self.consolidated_tags = [
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

        # Mapping from hierarchical tags to consolidated tags
        self.tag_mapping = self._create_tag_mapping()

    def _create_tag_mapping(self) -> Dict[str, str]:
        """
        Create mapping from hierarchical tags to consolidated tags.

        Returns
        -------
        Dict[str, str]
            Mapping from detailed tag names to consolidated tag names.
        """
        mapping = {
            # Access tags
            "Access_LargeISP": "Access",
            "Access_SmallISP": "Access",
            # Transit tags
            "Transit_Global": "Transit",
            "Transit_Regional": "Transit",
            "Transit_Domestic": "Transit",
            # Mobile remains Mobile
            "Mobile": "Mobile",
            # Satellite remains Satellite
            "Satellite": "Satellite",
            # Content Provider tags
            "ContentProvider_Cloud": "ContentProvider",
            "ContentProvider_Hosting": "ContentProvider",
            "ContentProvider_CDN": "ContentProvider",
            # Educational/Research tags
            "EducationalResearch_University": "EducationalResearch",
            "EducationalResearch_AcademicBackbone": "EducationalResearch",
            "EducationalResearch_Schools": "EducationalResearch",
            "EducationalResearch_ResearchInstitutes": "EducationalResearch",
            # Government tags
            "Government_Executive": "Government",
            "Government_Legislative": "Government",
            "Government_Judiciary": "Government",
            "Government_National": "Government",
            "Government_StateProvince": "Government",
            "Government_CityCountyMunicipality": "Government",
            "Government_Regulators": "Government",
            "Government_Agencies": "Government",
            # IXP remains IXP
            "IXP": "IXP",
            # DNS tags
            "DNS_Roots": "DNS",
            "DNS_ccTLD": "DNS",
            "DNS_ANS": "DNS",
            # Energy and Utility merged
            "EnergyAndUtility": "EnergyAndUtility",
            # Enterprise tags
            "Enterprise_E_commerce": "Enterprise",
            "Enterprise_Entertainment": "Enterprise",
            "Enterprise_Industrial_Manufacturing": "Enterprise",
            "Enterprise_Technology": "Enterprise",
            # Financial tags
            "Financial_Bank": "Finance",
            "Financial_CentralBanks": "Finance",
            "Financial_CreditUnion": "Finance",
            "Financial_Entities": "Finance",
            "Financial_StockExchanges": "Finance",
            # Law Enforcement remains LawEnforcement
            "LawEnforcement": "LawEnforcement",
            # Health tags
            "Health_Insurances": "Health",
            "Health_Hospitals": "Health",
            # Cooperatives remains Cooperatives
            "Cooperatives": "Cooperatives",
            # TV/Radio/Cultural remains TvRadioCulturalAmenities
            "TvRadioCulturalAmenities": "TvRadioCulturalAmenities",
            # Transportation tags
            "Transportation_Trains": "Transportation",
            "Transportation_Ships": "Transportation",
            "Transportation_TransitAuthority": "Transportation",
            "Transportation_Airports": "Transportation",
            # Personal remains Personal
            "Personal": "Personal",
        }

        return mapping

    def load_hierarchical_labels(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load hierarchical labeled data from CSV.

        Parameters
        ----------
        csv_path : Optional[Path]
            Path to the CSV file. If None, uses default location.

        Returns
        -------
        pd.DataFrame
            DataFrame with ASN and hierarchical tags.
        """
        if csv_path is None:
            csv_path = self.data_dir / "labeled_data" / "labeled_data.csv"

        logger.info(f"Loading hierarchical labels from {csv_path}")
        df = pd.read_csv(csv_path)

        # Clean column names (remove any whitespace)
        df.columns = df.columns.str.strip()

        # Set ASN as index
        df.set_index("ASN", inplace=True)

        # Drop the ASName column if present
        if "ASName / Org." in df.columns:
            df.drop("ASName / Org.", axis=1, inplace=True)

        # Convert all values to int (handle empty strings)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        return df

    def consolidate_labels(self, hierarchical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert hierarchical labels to consolidated 20-tag format.

        Parameters
        ----------
        hierarchical_df : pd.DataFrame
            DataFrame with hierarchical tags.

        Returns
        -------
        pd.DataFrame
            DataFrame with consolidated tags.
        """
        logger.info("Consolidating hierarchical labels to 20 tags")

        # Create empty DataFrame with consolidated tags
        consolidated_df = pd.DataFrame(
            0, index=hierarchical_df.index, columns=self.consolidated_tags
        )

        # Map hierarchical tags to consolidated tags
        for hier_tag, cons_tag in self.tag_mapping.items():
            if hier_tag in hierarchical_df.columns:
                # Use logical OR for consolidation
                consolidated_df[cons_tag] = (
                    consolidated_df[cons_tag] | hierarchical_df[hier_tag]
                ).astype(int)

        return consolidated_df

    def create_train_val_test_splits(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits for the data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with labels.
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

        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            df.index, test_size=test_size, random_state=random_state
        )

        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (train_size + val_size),
            random_state=random_state,
        )

        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[test_idx]

        logger.info(
            f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
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
        df.to_parquet(path, engine="pyarrow", compression="snappy")

    def save_metadata(self, metadata: Dict) -> None:
        """
        Save metadata about the labeled data.

        Parameters
        ----------
        metadata : Dict
            Metadata dictionary to save.
        """
        metadata_path = self.labeled_dir / "metadata.json"
        logger.info(f"Saving metadata to {metadata_path}")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def process_and_save_all(
        self,
        hierarchical_csv_path: Optional[Path] = None,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> Dict[str, Path]:
        """
        Process hierarchical labels and save all outputs.

        Parameters
        ----------
        hierarchical_csv_path : Optional[Path]
            Path to hierarchical CSV file.
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
        # Load hierarchical labels
        hier_df = self.load_hierarchical_labels(hierarchical_csv_path)

        # Consolidate to 20 tags
        cons_df = self.consolidate_labels(hier_df)

        # Save consolidated data
        consolidated_path = self.consolidated_dir / "labeled_data.parquet"
        self.save_to_parquet(cons_df, consolidated_path)

        # Create splits
        train_df, val_df, test_df = self.create_train_val_test_splits(
            cons_df, train_size, val_size, test_size, random_state
        )

        # Save splits
        train_path = self.splits_dir / "train.parquet"
        val_path = self.splits_dir / "validation.parquet"
        test_path = self.splits_dir / "test.parquet"

        self.save_to_parquet(train_df, train_path)
        self.save_to_parquet(val_df, val_path)
        self.save_to_parquet(test_df, test_path)

        # Save metadata
        metadata = {
            "total_asns": len(cons_df),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "tags": self.consolidated_tags,
            "tag_counts": cons_df.sum().to_dict(),
            "split_proportions": {
                "train": train_size,
                "validation": val_size,
                "test": test_size,
            },
            "random_state": random_state,
        }
        self.save_metadata(metadata)

        # Also save hierarchical data for reference
        hier_path = self.consolidated_dir / "labeled_data_hierarchical.parquet"
        self.save_to_parquet(hier_df, hier_path)

        return {
            "consolidated": consolidated_path,
            "hierarchical": hier_path,
            "train": train_path,
            "validation": val_path,
            "test": test_path,
            "metadata": self.labeled_dir / "metadata.json",
        }

    def load_consolidated_labels(self) -> pd.DataFrame:
        """
        Load consolidated labeled data from Parquet.

        Returns
        -------
        pd.DataFrame
            DataFrame with consolidated labels.
        """
        path = self.consolidated_dir / "labeled_data.parquet"
        logger.info(f"Loading consolidated labels from {path}")
        return pd.read_parquet(path)

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

    def validate_label_consistency(self, df: pd.DataFrame) -> List[str]:
        """
        Validate label consistency in the data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with labels to validate.

        Returns
        -------
        List[str]
            List of validation errors, empty if all valid.
        """
        errors = []

        # Check that all values are 0 or 1
        invalid_values = ((df != 0) & (df != 1)).any(axis=1)
        if invalid_values.any():
            invalid_asns = df[invalid_values].index.tolist()
            errors.append(f"ASNs with non-binary values: {invalid_asns}")

        # Check for ASNs with no labels
        no_labels = df.sum(axis=1) == 0
        if no_labels.any():
            unlabeled_asns = df[no_labels].index.tolist()
            errors.append(f"ASNs with no labels: {unlabeled_asns}")

        return errors

    def export_to_formats(
        self,
        df: pd.DataFrame,
        base_path: Path,
        formats: List[str] = ["json", "csv", "parquet"],
    ) -> Dict[str, Path]:
        """
        Export DataFrame to multiple formats.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to export.
        base_path : Path
            Base path without extension.
        formats : List[str]
            List of formats to export to.

        Returns
        -------
        Dict[str, Path]
            Mapping of format to output path.
        """
        output_paths = {}

        for fmt in formats:
            if fmt == "json":
                path = base_path.with_suffix(".json")
                df.to_json(path, orient="index", indent=2)
            elif fmt == "csv":
                path = base_path.with_suffix(".csv")
                df.to_csv(path)
            elif fmt == "parquet":
                path = base_path.with_suffix(".parquet")
                df.to_parquet(path, engine="pyarrow", compression="snappy")
            else:
                logger.warning(f"Unknown format: {fmt}")
                continue

            output_paths[fmt] = path
            logger.info(f"Exported to {fmt}: {path}")

        return output_paths


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Process labeled AS data")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Root data directory"
    )
    parser.add_argument("--input-csv", type=Path, help="Input hierarchical CSV file")

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
