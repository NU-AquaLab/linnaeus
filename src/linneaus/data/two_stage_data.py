"""
Enhanced data management for two-stage hierarchical classification pipeline.

This module provides comprehensive data loading, validation, and preprocessing
capabilities for the two-stage pipeline with proper handling of hierarchical
labels and train/validation/test splits.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from ..models.hierarchical_schemas import TopLevelCategory
from ..models.svm.feature_engineering import ASNFeatureEngineer
from .access import DataAccessLayer
from .labeled_data import LabeledDataManager

logger = logging.getLogger(__name__)


class DataSplitManager:
    """
    Manager for handling train/validation/test splits for two-stage pipeline.

    This class ensures consistent splits across both stages and provides
    proper data validation and preprocessing capabilities.
    """

    def __init__(
        self,
        data_dir: Path,
        random_state: int = 42,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Initialize data split manager.

        Parameters
        ----------
        data_dir : Path
            Root data directory.
        random_state : int
            Random state for reproducible splits.
        train_ratio : float
            Ratio of data for training.
        validation_ratio : float
            Ratio of data for validation.
        test_ratio : float
            Ratio of data for testing.
        """
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        # Validate ratios
        total_ratio = train_ratio + validation_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Initialize components
        self.labeled_data_manager = LabeledDataManager(data_dir)
        self.splits_dir = self.data_dir / "labeled" / "splits"
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded data
        self._train_data: Optional[pd.DataFrame] = None
        self._validation_data: Optional[pd.DataFrame] = None
        self._test_data: Optional[pd.DataFrame] = None
        self._feature_data: Optional[pd.DataFrame] = None

        logger.info(
            f"Initialized DataSplitManager with splits: {train_ratio:.2f}/{validation_ratio:.2f}/{test_ratio:.2f}"
        )

    def create_splits(
        self,
        stratify_by: str = "TopLevelCategory",
        min_samples_per_class: int = 2,
        force_recreate: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits with stratification.

        Parameters
        ----------
        stratify_by : str
            Column to use for stratification.
        min_samples_per_class : int
            Minimum samples required per class for stratification.
        force_recreate : bool
            Whether to force recreation of splits even if they exist.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Training, validation, and test dataframes.
        """
        train_file = self.splits_dir / "train.parquet"
        val_file = self.splits_dir / "validation.parquet"
        test_file = self.splits_dir / "test.parquet"

        # Check if splits already exist
        if not force_recreate and all(
            f.exists() for f in [train_file, val_file, test_file]
        ):
            logger.info("Loading existing data splits")
            train_df = pd.read_parquet(train_file)
            val_df = pd.read_parquet(val_file)
            test_df = pd.read_parquet(test_file)
            return train_df, val_df, test_df

        logger.info("Creating new data splits")

        # Load hierarchical data
        data = self.labeled_data_manager.load_hierarchical_data()
        logger.info(
            f"Loaded {len(data)} records from {data['ASN'].nunique()} unique ASNs"
        )

        # Group by ASN to ensure ASNs don't appear in multiple splits
        asn_groups = data.groupby("ASN").first().reset_index()
        logger.info(f"Created {len(asn_groups)} ASN-level samples for splitting")

        # Prepare stratification labels
        if stratify_by in asn_groups.columns:
            stratify_labels = asn_groups[stratify_by].values

            # Check if we have enough samples per class for stratification
            label_counts = pd.Series(stratify_labels).value_counts()
            min_count = label_counts.min()

            if min_count < min_samples_per_class:
                logger.warning(
                    f"Minimum class count ({min_count}) < {min_samples_per_class}, "
                    f"disabling stratification"
                )
                stratify_labels = None
        else:
            logger.warning(
                f"Stratification column '{stratify_by}' not found, using random split"
            )
            stratify_labels = None

        # Create first split: train vs (validation + test)
        temp_val_test_ratio = self.validation_ratio + self.test_ratio

        train_asns, val_test_asns = train_test_split(
            asn_groups,
            test_size=temp_val_test_ratio,
            random_state=self.random_state,
            stratify=stratify_labels,
        )

        # Create second split: validation vs test
        if len(val_test_asns) > 1:
            # Calculate relative split ratio for val vs test
            relative_test_ratio = self.test_ratio / temp_val_test_ratio

            # Prepare stratification for second split if needed
            val_test_stratify = None
            if stratify_labels is not None:
                val_test_indices = val_test_asns.index
                val_test_stratify = (
                    pd.Series(stratify_labels).iloc[val_test_indices].values
                )

                # Check if we still have enough samples
                val_test_counts = pd.Series(val_test_stratify).value_counts()
                if val_test_counts.min() < 2:
                    val_test_stratify = None

            val_asns, test_asns = train_test_split(
                val_test_asns,
                test_size=relative_test_ratio,
                random_state=self.random_state,
                stratify=val_test_stratify,
            )
        else:
            # Edge case: only one sample for val+test
            val_asns = val_test_asns
            test_asns = pd.DataFrame()

        # Create final splits by filtering original data
        train_df = data[data["ASN"].isin(train_asns["ASN"])].copy()
        val_df = data[data["ASN"].isin(val_asns["ASN"])].copy()
        test_df = (
            data[data["ASN"].isin(test_asns["ASN"])].copy()
            if len(test_asns) > 0
            else pd.DataFrame()
        )

        logger.info(
            f"Created splits - Train: {len(train_df)} records ({train_asns.shape[0]} ASNs), "
            f"Val: {len(val_df)} records ({val_asns.shape[0]} ASNs), "
            f"Test: {len(test_df)} records ({test_asns.shape[0]} ASNs)"
        )

        # Save splits
        train_df.to_parquet(train_file)
        val_df.to_parquet(val_file)
        if len(test_df) > 0:
            test_df.to_parquet(test_file)
        else:
            # Create empty file for consistency
            pd.DataFrame().to_parquet(test_file)

        # Validate splits (no ASN overlap)
        self._validate_splits(train_df, val_df, test_df)

        logger.info("Data splits created and saved successfully")
        return train_df, val_df, test_df

    def load_splits(
        self, split: str = "all"
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Load existing data splits.

        Parameters
        ----------
        split : str
            Which split to load: 'train', 'validation', 'test', or 'all'.

        Returns
        -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
            Requested split(s).
        """
        train_file = self.splits_dir / "train.parquet"
        val_file = self.splits_dir / "validation.parquet"
        test_file = self.splits_dir / "test.parquet"

        if not all(f.exists() for f in [train_file, val_file, test_file]):
            raise FileNotFoundError("Data splits not found. Run create_splits() first.")

        if split == "train":
            return pd.read_parquet(train_file)
        elif split == "validation":
            return pd.read_parquet(val_file)
        elif split == "test":
            return pd.read_parquet(test_file)
        elif split == "all":
            train_df = pd.read_parquet(train_file)
            val_df = pd.read_parquet(val_file)
            test_df = pd.read_parquet(test_file)
            return train_df, val_df, test_df
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'validation', 'test', or 'all'"
            )

    def get_stage1_data(
        self, split: str = "train", include_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get Stage 1 (top-level) classification data.

        Parameters
        ----------
        split : str
            Which split to use.
        include_features : bool
            Whether to include engineered features.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Features (X) and labels (y) for Stage 1.
        """
        # Load split data
        data = self.load_splits(split)

        # Group by ASN to get one sample per organization
        asn_data = (
            data.groupby("ASN")
            .agg(
                {
                    "ASName": "first",
                    "TopLevelCategory": lambda x: list(
                        set(x)
                    ),  # Collect unique categories
                }
            )
            .reset_index()
        )

        # Create binary encoding for top-level categories
        all_categories = [cat.value for cat in TopLevelCategory]
        y_binary = pd.DataFrame(0, index=asn_data.index, columns=all_categories)
        y_binary["asn"] = asn_data["ASN"]

        for idx, row in asn_data.iterrows():
            for category in row["TopLevelCategory"]:
                if category in all_categories:
                    y_binary.loc[idx, category] = 1

        # Prepare features
        if include_features:
            X = self._get_engineered_features(asn_data["ASN"].tolist())
        else:
            X = asn_data[["ASN", "ASName"]].copy()
            X.columns = ["asn", "name"]

        logger.info(
            f"Prepared Stage 1 data: {len(X)} samples, {len(all_categories)} categories"
        )
        return X, y_binary

    def get_stage2_data(
        self, split: str = "train", category: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Get Stage 2 (sublevel) classification data.

        Parameters
        ----------
        split : str
            Which split to use.
        category : Optional[str]
            Specific category to filter by (None for all).

        Returns
        -------
        Tuple[List[Dict[str, Any]], pd.DataFrame]
            Organization data and sublevel labels.
        """
        # Load split data
        data = self.load_splits(split)

        # Filter by category if specified
        if category:
            data = data[data["TopLevelCategory"] == category].copy()

        # Prepare organization data
        organizations = []
        labels = []

        for asn in data["ASN"].unique():
            asn_data = data[data["ASN"] == asn]
            org_name = asn_data["ASName"].iloc[0]

            org_dict = {
                "asn": int(asn),
                "name": org_name,
                "description": f"Autonomous System {asn} operated by {org_name}",
            }
            organizations.append(org_dict)

            # Collect sublevel labels for this ASN
            asn_labels = []
            for _, row in asn_data.iterrows():
                if pd.notna(row["SubCategory"]):
                    asn_labels.append(
                        {
                            "asn": int(asn),
                            "category": row["TopLevelCategory"],
                            "subcategory": row["SubCategory"],
                        }
                    )

            labels.extend(asn_labels)

        # Convert labels to DataFrame
        labels_df = (
            pd.DataFrame(labels)
            if labels
            else pd.DataFrame(columns=["asn", "category", "subcategory"])
        )

        logger.info(
            f"Prepared Stage 2 data: {len(organizations)} organizations, {len(labels_df)} sublevel labels"
        )
        return organizations, labels_df

    def _get_engineered_features(self, asns: List[int]) -> pd.DataFrame:
        """
        Get engineered features for the given ASNs.

        Parameters
        ----------
        asns : List[int]
            List of ASNs to get features for.

        Returns
        -------
        pd.DataFrame
            Feature dataframe.
        """
        # Initialize feature engineer if not already done
        if not hasattr(self, "_feature_engineer"):
            self._feature_engineer = ASNFeatureEngineer(
                include_asrank=True,
                include_peeringdb=True,
                include_aspop=True,
                include_ipinfo=True,
                normalize_features=True,
            )
            logger.info("Initialized ASN feature engineer")

        # Create input DataFrame
        asn_df = pd.DataFrame({"asn": asns})

        try:
            # Extract features using the feature engineering pipeline
            logger.info(f"Extracting features for {len(asns)} ASNs")
            features_df = self._feature_engineer.transform(asn_df)

            # Ensure ASN column is included
            if "asn" not in features_df.columns:
                features_df["asn"] = asns

            # Add organization names for completeness
            if not hasattr(self, "_data_access"):
                self._data_access = DataAccessLayer()

            names = []
            for asn in asns:
                name = self._data_access.get_organization_name(asn)
                names.append(name or f"AS{asn}")

            features_df["name"] = names

            logger.info(
                f"Extracted {len(features_df)} feature vectors with {len(features_df.columns)} features"
            )
            return features_df

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            logger.warning("Falling back to basic features")

            # Fallback to basic features if extraction fails
            return self._get_basic_features(asns)

    def _get_basic_features(self, asns: List[int]) -> pd.DataFrame:
        """
        Get basic fallback features when feature engineering fails.

        Parameters
        ----------
        asns : List[int]
            List of ASNs to get features for.

        Returns
        -------
        pd.DataFrame
            Basic feature dataframe.
        """
        if not hasattr(self, "_data_access"):
            self._data_access = DataAccessLayer()

        features_data = []
        for asn in asns:
            # Get basic organization data
            org_data = self._data_access.get_organization_data(asn)
            org_name = self._data_access.get_organization_name(asn)

            # Extract basic features
            features = {
                "asn": asn,
                "name": org_name or f"AS{asn}",
                "has_peeringdb": 1 if org_data and org_data.peeringdb else 0,
                "has_aspop": 1 if org_data and org_data.aspop else 0,
                "has_ipinfo": 1 if org_data and org_data.ipinfo else 0,
                "has_website": 1 if org_data and org_data.website else 0,
            }

            # Add ASRank features if available
            if org_data and org_data.asrank:
                features.update(
                    {
                        "asrank_rank": getattr(org_data.asrank, "rank", 0) or 0,
                        "asrank_degree": getattr(org_data.asrank, "degree", 0) or 0,
                        "asrank_customer_count": getattr(
                            org_data.asrank, "customer_count", 0
                        )
                        or 0,
                        "asrank_provider_count": getattr(
                            org_data.asrank, "provider_count", 0
                        )
                        or 0,
                        "asrank_peer_count": getattr(org_data.asrank, "peer_count", 0)
                        or 0,
                    }
                )
            else:
                features.update(
                    {
                        "asrank_rank": 0,
                        "asrank_degree": 0,
                        "asrank_customer_count": 0,
                        "asrank_provider_count": 0,
                        "asrank_peer_count": 0,
                    }
                )

            # Add ASPOP features if available
            if org_data and org_data.aspop:
                features.update(
                    {
                        "aspop_population": getattr(org_data.aspop, "population", 0)
                        or 0,
                        "aspop_density": getattr(org_data.aspop, "density", 0.0) or 0.0,
                    }
                )
            else:
                features.update(
                    {
                        "aspop_population": 0,
                        "aspop_density": 0.0,
                    }
                )

            features_data.append(features)

        features_df = pd.DataFrame(features_data)
        logger.info(
            f"Generated {len(features_df)} basic feature vectors with {len(features_df.columns)} features"
        )
        return features_df

    def _validate_splits(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        """
        Validate that splits have no ASN overlap.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data.
        val_df : pd.DataFrame
            Validation data.
        test_df : pd.DataFrame
            Test data.
        """
        train_asns = set(train_df["ASN"].unique())
        val_asns = set(val_df["ASN"].unique())
        test_asns = set(test_df["ASN"].unique()) if len(test_df) > 0 else set()

        # Check for overlaps
        train_val_overlap = train_asns & val_asns
        train_test_overlap = train_asns & test_asns
        val_test_overlap = val_asns & test_asns

        if train_val_overlap:
            raise ValueError(f"Train/validation overlap: {len(train_val_overlap)} ASNs")

        if train_test_overlap:
            raise ValueError(f"Train/test overlap: {len(train_test_overlap)} ASNs")

        if val_test_overlap:
            raise ValueError(f"Validation/test overlap: {len(val_test_overlap)} ASNs")

        logger.info("Data split validation passed - no ASN overlaps found")

    def get_split_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the data splits.

        Returns
        -------
        Dict[str, Any]
            Statistics about splits.
        """
        try:
            train_df, val_df, test_df = self.load_splits()
        except FileNotFoundError:
            return {"error": "Splits not found"}

        stats = {
            "train": {
                "records": len(train_df),
                "unique_asns": train_df["ASN"].nunique(),
                "categories": train_df["TopLevelCategory"].value_counts().to_dict(),
            },
            "validation": {
                "records": len(val_df),
                "unique_asns": val_df["ASN"].nunique(),
                "categories": val_df["TopLevelCategory"].value_counts().to_dict(),
            },
            "test": {
                "records": len(test_df),
                "unique_asns": test_df["ASN"].nunique(),
                "categories": (
                    test_df["TopLevelCategory"].value_counts().to_dict()
                    if len(test_df) > 0
                    else {}
                ),
            },
        }

        return stats

    def export_splits_summary(self, output_path: Path) -> None:
        """
        Export a summary of splits to a file.

        Parameters
        ----------
        output_path : Path
            Path to save the summary.
        """
        stats = self.get_split_statistics()

        with open(output_path, "w") as f:
            f.write("Two-Stage Pipeline Data Splits Summary\n")
            f.write("=" * 40 + "\n\n")

            for split_name, split_stats in stats.items():
                if split_name == "error":
                    f.write(f"Error: {split_stats}\n")
                    continue

                f.write(f"{split_name.upper()} SPLIT:\n")
                f.write(f"  Records: {split_stats['records']}\n")
                f.write(f"  Unique ASNs: {split_stats['unique_asns']}\n")
                f.write(f"  Categories:\n")

                for category, count in split_stats["categories"].items():
                    f.write(f"    {category}: {count}\n")
                f.write("\n")

        logger.info(f"Splits summary exported to {output_path}")


# Export main classes
__all__ = ["DataSplitManager"]
