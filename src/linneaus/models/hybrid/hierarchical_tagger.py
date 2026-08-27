"""
Modernized hierarchical AS tagger based on the StackingClassifier_ignacio branch.

This module provides an updated version of the HierarchicalASNTagger that
integrates with the new unified schema system and modern codebase structure.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from ...data.access import DataAccessLayer
from ..svm.svm_models import SVMClassifier
from ..unified_schemas import HierarchicalTags, TopLevelTags, UnifiedASClassification

logger = logging.getLogger(__name__)


class ModernHierarchicalTagger(BaseEstimator, ClassifierMixin):
    """
    Modern two-stage hierarchical tagger for Autonomous Systems.

    This classifier follows the original hierarchical approach:
    1. Predict general categories (Access, Transit, Government, etc.)
    2. For each predicted general category, apply specialized sub-models

    Updated to work with the unified schema system and modern infrastructure.
    """

    def __init__(
        self,
        general_model_path: Optional[str] = None,
        submodel_paths: Optional[Dict[str, str]] = None,
        data_access: Optional[DataAccessLayer] = None,
    ):
        """
        Initialize hierarchical tagger.

        Parameters
        ----------
        general_model_path : str, optional
            Path to the general category classifier model file.
        submodel_paths : Dict[str, str], optional
            Mapping from general category name to specialized model path.
        data_access : DataAccessLayer, optional
            Data access layer for feature extraction.
        """
        self.general_model_path = general_model_path
        self.submodel_paths = submodel_paths or {}
        self.data_access = data_access or DataAccessLayer()

        # Model instances
        self.general_model: Optional[SVMClassifier] = None
        self.specialized_models: Dict[str, SVMClassifier] = {}

        # Fitted attributes
        self.classes_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def load_models(self) -> None:
        """Load all models from disk into memory."""
        logger.info("Loading hierarchical models")

        # Load general model
        if self.general_model_path and os.path.exists(self.general_model_path):
            logger.info(f"Loading general model from {self.general_model_path}")
            self.general_model = SVMClassifier(approach="flat")
            self.general_model.load_model(self.general_model_path)
        else:
            logger.warning("General model path not provided or does not exist")

        # Load specialized models
        for category, model_path in self.submodel_paths.items():
            if os.path.exists(model_path):
                logger.info(
                    f"Loading specialized model for {category} from {model_path}"
                )
                specialized_model = SVMClassifier(approach="hierarchical")
                specialized_model.load_model(model_path)
                self.specialized_models[category] = specialized_model
            else:
                logger.warning(
                    f"Specialized model for {category} not found at {model_path}"
                )

        self.is_fitted_ = True
        logger.info(
            f"Loaded general model and {len(self.specialized_models)} specialized models"
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "ModernHierarchicalTagger":
        """
        Fit the hierarchical tagger.

        This method trains both general and specialized models from scratch.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing ASN and features.
        y : pd.DataFrame
            Label DataFrame with hierarchical tag columns.

        Returns
        -------
        self
            Fitted hierarchical tagger.
        """
        logger.info("Training hierarchical tagger from scratch")

        # Train general model for top-level categories
        logger.info("Training general classifier")
        y_general = self._create_general_labels(y)

        self.general_model = SVMClassifier(approach="flat")
        self.general_model.fit(X, y_general)

        # Train specialized models for each general category
        for general_tag in TopLevelTags:
            category_name = general_tag.value
            logger.info(f"Training specialized model for {category_name}")

            # Get samples that belong to this general category
            general_mask = y_general[category_name] == 1
            if general_mask.sum() == 0:
                logger.warning(f"No training samples for {category_name}")
                continue

            # Create specialized labels for this category
            y_specialized = self._create_specialized_labels(
                y[general_mask], general_tag
            )

            if y_specialized.empty or y_specialized.sum().sum() == 0:
                logger.warning(f"No specialized labels for {category_name}")
                continue

            # Train specialized model
            X_specialized = X[general_mask]
            specialized_model = SVMClassifier(approach="hierarchical")
            specialized_model.fit(X_specialized, y_specialized)

            self.specialized_models[category_name] = specialized_model

        # Set fitted attributes
        self.classes_ = list(y.columns)
        if hasattr(self.general_model, "feature_names_"):
            self.feature_names_ = self.general_model.feature_names_
        self.is_fitted_ = True

        logger.info("Hierarchical tagger training completed")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict hierarchical tags for input samples.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing ASN and features.

        Returns
        -------
        pd.DataFrame
            Predictions with general + specialized tag columns.
        """
        if not self.is_fitted_:
            if self.general_model_path:
                self.load_models()
            else:
                raise ValueError(
                    "Model not fitted and no model paths provided. Call fit() first."
                )

        if self.general_model is None:
            raise ValueError("General model not available")

        logger.info(f"Predicting hierarchical tags for {len(X)} samples")

        # Step 1: Predict general categories
        general_predictions = self._predict_general(X)

        # Step 2: Apply specialized models
        specialized_predictions = self._predict_specialized(X, general_predictions)

        # Combine predictions
        result = self._combine_predictions(
            general_predictions, specialized_predictions, X.index
        )

        logger.info("Hierarchical prediction completed")
        return result

    def predict_unified(self, X: pd.DataFrame) -> List[UnifiedASClassification]:
        """
        Predict and return results in unified format.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing ASN.

        Returns
        -------
        List[UnifiedASClassification]
            List of unified classification results.
        """
        predictions_df = self.predict(X)
        results = []

        for idx, row in predictions_df.iterrows():
            # Extract ASN
            asn = (
                int(idx)
                if isinstance(idx, (int, np.integer))
                else int(row.get("asn", 0))
            )

            # Get predicted tags
            top_level_tags = []
            hierarchical_tags = []

            for col, value in row.items():
                if col == "asn":
                    continue

                if value > 0:  # Positive prediction
                    # Check if it's a hierarchical tag
                    hierarchical_tag = None
                    for tag in HierarchicalTags:
                        if tag.value == col:
                            hierarchical_tag = tag
                            break

                    if hierarchical_tag:
                        hierarchical_tags.append(hierarchical_tag)
                    else:
                        # Check if it's a top-level tag
                        top_level_tag = None
                        for tag in TopLevelTags:
                            if tag.value == col:
                                top_level_tag = tag
                                break
                        if top_level_tag:
                            top_level_tags.append(top_level_tag)

            # Create unified result
            result = UnifiedASClassification(
                asn=asn,
                organization_name=f"ASN{asn}",  # Would be filled from data access
                top_level_tags=top_level_tags,
                hierarchical_tags=hierarchical_tags,
                model_used="HierarchicalSVM",
                classification_approach="hierarchical",
            )

            results.append(result)

        return results

    def _predict_general(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict general categories."""
        general_preds = self.general_model.predict(X)

        # Convert to DataFrame
        general_df = pd.DataFrame(
            general_preds, index=X.index, columns=self.general_model.classes_
        )

        return general_df

    def _predict_specialized(
        self, X: pd.DataFrame, general_preds: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Apply specialized models based on general predictions."""
        specialized_results = {}

        for category_name, specialized_model in self.specialized_models.items():
            if category_name not in general_preds.columns:
                continue

            # Get samples predicted for this general category
            category_mask = general_preds[category_name] == 1

            if not category_mask.any():
                continue

            # Apply specialized model to relevant samples
            X_category = X[category_mask]
            specialized_preds = specialized_model.predict(X_category)

            # Convert to DataFrame
            specialized_df = pd.DataFrame(
                specialized_preds,
                index=X_category.index,
                columns=specialized_model.classes_,
            )

            # Mask out predictions where general category was not predicted
            # (This ensures consistency between general and specialized predictions)
            specialized_df = specialized_df.mul(category_mask[category_mask], axis=0)

            specialized_results[category_name] = specialized_df

        return specialized_results

    def _combine_predictions(
        self,
        general_preds: pd.DataFrame,
        specialized_preds: Dict[str, pd.DataFrame],
        index: pd.Index,
    ) -> pd.DataFrame:
        """Combine general and specialized predictions."""

        # Start with general predictions
        all_columns = list(general_preds.columns)

        # Add specialized columns
        for category_df in specialized_preds.values():
            all_columns.extend(category_df.columns)

        # Remove duplicates while preserving order
        all_columns = list(dict.fromkeys(all_columns))

        # Create result DataFrame
        result = pd.DataFrame(0, index=index, columns=all_columns)

        # Fill in general predictions
        for col in general_preds.columns:
            if col in result.columns:
                result[col] = general_preds[col]

        # Fill in specialized predictions
        for category_df in specialized_preds.values():
            for col in category_df.columns:
                if col in result.columns:
                    # Use max to handle overlaps (shouldn't happen in well-designed hierarchy)
                    result[col] = np.maximum(
                        result[col], category_df[col].reindex(index, fill_value=0)
                    )

        return result

    def _create_general_labels(self, y: pd.DataFrame) -> pd.DataFrame:
        """Create general category labels from hierarchical labels."""
        general_columns = [tag.value for tag in TopLevelTags]
        general_labels = pd.DataFrame(0, index=y.index, columns=general_columns)

        for col in y.columns:
            # Find corresponding hierarchical tag
            hierarchical_tag = None
            for tag in HierarchicalTags:
                if tag.value == col:
                    hierarchical_tag = tag
                    break

            if hierarchical_tag:
                # Map to general category
                from ..unified_schemas import TagHierarchy

                general_tag = TagHierarchy.get_top_level(hierarchical_tag)
                if general_tag and general_tag.value in general_labels.columns:
                    general_labels[general_tag.value] = np.maximum(
                        general_labels[general_tag.value], y[col]
                    )

        return general_labels

    def _create_specialized_labels(
        self, y: pd.DataFrame, general_tag: TopLevelTags
    ) -> pd.DataFrame:
        """Create specialized labels for a specific general category."""
        from ..unified_schemas import TagHierarchy

        # Get hierarchical tags for this general category
        subtags = TagHierarchy.get_subtags(general_tag)
        subtag_values = [tag.value for tag in subtags]

        # Filter y to only include relevant subtags
        relevant_columns = [col for col in y.columns if col in subtag_values]

        return y[relevant_columns] if relevant_columns else pd.DataFrame()

    def save_models(self, save_dir: Path) -> None:
        """
        Save all models to directory.

        Parameters
        ----------
        save_dir : Path
            Directory to save models.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save general model
        if self.general_model:
            general_path = save_dir / "general_model.pkl"
            self.general_model.save_model(str(general_path))
            logger.info(f"General model saved to {general_path}")

        # Save specialized models
        specialized_dir = save_dir / "specialized"
        specialized_dir.mkdir(exist_ok=True)

        for category, model in self.specialized_models.items():
            model_path = (
                specialized_dir / f"{category.replace(' ', '_').lower()}_model.pkl"
            )
            model.save_model(str(model_path))
            logger.info(f"Specialized model for {category} saved to {model_path}")

        # Save model configuration
        config = {
            "general_model_path": str(save_dir / "general_model.pkl"),
            "submodel_paths": {
                category: str(
                    specialized_dir / f"{category.replace(' ', '_').lower()}_model.pkl"
                )
                for category in self.specialized_models.keys()
            },
        }

        import json

        config_path = save_dir / "hierarchy_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Hierarchical tagger saved to {save_dir}")

    @classmethod
    def load_from_directory(
        cls, model_dir: Path, data_access: Optional[DataAccessLayer] = None
    ) -> "ModernHierarchicalTagger":
        """
        Load hierarchical tagger from directory.

        Parameters
        ----------
        model_dir : Path
            Directory containing saved models.
        data_access : DataAccessLayer, optional
            Data access layer.

        Returns
        -------
        ModernHierarchicalTagger
            Loaded hierarchical tagger.
        """
        model_dir = Path(model_dir)

        # Load configuration
        config_path = model_dir / "hierarchy_config.json"
        if config_path.exists():
            import json

            with open(config_path) as f:
                config = json.load(f)

            general_model_path = config.get("general_model_path")
            submodel_paths = config.get("submodel_paths", {})
        else:
            # Fallback to manual path construction
            general_model_path = str(model_dir / "general_model.pkl")
            specialized_dir = model_dir / "specialized"
            submodel_paths = {}

            if specialized_dir.exists():
                for model_file in specialized_dir.glob("*_model.pkl"):
                    category = (
                        model_file.stem.replace("_model", "").replace("_", " ").title()
                    )
                    submodel_paths[category] = str(model_file)

        # Create and load tagger
        tagger = cls(
            general_model_path=general_model_path,
            submodel_paths=submodel_paths,
            data_access=data_access,
        )

        tagger.load_models()
        return tagger
