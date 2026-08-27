"""
SVM-based classifiers for AS classification.

This module provides traditional machine learning classifiers using
Support Vector Machines for AS organization classification.
"""

import logging
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

from ..unified_schemas import HierarchicalTags, TagHierarchy, TopLevelTags
from .feature_engineering import ASNFeatureEngineer

logger = logging.getLogger(__name__)


class SVMClassifier(BaseEstimator, ClassifierMixin):
    """
    SVM-based classifier for AS organization classification.

    Provides both flat and hierarchical classification capabilities
    using engineered features from network topology and metadata.
    """

    def __init__(
        self,
        approach: str = "flat",
        svm_params: Optional[Dict] = None,
        feature_selection: bool = True,
        n_features: int = 50,
        random_state: int = 42,
    ):
        """
        Initialize SVM classifier.

        Parameters
        ----------
        approach : str
            Classification approach: 'flat' or 'hierarchical'.
        svm_params : Dict, optional
            SVM hyperparameters.
        feature_selection : bool
            Whether to perform feature selection.
        n_features : int
            Number of features to select.
        random_state : int
            Random state for reproducibility.
        """
        self.approach = approach
        self.svm_params = svm_params or {"kernel": "rbf", "C": 1.0, "probability": True}
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.random_state = random_state

        # Initialize components
        self.feature_engineer = ASNFeatureEngineer()
        self.model = None
        self.classes_ = None
        self.feature_names_ = None

        # Set random state in SVM params
        if "random_state" not in self.svm_params:
            self.svm_params["random_state"] = random_state

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "SVMClassifier":
        """
        Fit the SVM classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing ASN and features.
        y : pd.DataFrame
            Label DataFrame with tag columns.

        Returns
        -------
        self
            Fitted classifier.
        """
        logger.info(f"Fitting SVM classifier with approach: {self.approach}")

        # Extract ASNs
        asns = X["asn"].tolist() if "asn" in X.columns else X.index.tolist()

        # Extract or engineer features
        if len(X.columns) > 1:  # Assume pre-extracted features
            X_features = X.drop("asn", axis=1, errors="ignore")
        else:  # Extract features from ASNs
            X_features = self.feature_engineer.extract_features(asns)
            X_features = self.feature_engineer.preprocess_features(X_features, fit=True)

        # Prepare target labels
        y_clean = y.drop("asn", axis=1, errors="ignore")

        # Feature selection
        if self.feature_selection:
            X_features = self.feature_engineer.select_features(
                pd.concat([pd.Series(asns, name="asn"), X_features], axis=1),
                y_clean,
                k=self.n_features,
            ).drop("asn", axis=1, errors="ignore")

        # Store feature names and classes
        self.feature_names_ = list(X_features.columns)
        self.classes_ = list(y_clean.columns)

        # Create and fit model
        base_svm = SVC(**self.svm_params)
        self.model = MultiOutputClassifier(base_svm, n_jobs=-1)

        logger.info(
            f"Training SVM with {X_features.shape[1]} features and {len(self.classes_)} labels"
        )
        self.model.fit(X_features, y_clean)

        logger.info("SVM classifier fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict labels for new samples.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Process features
        X_processed = self._prepare_features(X)

        # Make predictions
        predictions = self.model.predict(X_processed)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Check if base classifier supports probability prediction
        base_estimators = [estimator.estimator for estimator in self.model.estimators_]
        if not all(hasattr(est, "predict_proba") for est in base_estimators):
            raise ValueError("Base classifier does not support probability prediction")

        # Process features
        X_processed = self._prepare_features(X)

        # Get probabilities for each label
        probabilities = []
        for i, estimator in enumerate(self.model.estimators_):
            prob = estimator.predict_proba(X_processed)
            # Take probability of positive class (index 1)
            if prob.shape[1] > 1:
                probabilities.append(prob[:, 1])
            else:
                probabilities.append(prob[:, 0])

        return np.column_stack(probabilities)

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Extract ASNs
        asns = X["asn"].tolist() if "asn" in X.columns else X.index.tolist()

        # Extract or engineer features
        if len(X.columns) > 1 and all(
            col in self.feature_names_ for col in X.columns if col != "asn"
        ):
            # Features already extracted and match training features
            X_features = X.drop("asn", axis=1, errors="ignore")
            # Ensure feature order matches training
            X_features = X_features[self.feature_names_]
        else:
            # Extract features from ASNs
            X_features = self.feature_engineer.extract_features(asns)
            X_features = self.feature_engineer.preprocess_features(
                X_features, fit=False
            )

            # Apply feature selection if it was used during training
            if self.feature_selection:
                X_features = X_features[self.feature_names_]

        return X_features

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
    ) -> Dict:
        """
        Optimize SVM hyperparameters using grid search.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.DataFrame
            Training labels.
        param_grid : Dict, optional
            Parameter grid for search.
        cv : int
            Number of cross-validation folds.

        Returns
        -------
        Dict
            Best parameters found.
        """
        if param_grid is None:
            param_grid = {
                "estimator__C": [0.1, 1, 10, 100],
                "estimator__kernel": ["rbf", "linear", "poly"],
                "estimator__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            }

        logger.info(f"Optimizing hyperparameters with {cv}-fold CV")

        # Prepare data
        asns = X["asn"].tolist() if "asn" in X.columns else X.index.tolist()

        if len(X.columns) > 1:
            X_features = X.drop("asn", axis=1, errors="ignore")
        else:
            X_features = self.feature_engineer.extract_features(asns)
            X_features = self.feature_engineer.preprocess_features(X_features, fit=True)

        y_clean = y.drop("asn", axis=1, errors="ignore")

        # Feature selection
        if self.feature_selection:
            X_features = self.feature_engineer.select_features(
                pd.concat([pd.Series(asns, name="asn"), X_features], axis=1),
                y_clean,
                k=self.n_features,
            ).drop("asn", axis=1, errors="ignore")

        # Create base model for grid search
        base_svm = SVC(probability=True, random_state=self.random_state)
        model = MultiOutputClassifier(
            base_svm, n_jobs=1
        )  # Use single job for grid search

        # Perform grid search on first label (for efficiency)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1
        )

        # Use only first label for hyperparameter optimization
        y_first_label = y_clean.iloc[:, 0]
        grid_search.fit(X_features, y_first_label)

        best_params = grid_search.best_params_
        logger.info(f"Best parameters: {best_params}")

        # Update model parameters
        self.svm_params.update(
            {
                key.replace("estimator__", ""): value
                for key, value in best_params.items()
            }
        )

        return best_params

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """
        Evaluate model performance.

        Parameters
        ----------
        X : pd.DataFrame
            Test features.
        y : pd.DataFrame
            True labels.

        Returns
        -------
        Dict
            Evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Make predictions
        y_pred = self.predict(X)
        y_true = y.drop("asn", axis=1, errors="ignore")

        # Calculate metrics for each label
        results = {}
        for i, label in enumerate(self.classes_):
            report = classification_report(
                y_true.iloc[:, i], y_pred[:, i], output_dict=True, zero_division=0
            )
            results[label] = report

        return results

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        model_data = {
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "approach": self.approach,
            "classes_": self.classes_,
            "feature_names_": self.feature_names_,
            "svm_params": self.svm_params,
            "feature_selection": self.feature_selection,
            "n_features": self.n_features,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> "SVMClassifier":
        """Load a previously saved model."""
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.feature_engineer = model_data["feature_engineer"]
        self.approach = model_data["approach"]
        self.classes_ = model_data["classes_"]
        self.feature_names_ = model_data["feature_names_"]
        self.svm_params = model_data["svm_params"]
        self.feature_selection = model_data["feature_selection"]
        self.n_features = model_data["n_features"]

        logger.info(f"Model loaded from {filepath}")
        return self

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from feature engineering step."""
        return self.feature_engineer.get_feature_importance()


class HierarchicalSVMClassifier(SVMClassifier):
    """
    Hierarchical SVM classifier that predicts general categories first,
    then specialized sub-categories.
    """

    def __init__(self, **kwargs):
        """Initialize hierarchical SVM classifier."""
        super().__init__(approach="hierarchical", **kwargs)
        self.general_model = None
        self.specialized_models = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "HierarchicalSVMClassifier":
        """
        Fit hierarchical models.

        First trains a general classifier for top-level categories,
        then trains specialized classifiers for each category's sub-tags.
        """
        logger.info("Fitting hierarchical SVM classifier")

        # Prepare features
        asns = X["asn"].tolist() if "asn" in X.columns else X.index.tolist()

        if len(X.columns) > 1:
            X_features = X.drop("asn", axis=1, errors="ignore")
        else:
            X_features = self.feature_engineer.extract_features(asns)
            X_features = self.feature_engineer.preprocess_features(X_features, fit=True)

        y_clean = y.drop("asn", axis=1, errors="ignore")

        # Create general labels (top-level categories)
        y_general = self._create_general_labels(y_clean)

        # Train general classifier
        logger.info("Training general classifier")
        self.general_model = MultiOutputClassifier(SVC(**self.svm_params), n_jobs=-1)
        self.general_model.fit(X_features, y_general)

        # Train specialized classifiers for each general category
        for general_tag in TopLevelTags:
            logger.info(f"Training specialized classifier for {general_tag.value}")

            # Get samples that belong to this general category
            general_mask = y_general[general_tag.value] == 1
            if general_mask.sum() == 0:
                logger.warning(f"No training samples for {general_tag.value}")
                continue

            X_specialized = X_features[general_mask]
            y_specialized = self._create_specialized_labels(
                y_clean[general_mask], general_tag
            )

            if y_specialized.sum().sum() > 0:  # Check if there are any positive labels
                specialized_model = MultiOutputClassifier(
                    SVC(**self.svm_params), n_jobs=-1
                )
                specialized_model.fit(X_specialized, y_specialized)
                self.specialized_models[general_tag.value] = specialized_model

        # Store classes and features
        self.classes_ = list(y_clean.columns)
        self.feature_names_ = list(X_features.columns)

        logger.info("Hierarchical SVM classifier fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using hierarchical approach."""
        if self.general_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        X_processed = self._prepare_features(X)

        # Predict general categories
        general_preds = self.general_model.predict(X_processed)
        general_labels = [tag.value for tag in TopLevelTags]

        # Initialize final predictions
        all_predictions = np.zeros((X_processed.shape[0], len(self.classes_)))

        # For each sample and general category, apply specialized models
        for i, sample_features in enumerate(X_processed.iterrows()):
            sample_features = sample_features[1].values.reshape(1, -1)

            for j, general_tag in enumerate(general_labels):
                if general_preds[i, j] == 1 and general_tag in self.specialized_models:
                    # Apply specialized model
                    specialized_preds = self.specialized_models[general_tag].predict(
                        sample_features
                    )

                    # Map specialized predictions to final output
                    specialized_indices = self._get_specialized_indices(general_tag)
                    all_predictions[i, specialized_indices] = specialized_preds[0]
                elif general_preds[i, j] == 1:
                    # No specialized model, use general prediction
                    general_index = self._get_general_index(general_tag)
                    if general_index is not None:
                        all_predictions[i, general_index] = 1

        return all_predictions

    def _create_general_labels(self, y: pd.DataFrame) -> pd.DataFrame:
        """Create general category labels from hierarchical labels."""
        general_labels = pd.DataFrame(
            0, index=y.index, columns=[tag.value for tag in TopLevelTags]
        )

        for col in y.columns:
            # Find corresponding hierarchical tag
            hierarchical_tag = None
            for tag in HierarchicalTags:
                if tag.value == col:
                    hierarchical_tag = tag
                    break

            if hierarchical_tag:
                general_tag = TagHierarchy.get_top_level(hierarchical_tag)
                if general_tag:
                    general_labels[general_tag.value] = np.maximum(
                        general_labels[general_tag.value], y[col]
                    )

        return general_labels

    def _create_specialized_labels(
        self, y: pd.DataFrame, general_tag: TopLevelTags
    ) -> pd.DataFrame:
        """Create specialized labels for a specific general category."""
        # Get hierarchical tags for this general category
        subtags = TagHierarchy.get_subtags(general_tag)
        subtag_values = [tag.value for tag in subtags]

        # Filter y to only include relevant subtags
        relevant_columns = [col for col in y.columns if col in subtag_values]
        return y[relevant_columns] if relevant_columns else pd.DataFrame()

    def _get_specialized_indices(self, general_tag: str) -> List[int]:
        """Get column indices for specialized tags of a general category."""
        general_tag_enum = None
        for tag in TopLevelTags:
            if tag.value == general_tag:
                general_tag_enum = tag
                break

        if not general_tag_enum:
            return []

        subtags = TagHierarchy.get_subtags(general_tag_enum)
        indices = []
        for tag in subtags:
            if tag.value in self.classes_:
                indices.append(self.classes_.index(tag.value))

        return indices

    def _get_general_index(self, general_tag: str) -> Optional[int]:
        """Get column index for general tag if it exists in classes."""
        if general_tag in self.classes_:
            return self.classes_.index(general_tag)
        return None
