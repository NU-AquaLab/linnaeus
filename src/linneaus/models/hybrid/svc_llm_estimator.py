"""
Stacking classifier combining SVM and precomputed LLM predictions.

This module implements a two-branch stacking architecture:

* **SVM branch** -- drops the ASN column, imputes missing values, scales
  features, and classifies with a support-vector classifier.
* **LLM branch** -- selects the ASN column and looks up precomputed LLM
  predictions via :class:`LLMPredictor`.

A meta-learner (by default a linear SVC) combines both branches, and the
whole pipeline is wrapped in :class:`~sklearn.multioutput.MultiOutputClassifier`
for multi-label classification.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from .llm_predictor import LLMPredictor

warnings.simplefilter("ignore", UserWarning)

logger = logging.getLogger(__name__)


class SVCLLMEstimator:
    """
    Stacking classifier combining SVM and LLM-based features.

    Two-branch pipeline:

    * **SVM branch**: drop ASN, impute, scale, SVM.
    * **LLM branch**: select ASN, look up precomputed LLM predictions.

    Both branches are stacked using a meta-learner and wrapped in
    :class:`~sklearn.multioutput.MultiOutputClassifier`.

    Parameters
    ----------
    svm_params : dict, optional
        Keyword arguments forwarded to :class:`~sklearn.svm.SVC`.
    imputer : sklearn estimator, optional
        Imputer for missing feature values. Defaults to
        ``KNNImputer(n_neighbors=5)``.
    meta_learner : sklearn estimator, optional
        Final estimator for the stacking classifier. Defaults to a linear
        SVC with probability estimates.
    cv_stacking : cross-validation splitter, optional
        Cross-validation strategy for the stacking classifier.
    random_state : int, optional
        Random seed for reproducibility.
    llm_preds_df : pd.DataFrame, optional
        Precomputed LLM predictions indexed (or with column) ``asn``.

    Examples
    --------
    >>> estimator = SVCLLMEstimator(
    ...     svm_params={"kernel": "rbf", "C": 1.0, "probability": True},
    ...     llm_preds_df=llm_predictions,
    ... )
    >>> estimator.fit(X_train, y_train)
    >>> preds = estimator.predict(X_test)
    """

    def __init__(
        self,
        svm_params: Optional[Dict[str, Any]] = None,
        imputer: Any = None,
        meta_learner: Any = None,
        cv_stacking: Any = None,
        random_state: int = 42,
        llm_preds_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.svm_params = svm_params or {}
        self.imputer = imputer if imputer is not None else KNNImputer(n_neighbors=5)
        self.meta_learner = meta_learner or SVC(
            kernel="linear", probability=True, random_state=random_state
        )
        self.cv_stacking = cv_stacking or StratifiedKFold(
            n_splits=3, shuffle=True, random_state=random_state
        )
        self.random_state = random_state

        self.llm_predictor: Optional[LLMPredictor] = None
        if llm_preds_df is not None:
            self.llm_predictor = LLMPredictor(llm_preds_df=llm_preds_df)

        self.model: Optional[MultiOutputClassifier] = None

        logger.debug(
            "SVCLLMEstimator initialised (SVM params=%s, LLM preds available=%s)",
            self.svm_params,
            llm_preds_df is not None,
        )

    # ------------------------------------------------------------------
    # Pipeline construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def drop_asn(X: pd.DataFrame) -> pd.DataFrame:
        """Drop the ``asn`` column from the feature matrix."""
        return X.drop(columns=["asn"], errors="ignore")

    @staticmethod
    def select_asn(X: pd.DataFrame) -> np.ndarray:
        """Select only the ``asn`` column and return as a numpy array."""
        return X["asn"].values

    def build_model(self) -> MultiOutputClassifier:
        """
        Construct the two-branch stacking classifier.

        Returns
        -------
        MultiOutputClassifier
            Stacking classifier wrapped for multi-label output.
        """
        svm_pipeline = Pipeline(
            [
                ("drop_asn", FunctionTransformer(self.drop_asn)),
                ("imputer", self.imputer),
                ("scaler", StandardScaler()),
                ("svm", SVC(**self.svm_params, random_state=self.random_state)),
            ]
        )
        llm_pipeline = Pipeline(
            [
                ("select_asn", FunctionTransformer(self.select_asn)),
                ("llm", self.llm_predictor),
            ]
        )

        stacking = StackingClassifier(
            estimators=[("svm", svm_pipeline), ("llm", llm_pipeline)],
            final_estimator=self.meta_learner,
            cv=self.cv_stacking,
            n_jobs=-1,
        )

        logger.info("Built stacking classifier model")
        return MultiOutputClassifier(stacking, n_jobs=-1)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "SVCLLMEstimator":
        """
        Build the model and fit it on ``(X, y)``.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Must include an ``asn`` column used by the LLM
            branch.
        y : pd.DataFrame
            Label matrix. An ``asn`` column, if present, is dropped
            automatically.

        Returns
        -------
        self
            The fitted estimator.
        """
        y_clean = y.drop(columns=["asn"], errors="ignore")
        self.model = self.build_model()

        logger.info("Fitting SVCLLMEstimator on %d samples", len(X))
        self.model.fit(X, y_clean)
        logger.info("SVCLLMEstimator fitting complete")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict labels for the given samples.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (must include ``asn`` column).

        Returns
        -------
        np.ndarray
            Predicted label matrix.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit first.")
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filepath: Union[str, Path]) -> "SVCLLMEstimator":
        """
        Serialize the fitted model and LLM predictions to disk.

        Parameters
        ----------
        filepath : str or Path
            Destination file path.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        data = {
            "model": self.model,
            "llm_preds_df": (
                self.llm_predictor.llm_preds_df
                if self.llm_predictor is not None
                else None
            ),
        }
        joblib.dump(data, filepath)
        logger.info("Model saved to %s", filepath)

        return self

    def save_preds(self, filepath: Union[str, Path]) -> "SVCLLMEstimator":
        """
        Save the precomputed LLM predictions DataFrame to disk.

        Parameters
        ----------
        filepath : str or Path
            Destination file path (CSV or Parquet, determined by extension).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If there are no LLM predictions to save.
        """
        if self.llm_predictor is None or self.llm_predictor.llm_preds_df is None:
            raise ValueError("No LLM predictions available to save.")

        filepath = Path(filepath)
        if filepath.suffix == ".parquet":
            self.llm_predictor.llm_preds_df.to_parquet(filepath)
        else:
            self.llm_predictor.llm_preds_df.to_csv(filepath)

        logger.info("LLM predictions saved to %s", filepath)
        return self

    def load_weights(self, filepath: Union[str, Path]) -> "SVCLLMEstimator":
        """
        Load a previously saved model and LLM predictions from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to the serialized model file.

        Returns
        -------
        self

        Raises
        ------
        FileNotFoundError
            If the given path does not exist.
        """
        filepath = str(filepath)
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        data = joblib.load(filepath)
        self.model = data["model"]

        if data.get("llm_preds_df") is not None:
            self.llm_predictor = LLMPredictor(llm_preds_df=data["llm_preds_df"])

        logger.info("Model loaded from %s", filepath)
        return self
