"""
Scikit-learn-compatible estimator for precomputed LLM predictions.

This module provides an sklearn-compatible wrapper that looks up precomputed
LLM predictions by ASN, allowing LLM outputs to be used as a component
in standard sklearn pipelines and ensemble methods (e.g. StackingClassifier).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class LLMPredictor(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn-compatible estimator that retrieves precomputed LLM predictions by ASN.

    This estimator wraps a DataFrame of precomputed LLM predictions (indexed by
    ASN) so that it can participate in sklearn pipelines and stacking classifiers.
    For any ASN not found in the precomputed predictions, a zero vector is returned.

    Parameters
    ----------
    llm_preds_df : pd.DataFrame
        DataFrame containing precomputed LLM predictions. Must either have an
        ``asn`` column or be indexed by ``asn``.

    Attributes
    ----------
    llm_preds_df : pd.DataFrame
        Precomputed predictions indexed by ASN.
    default_pred : np.ndarray
        Zero vector returned for unknown ASNs.
    classes_ : np.ndarray or None
        Unique class labels, set during ``fit()``.

    Examples
    --------
    >>> preds_df = pd.DataFrame(
    ...     {"Content": [1, 0], "Access": [0, 1]},
    ...     index=pd.Index([13335, 7922], name="asn"),
    ... )
    >>> predictor = LLMPredictor(llm_preds_df=preds_df)
    >>> predictor.fit()
    LLMPredictor(...)
    >>> predictor.predict(np.array([13335, 99999]))
    array([[1, 0],
           [0, 0]])
    """

    def __init__(self, llm_preds_df: pd.DataFrame) -> None:
        if llm_preds_df.index.name == "asn":
            self.llm_preds_df = llm_preds_df
        else:
            if "asn" not in llm_preds_df.columns:
                raise ValueError(
                    "Input DataFrame must contain an 'asn' column or be indexed by it"
                )
            self.llm_preds_df = llm_preds_df.set_index("asn")

        n_labels = self.llm_preds_df.shape[1]
        dtype = self.llm_preds_df.dtypes.values[0]
        self.default_pred: np.ndarray = np.zeros(n_labels, dtype=dtype)

        logger.debug(
            "LLMPredictor initialised with %d ASNs and %d labels",
            len(self.llm_preds_df),
            n_labels,
        )

    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> "LLMPredictor":
        """
        Fit the estimator (no-op beyond recording class labels).

        Parameters
        ----------
        X : array-like, optional
            Ignored. Present for API compatibility.
        y : array-like, optional
            Target labels used to infer ``classes_``.

        Returns
        -------
        self
            The fitted estimator.
        """
        if y is not None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = None
        return self

    def predict(self, asn_array: np.ndarray) -> np.ndarray:
        """
        Look up precomputed predictions for the given ASNs.

        Parameters
        ----------
        asn_array : array-like
            1-D array of ASN identifiers.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(len(asn_array), n_labels)`` with the
            precomputed predictions for known ASNs and zeros otherwise.
        """
        asns = np.asarray(asn_array)
        output = np.tile(self.default_pred, (len(asns), 1))
        missing_count = 0

        for i, asn in enumerate(asns):
            try:
                output[i] = self.llm_preds_df.loc[asn].values
            except KeyError:
                missing_count += 1

        if missing_count:
            logger.warning(
                "LLMPredictor: %d / %d ASNs not found in precomputed predictions",
                missing_count,
                len(asns),
            )

        return output

    def predict_proba(self, asn_array: np.ndarray) -> np.ndarray:
        """
        Return precomputed predictions as pseudo-probabilities.

        Delegates to :meth:`predict`, since the stored values may already
        represent probabilities or binary predictions.

        Parameters
        ----------
        asn_array : array-like
            1-D array of ASN identifiers.

        Returns
        -------
        np.ndarray
            2-D array identical to the output of :meth:`predict`.
        """
        return self.predict(asn_array)
