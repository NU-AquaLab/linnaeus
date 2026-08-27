"""
ASN-aware cross-validation utilities for multi-label classification.

This module provides custom cross-validation splitters that respect ASN boundaries
to prevent data leakage in model evaluation.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, StratifiedKFold
from sklearn.utils import check_random_state
from tqdm import tqdm

try:
    from skmultilearn.model_selection import iterative_train_test_split

    SKMULTILEARN_AVAILABLE = True
except ImportError:
    SKMULTILEARN_AVAILABLE = False
    logging.warning(
        "skmultilearn not available. Multi-label stratification will fall back to simple splitting."
    )

logger = logging.getLogger(__name__)


class ASNStratifiedKFold(BaseCrossValidator):
    """
    K-fold cross-validator that ensures ASNs don't appear in multiple folds.

    This splitter is designed for AS classification tasks where each ASN
    must appear in only one fold to prevent data leakage. It supports
    both single-label and multi-label classification with stratification.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int or RandomState, default=None
        Controls the randomness of the splits.
    stratify : bool, default=True
        Whether to perform stratified splitting based on labels.
    """

    def __init__(
        self,
        n_splits: int = 3,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        stratify: bool = True,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_labels)
            Target variable for stratification.
        groups : array-like of shape (n_samples,), default=None
            Group labels (e.g., ASNs) for the samples.

        Yields
        ------
        train : ndarray
            Training set indices for that split.
        test : ndarray
            Testing set indices for that split.
        """
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)

        # Get ASN information
        if groups is not None:
            asns = np.array(groups)
        elif isinstance(X, pd.DataFrame) and "asn" in X.columns:
            asns = X["asn"].values
        elif isinstance(X, pd.DataFrame) and X.index.name == "asn":
            asns = X.index.values
        else:
            # No ASN information, fall back to regular splitting
            logger.warning("No ASN information found. Using regular k-fold splitting.")
            asns = np.arange(n_samples)

        # Get unique ASNs
        unique_asns = np.unique(asns)
        n_asns = len(unique_asns)

        if n_asns < self.n_splits:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater than "
                f"the number of unique ASNs: {n_asns}."
            )

        # Create ASN to indices mapping
        asn_to_indices = {}
        for idx, asn in enumerate(asns):
            if asn not in asn_to_indices:
                asn_to_indices[asn] = []
            asn_to_indices[asn].append(idx)

        # Determine splitting strategy based on label type
        if y is not None and self.stratify:
            y_array = np.array(y) if not isinstance(y, np.ndarray) else y

            # Check if multi-label
            is_multilabel = len(y_array.shape) > 1 and y_array.shape[1] > 1

            if is_multilabel and SKMULTILEARN_AVAILABLE:
                yield from self._multilabel_stratified_split(
                    asn_to_indices, y_array, unique_asns
                )
            else:
                yield from self._single_label_stratified_split(
                    asn_to_indices, y_array, unique_asns
                )
        else:
            # Simple splitting without stratification
            yield from self._simple_split(asn_to_indices, unique_asns)

    def _simple_split(
        self, asn_to_indices: dict, unique_asns: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Simple k-fold split without stratification."""
        rng = check_random_state(self.random_state)

        if self.shuffle:
            unique_asns = unique_asns.copy()
            rng.shuffle(unique_asns)

        fold_sizes = np.full(self.n_splits, len(unique_asns) // self.n_splits)
        fold_sizes[: len(unique_asns) % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_asns = unique_asns[current : current + fold_size]
            train_asns = np.setdiff1d(unique_asns, test_asns)

            train_indices = []
            test_indices = []

            for asn in train_asns:
                train_indices.extend(asn_to_indices[asn])
            for asn in test_asns:
                test_indices.extend(asn_to_indices[asn])

            yield np.array(train_indices), np.array(test_indices)
            current += fold_size

    def _single_label_stratified_split(
        self, asn_to_indices: dict, y: np.ndarray, unique_asns: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Stratified k-fold split for single-label classification."""
        # Get majority label for each ASN
        asn_labels = {}
        for asn in unique_asns:
            indices = asn_to_indices[asn]
            if len(y.shape) > 1:
                # Multi-class represented as one-hot
                labels = y[indices].argmax(axis=1)
            else:
                labels = y[indices]
            # Use majority vote
            unique_labels, counts = np.unique(labels, return_counts=True)
            asn_labels[asn] = unique_labels[counts.argmax()]

        # Group ASNs by label
        label_to_asns = {}
        for asn, label in asn_labels.items():
            if label not in label_to_asns:
                label_to_asns[label] = []
            label_to_asns[label].append(asn)

        # Distribute ASNs across folds maintaining stratification
        rng = check_random_state(self.random_state)
        fold_asns = [[] for _ in range(self.n_splits)]

        for label, label_asns in label_to_asns.items():
            if self.shuffle:
                label_asns = label_asns.copy()
                rng.shuffle(label_asns)

            # Distribute this label's ASNs across folds
            for i, asn in enumerate(label_asns):
                fold_asns[i % self.n_splits].append(asn)

        # Generate train/test splits
        for i in range(self.n_splits):
            test_asns = fold_asns[i]
            train_asns = []
            for j in range(self.n_splits):
                if j != i:
                    train_asns.extend(fold_asns[j])

            train_indices = []
            test_indices = []

            for asn in train_asns:
                train_indices.extend(asn_to_indices[asn])
            for asn in test_asns:
                test_indices.extend(asn_to_indices[asn])

            yield np.array(train_indices), np.array(test_indices)

    def _multilabel_stratified_split(
        self, asn_to_indices: dict, y: np.ndarray, unique_asns: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Stratified k-fold split for multi-label classification."""
        # Create ASN-level features by aggregating labels
        asn_features = []
        asn_list = []

        for asn in unique_asns:
            indices = asn_to_indices[asn]
            # Use max aggregation for multi-label
            asn_label = y[indices].max(axis=0)
            asn_features.append(asn_label)
            asn_list.append(asn)

        asn_features = np.array(asn_features)
        asn_list = np.array(asn_list)

        # Use iterative stratification for multi-label if available
        1.0 / self.n_splits
        check_random_state(self.random_state)

        # Generate k folds using iterative train/test splits
        remaining_asns = np.arange(len(asn_list))

        for i in range(self.n_splits):
            if i == self.n_splits - 1:
                # Last fold gets all remaining
                test_fold_asns = remaining_asns
                train_fold_asns = np.setdiff1d(np.arange(len(asn_list)), test_fold_asns)
            else:
                # Calculate test size for remaining data
                current_test_size = 1.0 / (self.n_splits - i)

                # Split remaining data
                train_idx, _, test_idx, _ = iterative_train_test_split(
                    np.arange(len(remaining_asns)).reshape(-1, 1),
                    asn_features[remaining_asns],
                    test_size=current_test_size,
                )

                test_fold_asns = remaining_asns[test_idx.ravel()]
                remaining_asns = remaining_asns[train_idx.ravel()]
                train_fold_asns = np.setdiff1d(np.arange(len(asn_list)), test_fold_asns)

            # Convert ASN indices to sample indices
            train_indices = []
            test_indices = []

            for asn_idx in train_fold_asns:
                asn = asn_list[asn_idx]
                train_indices.extend(asn_to_indices[asn])

            for asn_idx in test_fold_asns:
                asn = asn_list[asn_idx]
                test_indices.extend(asn_to_indices[asn])

            yield np.array(train_indices), np.array(test_indices)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


def create_asn_aware_splits(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 3,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation splits that respect ASN boundaries.

    This is a convenience function that mimics the interface from the reference
    implementation but ensures ASN-aware splitting.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe with ASN information.
    y : pd.DataFrame
        Label dataframe (single or multi-label).
    n_splits : int
        Number of folds for cross-validation.
    test_size : float
        Proportion of data to use for validation.
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        Whether to use stratified splitting.

    Returns
    -------
    X_train, X_val, y_train, y_val : pd.DataFrames
        Train and validation splits.
    """
    # Ensure ASN information is available
    if "asn" not in X.columns and X.index.name != "asn":
        raise ValueError("X must have ASN information in columns or index")

    # Use ASN as groups
    if "asn" in X.columns:
        groups = X["asn"].values
    else:
        groups = X.index.values

    # For train/test split, use 2-fold cross-validation and take first split
    splitter = ASNStratifiedKFold(
        n_splits=int(1 / test_size),
        shuffle=True,
        random_state=random_state,
        stratify=stratify,
    )

    train_idx, val_idx = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Print distribution info
    logger.info(
        f"Train set: {len(X_train)} samples, Validation set: {len(X_val)} samples"
    )

    if stratify and y is not None:
        # Print label distribution
        train_dist = y_train.sum()
        val_dist = y_val.sum()
        logger.info("Label distribution in train/validation sets:")
        for col in y.columns:
            logger.info(f"  {col}: Train={train_dist[col]}, Val={val_dist[col]}")

    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Nested cross-validation pipeline
# ---------------------------------------------------------------------------


def nested_cross_val(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    llm_preds: pd.DataFrame,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    inner_folds: int = 5,
    outer_folds: int = 3,
    grid_search_scoring: str = "jaccard_micro",
    imputer: KNNImputer = KNNImputer(n_neighbors=5),
    random_state: int = 42,
    cv_stacking: int = 3,
    meta_learner: Optional[Any] = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, Dict[str, int]], Dict[str, dict]
]:
    """Perform nested cross-validation with SVM hyperparameter tuning and stacking.

    The outer loop uses :class:`ASNStratifiedKFold` with *outer_folds* splits to
    partition data into train and test sets while respecting ASN boundaries.
    Within each outer fold an inner :class:`~sklearn.model_selection.GridSearchCV`
    (with *inner_folds* splits) tunes SVM parameters (passed via *param_grid*)
    on the training portion.  The best estimator is then evaluated on the
    held-out outer test split.  At each fold we also record per-label counts
    and best-found parameters.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame of input features, **including** an ``'asn'`` column used for
        LLM prediction lookup.
    labels : pd.DataFrame
        DataFrame of true multi-label targets.  May contain an ``'asn'`` column,
        which will be dropped before fitting.
    llm_preds : pd.DataFrame
        DataFrame of precomputed LLM-based probabilities, indexed or keyed by
        ASN.
    param_grid : dict or None
        Dictionary mapping SVM parameter names to lists of values for
        GridSearchCV.  If ``None``, no tuning is performed and default SVM
        settings are used.
    inner_folds : int
        Number of folds in the inner cross-validation (GridSearchCV).
    outer_folds : int
        Number of folds in the outer cross-validation.
    grid_search_scoring : str
        Scoring metric name passed to GridSearchCV.
    imputer : KNNImputer
        Imputer instance applied before scaling in the SVM pipeline.
    random_state : int
        Random seed for reproducibility of cross-validation splits.
    cv_stacking : int
        Number of folds used by the stacking classifier itself.
    meta_learner : estimator or None
        Final estimator for stacking.  If ``None``, defaults to a linear-kernel
        SVM inside the estimator.

    Returns
    -------
    all_true : pd.DataFrame
        Concatenation of true labels from all outer test folds.
    all_pred : pd.DataFrame
        Concatenation of predicted labels for each outer test fold.
    all_asns : pd.Series
        Concatenation of the ASN identifiers for each test sample in order.
        Empty ``pd.Series`` if no ``'asn'`` column was present in *labels*.
    fold_label_counts : dict[str, dict[str, int]]
        Mapping of fold names (e.g. ``"Outer_Fold_1_train"``) to
        ``{label_name: count}`` for train/test.
    fold_best_params : dict[str, dict]
        Mapping of fold names to best parameter dict from GridSearchCV.
    """
    # Lazy import to avoid circular dependency at module level
    from ..hybrid.svc_llm_estimator import SVCLLMEstimator

    # --- Build CV splitters ------------------------------------------------
    inner_cv = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=random_state
    )
    outer_cv = ASNStratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=random_state, stratify=True
    )
    stacking_cv = StratifiedKFold(
        n_splits=cv_stacking, shuffle=True, random_state=random_state
    )

    # --- Build the stacking estimator --------------------------------------
    estimator = SVCLLMEstimator(
        svm_params={"kernel": "rbf", "probability": True},
        imputer=imputer,
        meta_learner=meta_learner,
        cv_stacking=stacking_cv,
        random_state=random_state,
        llm_preds_df=llm_preds,
    )
    classifier = estimator.build_model()

    # --- GridSearchCV wrapper ----------------------------------------------
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid or {},
        cv=inner_cv,
        scoring=grid_search_scoring,
        n_jobs=-1,
    )

    # --- Outer CV loop -----------------------------------------------------
    labels_list: List[pd.DataFrame] = []
    preds_list: List[pd.DataFrame] = []
    asn_list: List[pd.Series] = []
    fold_label_counts: Dict[str, Dict[str, int]] = {}
    fold_best_params: Dict[str, dict] = {}

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(outer_cv.split(features, labels), total=outer_folds, desc="Outer CV"),
        start=1,
    ):
        X_train = features.iloc[train_idx]
        X_test = features.iloc[test_idx]
        y_train = labels.iloc[train_idx]
        y_test = labels.iloc[test_idx]

        # Drop the 'asn' column from labels before fitting
        y_train_no_asn = y_train.drop(columns=["asn"], errors="ignore")
        y_test_no_asn = y_test.drop(columns=["asn"], errors="ignore")

        # Record per-label positive counts for this fold
        log_fold_label_counts(fold, fold_label_counts, y_train_no_asn, y_test_no_asn)

        # Fit with inner CV hyperparameter search
        logger.info(
            "Fold %d/%d: fitting GridSearchCV on %d samples",
            fold,
            outer_folds,
            len(X_train),
        )
        grid_search.fit(X_train, y_train_no_asn)
        best = grid_search.best_estimator_

        # Record best hyperparameters
        log_fold_best_model_params(fold, fold_best_params, grid_search)
        logger.info(
            "Fold %d/%d: best params = %s", fold, outer_folds, grid_search.best_params_
        )

        # Predict on the held-out test set
        y_pred = best.predict(X_test)

        labels_list.append(y_test_no_asn)
        preds_list.append(pd.DataFrame(y_pred, columns=y_test_no_asn.columns))
        if "asn" in y_test.columns:
            asn_list.append(y_test["asn"])

    all_true = pd.concat(labels_list, ignore_index=True)
    all_pred = pd.concat(preds_list, ignore_index=True)
    all_asns = (
        pd.concat(asn_list, ignore_index=True) if asn_list else pd.Series(dtype=object)
    )

    return all_true, all_pred, all_asns, fold_label_counts, fold_best_params


def log_fold_label_counts(
    fold_number: int,
    fold_label_counts: Dict[str, Dict[str, int]],
    y_train_no_asn: pd.DataFrame,
    y_test_no_asn: pd.DataFrame,
) -> None:
    """Record per-label positive counts in train and test for a given fold.

    Populates *fold_label_counts* in-place with two new entries keyed as
    ``"Outer_Fold_{fold_number}_train"`` and ``"Outer_Fold_{fold_number}_test"``.

    Parameters
    ----------
    fold_number : int
        1-based index of the current outer fold.
    fold_label_counts : dict
        Dictionary to be populated with label counts.
    y_train_no_asn : pd.DataFrame
        Training labels DataFrame (without ``'asn'`` column).
    y_test_no_asn : pd.DataFrame
        Test labels DataFrame (without ``'asn'`` column).
    """
    fold_label_counts[f"Outer_Fold_{fold_number}_train"] = y_train_no_asn.sum(
        axis=0
    ).to_dict()
    fold_label_counts[f"Outer_Fold_{fold_number}_test"] = y_test_no_asn.sum(
        axis=0
    ).to_dict()


def log_fold_best_model_params(
    fold_number: int,
    fold_best_model_params: Dict[str, dict],
    grid_search: GridSearchCV,
) -> None:
    """Store the best-found hyperparameters from GridSearchCV for a given fold.

    Populates *fold_best_model_params* in-place with a new entry keyed as
    ``"Outer_Fold_{fold_number}"``.

    Parameters
    ----------
    fold_number : int
        1-based index of the current outer fold.
    fold_best_model_params : dict
        Dictionary to be populated with best parameters.
    grid_search : GridSearchCV
        The fitted GridSearchCV object from which ``.best_params_`` is read.
    """
    fold_best_model_params[f"Outer_Fold_{fold_number}"] = grid_search.best_params_
