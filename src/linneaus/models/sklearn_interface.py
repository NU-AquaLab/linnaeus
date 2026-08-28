"""
Scikit-learn compatible interface for AS classification.

This module provides a scikit-learn compatible classifier interface
that wraps the OpenAI-based classification pipeline, enabling easy
integration with existing ML workflows and evaluation tools.
"""

import asyncio
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..config import get_config
from ..data.access import DataAccessLayer
from .hybrid.stacking_classifier import HybridASClassifier
from .schemas import ASClassification, ClassificationTags, OrganizationData
from .unified_schemas import HierarchicalTags, TopLevelTags


class ASClassifier:
    """Stub wrapper for LLM-based AS classification via sklearn interface."""

    def __init__(self, model_id: str, api_key: Optional[str] = None):
        self.model_id = model_id
        self.api_key = api_key
        self.batch_size = 10

    async def classify_batch(self, organizations, include_confidence: bool = False):
        raise NotImplementedError(
            "Direct LLM classification requires OpenAI API. "
            "Use HybridASNClassifier or the CLI instead."
        )


class ASNClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible AS classification estimator.

    This class provides a familiar fit/predict interface for AS classification
    using fine-tuned language models, making it compatible with scikit-learn
    pipelines, cross-validation, and other ML tools.

    Parameters
    ----------
    model_id : str, optional
        OpenAI model identifier. If None, will attempt to train a new model
        during fit() or use default from config.
    api_key : str, optional
        OpenAI API key. Uses config if None.
    batch_size : int, optional
        Batch size for inference. Uses config default if None.
    base_model : str, default "gpt-4o-mini-2024-07-18"
        Base model to use for training if model_id is None.
    auto_train : bool, default False
        Whether to automatically train a model if none is specified.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels (ClassificationTags enum values).
    n_features_in_ : int
        Number of features (always 1 for ASN input).
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names (always ['asn']).
    model_id_ : str
        The model ID used for classification after fitting.

    Examples
    --------
    >>> from linnaeus.models import ASNClassifier
    >>> from linnaeus.data import DataAccessLayer
    >>>
    >>> # Initialize classifier
    >>> clf = ASNClassifier(model_id="ft:gpt-4o-mini-...")
    >>>
    >>> # Prepare data (ASNs)
    >>> X = np.array([174, 15169, 32934]).reshape(-1, 1)
    >>>
    >>> # Fit (loads data and prepares classifier)
    >>> clf.fit(X)
    >>>
    >>> # Predict
    >>> predictions = clf.predict(X)
    >>>
    >>> # Get prediction probabilities (if model supports it)
    >>> probabilities = clf.predict_proba(X)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: Optional[int] = None,
        base_model: str = "gpt-4o-mini-2024-07-18",
        auto_train: bool = False,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.base_model = base_model
        self.auto_train = auto_train

        # Initialize config
        self.config = get_config()

        # Use config API key if not provided
        self.api_key = api_key or self.config.openai.api_key

        # Set up data access
        self.data_access = DataAccessLayer()

        # Initialize label binarizer for multi-label encoding
        self.label_binarizer = MultiLabelBinarizer()

        # Attributes set during fit
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.model_id_ = None
        self._classifier = None

    def fit(self, X, y=None):
        """
        Fit the AS classifier.

        For this classifier, fitting mainly involves:
        1. Validating the model_id or training a new model if needed
        2. Setting up the classifier with the specified model
        3. Preparing the label encoding

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            ASN numbers to fit on. Used for validation and setup.
        y : array-like of shape (n_samples, n_labels), optional
            Target labels. If provided and auto_train=True, will train a model.
            If None, assumes model_id is provided for pre-trained model.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        # Input validation
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)
        else:
            X = check_array(X, accept_sparse=False)

        # Validate X shape (should be ASN numbers)
        if X.shape[1] != 1:
            raise ValueError("X should have exactly 1 feature (ASN number)")

        # Set attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(["asn"])

        # Set up classes
        self.classes_ = np.array([tag.value for tag in ClassificationTags])
        self.label_binarizer.fit([self.classes_])  # Fit on all possible classes

        # Determine model to use
        if self.model_id is None:
            if self.auto_train and y is not None:
                # Train a new model
                self.model_id_ = self._train_model(X, y)
            else:
                # Use default model from config
                self.model_id_ = self.config.openai.default_model
        else:
            self.model_id_ = self.model_id

        # Initialize classifier
        self._classifier = ASClassifier(model_id=self.model_id_, api_key=self.api_key)

        if self.batch_size is not None:
            self._classifier.batch_size = self.batch_size

        return self

    def predict(self, X):
        """
        Predict AS classifications.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            ASN numbers to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_classes)
            Binary prediction matrix where each row represents an ASN
            and each column represents a classification tag.
        """
        check_is_fitted(self, attributes=["n_features_in_", "classes_", "_classifier"])
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        # Convert ASNs to OrganizationData
        organizations = self._asns_to_organizations(X.flatten())

        # Run classification
        results = asyncio.run(
            self._classifier.classify_batch(organizations, include_confidence=False)
        )

        # Convert to binary matrix
        y_pred = self._classifications_to_binary_matrix(results.classifications)

        return y_pred

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            ASN numbers to classify.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Probability matrix where each row represents an ASN
            and each column represents the probability for a classification tag.
        """
        check_is_fitted(self, attributes=["n_features_in_", "classes_", "_classifier"])
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        # Convert ASNs to OrganizationData
        organizations = self._asns_to_organizations(X.flatten())

        # Run classification with confidence scores
        results = asyncio.run(
            self._classifier.classify_batch(organizations, include_confidence=True)
        )

        # Convert to probability matrix
        y_proba = self._classifications_to_probability_matrix(results.classifications)

        return y_proba

    def decision_function(self, X):
        """
        Compute decision function (same as predict_proba for this classifier).

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            ASN numbers.

        Returns
        -------
        decision : ndarray of shape (n_samples, n_classes)
            Decision function values.
        """
        return self.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean Jaccard score on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Test ASNs.
        y : array-like of shape (n_samples, n_classes)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights (not supported, will be ignored).

        Returns
        -------
        score : float
            Mean Jaccard score.
        """
        from sklearn.metrics import jaccard_score

        y_pred = self.predict(X)
        return jaccard_score(y, y_pred, average="macro", zero_division=0)

    def _asns_to_organizations(self, asns: np.ndarray) -> List[OrganizationData]:
        """Convert ASN array to OrganizationData list."""
        organizations = []

        for asn in asns:
            asn_int = int(asn)
            org_data = self.data_access.get_organization_data(asn_int)

            if org_data is None:
                # Create minimal organization data if not found
                org_data = OrganizationData(
                    asn=asn_int,
                    name=f"AS{asn_int}",
                    country="Unknown",
                    website="Not available",
                )

            organizations.append(org_data)

        return organizations

    def _classifications_to_binary_matrix(
        self, classifications: List[ASClassification]
    ) -> np.ndarray:
        """Convert classification results to binary matrix."""
        n_samples = len(classifications)
        n_classes = len(self.classes_)

        y_pred = np.zeros((n_samples, n_classes), dtype=int)

        # Create tag to index mapping
        tag_to_idx = {tag: idx for idx, tag in enumerate(self.classes_)}

        for i, classification in enumerate(classifications):
            for tag in classification.tags:
                if tag.value in tag_to_idx:
                    y_pred[i, tag_to_idx[tag.value]] = 1

        return y_pred

    def _classifications_to_probability_matrix(
        self, classifications: List[ASClassification]
    ) -> np.ndarray:
        """Convert classification results to probability matrix."""
        n_samples = len(classifications)
        n_classes = len(self.classes_)

        y_proba = np.zeros((n_samples, n_classes), dtype=float)

        # Create tag to index mapping
        tag_to_idx = {tag: idx for idx, tag in enumerate(self.classes_)}

        for i, classification in enumerate(classifications):
            if classification.confidence_scores:
                # Use confidence scores
                for tag, score in classification.confidence_scores.items():
                    if tag in tag_to_idx:
                        y_proba[i, tag_to_idx[tag]] = score
            else:
                # Use binary predictions as probabilities
                for tag in classification.tags:
                    if tag.value in tag_to_idx:
                        y_proba[i, tag_to_idx[tag.value]] = 1.0

        return y_proba

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> str:
        """Train a new model using the provided data."""
        # This would require converting X, y back to the training format
        # For now, raise an error as this requires more complex implementation
        raise NotImplementedError(
            "Auto-training is not yet implemented. Please provide a model_id "
            "or use the TrainingPipeline class directly."
        )

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default True
            If True, return parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "model_id": self.model_id,
            "api_key": self.api_key,
            "batch_size": self.batch_size,
            "base_model": self.base_model,
            "auto_train": self.auto_train,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {type(self).__name__}"
                )

        return self


class ASNDataTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to convert ASN numbers to feature vectors.

    This transformer takes ASN numbers and converts them to feature
    vectors that can be used with traditional ML algorithms by extracting
    network topology and metadata features.

    Parameters
    ----------
    include_asrank : bool, default True
        Whether to include ASRank features.
    include_peeringdb : bool, default True
        Whether to include PeeringDB features.
    include_aspop : bool, default True
        Whether to include ASPOP features.
    normalize_features : bool, default True
        Whether to normalize numerical features.

    Examples
    --------
    >>> from linnaeus.models import ASNDataTransformer
    >>> transformer = ASNDataTransformer()
    >>> X = np.array([174, 15169, 32934]).reshape(-1, 1)
    >>> X_transformed = transformer.fit_transform(X)
    """

    def __init__(
        self,
        include_asrank: bool = True,
        include_peeringdb: bool = True,
        include_aspop: bool = True,
        normalize_features: bool = True,
    ):
        self.include_asrank = include_asrank
        self.include_peeringdb = include_peeringdb
        self.include_aspop = include_aspop
        self.normalize_features = normalize_features

        # Initialize data access
        self.data_access = DataAccessLayer()

        # Attributes set during fit
        self.feature_names_ = None
        self.n_features_in_ = None
        self.n_features_out_ = None
        self._scaler = None

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            ASN numbers.
        y : ignored
            Not used, present for scikit-learn compatibility.

        Returns
        -------
        self
        """
        X = check_array(X, accept_sparse=False)

        self.n_features_in_ = X.shape[1]

        # Determine output features
        self.feature_names_ = self._get_feature_names()
        self.n_features_out_ = len(self.feature_names_)

        # Fit scaler if normalization is requested
        if self.normalize_features:
            from sklearn.preprocessing import StandardScaler

            # Transform a sample to fit the scaler
            X_sample = self._transform_asns(
                X[: min(100, len(X))]
            )  # Use sample for efficiency
            self._scaler = StandardScaler()
            self._scaler.fit(X_sample)

        return self

    def transform(self, X):
        """
        Transform ASN numbers to feature vectors.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            ASN numbers to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_out_)
            Transformed feature vectors.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        X_transformed = self._transform_asns(X)

        if self.normalize_features and self._scaler is not None:
            X_transformed = self._scaler.transform(X_transformed)

        return X_transformed

    def _transform_asns(self, X: np.ndarray) -> np.ndarray:
        """Transform ASN array to feature matrix."""
        features_list = []

        for asn_array in X:
            asn = int(asn_array[0])
            features = []

            # ASRank features
            if self.include_asrank:
                asrank_data = self.data_access.get_asrank_data(asn)
                if asrank_data:
                    features.extend(
                        [
                            asrank_data.rank or 0,
                            asrank_data.longitude or 0,
                            asrank_data.latitude or 0,
                            asrank_data.cone_asns or 0,
                            asrank_data.cone_prefixes or 0,
                            asrank_data.cone_addresses or 0,
                            asrank_data.degree_provider or 0,
                            asrank_data.degree_peer or 0,
                            asrank_data.degree_customer or 0,
                            asrank_data.degree_total or 0,
                            asrank_data.announcing_prefixes or 0,
                            asrank_data.announcing_addresses or 0,
                        ]
                    )
                else:
                    features.extend([0] * 12)

            # PeeringDB features
            if self.include_peeringdb:
                peeringdb_data = self.data_access.get_peeringdb_data(asn)
                if peeringdb_data:
                    features.extend(
                        [
                            1 if peeringdb_data.website else 0,
                            1 if peeringdb_data.looking_glass else 0,
                            1 if peeringdb_data.info_unicast else 0,
                            1 if peeringdb_data.info_multicast else 0,
                            1 if peeringdb_data.info_ipv6 else 0,
                        ]
                    )
                else:
                    features.extend([0] * 5)

            # ASPOP features
            if self.include_aspop:
                aspop_data = self.data_access.get_aspop_data(asn)
                if aspop_data:
                    features.extend(
                        [
                            aspop_data.customer_cone_asns or 0,
                            aspop_data.customer_cone_prefixes or 0,
                            aspop_data.customer_cone_addresses or 0,
                        ]
                    )
                else:
                    features.extend([0] * 3)

            features_list.append(features)

        return np.array(features_list, dtype=float)

    def _get_feature_names(self) -> List[str]:
        """Get feature names for the output."""
        feature_names = []

        if self.include_asrank:
            feature_names.extend(
                [
                    "asrank_rank",
                    "asrank_longitude",
                    "asrank_latitude",
                    "asrank_cone_asns",
                    "asrank_cone_prefixes",
                    "asrank_cone_addresses",
                    "asrank_degree_provider",
                    "asrank_degree_peer",
                    "asrank_degree_customer",
                    "asrank_degree_total",
                    "asrank_announcing_prefixes",
                    "asrank_announcing_addresses",
                ]
            )

        if self.include_peeringdb:
            feature_names.extend(
                [
                    "peeringdb_has_website",
                    "peeringdb_has_looking_glass",
                    "peeringdb_info_unicast",
                    "peeringdb_info_multicast",
                    "peeringdb_info_ipv6",
                ]
            )

        if self.include_aspop:
            feature_names.extend(
                [
                    "aspop_customer_cone_asns",
                    "aspop_customer_cone_prefixes",
                    "aspop_customer_cone_addresses",
                ]
            )

        return feature_names

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self)
        return np.array(self.feature_names_)


class HybridASNClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible hybrid AS classification estimator.

    This class provides a familiar fit/predict interface that combines
    SVM and LLM approaches using the HybridASClassifier, making it
    compatible with scikit-learn pipelines and evaluation tools.

    Parameters
    ----------
    approach : str, default "hybrid"
        Classification approach: 'flat', 'hierarchical', or 'hybrid'.
    svm_weight : float, default 0.4
        Weight for SVM predictions in ensemble.
    llm_weight : float, default 0.6
        Weight for LLM predictions in ensemble.
    use_hierarchical_svm : bool, default True
        Whether to use hierarchical SVM approach.
    llm_model_id : str, optional
        LLM model identifier for inference.
    svm_params : dict, optional
        Parameters for SVM classifier.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    n_features_in_ : int
        Number of input features.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of input features.
    """

    def __init__(
        self,
        approach: str = "hybrid",
        svm_weight: float = 0.4,
        llm_weight: float = 0.6,
        use_hierarchical_svm: bool = True,
        llm_model_id: Optional[str] = None,
        svm_params: Optional[Dict] = None,
    ):
        self.approach = approach
        self.svm_weight = svm_weight
        self.llm_weight = llm_weight
        self.use_hierarchical_svm = use_hierarchical_svm
        self.llm_model_id = llm_model_id
        self.svm_params = svm_params

        # Initialize components
        self.hybrid_classifier = None
        self.label_binarizer = MultiLabelBinarizer()

        # Fitted attributes
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Fit the hybrid AS classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            ASN numbers to fit on.
        y : array-like of shape (n_samples, n_labels), optional
            Target labels. For hybrid training, this should include
            hierarchical tag information.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        X = check_array(X, accept_sparse=False, dtype=None)

        if X.shape[1] != 1:
            raise ValueError("X must have exactly 1 feature (ASN column)")

        # Store input information
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(["asn"])

        # Convert ASNs to DataFrame
        asns_df = pd.DataFrame(X, columns=["asn"])

        # Set up classes for unified system
        if self.approach == "hierarchical":
            # Use hierarchical tags as classes
            all_tags = [tag.value for tag in HierarchicalTags]
        else:
            # Use top-level tags as classes
            all_tags = [tag.value for tag in TopLevelTags]

        self.classes_ = np.array(all_tags)

        # Initialize hybrid classifier
        self.hybrid_classifier = HybridASClassifier(
            approach=self.approach,
            svm_weight=self.svm_weight,
            llm_weight=self.llm_weight,
            use_hierarchical_svm=self.use_hierarchical_svm,
            llm_model_id=self.llm_model_id,
            svm_params=self.svm_params,
        )

        # If we have labels, use them for training
        if y is not None:
            y = check_array(y, accept_sparse=False, force_all_finite=False)

            # Convert y to DataFrame format expected by hybrid classifier
            if y.ndim == 1:
                # Single label case - convert to multi-label format
                y_df = pd.DataFrame(0, index=range(len(y)), columns=self.classes_)
                for i, label_idx in enumerate(y):
                    if 0 <= label_idx < len(self.classes_):
                        y_df.iloc[i, label_idx] = 1
            else:
                # Multi-label case
                y_df = pd.DataFrame(y, columns=self.classes_[: y.shape[1]])

            # Add ASN column
            y_df["asn"] = asns_df["asn"]

            # Train hybrid classifier
            self.hybrid_classifier.fit(asns_df, y_df)
        else:
            # No training data - just initialize for prediction
            # This assumes pre-trained models are available
            pass

        return self

    def predict(self, X):
        """
        Predict AS classifications.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            ASN numbers to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_classes)
            Predicted multi-label binary matrix.
        """
        check_is_fitted(self, attributes=["classes_", "hybrid_classifier"])
        X = check_array(X, accept_sparse=False, dtype=None)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        # Convert to DataFrame
        asns_df = pd.DataFrame(X, columns=["asn"])

        # Get predictions from hybrid classifier
        try:
            predictions = self.hybrid_classifier.predict(asns_df)
            return predictions
        except Exception:
            # Fallback to unified predictions if regular predict fails
            unified_results = self.hybrid_classifier.predict_unified(asns_df)

            # Convert unified results to binary matrix
            y_pred = np.zeros((len(asns_df), len(self.classes_)))

            for i, result in enumerate(unified_results):
                # Get all predicted tags
                all_tags = []
                if self.approach == "hierarchical":
                    all_tags.extend([tag.value for tag in result.hierarchical_tags])
                else:
                    all_tags.extend([tag.value for tag in result.top_level_tags])

                # Set binary indicators
                for tag in all_tags:
                    if tag in self.classes_:
                        tag_idx = np.where(self.classes_ == tag)[0]
                        if len(tag_idx) > 0:
                            y_pred[i, tag_idx[0]] = 1

            return y_pred

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            ASN numbers to classify.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Probability matrix.
        """
        check_is_fitted(self, attributes=["classes_", "hybrid_classifier"])
        X = check_array(X, accept_sparse=False, dtype=None)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        # Convert to DataFrame
        asns_df = pd.DataFrame(X, columns=["asn"])

        # Get probabilities from hybrid classifier
        try:
            probabilities = self.hybrid_classifier.predict_proba(asns_df)
            return probabilities
        except Exception:
            # Fallback to binary predictions as probabilities
            predictions = self.predict(X)
            return predictions.astype(float)

    def decision_function(self, X):
        """
        Compute decision function (same as predict_proba for this classifier).
        """
        return self.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean Jaccard score on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Test ASNs.
        y : array-like of shape (n_samples, n_classes)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights (not supported, will be ignored).

        Returns
        -------
        score : float
            Mean Jaccard score.
        """
        from sklearn.metrics import jaccard_score

        y_pred = self.predict(X)
        return jaccard_score(y, y_pred, average="macro", zero_division=0)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self)
        return self.classes_
