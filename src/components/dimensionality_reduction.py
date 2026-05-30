import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.logger import get_logger
logger = get_logger(__name__)


class DimensionalityReduction:
    def __init__(self):
        self.pca_model = None
    # --------------------
    # PCA
    # --------------------

    def train_pca(self, X, n_components=None):
        logger.info(f"Training PCA with n_components={n_components}")

        if n_components is not None and n_components > X.shape[1]:
            raise ValueError(
                f"n_components ({n_components}) cannot be greater than "
                f"number of features ({X.shape[1]})"
            )
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(X)
        self.pca_model = pca

        logger.info("PCA training completed")
        return pca, transformed

    # --------------------
    # Explained Variance
    # --------------------

    def explained_variance(self):
        if self.pca_model is None:
            raise ValueError("PCA model not trained")

        variance_ratio = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)

        logger.info(f"Cumulative variance explained: " f"{cumulative_variance[-1]:.4f}")
        return (variance_ratio, cumulative_variance)

    # --------------------
    # Feature Importance
    # --------------------

    def feature_importance(self, feature_names):
        if self.pca_model is None:
            raise ValueError("PCA model not trained")

        loadings = pd.DataFrame(self.pca_model.components_, columns=feature_names)
        importance = loadings.abs().mean(axis=0).sort_values(ascending=False)

        logger.info("PCA feature importance calculated")
        return importance

    # --------------------
    # t-SNE
    # --------------------

    def run_tsne(self, X, n_samples=10000):
        logger.info("Starting t-SNE transformation")

        if isinstance(X, pd.DataFrame):
            X_sample = X.sample(n=min(n_samples, len(X)), random_state=42)
        else:
            X_sample = X[: min(n_samples, len(X))]
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        transformed = tsne.fit_transform(X_sample)

        logger.info(f"t-SNE completed on " f"{len(X_sample)} samples")
        return transformed
