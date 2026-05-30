import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage

from src.logger import get_logger
logger = get_logger(__name__)


class GeoClustering:
    def __init__(self):
        pass
    # --------------------
    # KMeans
    # --------------------

    def train_kmeans(self, X, n_clusters=4):
        logger.info(f"Training KMeans with {n_clusters} clusters")

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        logger.info("KMeans training completed")
        return model, labels

    # --------------------
    # DBSCAN
    # --------------------

    def train_dbscan(self, X, eps=0.3, min_samples=30):

        logger.info("Training DBSCAN")

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        logger.info("DBSCAN training completed")
        return model, labels

    # --------------------
    # Hierarchical
    # --------------------

    def hierarchical_linkage(self, X):

        logger.info("Generating hierarchical linkage matrix")

        Z = linkage(X, method="ward")
        return Z

    # --------------------
    # Evaluation
    # --------------------

    def evaluate(self, X, labels):
        unique_labels = set(labels)
        if len(unique_labels) < 2:

            logger.warning("Cannot compute clustering metrics with less than 2 clusters")
            return None, None

        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        logger.info(f"Silhouette Score: {sil:.4f}")
        logger.info(f"Davies-Bouldin Score: {db:.4f}")
        return sil, db
    # --------------------
    # Elbow Method
    # --------------------

    def elbow(self, X, k_range=range(2, 10)):
        inertias = {}
        logger.info("Calculating elbow curve")

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(X)
            inertias[k] = model.inertia_
        return inertias
