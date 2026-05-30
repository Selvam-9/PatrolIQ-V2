import os
import joblib
import mlflow
import mlflow.sklearn
import json

from sklearn.preprocessing import StandardScaler

from src.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering
from src.components.clustering import GeoClustering
from src.components.dimensionality_reduction import DimensionalityReduction


class TrainingPipeline:
    def __init__(self):
        self.logger = get_logger(__name__)

    def run(self):
        self.logger.info("Starting PatrolIQ Training Pipeline")

        # ---------------------------------
        # MLflow Setup
        # ---------------------------------
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("PatrolIQ_Clustering")

        # ---------------------------------
        # Artifact Folder Structure
        # ---------------------------------

        MODELS_DIR = os.path.join("artifacts", "models")
        PROCESSED_DIR = os.path.join("artifacts", "processed")
        REPORTS_DIR = os.path.join("artifacts", "reports")
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

        # ---------------------------------
        # Data Ingestion
        # ---------------------------------
        ingestion = DataIngestion(
            raw_path=("data/raw/Crimes_-_2001_to_Present_20260115.csv"),
            processed_path=("data/processed/sample.csv"),
        )

        df_sample = ingestion.run()
        self.logger.info(f"Sample Data Shape: {df_sample.shape}")

        # ---------------------------------
        # Data Preprocessing
        # ---------------------------------
        preprocessing = DataPreprocessing()
        df_clean = preprocessing.run(df_sample)
        self.logger.info(f"Clean Data Shape: {df_clean.shape}")

        # ---------------------------------
        # Feature Engineering
        # ---------------------------------
        feature_engineering = FeatureEngineering()
        df_featured = feature_engineering.run(df_clean)
        self.logger.info(f"Feature Data Shape: {df_featured.shape}")

        # ---------------------------------
        # Geographic Scaling
        # ---------------------------------
        geo_scaler = StandardScaler()
        geo_features = df_featured[["Latitude", "Longitude"]]

        scaled_geo = geo_scaler.fit_transform(geo_features)
        df_featured["Lat_scaled"] = scaled_geo[:, 0]
        df_featured["Long_scaled"] = scaled_geo[:, 1]

        self.logger.info("Geographic scaling completed")

        # ---------------------------------
        # KMeans Clustering
        # ---------------------------------

        geo_X = df_featured[["Lat_scaled", "Long_scaled", "Crime_Severity_Score"]]
        geo_sample = geo_X.sample(n=min(50000, len(geo_X)), random_state=42)

        clustering = GeoClustering()
        kmeans_model, _ = clustering.train_kmeans(geo_X, n_clusters=6)
        sample_labels = kmeans_model.predict(geo_sample)
        sil, db = clustering.evaluate(geo_sample, sample_labels)

        self.logger.info(f"Silhouette Score: {sil}")
        self.logger.info(f"Davies Bouldin Score: {db}")

        # ---------------------------------
        # PCA Analysis
        # ---------------------------------

        pca_features = df_featured[
            [
                "Lat_scaled",
                "Long_scaled",
                "Hour",
                "DayOfWeek_Num",
                "Month",
                "Is_Weekend",
                "Crime_Severity_Score",
            ]
        ].dropna()

        dr = DimensionalityReduction()
        pca_model, _ = dr.train_pca(pca_features, n_components=2)
        (variance_ratio, cumulative_variance) = dr.explained_variance()

        self.logger.info(f"Cumulative Variance: " f"{cumulative_variance[-1]}")

        # ---------------------------------
        # MLflow Tracking
        # ---------------------------------
        with mlflow.start_run():

            mlflow.log_param("geo_kmeans_clusters", 6)

            mlflow.log_param("pca_components", 2)

            mlflow.log_metric("silhouette_score", sil)

            mlflow.log_metric("davies_bouldin_score", db)

            mlflow.log_metric("pca_component_1_variance", variance_ratio[0])

            mlflow.log_metric("pca_component_2_variance", variance_ratio[1])

            mlflow.log_metric("pca_cumulative_variance", cumulative_variance[-1])

            mlflow.sklearn.log_model(kmeans_model, artifact_path="geo_kmeans_model")

            mlflow.sklearn.log_model(pca_model, artifact_path="pca_model")

        # ---------------------------------
        # Save Models
        # ---------------------------------
        joblib.dump(geo_scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        joblib.dump(kmeans_model, os.path.join(MODELS_DIR, "geo_kmeans_model.pkl"))
        joblib.dump(pca_model,os.path.join(MODELS_DIR, "pca_model.pkl"))

        self.logger.info("Models saved successfully")

        # ---------------------------------
        # Save Processed Dataset
        # ---------------------------------
        df_featured.to_csv(os.path.join(PROCESSED_DIR, "processed_dataset.csv"), index=False)
                
        # ---------------------------------
        # Save Metrics
        # ---------------------------------

        metrics = {
            "n_clusters": 6,
            "pca_components": 2,
            "silhouette_score": float(sil),
            "davies_bouldin_score": float(db),
            "pca_component_1_variance": float(variance_ratio[0]),
            "pca_component_2_variance": float(variance_ratio[1]),
            "pca_total_variance": float(cumulative_variance[-1]),
        }

        with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        self.logger.info("Metrics saved successfully")
        self.logger.info("Training Pipeline Completed Successfully")


