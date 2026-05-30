import joblib
import pandas as pd

class InferencePipeline:
    def __init__(self):
        self.kmeans_model = joblib.load("artifacts/models/geo_kmeans_model.pkl")
        self.pca_model = joblib.load("artifacts/models/pca_model.pkl")
        self.scaler = joblib.load("artifacts/models/scaler.pkl")
        self.dataset = pd.read_csv("artifacts/processed/processed_dataset.csv")

    # --------------------
    # Dashboard Methods
    # --------------------
    def get_dataset(self):
        return self.dataset

    def get_models(self):
        return (self.kmeans_model, self.pca_model)

    # --------------------
    # Prediction Method
    # --------------------
    def predict_cluster(self, latitude, longitude, severity):
        geo_df = pd.DataFrame({"Latitude": [latitude], "Longitude": [longitude]})
        scaled = self.scaler.transform(geo_df)
        features = [[scaled[0][0], scaled[0][1], severity]]
        cluster = self.kmeans_model.predict(features)[0]
        return cluster

