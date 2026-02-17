import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
from src.pipeline.inference_pipeline import InferencePipeline

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("📊 Model Performance Monitoring")


pipeline = InferencePipeline()
_, pca_model = pipeline.get_models()

st.subheader("Model Configuration")

st.json({
    "Geo KMeans Clusters": 6,
    "PCA Components": 2
})

st.subheader("Model Metrics")

explained = pca_model.explained_variance_ratio_

st.json({
    "PCA Component 1 Variance": float(explained[0]),
    "PCA Component 2 Variance": float(explained[1]),
    "Total PCA Variance": float(explained.sum()),
})
