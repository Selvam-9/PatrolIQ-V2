import sys
import os
import json
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.pipeline.inference_pipeline import InferencePipeline

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("📊 Model Performance Monitoring")

@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipeline = load_pipeline()
df = pipeline.get_dataset()

# --------------------------
# Load Metrics
# --------------------------
METRICS_PATH = "artifacts/reports/metrics.json"

if not os.path.exists(METRICS_PATH):
    st.error("metrics.json not found. " "Please run training pipeline.")
    st.stop()
with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

# --------------------------
# Model Configuration
# --------------------------
st.subheader("Model Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("KMeans Clusters", metrics["n_clusters"])
with col2:
    st.metric("PCA Components", metrics["pca_components"])
with col3:
    st.metric("Records", f"{len(df):,}")

# --------------------------
# Clustering Metrics
# --------------------------
st.subheader("Clustering Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Silhouette Score", round(metrics["silhouette_score"], 3))
with col2:
    st.metric("Davies-Bouldin Score", round(metrics["davies_bouldin_score"], 3))

# --------------------------
# PCA Metrics
# --------------------------
st.subheader("PCA Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("PC1 Variance", f"{metrics['pca_component_1_variance']:.2%}")
with col2:
    st.metric("PC2 Variance", f"{metrics['pca_component_2_variance']:.2%}")
with col3:
    st.metric("Total Variance", f"{metrics['pca_total_variance']:.2%}")

# --------------------------
# Interpretation
# --------------------------
st.subheader("Model Interpretation")
st.info(
    f"""
    • Silhouette Score:
      {metrics['silhouette_score']:.3f}

    • Davies-Bouldin Score:
      {metrics['davies_bouldin_score']:.3f}

    • PCA retained
      {metrics['pca_total_variance']:.2%}
      of total variance.

    Higher Silhouette and
    lower Davies-Bouldin values
    indicate stronger cluster separation.
    """
)
