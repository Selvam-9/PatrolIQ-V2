import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import plotly.express as px
import pandas as pd

from src.pipeline.inference_pipeline import InferencePipeline


# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Geographic Hotspots", layout="wide")
st.title("📍 Geographic Crime Hotspots")

# ----------------------------------
# Cached Loader
# ----------------------------------
@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipeline = load_pipeline()

# ----------------------------------
# Load Data & Models
# ----------------------------------

df = pipeline.get_dataset()
kmeans_model, _ = pipeline.get_models()

# ----------------------------------
# Validation
# ----------------------------------

required_cols = [
    "Lat_scaled",
    "Long_scaled",
    "Crime_Severity_Score",
    "Latitude",
    "Longitude",
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()
# ----------------------------------
# Cluster Prediction
# ----------------------------------
df_clustered = df.copy()
geo_features = df_clustered[["Lat_scaled", "Long_scaled", "Crime_Severity_Score"]]
df_clustered["Cluster"] = kmeans_model.predict(geo_features)

# ----------------------------------
# Map Visualization
# ----------------------------------
st.subheader("Crime Hotspot Clusters")

sample_size = min(20000, len(df_clustered))
df_sample = df_clustered.sample(n=sample_size, random_state=42)

fig = px.scatter_mapbox(df_sample, lat="Latitude", lon="Longitude", color="Cluster", zoom=10, height=700)
fig.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# Cluster Summary
# ----------------------------------

st.subheader("Cluster Intelligence Summary")

cluster_summary = (df_clustered.groupby("Cluster")
    .agg(Total_Crimes=("Cluster", "size"), Avg_Severity=("Crime_Severity_Score", "mean")
    ).reset_index().sort_values(by="Total_Crimes", ascending=False))

cluster_summary["Risk_Score"] = (
    cluster_summary["Total_Crimes"] * cluster_summary["Avg_Severity"]
)
cluster_summary = cluster_summary.sort_values(
    by="Risk_Score", ascending=False
).reset_index(drop=True)

# Risk Labels based on ranking

risk_labels = ["Critical", "Very High", "High", "Medium", "Low", "Very Low"]
cluster_summary["Risk_Level"] = risk_labels[: len(cluster_summary)]


# Top Crime Type per Cluster
top_crimes = (
    df_clustered.groupby("Cluster")["Primary Type"]
    .apply(lambda x: ", ".join(x.value_counts().head(3).index.tolist()))
    .reset_index()
)
top_crimes = top_crimes.rename(columns={"Primary Type": "Top 3 Crimes"})

cluster_summary = cluster_summary.merge(top_crimes, on="Cluster", how="left")
cluster_summary = cluster_summary.rename(columns={"Primary Type": "Top Crime Type"})

st.dataframe(cluster_summary, use_container_width=True)

# ----------------------------------
# Dataset Info
# ----------------------------------
st.subheader("Dataset Information")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", f"{len(df_clustered):,}")
with col2:
    st.metric("Number of Clusters", df_clustered["Cluster"].nunique())
