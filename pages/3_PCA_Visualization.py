import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import plotly.express as px
import pandas as pd

from src.pipeline.inference_pipeline import InferencePipeline


# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(page_title="PCA Visualization", layout="wide")
st.title("📉 PCA Crime Pattern Visualization")

# ----------------------------------
# Cached Loader
# ----------------------------------
@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipeline = load_pipeline()
df = pipeline.get_dataset()
kmeans_model, pca_model = pipeline.get_models()

# ----------------------------------
# Validation
# ----------------------------------
required_cols = [
    "Lat_scaled",
    "Long_scaled",
    "Hour",
    "DayOfWeek_Num",
    "Month",
    "Is_Weekend",
    "Crime_Severity_Score",
]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# ----------------------------------
# Generate Cluster Labels
# ----------------------------------
geo_features = df[["Lat_scaled", "Long_scaled", "Crime_Severity_Score"]]
clusters = kmeans_model.predict(geo_features)
df_clustered = df.copy()
df_clustered["Cluster"] = clusters

# ----------------------------------
# PCA Features
# ----------------------------------
pca_features = df_clustered[
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

# ----------------------------------
# PCA Transform
# ----------------------------------
pca_transformed = pca_model.transform(pca_features)

pca_df = pd.DataFrame(pca_transformed, columns=["PC1", "PC2"])
pca_df["Cluster"] = df_clustered.loc[pca_features.index, "Cluster"]

# ----------------------------------
# PCA Scatter Plot
# ----------------------------------
explained = pca_model.explained_variance_ratio_
# st.subheader("2D PCA Projection")

# sample_size = min(20000, len(pca_df))
# pca_sample = pca_df.sample(n=sample_size, random_state=42)

# fig = px.scatter(pca_sample, x="PC1", y="PC2", color="Cluster", opacity=0.6)
# st.plotly_chart(fig, use_container_width=True)

feature_names = [
    "Lat_scaled",
    "Long_scaled",
    "Hour",
    "DayOfWeek_Num",
    "Month",
    "Is_Weekend",
    "Crime_Severity_Score",
]

loadings = pd.DataFrame(
    pca_model.components_.T, columns=["PC1", "PC2"], index=feature_names
)

loadings["Importance"] = loadings.abs().mean(axis=1)

loadings = loadings.sort_values(by="Importance", ascending=False)

st.subheader(
    "Feature Contribution"
)

fig = px.bar(
    loadings,
    x=loadings.index,
    y="Importance"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(loadings)

# ----------------------------------
# Explained Variance
# ----------------------------------
explained = pca_model.explained_variance_ratio_

st.subheader("Explained Variance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("PC1 Variance", f"{explained[0]:.2%}")
with col2:
    st.metric("PC2 Variance", f"{explained[1]:.2%}")
with col3:
    st.metric("Total Variance", f"{explained.sum():.2%}")

# ----------------------------------
# PCA Dataset Summary
# ----------------------------------
st.subheader("PCA Summary")

summary_df = pd.DataFrame(
    {
        "Metric": ["Records Used", "Components", "Clusters"],
        "Value": [len(pca_features), 2, pca_df["Cluster"].nunique()],
    }
)
st.dataframe(summary_df, use_container_width=True)
