import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import plotly.express as px

from src.pipeline.inference_pipeline import InferencePipeline

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Temporal Patterns", layout="wide")
st.title("⏰ Temporal Crime Patterns")

# --------------------------
# Cached Loader
# --------------------------
@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipeline = load_pipeline()
df = pipeline.get_dataset()

# --------------------------
# Validation
# --------------------------
required_cols = ["Hour", "Month", "Is_Weekend"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# --------------------------
# Hourly Distribution
# --------------------------
st.subheader("Hourly Crime Distribution")

hourly = df.groupby("Hour").size().reset_index(name="Total Crimes").sort_values("Hour")
fig_hour = px.bar(hourly, x="Hour", y="Total Crimes")

st.plotly_chart(fig_hour, use_container_width=True)

# --------------------------
# Weekday vs Weekend
# --------------------------
st.subheader("Weekday vs Weekend")

weekend_data = df.groupby("Is_Weekend").size().reset_index(name="Total Crimes")
weekend_data["Is_Weekend"] = weekend_data["Is_Weekend"].map({0: "Weekday", 1: "Weekend"})

fig_weekend = px.bar(weekend_data, x="Is_Weekend", y="Total Crimes")
st.plotly_chart(fig_weekend, use_container_width=True)

# --------------------------
# Monthly Trend
# --------------------------
st.subheader("Monthly Crime Trend")

monthly = (
    df.groupby("Month").size().reset_index(name="Total Crimes").sort_values("Month")
)
fig_month = px.line(monthly, x="Month", y="Total Crimes")

st.plotly_chart(fig_month, use_container_width=True)

# --------------------------
# Dataset Metrics
# --------------------------
st.subheader("Summary")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    st.metric("Hours Tracked", df["Hour"].nunique())
