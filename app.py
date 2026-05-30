import streamlit as st

st.set_page_config(
    page_title="PatrolIQ",
    layout="wide"
)

st.title("🚔 PatrolIQ – Crime Intelligence Platform")

st.markdown("""
### 🔍 Overview

PatrolIQ is an AI-powered crime hotspot intelligence system built using:

- 📍 Geographic Clustering (KMeans)
- 📉 Dimensionality Reduction (PCA)
- ⏰ Temporal Crime Pattern Analysis
- 📊 MLflow Experiment Tracking
---

### 🎯 Project Objectives

✔ Identify geographic crime hotspots  
✔ Detect high-risk time periods  
✔ Reduce high-dimensional crime features into interpretable components  
✔ Compare clustering performance using evaluation metrics  
✔ Modular ML architecture  

---

### 🧠 Machine Learning Techniques Used

- **K-Means Clustering** for hotspot detection  
- **Evaluated KMeans** using DBSCAN and Hierarchical Clustering as comparative clustering approaches.  
- **PCA (Principal Component Analysis)** for feature reduction  
- **Silhouette Score & Davies-Bouldin Index** for model evaluation  

---

### 📊 Dashboard Pages

Use the sidebar to explore:

1. **Geographic Hotspots** – Crime cluster map  
2. **Temporal Patterns** – Hourly & seasonal crime trends  
3. **PCA Visualization** – 2D projection of crime features  
4. **Model Performance** – MLflow experiment metrics  

---

### 🏙 Dataset

Chicago Crime Dataset (2001–Present)  
Sample Size: ~500,000 records  
Features Used: 7 engineered analytical features  

---

Built for production deployment using modular architecture and MLflow tracking.
""")
