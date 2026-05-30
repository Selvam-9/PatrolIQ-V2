# 🚔 PatrolIQ – Crime Intelligence Platform

PatrolIQ is an AI-powered system designed to analyze and visualize crime patterns using the Chicago Crime Dataset.

---

## 🎯 Project Features
* **📍 Hotspot Detection:** Uses **K-Means Clustering** to find high-density crime areas.
* **⏰ Trend Analysis:** Explores hourly, weekly, and seasonal crime frequencies.
* **📉 2D Visualization:** Uses **PCA** to reduce complex data for easy plotting.
* **📊 Model Tracking:** Uses **MLflow** to log and evaluate machine learning metrics.

---

## 🖥️ Dashboard Pages
The Streamlit app contains four simple views:
1. **Geographic Hotspots** – Interactive crime cluster maps.
2. **Temporal Patterns** – Charts showing peak crime times.
3. **PCA Visualization** – 2D scatter plots of crime features.
4. **Model Performance** – Saved metrics like Silhouette Score and Davies-Bouldin Index.
