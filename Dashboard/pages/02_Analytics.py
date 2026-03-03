import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

st.title("Analytics")

# Correlation heatmap
corr = df.corr()
fig_corr = px.imshow(corr, title="Correlation Heatmap", color_continuous_scale="RdBu_r")
fig_corr.update_layout(template="plotly_dark")
st.plotly_chart(fig_corr, use_container_width=True)

# PCA distribution if possible
from sklearn.decomposition import PCA

features = df.drop(columns=["Class"])
if not features.empty:
    pca = PCA(n_components=2)
    comps = pca.fit_transform(features)
    pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
    pca_df["Class"] = df["Class"]
    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Class"].map({0: "Non-Fraud", 1: "Fraud"}),
                         title="PCA Feature Distribution",
                         color_discrete_map={"Non-Fraud": "green", "Fraud": "red"})
    fig_pca.update_layout(template="plotly_dark")
    st.plotly_chart(fig_pca, use_container_width=True)

# Fraud trend simulation
if "Time" in df.columns:
    df_temp = df.copy()
    df_temp["TimeIndex"] = pd.to_datetime(df_temp["Time"], unit="s", origin="unix")
    trend = df_temp.resample("1H", on="TimeIndex")["Class"].sum().reset_index()
    fig_trend = px.line(trend, x="TimeIndex", y="Class", title="Fraud Trend Over Time")
    fig_trend.update_layout(template="plotly_dark")
    st.plotly_chart(fig_trend, use_container_width=True)

# Feature importance from model
try:
    model = joblib.load(MODEL_PATH)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        feat = df.drop(columns=["Class"]).columns
        imp_df = pd.DataFrame({"feature": feat, "importance": imp})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Feature Importance")
        fig_imp.update_layout(template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)
except Exception as e:
    st.error(f"Could not load model for feature importance: {e}")
