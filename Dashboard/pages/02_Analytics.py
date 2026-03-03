import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import sys
from sklearn.decomposition import PCA

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
from styles import apply
apply()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
# ensure src on path for risk
SRC_PATH = BASE_DIR / "src"
sys.path.append(str(SRC_PATH))
from risk import calculate_risk

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# load data

df = load_data()

st.markdown("<h1>📊 Analytics</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# correlation heatmap
corr = df.corr()
fig_corr = px.imshow(corr, title="Correlation Heatmap", color_continuous_scale="RdBu_r")
fig_corr.update_layout(template="plotly_dark")
st.plotly_chart(fig_corr, use_container_width=True)

# fraud amount trend area chart
if "Time" in df.columns:
    df_temp = df.copy()
    df_temp["TimeIndex"] = pd.to_datetime(df_temp["Time"], unit="s", origin="unix")
    trend = df_temp.resample("1H", on="TimeIndex")["Amount"].sum().reset_index()
    fig_area = px.area(trend, x="TimeIndex", y="Amount", title="Fraud Amount Trend", color_discrete_sequence=["red"])
    fig_area.update_layout(template="plotly_dark")
    st.plotly_chart(fig_area, use_container_width=True)

# feature importance
try:
    model = joblib.load(MODEL_PATH)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        feat = df.drop(columns=["Class"]).columns
        imp_df = pd.DataFrame({"feature": feat, "importance": imp})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Feature Importance",
                         color_discrete_sequence=["#0af"])
        fig_imp.update_layout(template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)
except Exception as e:
    st.error(f"Could not load model for feature importance: {e}")

# risk segmentation
probs = []
if "Class" in df.columns:
    # compute sample risk scores for segmentation
    X = df.drop(columns=["Class"])
    scaler_path = BASE_DIR / "models" / "scaler.pkl"
    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(MODEL_PATH)
        arr = scaler.transform(X)
        probs = model.predict_proba(arr)[:,1]
    except Exception:
        probs = np.zeros(len(df))
risk_scores = [calculate_risk(p) for p in probs]
labels = pd.cut(risk_scores, bins=[-1,33,66,100], labels=["Low","Medium","High"])
seg = labels.value_counts().reset_index()
seg.columns=["Risk","Count"]
fig_seg = px.bar(seg, x="Risk", y="Count", title="Risk Segmentation", color="Risk",
                 color_discrete_map={"Low":"green","Medium":"orange","High":"red"})
fig_seg.update_layout(template="plotly_dark")
st.plotly_chart(fig_seg, use_container_width=True)
