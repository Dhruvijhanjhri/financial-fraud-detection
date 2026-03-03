import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
from styles import apply
apply()

from predict import predict_transaction, load_model_scaler
from risk import calculate_risk

BASE_DIR = Path(__file__).resolve().parents[2]
# ensure src path for import
SRC_PATH = BASE_DIR / "src"
sys.path.append(str(SRC_PATH))
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

st.markdown("<h1>📡 Live Monitoring</h1>", unsafe_allow_html=True)

# placeholder for live updates
placeholder = st.empty()

def highlight_risk(df):
    def color_row(row):
        return ['background-color: red' if row.Risk>70 else '' for _ in row]
    return df.style.apply(color_row, axis=1)

with placeholder.container():
    sample = df.sample(5)
    results = []
    for _, row in sample.iterrows():
        features = [0] + row[[c for c in df.columns if c.startswith("V")]].tolist()
        features += [0] * (30 - len(features) - 1)
        features.append(row["Amount"])
        try:
            pred, prob = predict_transaction(features)
        except Exception:
            pred, prob = 0, 0.0
        score = calculate_risk(prob)
        results.append({"Time": pd.Timestamp.now(), "Amount": row["Amount"], "Risk": score, "Prediction": pred})

    results_df = pd.DataFrame(results)

    # fraud alert ticker
    alerts = results_df[results_df["Prediction"] == 1]
    if not alerts.empty:
        st.markdown("<marquee style='color:red;'>FRAUD ALERT! Check transactions immediately.</marquee>", unsafe_allow_html=True)

    # live risk trend
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.extend(results_df["Risk"].tolist())
    hist = pd.DataFrame({"Risk": st.session_state.history, "Index": range(len(st.session_state.history))})
    fig = px.line(hist, x="Index", y="Risk", title="Risk Score Over Time", color_discrete_sequence=["orange"])
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Latest High-Risk Transactions")
    if not results_df.empty:
        top = results_df.sort_values("Risk", ascending=False).head(10)
        st.dataframe(highlight_risk(top))
