import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
from styles import apply
apply()

# ensure src directory in path
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_PATH = BASE_DIR / "src"
sys.path.append(str(SRC_PATH))

from predict import predict_transaction, load_model_scaler
from risk import calculate_risk

st.markdown("<h1>🔮 Prediction</h1>", unsafe_allow_html=True)

# load data for slider ranges
@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "creditcard.csv")

df = load_data()

v_cols = [c for c in df.columns if c.startswith("V")]
sliders = {}
with st.form("input_form"):
    amount = st.number_input("Amount", min_value=0.0, value=1.0)
    for v in v_cols[:6]:
        mn = float(df[v].min())
        mx = float(df[v].max())
        sliders[v] = st.slider(v, min_value=mn, max_value=mx, value=0.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        features = [0]
        for v in v_cols[:6]:
            features.append(sliders[v])
        remaining = 30 - len(features) - 1
        features += [0] * remaining
        features.append(amount)
        pred, prob = predict_transaction(features)
        score = calculate_risk(prob)

        if pred == 0:
            st.markdown("<div class='glass-card' style='border-left:5px solid green;'><strong>Transaction predicted as SAFE.</strong></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='glass-card' style='border-left:5px solid red;'><strong>Transaction predicted as FRAUD.</strong></div>", unsafe_allow_html=True)

        # animated risk score
        st.markdown(f"<h3>Risk Score: <span class='animated'>{score}%</span></h3>", unsafe_allow_html=True)
        st.markdown("<style>@keyframes pulse{0%{opacity:1;}50%{opacity:.4;}100%{opacity:1;}}.animated{animation:pulse 2s infinite;}</style>", unsafe_allow_html=True)

        # gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            title={'text': "Fraud Probability"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red"}},
        ))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError as e:
        st.error(f"Model not available: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
