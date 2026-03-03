import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# ensure src path
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_PATH = BASE_DIR / "src"
sys.path.append(str(SRC_PATH))

from predict import load_model_scaler, predict_transaction
from risk import calculate_risk

st.title("Risk Analysis")

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "creditcard.csv")

df = load_data()

# compute risk scores for sample or full data
try:
    model, scaler = load_model_scaler()
    X = df.drop(columns=["Class"])
    arr = scaler.transform(X)
    probs = model.predict_proba(arr)[:, 1]
    risks = [calculate_risk(p) for p in probs]
    df_risk = df.copy()
    df_risk["RiskScore"] = risks
except Exception as e:
    st.error(f"Could not load model/scaler: {e}")
    df_risk = df.copy()
    df_risk["RiskScore"] = 0

# show histogram of risk
fig = px.histogram(df_risk, x="RiskScore", nbins=50, title="Risk Score Distribution",
                   color_discrete_sequence=["orange"])
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# show high risk transactions
threshold = st.slider("Risk threshold", min_value=0, max_value=100, value=50)
high = df_risk[df_risk["RiskScore"] >= threshold]
st.markdown(f"### Transactions with risk &ge; {threshold}% ({len(high)})")
st.dataframe(high.head(20))

with st.expander("Full risk-scored dataset"):
    st.dataframe(df_risk.head())
