import streamlit as st
import pandas as pd
from pathlib import Path

# page configuration
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="💳",
    layout="wide",
)

from styles import apply

# custom CSS
apply()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

@st.cache_data

def load_data():
    return pd.read_csv(DATA_PATH)


# sidebar content
st.sidebar.title("Financial Fraud Detection")
st.sidebar.markdown("A multi-page dashboard for monitoring and analysis of transaction fraud.")

df = load_data()
total = len(df)
fraud = int(df["Class"].sum())
nonfraud = total - fraud
fraud_pct = round(fraud / total * 100, 2)
accuracy = "--"

# sidebar metrics
st.sidebar.metric("Total Transactions", total)
st.sidebar.metric("Fraud Transactions", fraud, delta=f"{fraud_pct}%")
st.sidebar.metric("Fraud %", f"{fraud_pct}%")
st.sidebar.metric("Model Accuracy", accuracy)

# hero section
st.markdown("<div class='hero'>", unsafe_allow_html=True)
st.markdown("<h1>Financial Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p>A comprehensive enterprise-grade dashboard for monitoring, analyzing and predicting transaction fraud.</p>", unsafe_allow_html=True)
st.markdown(f"<div class='progress-bar'><span class='progress-fill' style='--pct:{fraud_pct}%;'></span></div>", unsafe_allow_html=True)
st.markdown(f"<p>Fraud Rate: {fraud_pct}%</p>", unsafe_allow_html=True)
st.markdown("<a href='?page=Analytics'><button class='btn'>Go to Analytics</button></a>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# KPI cards below hero
col1, col2, col3 = st.columns(3)
for col,name,val in zip([col1,col2,col3],["Total Transactions","Fraud Transactions","Model Accuracy"],[total,fraud,accuracy]):
    col.markdown(f"<div class='glass-card'><h3>{name}</h3><h2>{val}</h2></div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<h2>Executive Summary</h2>", unsafe_allow_html=True)
st.markdown("<p>This dashboard provides real-time insights into transaction patterns, fraud rates, model performance and risk intelligence. Ideal for banking and financial compliance teams.</p>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'><h3>Risk Intelligence Score</h3><h2>--</h2></div>", unsafe_allow_html=True)

