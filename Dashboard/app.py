import streamlit as st
import pandas as pd
from pathlib import Path

# page configuration
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="💳",
    layout="wide",
)

# custom CSS
st.markdown(
    """
    <style>
    .reportview-container .main {
        background-color: #f0f2f6;
    }
    .css-1d391kg {padding-top: 0rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

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
fraud_pct = round(fraud / total * 100, 4)

st.sidebar.metric("Total Transactions", total)
st.sidebar.metric("Fraud Transactions", fraud, delta=f"{fraud_pct}%")
st.sidebar.metric("Fraud %", f"{fraud_pct}%")
st.sidebar.metric("Model Accuracy", "--")

st.title("Financial Fraud Detection System")

# central overview metrics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", total)
col2.metric("Fraud Transactions", fraud)
col3.metric("Fraud %", f"{fraud_pct}%")
col4.metric("Non-Fraud", nonfraud)

st.markdown("---")

st.markdown(
    """ 
    <div style='text-align:center;'>
    <h4>Welcome to the Financial Fraud Detection dashboard. Use the sidebar to explore analytics, predictions, model performance, live monitoring, and risk analysis.</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

