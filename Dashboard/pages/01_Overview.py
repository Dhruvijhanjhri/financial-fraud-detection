import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
from styles import apply
apply()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# load

df = load_data()
# downsample for heavy plots
sample_df = df.sample(n=min(10000, len(df)), random_state=1)

# compute metrics

total = len(df)
fraud = int(df["Class"].sum())
nonfraud = total - fraud
fraud_pct = round(fraud / total * 100, 2)

# header
st.markdown(f"<div class='glass-card'><h2>Fraud Rate: {fraud_pct}%</h2></div>", unsafe_allow_html=True)

# KPI cards
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='glass-card'><h3>Total Transactions</h3><h2>{total}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='glass-card'><h3>Fraudulent</h3><h2>{fraud}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='glass-card'><h3>Non-Fraud</h3><h2>{nonfraud}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

@st.cache_data
def make_donut(data):
    fig = px.pie(data, names=data["Class"].map({0: "Safe", 1: "Fraud"}), title="Fraud vs Non-Fraud",
                 color_discrete_map={"Safe": "green", "Fraud": "red"}, hole=0.4)
    fig.update_layout(template="plotly_dark")
    return fig

@st.cache_data
def make_hist(data):
    fig = px.histogram(data, x="Amount", nbins=50, title="Transaction Amount Distribution")
    fig.update_layout(template="plotly_dark", xaxis=dict(rangeslider=dict(visible=True)))
    return fig

@st.cache_data
def make_violin(data):
    fig = px.violin(data, x="Class", y="Amount", color=data["Class"].map({0: "Safe", 1: "Fraud"}),
                     box=True, points="all", title="Amount Distribution by Class",
                     color_discrete_map={"Safe": "green", "Fraud": "red"})
    fig.update_layout(template="plotly_dark")
    return fig

# donut chart
st.plotly_chart(make_donut(sample_df), use_container_width=True)

# histogram with range slider
st.plotly_chart(make_hist(sample_df), use_container_width=True)

# violin plot
st.plotly_chart(make_violin(sample_df), use_container_width=True)

# boxplot
fig_box = px.box(df, x="Class", y="Amount", title="Amount Distribution Boxplot",
                 color=df["Class"].map({0: "Safe", 1: "Fraud"}),
                 color_discrete_map={"Safe": "green", "Fraud": "red"})
fig_box.update_layout(template="plotly_dark")
st.plotly_chart(fig_box, use_container_width=True)

with st.expander("Dataset Preview"):
    st.dataframe(df.head())
