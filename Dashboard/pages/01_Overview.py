import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

# KPI cards
total = len(df)
fraud = int(df["Class"].sum())
nonfraud = total - fraud
col1, col2, col3 = st.columns(3)
col1.metric("Total", total)
col2.metric("Fraud", fraud)
col3.metric("Non-Fraud", nonfraud)

st.markdown("---")

# charts
fig_pie = px.pie(
    df,
    names=df["Class"].map({0: "Non-Fraud", 1: "Fraud"}),
    title="Fraud vs Non-Fraud",
    color_discrete_map={"Non-Fraud": "green", "Fraud": "red"},
)
fig_pie.update_layout(template="plotly_dark")
st.plotly_chart(fig_pie, use_container_width=True)

fig_hist = px.histogram(df, x="Amount", nbins=50, title="Transaction Amount Distribution")
fig_hist.update_layout(template="plotly_dark")
st.plotly_chart(fig_hist, use_container_width=True)

fig_box = px.box(df, x="Class", y="Amount", title="Fraud by Amount",
                 color=df["Class"].map({0: "Non-Fraud", 1: "Fraud"}),
                 color_discrete_map={0: "green", 1: "red"})
fig_box.update_layout(template="plotly_dark")
st.plotly_chart(fig_box, use_container_width=True)

with st.expander("Dataset Preview"):
    st.dataframe(df.head())
