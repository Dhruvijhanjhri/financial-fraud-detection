import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
from styles import apply
apply()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# load

df = load_data()
X = df.drop(columns=["Class"])
y = df["Class"]

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)

    # KPI row
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='glass-card'><h4>Accuracy</h4><h2>{acc:.4f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='glass-card'><h4>Precision</h4><h2>{prec:.4f}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='glass-card'><h4>Recall</h4><h2>{rec:.4f}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")

    # classification report
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # styled confusion matrix
    cm = confusion_matrix(y, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                     title="Confusion Matrix", color_continuous_scale="RdBu")
    fig_cm.update_layout(template="plotly_dark")
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve filled
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, fill='tozeroy', name="ROC", line=dict(color="cyan")))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="white"), showlegend=False))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", template="plotly_dark")
    st.plotly_chart(fig_roc, use_container_width=True)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y, y_prob)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", fill='tozeroy', line=dict(color="magenta")))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark")
    st.plotly_chart(fig_pr, use_container_width=True)

    # model comparison placeholder
    st.markdown("<h3>Model comparison (coming soon)</h3>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Could not compute performance metrics: {e}")
