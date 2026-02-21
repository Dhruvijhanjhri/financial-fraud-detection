import streamlit as st
import numpy as np
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(page_title="Financial Fraud Detection", layout="wide")

st.title("💳 Financial Fraud Detection System")
st.write("Enter transaction details below:")

# Create 30 feature inputs (V1–V28 + Time + Amount)
features = []

col1, col2 = st.columns(2)

for i in range(30):
    with col1 if i < 15 else col2:
        value = st.number_input(f"Feature {i+1}", value=0.0)
        features.append(value)

if st.button("Predict"):
    input_array = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected! Risk Score: {probability:.2f}")
    else:
        st.success(f"✅ Transaction is Safe. Risk Score: {probability:.2f}")