import joblib
import numpy as np
from pathlib import Path


def load_model_scaler():
    """Load the trained model and scaler from disk."""
    base = Path(__file__).parents[1]
    model_path = base / "models" / "fraud_model.pkl"
    scaler_path = base / "models" / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model or scaler file not found. Please train the model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_transaction(features):
    """Make a prediction given a list or array of feature values.

    Args:
        features (list or np.ndarray): Ordered list of 30 feature values (Time, V1..V28, Amount).

    Returns:
        tuple: (prediction, probability_of_fraud)
    """
    model, scaler = load_model_scaler()
    arr = np.array(features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    prob = model.predict_proba(arr_scaled)[0][1]
    return int(pred), float(prob)