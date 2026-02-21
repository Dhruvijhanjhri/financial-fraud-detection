import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import load_data, split_data, scale_data, apply_smote
import os

def train_model():
    # Load data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

    df = load_data(DATA_PATH)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # SMOTE
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)

    # Model
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_train_res, y_train_res)

    # Evaluation
    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    # Save
    # Create models folder if not exists
    MODELS_PATH = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_PATH, exist_ok=True)

    model_file = os.path.join(MODELS_PATH, "fraud_model.pkl")
    scaler_file = os.path.join(MODELS_PATH, "scaler.pkl")

    joblib.dump(rf, model_file)
    joblib.dump(scaler, scaler_file)



    print("\nModel saved successfully!")


if __name__ == "__main__":
    train_model()