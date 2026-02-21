# 💳 Financial Fraud Detection System

An end-to-end Machine Learning project to detect fraudulent financial transactions using Random Forest.

## 🚀 Features
- Handles imbalanced dataset using SMOTE
- Random Forest model with high ROC-AUC (~0.98)
- Streamlit interactive UI
- Production-ready folder structure
- Model persistence using Joblib

## 📊 Model Performance
- ROC-AUC: ~0.98
- Fraud Recall: 0.86
- Fraud Precision: 0.42

## 🛠 Tech Stack
- Python
- Scikit-learn
- Pandas
- Streamlit
- SMOTE (Imbalanced-learn)

## ▶ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py