import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path):
    df = pd.read_csv(path)
    return df


def split_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X_train, y_train)