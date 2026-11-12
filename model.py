"""model.py — 데이터 분할, 모델 학습 및 예측 (RobustBreast 계획서 4절 사양)"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MALIGNANT = 0  # sklearn WDBC: target 0 = malignant, 1 = benign


def load_data():
    ds = load_breast_cancer(as_frame=True)
    X = ds.data.copy()
    y = ds.target.copy()
    return X, y, list(ds.feature_names)


def split(X, y, seed=42):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def train(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def malignant_proba(model, X):
    idx = list(model.classes_).index(MALIGNANT)
    return model.predict_proba(X)[:, idx]


def predict_label(p_malignant, threshold=0.5):
    """1 = 악성 판정, 0 = 양성 판정"""
    return (np.asarray(p_malignant) >= threshold).astype(int)
