"""robustness.py — 특징별 교란 및 전후 결과 계산"""
import numpy as np
import pandas as pd
from model import malignant_proba, predict_label


def perturb(X, feature, delta):
    """지정 변수 하나에만 x * (1 + delta) 적용. 원본은 바꾸지 않는다."""
    Xp = X.copy()
    Xp[feature] = Xp[feature] * (1.0 + delta)
    return Xp


def compare(model, X, feature, delta, threshold=0.5):
    p0 = malignant_proba(model, X)
    p1 = malignant_proba(model, perturb(X, feature, delta))
    l0, l1 = predict_label(p0, threshold), predict_label(p1, threshold)
    return pd.DataFrame({
        "case_id": X.index,
        "feature": feature,
        "delta": delta,
        "p_orig": p0,
        "p_pert": p1,
        "dp": p1 - p0,
        "label_orig": l0,
        "label_pert": l1,
        "flipped": l0 != l1,
    })


def stability_check(model, x_row, features, max_delta, threshold=0.5):
    """한 사례에서 ±max_delta 교란 시 판정이 반전되는 특징 목록 (자동 안정성 검사)"""
    rows = []
    for f in features:
        for d in (+max_delta, -max_delta):
            r = compare(model, x_row, f, d, threshold).iloc[0]
            rows.append(r)
    out = pd.DataFrame(rows)
    return out
