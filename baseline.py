import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from model import *

X, y, feats = load_data()
print("N", len(X), "features", len(feats), "missing", int(X.isna().sum().sum()))
ym = (y == MALIGNANT).astype(int)
print("malignant", ym.sum(), "benign", (1-ym).sum())
Xtr, Xte, ytr, yte = split(X, y)
print("train", len(Xtr), "malig", int((ytr==0).sum()), "test", len(Xte), "malig", int((yte==0).sum()))
m = train(Xtr, ytr)
print("n_iter", m.named_steps['clf'].n_iter_)
p = malignant_proba(m, Xte)
yt = (yte == MALIGNANT).astype(int).values
yp = predict_label(p)
tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
print(f"TP {tp} FN {fn} FP {fp} TN {tn}")
print("sens", tp/(tp+fn), "spec", tn/(tn+fp), "acc", accuracy_score(yt, yp), "auroc", roc_auc_score(yt, p))
print("PPV", tp/(tp+fp), "NPV", tn/(tn+fn))
# train metrics
ptr = malignant_proba(m, Xtr); ytt=(ytr==MALIGNANT).astype(int).values
print("train acc", accuracy_score(ytt, predict_label(ptr)), "train auroc", roc_auc_score(ytt, ptr))
# boundary distribution
d = np.abs(p-0.5)
print("dist bins", np.histogram(d, bins=[0,0.1,0.2,0.3,0.4,0.5001])[0])
# summary stats for a few features
desc = X.describe().T[['mean','std','min','max']]
desc['cv'] = desc['std']/desc['mean']
print(desc.sort_values('cv').to_string())
# by class means for key features
print(X.groupby(y).mean().T.head(10))
