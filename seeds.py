import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from model import *
from robustness import compare
X, y, feats = load_data()
rows=[]; top=[]
for seed in range(20):
    Xtr, Xte, ytr, yte = split(X, y, seed=seed)
    m = train(Xtr, ytr)
    yt = (yte==MALIGNANT).astype(int).values
    p = malignant_proba(m, Xte); l = predict_label(p)
    tn, fp, fn, tp = confusion_matrix(yt, l, labels=[0,1]).ravel()
    res = pd.concat([compare(m, Xte, f, d) for f in feats for d in (-0.10, 0.10)])
    res["absdp"]=res.dp.abs()
    rank = res.groupby("feature").absdp.mean().sort_values(ascending=False)
    flip_cases = res[res.flipped].case_id.unique()
    dist = pd.Series(np.abs(p-0.5), index=Xte.index)
    near = (dist < 0.2)
    rows.append(dict(seed=seed, sens=tp/(tp+fn), spec=tn/(tn+fp), auroc=roc_auc_score(yt,p),
                     flip_rate10=res.flipped.mean()*100, cases_flip=len(flip_cases),
                     near_cases=int(near.sum()), flip_cases_near=int(near.loc[flip_cases].sum()) if len(flip_cases) else 0,
                     max_dist_flip=float(dist.loc[flip_cases].max()) if len(flip_cases) else np.nan,
                     top1=rank.index[0], top3=list(rank.index[:3])))
    top.append(list(rank.index[:5]))
df = pd.DataFrame(rows)
print(df.drop(columns=["top3"]).to_string())
print(df[["sens","spec","auroc","flip_rate10","cases_flip"]].describe().loc[["mean","std","min","max"]])
from collections import Counter
c1 = Counter(df.top1); print("top1 counts", c1)
c3 = Counter([f for t in df.top3 for f in t]); print("top3 membership", c3.most_common(8))
c5 = Counter([f for t in top for f in t]); print("top5 membership", c5.most_common(10))
print("flip cases total", df.cases_flip.sum(), "of which near(<0.2)", df.flip_cases_near.sum(), "max dist of any flip", df.max_dist_flip.max())
df.to_csv("seeds.csv", index=False)
