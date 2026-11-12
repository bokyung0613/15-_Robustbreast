import numpy as np, pandas as pd
from model import *
from robustness import compare
X, y, feats = load_data()
Xtr, Xte, ytr, yte = split(X, y)
m = train(Xtr, ytr)
p0 = malignant_proba(m, Xte); l0 = predict_label(p0)
yt=(yte==MALIGNANT).astype(int).values
out=[]
for prefix in ["mean","worst"]:
    for d in [-0.10,-0.05,0.05,0.10]:
        Xp = Xte.copy()
        Xp[f"{prefix} radius"] *= (1+d); Xp[f"{prefix} perimeter"] *= (1+d); Xp[f"{prefix} area"] *= (1+d)**2
        p1 = malignant_proba(m, Xp); l1 = predict_label(p1)
        single = compare(m, Xte, f"{prefix} radius", d)
        out.append(dict(group=prefix, delta=d, joint_mean_absdp=np.abs(p1-p0).mean(), joint_flips=int((l0!=l1).sum()),
                        radius_only_mean_absdp=single.dp.abs().mean(), radius_only_flips=int(single.flipped.sum()),
                        joint_fn=int(((yt==1)&(l1==0)).sum()), joint_fp=int(((yt==0)&(l1==1)).sum())))
df=pd.DataFrame(out); df["ratio"]=df.joint_mean_absdp/df.radius_only_mean_absdp
print(df.to_string()); df.to_csv("joint_size.csv", index=False)

# sign check: univariate direction vs model weight
g = pd.read_csv("coef_sensitivity.csv", index_col=0)
cm = X.groupby((y==MALIGNANT).astype(int)).mean().T  # cols 0 benign,1 malignant
g["malig_minus_benign"] = cm[1]-cm[0]
g["uni_dir"] = np.sign(g.malig_minus_benign)
g["model_dir"] = np.sign(g.w_malig_std)
mis = g[g.uni_dir != g.model_dir]
print(mis[["w_malig_std","malig_minus_benign"]].join(cm.rename(columns={0:"benign_mean",1:"malig_mean"})))
# FP/FN by delta
s = pd.read_csv("summary_feature_delta.csv")
print(s.groupby("delta")[["fn_after","fp_after"]].agg(["min","max"]))
