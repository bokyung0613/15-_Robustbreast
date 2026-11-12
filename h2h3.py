import numpy as np, pandas as pd
from scipy.stats import spearmanr
from model import *
res = pd.read_csv("results_all.csv")
X, y, feats = load_data()
Xtr, Xte, ytr, yte = split(X, y)
m = train(Xtr, ytr)
p0 = res.drop_duplicates("case_id").set_index("case_id")[["p_orig","label_orig","y_true_malig"]]
print("FN case:", p0[(p0.y_true_malig==1)&(p0.label_orig==0)])
print("FP case:", p0[(p0.y_true_malig==0)&(p0.label_orig==1)])

# H2 ranking at |delta| = 0.05 and 0.10
res["absdelta"] = res.delta.abs()
for ad in [0.05, 0.10]:
    sub = res[res.absdelta.round(2)==ad]
    r = sub.groupby("feature").agg(mean_absdp=("absdp","mean"), flips=("flipped","sum"), max_absdp=("absdp","max")).sort_values("mean_absdp", ascending=False)
    r["mean_absdp_pp"] = r.mean_absdp*100
    print(f"\nH2 |delta|={ad}\n", r.to_string())
    r.to_csv(f"h2_rank_{int(ad*100)}.csv")

# coefficients & analytic sensitivity
sc = m.named_steps["scaler"]; clf = m.named_steps["clf"]
# sklearn coef for class 1 (benign); malignant logit = -coef
w = -clf.coef_[0]
coef = pd.DataFrame({"feature": feats, "w_malig_std": w, "train_mean": sc.mean_, "train_sd": np.sqrt(sc.var_)})
coef["cv"] = coef.train_sd/coef.train_mean
coef["S_index"] = np.abs(coef.w_malig_std)/coef.cv   # |w| * mean/sd : logit change per 100% relative change at mean
coef = coef.set_index("feature")
r10 = pd.read_csv("h2_rank_10.csv", index_col=0)
coef = coef.join(r10[["mean_absdp","flips"]])
rho, pv = spearmanr(coef.S_index, coef.mean_absdp)
print("\nSpearman S_index vs observed mean|dp| @10%:", rho, pv)
print(coef.sort_values("mean_absdp", ascending=False).to_string())
coef.to_csv("coef_sensitivity.csv")
# sign check: features whose increase lowers malignant prob
print("\nNegative-weight (increase -> less malignant) features:", list(coef[coef.w_malig_std<0].index))
