import numpy as np, pandas as pd, json
from model import *
from robustness import compare, perturb

DELTAS = [-0.10, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.10]
X, y, feats = load_data()
Xtr, Xte, ytr, yte = split(X, y)
m = train(Xtr, ytr)
yt = (yte == MALIGNANT).astype(int).values

frames = []
for f in feats:
    for d in DELTAS:
        r = compare(m, Xte, f, d)
        r["y_true_malig"] = yt
        frames.append(r)
res = pd.concat(frames, ignore_index=True)
res["absdp"] = res["dp"].abs()
res.to_csv("results_all.csv", index=False)
print("rows", len(res))

# FN change per feature/delta
def fn_count(lbl, ytrue): return int(((ytrue == 1) & (lbl == 0)).sum())
def fp_count(lbl, ytrue): return int(((ytrue == 0) & (lbl == 1)).sum())
base_fn = fn_count(res[(res.feature==feats[0])&(res.delta==DELTAS[0])].label_orig.values, yt)
base_fp = fp_count(res[(res.feature==feats[0])&(res.delta==DELTAS[0])].label_orig.values, yt)
print("base FN", base_fn, "FP", base_fp)

g = res.groupby(["feature", "delta"]).agg(mean_absdp=("absdp", "mean"), max_absdp=("absdp","max"), flips=("flipped", "sum")).reset_index()
fnfp = res.groupby(["feature","delta"]).apply(lambda d: pd.Series({"fn_after": fn_count(d.label_pert.values, d.y_true_malig.values), "fp_after": fp_count(d.label_pert.values, d.y_true_malig.values)}), include_groups=False).reset_index()
g = g.merge(fnfp, on=["feature","delta"])
g.to_csv("summary_feature_delta.csv", index=False)

# H1: by delta
h1 = res.groupby("delta").agg(mean_absdp=("absdp","mean"), median_absdp=("absdp","median"), flips=("flipped","sum"), trials=("flipped","size")).reset_index()
h1["flip_rate_pct"] = h1.flips / h1.trials * 100
h1["features_with_flip"] = g.groupby("delta").apply(lambda d: int((d.flips > 0).sum()), include_groups=False).values
h1["fn_increase_total"] = g.groupby("delta").apply(lambda d: int((d.fn_after - base_fn).clip(lower=0).sum()), include_groups=False).values
h1["max_fn_after"] = g.groupby("delta").fn_after.max().values
print("\nH1\n", h1.to_string())
# by |delta|
res["absdelta"] = res.delta.abs()
h1b = res.groupby("absdelta").agg(mean_absdp=("absdp","mean"), flips=("flipped","sum"), trials=("flipped","size")).reset_index()
h1b["flip_rate_pct"] = h1b.flips/h1b.trials*100
# cases with any flip at each |delta|
anyflip = res.groupby(["absdelta","case_id"]).flipped.any().groupby("absdelta").sum()
h1b["cases_any_flip"] = anyflip.values
print("\nH1 by |delta|\n", h1b.to_string())
h1.to_csv("h1_by_delta.csv", index=False); h1b.to_csv("h1_by_absdelta.csv", index=False)
