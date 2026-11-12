import numpy as np, pandas as pd
from model import *
from robustness import compare, perturb, stability_check
res = pd.read_csv("results_all.csv")
X, y, feats = load_data()
Xtr, Xte, ytr, yte = split(X, y)
m = train(Xtr, ytr)
# H3
res["dist"] = (res.p_orig-0.5).abs()
bins = [0,0.1,0.2,0.3,0.4,0.5001]
labels = ["0.0–0.1","0.1–0.2","0.2–0.3","0.3–0.4","0.4–0.5"]
res["bin"] = pd.cut(res.dist, bins=bins, labels=labels, right=False)
h3 = res.groupby("bin", observed=False).agg(cases=("case_id","nunique"), trials=("flipped","size"), flips=("flipped","sum"), mean_absdp=("absdp","mean")).reset_index()
h3["flip_rate_pct"] = h3.flips/h3.trials*100
cf = res.groupby(["bin","case_id"], observed=False).flipped.any().reset_index().groupby("bin", observed=False).flipped.sum()
h3["cases_with_flip"] = cf.values
print(h3.to_string())
h3.to_csv("h3_boundary.csv", index=False)
# per-delta-magnitude within bins
res["absdelta"]=res.delta.abs().round(2)
print(res.pivot_table(index="bin", columns="absdelta", values="flipped", aggfunc="sum", observed=False))

# Scenario: case 291 mean radius +5% and +10%
x = Xte.loc[[291]]
for d in [0.0, 0.05, 0.10]:
    r = compare(m, x, "mean radius", d).iloc[0]
    print("case291 mean radius", d, round(r.p_orig,4), round(r.p_pert,4), r.flipped, "x:", x["mean radius"].iloc[0], "->", x["mean radius"].iloc[0]*(1+d))
# auto stability check case 291 at 5%
sc = stability_check(m, x, feats, 0.05)
print(sc[sc.flipped][["feature","delta","p_orig","p_pert"]])
print("case 291 n features flipping at ±5%:", sc[sc.flipped].feature.nunique())
sc3 = stability_check(m, x, feats, 0.03); print("±3%:", list(sc3[sc3.flipped].feature))
# a far-from-boundary case example: pick case with p ~ 0.99 malignant
far = res.drop_duplicates("case_id").sort_values("p_orig").iloc[[-1, len(res.drop_duplicates('case_id'))//2]]
print(far[["case_id","p_orig"]])
# FN case 73 stability
x73 = Xte.loc[[73]]
s73 = stability_check(m, x73, feats, 0.10)
print("case 73 max p_pert at ±10%", s73.p_pert.max(), "p_orig", s73.p_orig.iloc[0])
# zero perturbation unit test
z = pd.concat([compare(m, Xte, f, 0.0) for f in feats])
print("delta 0 -> max |dp|", z.dp.abs().max(), "flips", z.flipped.sum())
