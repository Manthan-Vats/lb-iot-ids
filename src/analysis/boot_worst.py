"""Bootstrap CIs for worst-case release (max over conditions of the mean over cells), paired between methods.
Usage: python boot_worst.py RESULT.json [RESULT2.json ...] -- A:B [A:B ...]
Resamples whole seeds (cells within a seed share peers), so the CIs account for that dependence."""
import json, sys, numpy as np

files = sys.argv[1:sys.argv.index("--")]; pairs = [p.split(":") for p in sys.argv[sys.argv.index("--") + 1:]]
R = [r for f in files for r in json.load(open(f))]
conds = sorted({r["cond"] for r in R}); idx = {(r["seed"], r["target"], r["v"], r["cond"]): r for r in R}
cells = sorted({(r["seed"], r["target"], r["v"]) for r in R}); seeds = sorted({c[0] for c in cells})
rows = {s: [i for i, c in enumerate(cells) if c[0] == s] for s in seeds}
rng = np.random.default_rng(0)
print(files, len(seeds), "seeds", len(cells), "cells", conds)
for a in ("0.001", "0.01"):
    for A, B in pairs:
        M = {m: np.array([[idx[c + (k,)][f"{m}@{a}"][2] for k in conds] for c in cells]) for m in (A, B)}
        d0 = M[A].mean(0).max() - M[B].mean(0).max(); ds = []
        for _ in range(2000):
            ii = np.concatenate([rows[s] for s in rng.choice(seeds, len(seeds))])
            ds.append(M[A][ii].mean(0).max() - M[B][ii].mean(0).max())
        lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"  a{a} worst({A}) - worst({B}) = {d0:+.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]")
