"""EXPLORATORY (not in P-FP): federated Lloyd partition. Each round clients send per-centroid sums and
counts of their own rows through secure aggregation; with honest clients this equals pooled k-means
on the reported rows. Attacks: 'omit' (attackers leave out their target rows) and 'pull' (attackers
report their target-row count at the midpoint between target mean and nearest omit-run centroid).
Same model-free metrics as explore/fedpart.py."""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedpart import (B, K, DEV, assign, support, scenarios, load_toniot, load_iot23, load_nbaiot,
                     Pool, ROOT)


def lloyd(X, extra=None, seed=2027, it=30):
    """k-means on rows X plus optional (points, weights) contributions; init as Pool.cells."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    sub = X[torch.randperm(len(X), generator=g, device=X.device)[:150000]]
    C = sub[torch.randperm(len(sub), generator=g, device=X.device)[:B]].clone()
    for _ in range(it):
        a = assign(sub, C)[0]; a = torch.from_numpy(a).to(X.device)
        S = torch.zeros_like(C).index_add_(0, a, sub)
        n = torch.bincount(a, minlength=B).float()
        if extra is not None:
            P, w = extra; scale = len(sub) / len(X)            # same subsampling rate as honest rows
            ap = torch.cdist(P, C).argmin(1)
            S.index_add_(0, ap, P * (w * scale)[:, None]); n.index_add_(0, ap, w * scale)
        m = n > 0
        C[m] = S[m] / n[m, None]
    return C


if __name__ == "__main__":
    res = {}
    for name, loader in [("toniot", load_toniot), ("iot23", load_iot23), ("nbaiot", load_nbaiot)]:
        fed = loader(); pool = Pool(fed); k = K[name]; Xte = torch.from_numpy(fed.Xte).to(DEV)
        allc = list(range(fed.n)); bmask = fed.yte == fed.benign
        C = lloyd(pool.X)
        base = assign(pool.X, C) + assign(Xte, C)
        out = {"FA": float((support(pool, *base, allc, False)[bmask] < k).mean() * 100),
               "FA_rad": float((support(pool, *base, allc, True)[bmask] < k).mean() * 100), "EXP": {}}
        print(name, "FA P3", out["FA"], "P3+radius", out["FA_rad"], flush=True)
        for sc, tg, byz, rows in scenarios(name, fed):
            hon = [i for i in allc if i not in byz]; need = k - len(byz)
            if need <= 0:
                continue
            e = {"P3": float((support(pool, *base, hon, False)[rows] >= need).mean() * 100)}
            keep = np.ones(len(pool.y), bool)
            for i in byz:
                ix = pool.client(i); keep[ix[np.isin(pool.y[ix], tg)]] = False
            Xk = pool.X[torch.from_numpy(keep).to(DEV)]
            Co = lloyd(Xk)
            pr = assign(pool.X, Co) + assign(Xte, Co)
            e["P3omit"] = float((support(pool, *pr, hon, False)[rows] >= need).mean() * 100)
            e["P3omit_rad"] = float((support(pool, *pr, hon, True)[rows] >= need).mean() * 100)
            pts, ws = [], []
            for i in byz:
                ix = pool.client(i); T = pool.X[ix[np.isin(pool.y[ix], tg)]]
                if not len(T):
                    continue
                mu = T.mean(0, keepdim=True); near = Co[torch.cdist(mu, Co).argmin()]
                pts.append((mu[0] + near) / 2); ws.append(float(len(T)))
            if pts:
                Cp = lloyd(Xk, (torch.stack(pts), torch.tensor(ws, device=DEV)))
                pr = assign(pool.X, Cp) + assign(Xte, Cp)
                e["P3pull"] = float((support(pool, *pr, hon, False)[rows] >= need).mean() * 100)
                e["P3pull_rad"] = float((support(pool, *pr, hon, True)[rows] >= need).mean() * 100)
            out["EXP"][sc] = e
            print(f"  {sc:28} n={rows.sum():6d} " + " ".join(f"{a}={v:5.1f}" for a, v in e.items()), flush=True)
        res[name] = out
        json.dump(res, open(ROOT / "results/explore/fedlloyd.json", "w"), indent=1)
