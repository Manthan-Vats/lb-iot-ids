"""E-SW4 (RESEARCH_PLAN): pull/drag attack on the federated partition vs own checks."""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fedlloyd
from fedpart import M, DEV, NB_OWNERS, assign, scenarios, load_toniot, load_iot23, load_nbaiot, Pool, ROOT
from datasets import MIRAI
from selfwit import nbaiot_src
CAP = 2000


def own_matrix(pool, C, Xte, B, victims):
    fed = pool.fed; ctr, dtr = assign(pool.X, C); cte, dte = assign(Xte, C)
    out = {v: np.zeros((fed.n, len(cte)), bool) for v in ("cell", "r100", "nn")}
    for i in victims:
        ix = pool.client(i); ix = ix[pool.y[ix] == fed.benign]
        c, d = ctr[ix], dtr[ix]
        cnt = np.bincount(c, minlength=B); ok = np.nonzero(cnt >= M)[0]
        out["cell"][i] = (cnt >= M)[cte]
        r = np.full(B, -1.0)
        for b in ok:
            m = c == b; r[b] = d[m].max()
            R = pool.X[ix[m]]
            if len(R) > CAP:
                R = R[torch.randperm(len(R), generator=torch.Generator(device=DEV).manual_seed(int(b)), device=DEV)[:CAP]]
            D = torch.cdist(R, R); D.fill_diagonal_(float("inf")); tau = D.min(1).values.max()
            t = np.nonzero(cte == b)[0]
            if len(t):
                dn = torch.cdist(Xte[torch.from_numpy(t).to(DEV)], R).min(1).values
                out["nn"][i][t] = (dn <= tau).cpu().numpy()
        out["r100"][i] = r[cte] >= dte
    return out


if __name__ == "__main__":
    res = {}
    for name, loader in [("toniot", load_toniot), ("iot23", load_iot23), ("nbaiot", load_nbaiot)]:
        fed = loader(); pool = Pool(fed); Xte = torch.from_numpy(fed.Xte).to(DEV)
        src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
        bmask = fed.yte == fed.benign; T = np.arange(len(src)); allc = list(range(fed.n))
        scs = list(scenarios(name, fed))
        if name == "nbaiot":
            scs = [s for s in scs if s[0] in ("mirai_j1", "mirai_j5")]
        out = {"scen": [s[0] for s in scs]}
        for B in (512, 128):
            fedlloyd.B = B
            own = own_matrix(pool, fedlloyd.lloyd(pool.X), Xte, B, allc)
            row = {f"FA_{v}": round(float((~o[src, T])[bmask].mean() * 100), 2) for v, o in own.items()}
            print(name, B, row, flush=True)
            for w in (1, 10):
                ex = {v: [] for v in own}
                for sc, tg, byz, rows in scs:
                    hon = [i for i in allc if i not in byz]
                    keep = np.ones(len(pool.y), bool); pts, ws = [], []
                    for i in byz:
                        ix = pool.client(i); keep[ix[np.isin(pool.y[ix], tg)]] = False
                    Xk = pool.X[torch.from_numpy(keep).to(DEV)]
                    Co = fedlloyd.lloyd(Xk)
                    # centroids holding honest benign rows
                    ctr = assign(pool.X, Co)[0]; hb = np.zeros(B, bool)
                    for h in hon:
                        ix = pool.client(h); ix = ix[pool.y[ix] == fed.benign]
                        hb |= np.bincount(ctr[ix], minlength=B) >= M
                    Hc = Co[torch.from_numpy(np.nonzero(hb)[0]).to(DEV)]
                    for i in byz:
                        ix = pool.client(i); Tm = pool.X[ix[np.isin(pool.y[ix], tg)]]
                        if len(Tm):
                            mu = Tm.mean(0, keepdim=True); near = Hc[torch.cdist(mu, Hc).argmin()]
                            pts.append((mu[0] + near) / 2); ws.append(float(len(Tm)) * w)
                    Cp = fedlloyd.lloyd(Xk, (torch.stack(pts), torch.tensor(ws, device=DEV)))
                    ownP = own_matrix(pool, Cp, Xte, B, hon)
                    for v in own:
                        ex[v].append(round(float(np.mean([ownP[v][h][rows].mean() for h in hon]) * 100), 2))
                row[f"EXPpull_w{w}"] = ex
                print(name, B, f"w{w}", {v: round(float(np.mean(e)), 2) for v, e in ex.items()}, flush=True)
            out[f"B{B}"] = row
            res[name] = out
            json.dump(res, open(ROOT / "results/explore/own_pull.json", "w"), indent=1)
