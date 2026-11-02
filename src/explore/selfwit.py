"""E-SW1 (see RESEARCH_PLAN, direction E-SW): self-witness corroboration, model-free.
At victim v a benign verdict is released iff v holds >= M benign training rows in the input's cell
(own) or >= k OTHER clients do (worst-mode liars add |B|). Own-only = k infinite.
FA: benign test rows of client i, evaluated at i, no attack. EXP: share of (target row, honest
victim) pairs released, an upper bound on gated ASR for any detector.
kNN variant: own(v, x) = 1-NN distance of x to v's benign rows <= v's 0.99 calibration quantile."""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedpart import K, M, DEV, NB_OWNERS, scenarios, load_toniot, load_iot23, load_nbaiot, Pool, ROOT
from datasets import MIRAI
from iot23_time import load_iot23_time

Q, NREF = 0.99, 20000


def nbaiot_src(fed):
    """Test rows are stacked per device, classes ascending within a device (datasets._split_scale),
    so a new device starts where the label decreases. Checked against the train counts."""
    y = fed.yte; src = np.zeros(len(y), np.int16); d = 0
    for t in range(1, len(y)):
        if y[t] < y[t - 1]:
            d += 1
        src[t] = d
    assert d == fed.n - 1
    for i in range(fed.n):
        for c in np.unique(fed.ys[i]):
            ntr, nte = int((fed.ys[i] == c).sum()), int(((src == i) & (y == c)).sum())
            kk = ntr + nte
            assert nte == min(max(int(round(kk * 0.15)), 1 if kk >= 3 else 0), kk - 1), (i, c)
    return src


def nn_dist(X, A):
    out = []
    for s in range(0, len(X), 8192):
        out.append(torch.cdist(X[s:s + 8192], A).min(1).values)
    return torch.cat(out).cpu().numpy()


def knn_own(pool, Xte):
    """own[i] over test rows, plus calibration thresholds."""
    fed = pool.fed; own = []
    for i in range(fed.n):
        ix = pool.client(i); ix = ix[pool.y[ix] == fed.benign]
        g = np.random.RandomState(i); ix = g.permutation(ix)[:NREF]
        if len(ix) < 20:
            own.append(np.zeros(len(Xte), bool)); continue
        cut = int(len(ix) * 0.8); A, H = pool.X[ix[:cut]], pool.X[ix[cut:]]
        thr = np.quantile(nn_dist(H, A), Q)
        own.append(nn_dist(Xte, A) <= thr)
    return np.array(own)


def rates(own, S_all, k, src, bmask, scs):
    """own: (n, T) bool per client and test row; S_all: support of all clients per test row."""
    n = own.shape[0]; T = np.arange(own.shape[1])
    o_src = own[src, T]
    fa = {"gate": (S_all[bmask] < k).mean(),
          "self": (~o_src & (S_all - o_src < k))[bmask].mean(),
          "own": (~o_src)[bmask].mean()}
    ex = {"gate": [], "self": [], "own": []}
    for sc, tg, byz, rows in scs:
        hon = [i for i in range(n) if i not in byz]
        S_h = own[hon].sum(0)
        ex["gate"].append(((S_h + len(byz)) >= k)[rows].mean() if k - len(byz) > 0 else 1.0)
        s, o = [], []
        for v in hon:
            s.append((own[v] | (S_h - own[v] + len(byz) >= k))[rows].mean()); o.append(own[v][rows].mean())
        ex["self"].append(np.mean(s)); ex["own"].append(np.mean(o))
    return {f"FA_{a}": float(v * 100) for a, v in fa.items()} | \
           {f"EXP_{a}": [round(float(e) * 100, 2) for e in v] for a, v in ex.items()}


if __name__ == "__main__":
    res = {}
    for name, loader in [(a, b) for a, b in [("nbaiot", load_nbaiot), ("toniot", load_toniot), ("iot23", load_iot23), ("iot23time", load_iot23_time)] if a in sys.argv[1:] or len(sys.argv) == 1]:
        fed = loader(); pool = Pool(fed); Xte = torch.from_numpy(fed.Xte).to(DEV)
        src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
        bmask = fed.yte == fed.benign; k = K[name.replace("time", "")]
        scs = list(scenarios(name.replace("time", ""), fed))
        if name == "nbaiot":
            scs.append(("mirai_j6", list(MIRAI), NB_OWNERS[:6], np.isin(fed.yte, MIRAI)))
        out = {"scen": [s[0] for s in scs]}
        for B in (32, 128, 512):
            ctr, cte = pool.cells(B, Xte)
            ctr, cte = np.asarray(ctr), np.asarray(cte)
            own = np.zeros((fed.n, len(cte)), bool)
            for i in range(fed.n):
                ix = pool.client(i); cnt = np.bincount(ctr[ix][pool.y[ix] == fed.benign], minlength=B)
                own[i] = (cnt >= M)[cte]
            r = rates(own, own.sum(0), k, src, bmask, scs); out[f"B{B}"] = r
            print(name, B, {a: (round(v, 1) if isinstance(v, float) else round(float(np.mean(v)), 1))
                            for a, v in r.items()}, flush=True)
        own = knn_own(pool, Xte)
        ctr, cte = pool.cells(32, Xte); ctr, cte = np.asarray(ctr), np.asarray(cte)
        cell_own = np.zeros((fed.n, len(cte)), bool)
        for i in range(fed.n):
            ix = pool.client(i); cnt = np.bincount(ctr[ix][pool.y[ix] == fed.benign], minlength=32)
            cell_own[i] = (cnt >= M)[cte]
        # kNN own-support for the self term, B32 cells for the peer count
        T = np.arange(len(cte)); o_src = own[src, T]; S_all = cell_own.sum(0)
        r = {"FA_own": float((~o_src)[bmask].mean() * 100),
             "FA_self": float((~o_src & (S_all - cell_own[src, T] < k))[bmask].mean() * 100)}
        es, eo = [], []
        for sc, tg, byz, rows in scs:
            hon = [i for i in range(fed.n) if i not in byz]; S_h = cell_own[hon].sum(0)
            es.append(np.mean([(own[v] | (S_h - cell_own[v] + len(byz) >= k))[rows].mean() for v in hon]) * 100)
            eo.append(np.mean([own[v][rows].mean() for v in hon]) * 100)
        r["EXP_self"] = [round(float(e), 2) for e in es]; r["EXP_own"] = [round(float(e), 2) for e in eo]
        out["knn"] = r
        print(name, "knn", {a: (round(v, 1) if isinstance(v, float) else round(float(np.mean(v)), 1))
                            for a, v in r.items()}, flush=True)
        res[name] = out
        json.dump(res, open(ROOT / f"results/explore/selfwit_{name}.json", "w"), indent=1)
