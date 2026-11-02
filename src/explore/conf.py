"""E-SW13 (RESEARCH_PLAN): one-sided conformal witness. Scores L (local 1-NN), F (federated AE), W
(partition witness) are calibrated only on v's own held-out benign rows; release iff every component's
conformal p-value > alpha/K."""
import sys, json, math
from pathlib import Path
from itertools import combinations
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fedlloyd, fae
from fedpart import DEV, assign, load_toniot, load_iot23, load_nbaiot, Pool, ROOT
from selfwit import nbaiot_src
from datasets import MIRAI
NB, ALPHAS, M = (100, 300, 1000, 5000), (0.01, 0.05), 5
RULES = ["L", "F", "W", "L&F", "L&W", "L&F&W"]


class Sub:
    """Pool view with a subset of rows (as explore/scarce.py, which runs on import)."""
    def __init__(self, pool, keep):
        self.X = pool.X[torch.from_numpy(keep).to(DEV)]


def nn_dist(Q, R):
    out = []
    for s in range(0, len(Q), 8192):
        out.append(torch.cdist(Q[s:s + 8192], R).min(1).values)
    return torch.cat(out)


def local_score(T):
    mu, sd = T.mean(0), T.std(0).clamp_min(1e-3); Tz = (T - mu) / sd
    return lambda Z: nn_dist((Z - mu) / sd, Tz).cpu().numpy()


def witness_score(T, C):
    """1-NN distance to own train rows in the query's cell; inf if the cell holds < M own rows."""
    ct = assign(T, C)[0]; cnt = np.bincount(ct, minlength=len(C))
    def f(Z):
        cz = assign(Z, C)[0]; s = np.full(len(Z), np.inf)
        for b in np.unique(cz):
            if cnt[b] < M:
                continue
            q = np.nonzero(cz == b)[0]
            s[q] = nn_dist(Z[torch.from_numpy(q).to(DEV)], T[torch.from_numpy(np.nonzero(ct == b)[0]).to(DEV)]).cpu().numpy()
        return s
    return f


def pvals(cal, s):
    """Split-conformal p-value (1 + #{cal >= s}) / (n + 1); inf scores tie with inf."""
    c = np.sort(cal); ge = len(c) - np.searchsorted(c, s, side="left")
    return (1 + ge) / (len(c) + 1)


def evaluate(fed, pool, Xte, src, tr, ca, part, fmodel, hon, tgrows):
    b = fed.yte == fed.benign; res = {f"{r}@{a}": {"FA": [], "EXP": [], "EXPt": []} for r in RULES for a in ALPHAS}
    reach = {}
    for v in hon:
        ob = b & (src == v)
        if len(ca[v]) < 2:
            continue
        T, Cr = pool.X[tr[v]], pool.X[ca[v]]
        fns = {"L": local_score(T), "W": witness_score(T, part), "F": lambda Z: fae.score(fmodel, Z)}
        P = {k: pvals(f(Cr), f(Xte)) for k, f in fns.items()}
        for r in RULES:
            ks = r.split("&")
            for a in ALPHAS:
                rel = np.all([P[k] > a / len(ks) for k in ks], axis=0)
                reach[f"{r}@{a}"] = reach.get(f"{r}@{a}", 0) + int(1 / (len(ca[v]) + 1) <= a / len(ks))
                x = res[f"{r}@{a}"]
                if ob.sum():
                    x["FA"].append((~rel[ob]).mean())
                x["EXP"].append(rel[~b].mean())
                if tgrows is not None:
                    x["EXPt"].append(rel[tgrows].mean())
    out = {}
    for k, x in res.items():
        out[k] = {m: round(float(np.mean(x[m]) * 100), 2) if x[m] else None for m in x}
        out[k]["reach"] = reach.get(k, 0)
    return out


def run(name, loader):
    fed = loader(); pool = Pool(fed); Xte = torch.from_numpy(fed.Xte).to(DEV); d = Xte.shape[1]
    src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
    n = fed.n; allc = list(range(n)); fedlloyd.B = 512
    tgs = [list(MIRAI)] if name == "nbaiot" else [[fed.class_names.index(c)] for c in fae.TG[name]]
    res = {"clean": {}, "attack": []}
    for nb in NB:
        rows = []
        for seed in range(3):
            keep = np.ones(len(pool.y), bool); tr, ca = {}, {}
            for i in allc:
                ix = pool.client(i); bi = ix[pool.y[ix] == fed.benign]
                bi = np.random.RandomState(100 * seed + i).permutation(bi)
                keep[bi[nb:]] = False; kb = bi[:nb]; c = int(round(len(kb) * 0.6))
                tr[i], ca[i] = kb[:c], kb[c:]
            sub = Sub(pool, keep)                      # partition from everyone's kept inputs
            # tr/ca index the full pool; Sub only defines what the partition sees
            part = fedlloyd.lloyd(sub.X)
            fm = fae.train_fed([pool.X[tr[i]] for i in allc if len(tr[i])], "avg", d, seed)
            r = evaluate(fed, pool, Xte, src, tr, ca, part, fm, allc, None)
            rows.append(r); print(name, nb, seed, json.dumps({k: (v["FA"], v["EXP"]) for k, v in r.items()}), flush=True)
            if nb == 1000:
                for j, tg in enumerate(tgs):
                    pool_t = np.nonzero(np.isin(pool.y, tg))[0]; bsz = n // 2
                    byz = sorted(np.random.RandomState(1000 * seed + bsz).choice(n, bsz, replace=False).tolist())
                    hon = [i for i in allc if i not in byz]; g = np.random.RandomState(seed)
                    data, inj = [], []
                    for i in allc:
                        x = pool.X[tr[i]]
                        if i in byz:
                            sel = pool_t[g.randint(0, len(pool_t), max(1, len(tr[i])))]
                            x = torch.cat([x, pool.X[sel]]); inj.append(pool.X[sel])
                        if len(x):
                            data.append(x)
                    fmp = fae.train_fed(data, "avg", d, seed)
                    partp = fedlloyd.lloyd(torch.cat([sub.X] + inj))
                    ra = evaluate(fed, pool, Xte, src, tr, ca, partp, fmp, hon, np.isin(fed.yte, tg))
                    res["attack"].append({"nb": nb, "seed": seed, "target": j, "b": bsz, "r": ra})
                    print(name, "ATTACK", j, seed, json.dumps({k: (v["FA"], v["EXPt"]) for k, v in ra.items()}), flush=True)
        res["clean"][str(nb)] = {k: {m: round(float(np.mean([x[k][m] for x in rows if x[k][m] is not None])), 2)
                                     if any(x[k][m] is not None for x in rows) else None for m in ("FA", "EXP")}
                                 for k in rows[0]}
        print(name, nb, "MEAN", json.dumps(res["clean"][str(nb)]), flush=True)
        json.dump(res, open(ROOT / f"results/explore/conf_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    for name, loader in [(a, l) for a, l in [("nbaiot", load_nbaiot), ("toniot", load_toniot),
                                             ("iot23", load_iot23)] if a in sys.argv[1:]]:
        run(name, loader)
