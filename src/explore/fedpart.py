"""P-FP: federated partitions for the k-corroboration gate, model-free bounds (see RESEARCH_PLAN P-FP).
EXP = share of target test rows whose cell has honest support >= k - |B| (upper bound on gated ASR for
any detector, worst-mode flags). FA = share of benign test rows with honest support < k, no attack."""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import MIRAI, load_nbaiot, load_iot23, load_toniot
from multirun import Pool
ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, L, M, Q = 32, 8, 5, 0.99
NB_OWNERS = [0, 1, 3, 4, 5, 7, 8]
K = {"nbaiot": 6, "toniot": 2, "iot23": 2}
TARGETS = {"toniot": ["ddos", "xss", "scanning"], "iot23": ["DDoS", "PortScan"]}


def kmeans(X, k, w=None, seed=0, it=30):
    g = torch.Generator(device=X.device).manual_seed(seed)
    w = torch.ones(len(X), device=X.device) if w is None else w
    if len(X) <= k:
        return X.clone()
    C = X[torch.multinomial(w, 1, generator=g)]                   # k-means++ init
    for _ in range(k - 1):
        d = torch.cdist(X, C).min(1).values ** 2 * w
        C = torch.cat([C, X[torch.multinomial(d + 1e-12, 1, generator=g)]])
    for _ in range(it):
        a = torch.cdist(X, C).argmin(1)
        S = torch.zeros_like(C).index_add_(0, a, X * w[:, None])
        n = torch.zeros(k, device=X.device).index_add_(0, a, w)
        C[n > 0] = S[n > 0] / n[n > 0, None]
    return C


def assign(X, C):
    a, d = [], []
    for s in range(0, len(X), 16384):
        D = torch.cdist(X[s:s + 16384], C)
        v, i = D.min(1); a.append(i); d.append(v)
    return torch.cat(a).cpu().numpy(), torch.cat(d).cpu().numpy()


def fed_centroids(pool, tg, byz=(), mode="none"):
    fed = pool.fed; pts, wts = [], []
    honest_C = []
    for i in range(fed.n):
        ix = pool.client(i)
        if i in byz and mode == "pull":      # all of its weight goes to the pull points below
            continue
        if i in byz and mode == "omit":
            ix = ix[~np.isin(pool.y[ix], tg)]
        if not len(ix):
            continue
        X = pool.X[ix]
        C = kmeans(X, L, seed=100 + i)
        a, _ = assign(X, C)
        w = np.bincount(a, minlength=len(C)) / len(a)
        pts.append(C); wts.append(torch.tensor(w, dtype=torch.float32, device=DEV))
        if i not in byz:
            honest_C.append(C)
    if mode == "pull":
        H = torch.cat(honest_C)
        for j, i in enumerate(byz):
            T = pool.X[pool.client(i)][np.isin(pool.y[pool.client(i)], tg)]
            if not len(T):
                continue
            mu = T.mean(0, keepdim=True)
            near = H[torch.cdist(mu, H).argmin()]
            pts.append(((mu[0] + near) / 2).repeat(L, 1))
            wts.append(torch.full((L,), 1 / L, device=DEV))
    P, W = torch.cat(pts), torch.cat(wts)
    return kmeans(P, B, W, seed=2027)


def support(pool, ctr, dtr, cte, dte, honest, radius):
    """Per test row: number of honest clients with >= M true-benign rows in its cell (and, with
    radius, whose 0.99-quantile benign distance in that cell reaches the row)."""
    fed = pool.fed; S = np.zeros(len(cte), int)
    for i in honest:
        ix = pool.client(i); bm = pool.y[ix] == fed.benign
        c, d = ctr[ix][bm], dtr[ix][bm]
        cnt = np.bincount(c, minlength=B)
        if radius:
            r = np.full(B, -1.0)
            for b in np.nonzero(cnt >= M)[0]:
                r[b] = np.quantile(d[c == b], Q)
            S += (r[cte] >= dte)
        else:
            S += (cnt >= M)[cte]
    return S


def scenarios(name, fed):
    yt, src = fed.yte, (fed.meta or {}).get("test_src")
    if name == "nbaiot":
        for j in range(1, 6):
            yield f"mirai_j{j}", list(MIRAI), NB_OWNERS[:j], np.isin(yt, MIRAI)
        return
    for cn in TARGETS[name]:
        c = fed.class_names.index(cn)
        for f in range(fed.n):
            own = (yt == c) & (src == f)
            if own.sum() >= 10 and (fed.ys[f] == c).any():
                yield f"{cn}|{fed.client_names[f]}", [c], [f], own


if __name__ == "__main__":
    res = {}
    for name, loader in [("toniot", load_toniot), ("iot23", load_iot23), ("nbaiot", load_nbaiot)]:
        fed = loader(); pool = Pool(fed); k = K[name]; Xte = torch.from_numpy(fed.Xte).to(DEV)
        allc = list(range(fed.n)); bmask = fed.yte == fed.benign
        P0 = pool.cells(B, Xte)
        out = {"FA": {}, "EXP": {}}
        # no-attack partitions
        Cf = fed_centroids(pool, [])
        parts = {}
        ctr, dtr = assign(pool.X, Cf); cte, dte = assign(Xte, Cf); parts["P1"] = (ctr, dtr, cte, dte)
        parts["P0"] = (P0[0], np.zeros(len(P0[0])), P0[1], np.zeros(len(P0[1])))   # exact pipeline cells
        for tag, (P, rad) in {"P0": ("P0", False), "P1": ("P1", False), "P2": ("P1", True)}.items():
            S = support(pool, *parts[P], allc, rad)
            out["FA"][tag] = float((S[bmask] < k).mean() * 100)
        print(name, "FA", out["FA"], flush=True)
        for sc, tg, byz, rows in scenarios(name, fed):
            hon = [i for i in allc if i not in byz]; need = k - len(byz); e = {}
            if need <= 0:
                continue
            e["P0"] = float((support(pool, *parts["P0"], hon, False)[rows] >= need).mean() * 100)
            e["P1"] = float((support(pool, *parts["P1"], hon, False)[rows] >= need).mean() * 100)
            for mode in ("omit", "pull"):
                C = fed_centroids(pool, tg, byz, mode)
                pr = assign(pool.X, C) + assign(Xte, C)
                e[f"P1{mode}"] = float((support(pool, *pr, hon, False)[rows] >= need).mean() * 100)
                e[f"P2{mode}"] = float((support(pool, *pr, hon, True)[rows] >= need).mean() * 100)
            out["EXP"][sc] = e
            print(f"  {sc:28} n={rows.sum():6d} " + " ".join(f"{a}={v:5.1f}" for a, v in e.items()), flush=True)
        res[name] = out
        json.dump(res, open(ROOT / "results/explore/fedpart.json", "w"), indent=1)
