"""E-SW14 (RESEARCH_PLAN): anytrust veto. Federation may only add alarms at victim v; every alarm
source (local 1-NN score L, aggregated model G, per-client vetoers V) is calibrated on v's own held-out
benign rows. V uses joint max-rank calibration (ranks on C1, conformal p on C2)."""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedpart import DEV, load_toniot, load_iot23, load_nbaiot, Pool, ROOT
from selfwit import nbaiot_src
from conf import local_score, pvals
from datasets import MIRAI
TG = {"nbaiot": [list(MIRAI)], "toniot": ["ddos", "scanning"], "iot23": ["DDoS", "PortScan"]}
ALPHAS, CAPB, CAPA, MINH = (0.01, 0.05), 5000, 20000, 20
RULES = ["L", "G", "V", "L|G", "AT", "ATG"]


def mlp(d, h):
    return nn.Sequential(nn.Linear(d, h[0]), nn.ReLU(), nn.Linear(h[0], h[1]), nn.ReLU(),
                         nn.Linear(h[1], 1)).to(DEV)


def train_clf(X, y, h, epochs, seed):
    torch.manual_seed(seed); g = torch.Generator(device=DEV).manual_seed(seed)
    m = mlp(X.shape[1], h); opt = torch.optim.Adam(m.parameters(), 1e-3)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEV); pos = max(float(y.sum()), 1.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / pos, device=DEV).clamp(0.01, 100))
    for _ in range(epochs):
        p = torch.randperm(len(X), generator=g, device=DEV)
        for s in range(0, len(X), 512):
            b = p[s:s + 512]; loss = lossf(m(X[b])[:, 0], yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def logits(m, X):
    with torch.no_grad():
        return torch.cat([m(X[s:s + 16384])[:, 0] for s in range(0, len(X), 16384)]).cpu().numpy()


def run(name, loader):
    fed = loader(); pool = Pool(fed); Xte = torch.from_numpy(fed.Xte).to(DEV); n = fed.n
    src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
    bte = fed.yte == fed.benign
    tgs = [t if isinstance(t, list) else [fed.class_names.index(t)] for t in TG[name]]
    out = {"clean": [], "attack": []}
    for seed in range(3):
        rs = np.random.RandomState(seed)
        tr, c1, c2, atk = {}, {}, {}, {}
        for i in range(n):
            ix = pool.client(i); bi = rs.permutation(ix[pool.y[ix] == fed.benign])[:CAPB]
            a, b = int(len(bi) * 0.6), int(len(bi) * 0.8)
            tr[i], c1[i], c2[i] = bi[:a], bi[a:b], bi[b:]
            atk[i] = rs.permutation(ix[pool.y[ix] != fed.benign])[:CAPA]
        cal = np.concatenate([np.concatenate([c1[i], c2[i]]) for i in range(n)])
        pos = {r: k for k, r in enumerate(cal)}
        Xcal = pool.X[torch.from_numpy(cal).to(DEV)]

        def vetoer(j, flip=()):
            ix = np.concatenate([tr[j], atk[j]]); y = (pool.y[ix] != fed.benign).astype(int)
            y[np.isin(pool.y[ix], flip)] = 0
            m = train_clf(pool.X[torch.from_numpy(ix).to(DEV)], y, (64, 32), 8, 1000 * seed + j)
            return logits(m, Xte), logits(m, Xcal)

        def aggregated(flip_clients=(), flip=()):
            ix = np.concatenate([np.concatenate([tr[i], atk[i]]) for i in range(n)])
            y = (pool.y[ix] != fed.benign).astype(int)
            if len(flip_clients):
                fc = np.concatenate([pool.client(i) for i in flip_clients])
                y[np.isin(ix, fc) & np.isin(pool.y[ix], flip)] = 0
            m = train_clf(pool.X[torch.from_numpy(ix).to(DEV)], y, (128, 64), 5, seed)
            return logits(m, Xte), logits(m, Xcal)

        V0 = {j: vetoer(j) for j in range(n) if len(atk[j]) >= MINH}
        G0 = aggregated()
        Lp = {}
        for v in range(n):
            if len(c1[v]) + len(c2[v]) < 2:
                continue
            f = local_score(pool.X[torch.from_numpy(tr[v]).to(DEV)])
            cc = np.concatenate([c1[v], c2[v]])
            Lp[v] = pvals(f(pool.X[torch.from_numpy(cc).to(DEV)]), f(Xte))

        def evaluate(V, G, victims, tg):
            res = []
            for v in victims:
                if v not in Lp or len(c1[v]) < 1 or len(c2[v]) < 1:
                    continue
                k1 = np.array([pos[r] for r in c1[v]]); k2 = np.array([pos[r] for r in c2[v]])
                kc = np.concatenate([k1, k2])
                pG = pvals(G[1][kc], G[0])

                def maxrank(items):
                    Mt, Mc = np.zeros(len(Xte)), np.zeros(len(k2))
                    for st, sc in items:
                        ref = np.sort(sc[k1])
                        Mt = np.maximum(Mt, np.searchsorted(ref, st, "left") / len(ref))
                        Mc = np.maximum(Mc, np.searchsorted(ref, sc[k2], "left") / len(ref))
                    return pvals(Mc, Mt)
                pV = maxrank(list(V.values())); pVG = maxrank(list(V.values()) + [G])
                ob = bte & (src == v); r = {"v": int(v), "ncal": int(len(kc))}
                for a in ALPHAS:
                    alarm = {"L": Lp[v] <= a, "G": pG <= a, "V": pV <= a,
                             "L|G": (Lp[v] <= a / 2) | (pG <= a / 2),
                             "AT": (Lp[v] <= a / 2) | (pV <= a / 2),
                             "ATG": (Lp[v] <= a / 2) | (pVG <= a / 2)}
                    for k, al in alarm.items():
                        rel = ~al
                        r[f"{k}@{a}"] = {"FA": float(al[ob].mean() * 100) if ob.sum() else None,
                                         "EXP": float(rel[~bte].mean() * 100),
                                         "EXPt": float(rel[np.isin(fed.yte, tg)].mean() * 100) if tg is not None else None}
                res.append(r)
            return res

        def summ(res, keys=None):
            return {k: {m: round(float(np.mean([x[k][m] for x in res if x[k][m] is not None])), 2)
                        if any(x[k][m] is not None for x in res) else None for m in ("FA", "EXP", "EXPt")}
                    for k in res[0] if "@" in k} if res else {}

        clean = evaluate(V0, G0, range(n), None)
        cl_t = {}
        for ti, tg in enumerate(tgs):
            cl_t[ti] = evaluate(V0, G0, range(n), tg)
        out["clean"].append({"seed": seed, "all": summ(clean),
                             "per_target": {ti: summ(x) for ti, x in cl_t.items()},
                             "per_victim": clean})
        print(name, seed, "CLEAN", json.dumps({k: (x["FA"], x["EXP"]) for k, x in summ(clean).items()}), flush=True)
        for ti, tg in enumerate(tgs):
            holders = [j for j in range(n) if (pool.y[pool.client(j)][:, None] == np.array(tg)).any(1).sum() >= MINH]
            if len(holders) < 2:
                print(name, seed, "target", ti, "holders", holders, "skip", flush=True); continue
            surv = int(rs.choice(holders))
            for cond, byz in (("one", [j for j in holders if j != surv]), ("none", holders)):
                G = aggregated(byz, tg); hon = [j for j in range(n) if j not in byz]
                flipV = {j: vetoer(j, tg) for j in byz if j in V0}
                honV = {j: V0[j] for j in V0 if j not in byz}
                for liar in ("flip", "anti", "noise"):
                    V = dict(honV)
                    for j in byz:
                        if liar == "flip" and j in flipV:
                            V[j] = flipV[j]
                        elif liar == "anti" and honV:
                            V[j] = (-np.max([x[0] for x in honV.values()], 0), -np.max([x[1] for x in honV.values()], 0))
                        elif liar == "noise":
                            g = np.random.RandomState(7 + j)
                            V[j] = (g.randn(len(Xte)), g.randn(len(cal)))
                    res = evaluate(V, G, hon, tg)
                    hs = [x for x in res if x["v"] in holders]; nh = [x for x in res if x["v"] not in holders]
                    rec = {"seed": seed, "target": ti, "cond": cond, "liar": liar, "surv": surv, "byz": byz,
                           "all": summ(res), "holder": summ(hs), "nonholder": summ(nh)}
                    out["attack"].append(rec)
                    print(name, seed, "T", ti, cond, liar, "nonholder",
                          json.dumps({k: (x["FA"], x["EXPt"]) for k, x in rec["nonholder"].items()}), flush=True)
        json.dump(out, open(ROOT / f"results/explore/anytrust_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    for name, loader in [("toniot", load_toniot), ("nbaiot", load_nbaiot), ("iot23", load_iot23)]:
        if name in sys.argv[1:]:
            run(name, loader)
