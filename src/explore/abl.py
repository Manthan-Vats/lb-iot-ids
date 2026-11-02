"""E-SW15 (RESEARCH_PLAN): share attacks, keep benign local (ABL). Each source j shares only labelled
ATTACK rows; victim v trains one classifier per source (v's own benign vs j's attack rows), calibrates
every score on its own held-out benign rows and alarms if the local score or the max-rank over
sources is extreme (anytrust veto, notes/anytrust_theory.md). No model or benign row ever leaves a client."""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedpart import DEV, load_toniot, load_iot23, load_nbaiot, Pool, ROOT
from selfwit import nbaiot_src
from conf import local_score, pvals
from anytrust import TG, ALPHAS, MINH
CAPB, CAPS = 5000, 5000          # own benign cap; shared attack rows per source
LIAR = next((a[5:] for a in sys.argv[1:] if a.startswith("liar=")), "junk")
RULES = ["L", "P", "S", "L|P", "L|S"]


def train_ens(Xb, As, evals, seed, steps=400, bs=256, h=(64, 32)):
    """m independent MLPs d-64-32-1 trained in parallel; model j sees balanced batches of v's benign
    rows Xb (label 0) and source j's rows As[j] (label 1). Returns [(scores on each eval set)] per j."""
    m, d = len(As), As[0].shape[1]; g = torch.Generator(device=DEV).manual_seed(seed); torch.manual_seed(seed)
    def lin(i, o):
        k = 1 / i ** 0.5
        return (torch.empty(m, i, o, device=DEV).uniform_(-k, k).requires_grad_(),
                torch.empty(m, 1, o, device=DEV).uniform_(-k, k).requires_grad_())
    P = [lin(d, h[0]), lin(h[0], h[1]), lin(h[1], 1)]
    opt = torch.optim.Adam([t for p in P for t in p], 1e-3)
    def fwd(X):                                        # X (m, B, d) -> (m, B)
        for k, (W, b) in enumerate(P):
            X = torch.baddbmm(b, X, W)
            if k < 2:
                X = torch.relu(X)
        return X[..., 0]
    A = torch.cat(As); off = torch.tensor(np.cumsum([0] + [len(a) for a in As[:-1]]), device=DEV)
    ln = torch.tensor([len(a) for a in As], device=DEV, dtype=torch.float32)
    Bs = Xb if isinstance(Xb, list) else [Xb] * m     # per-model benign sets (E-SW17) or one shared set
    Xb = torch.cat(Bs); offb = torch.tensor(np.cumsum([0] + [len(b) for b in Bs[:-1]]), device=DEV)
    lnb = torch.tensor([len(b) for b in Bs], device=DEV, dtype=torch.float32)
    y = torch.cat([torch.zeros(bs), torch.ones(bs)]).to(DEV).expand(m, -1)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        ib = offb[:, None] + (torch.rand(m, bs, generator=g, device=DEV) * lnb[:, None]).long()
        ia = off[:, None] + (torch.rand(m, bs, generator=g, device=DEV) * ln[:, None]).long()
        X = torch.cat([Xb[ib], A[ia]], 1); loss = lossf(fwd(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    outs = []
    with torch.no_grad():
        for E in evals:
            outs.append(torch.cat([fwd(E[s:s + 8192].expand(m, -1, -1)) for s in range(0, len(E), 8192)], 1).cpu().numpy())
    return [tuple(o[j] for o in outs) for j in range(m)]


def maxrank_p(items, k1, k2):
    """items: list of (test scores, cal scores over v's C1 u C2 order); ranks on C1, p on C2."""
    Mt, Mc = None, None
    for st, sc in items:
        ref = np.sort(sc[k1]); a = np.searchsorted(ref, st, "left") / len(ref)
        b = np.searchsorted(ref, sc[k2], "left") / len(ref)
        Mt = a if Mt is None else np.maximum(Mt, a); Mc = b if Mc is None else np.maximum(Mc, b)
    return pvals(Mc, Mt)


def run(name, loader, nbs):
    fed = loader(); pool = Pool(fed); Xte = torch.from_numpy(fed.Xte).to(DEV); n = fed.n
    src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
    bte = fed.yte == fed.benign; atk_te = ~bte
    tgs = [t if isinstance(t, list) else [fed.class_names.index(t)] for t in TG[name]]
    ally = pool.y; out = []
    T = lambda ix: pool.X[torch.from_numpy(np.asarray(ix)).to(DEV)]
    for nb in nbs:
        for seed in range(3):
            rs = np.random.RandomState(seed)
            tr, c1, c2, sh = {}, {}, {}, {}
            for i in range(n):
                ix = pool.client(i); bi = rs.permutation(ix[ally[ix] == fed.benign])[:min(nb, CAPB)]
                a, b = int(len(bi) * 0.6), int(len(bi) * 0.8)
                tr[i], c1[i], c2[i] = bi[:a], bi[a:b], bi[b:]
                sh[i] = rs.permutation(ix[ally[ix] != fed.benign])[:CAPS]
            sources = [j for j in range(n) if len(sh[j]) >= MINH]
            benign_all = np.concatenate([tr[i] for i in range(n)])
            conds = [("clean", None, [], None)]
            for ti, tg in enumerate(tgs):
                holders = [j for j in sources if np.isin(ally[sh[j]], tg).sum() >= MINH]
                if len(holders) < 2:
                    continue
                surv = int(rs.choice(holders))
                conds.append(("one", ti, [j for j in holders if j != surv], tg))
                conds.append(("none", ti, holders, tg))
            for v in range(n):
                if len(tr[v]) < 5 or len(c1[v]) < 1 or len(c2[v]) < 1:
                    continue
                cc = np.concatenate([c1[v], c2[v]]); k1 = np.arange(len(c1[v])); k2 = np.arange(len(c1[v]), len(cc))
                Xv, Xc = T(tr[v]), T(cc)
                f = local_score(Xv); pL = pvals(f(Xc), f(Xte))
                ob = bte & (src == v)
                Zv = (Xv - Xv.mean(0)) / Xv.std(0).clamp_min(1e-3)
                tail = tr[v][np.argsort(-Zv.norm(dim=1).cpu().numpy())[:max(1, int(0.3 * len(tr[v])))]]
                cache = {}

                def clfs(keys, rowsets):
                    """Trains the missing (key -> rows) classifiers as one batched ensemble."""
                    need = [(k, r) for k, r in zip(keys, rowsets) if k not in cache]
                    if need:
                        sc = train_ens(Xv, [T(r) for _, r in need], (Xte, Xc), 1000 * seed + v)
                        for (k, _), (a, b) in zip(need, sc):
                            cache[k] = (a, b)
                    return [cache[k] for k in keys]
                for cond, ti, byz, tg in conds:
                    if v in byz:
                        continue
                    keys, pooled = [], []
                    for j in sources:
                        if j == v:
                            continue
                        if j in byz:
                            # liar: withholds the target, sends benign-looking junk labelled attack
                            if LIAR == "mimic":     # oracle liar: v's own benign tail, labelled attack
                                junk = np.resize(tail, CAPS // 2)
                            else:
                                junk = rs.choice(benign_all[~np.isin(benign_all, pool.client(v))], CAPS // 2)
                            rows = np.concatenate([sh[j][~np.isin(ally[sh[j]], tg)], junk])
                            keys.append(("liar", j, ti)); pooled.append(rows)
                        else:
                            keys.append(("hon", j)); pooled.append(sh[j])
                    if v in sources:
                        keys.append(("own", v)); pooled.append(sh[v])
                    if not keys:
                        continue
                    keys.append(("pool", cond, ti)); pooled.append(np.concatenate(pooled))
                    items = clfs(keys, pooled); sp = items.pop(); pP = pvals(sp[1], sp[0])
                    pS = maxrank_p(items, k1, k2)
                    r = {"nb": nb, "seed": seed, "cond": cond, "target": ti, "v": int(v), "ncal": len(cc),
                         "holder": bool(tg is not None and np.isin(ally[sh[v]], tg).sum() >= MINH)}
                    for a in ALPHAS:
                        alarm = {"L": pL <= a, "P": pP <= a, "S": pS <= a,
                                 "L|P": (pL <= a / 2) | (pP <= a / 2), "L|S": (pL <= a / 2) | (pS <= a / 2),
                                 "L|P|S": (pL <= a / 3) | (pP <= a / 3) | (pS <= a / 3)}
                        for k, al in alarm.items():
                            rel = ~al
                            r[f"{k}@{a}"] = [float(al[ob].mean() * 100) if ob.sum() else None,
                                             float(rel[atk_te].mean() * 100),
                                             float(rel[np.isin(fed.yte, tg)].mean() * 100) if tg is not None else None,
                                             [float(rel[np.isin(fed.yte, t)].mean() * 100) for t in tgs]]
                    out.append(r)
            for cond in ("clean", "one", "none"):
                rr = [x for x in out if x["nb"] == nb and x["seed"] == seed and x["cond"] == cond and not x["holder"]]
                if rr:
                    print(name, nb, seed, cond, "nonholder", json.dumps({k: [round(float(np.mean([x[k][q] for x in rr if x[k][q] is not None])), 2)
                          if any(x[k][q] is not None for x in rr) else None for q in range(3)] for k in rr[0] if "@" in k}), flush=True)
            json.dump(out, open(ROOT / f"results/explore/abl_{name}{'' if LIAR == 'junk' else '_' + LIAR}{'' if nbs == [5000] else '_nb' + str(nbs[0])}.json", "w"))


if __name__ == "__main__":
    nbs = [int(a[3:]) for a in sys.argv[1:] if a.startswith("nb=")] or [5000]
    for name, loader in [("toniot", load_toniot), ("nbaiot", load_nbaiot), ("iot23", load_iot23)]:
        if name in sys.argv[1:]:
            run(name, loader, nbs)
