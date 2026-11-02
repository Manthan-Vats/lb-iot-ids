"""E-AOE1 (RESEARCH_PLAN): Anchored Outlier Exposure. The victim's detector is a classifier of peers' shared attack
rows (untrusted) vs its own small trusted benign anchor, never trained on its (possibly dirty) baseline; a split-
conformal threshold on a disjoint anchor part bounds the false-alarm rate whatever the peers send.
Leak-free split: each holder's target rows are halved into shared evidence E_j and victim-side rows R_j."""
import sys, json, math, time
import numpy as np, torch
from aw import (DEV, load_toniot, load_iot23, load_nbaiot, Pool, ROOT, nbaiot_src, nn_dist, pvals, TG, ALPHAS,
                MINH, CAPB, CAPS, ae, LPOOL, MINB, KCAP, kmeans)
CONDS = ("hon", "flip", "flipall", "mimic10", "tail", "flood")
EPS = 0.1
SEEDS = [int(a[5:]) for a in sys.argv[1:] if a.startswith("seed=")] or [0, 1, 2]


def train_clf(groups, N, seed, by_group, steps=400):
    """groups: list of (X, weight). Positives sampled by group uniformly (by_group) or proportional to weight."""
    torch.manual_seed(seed); g = np.random.RandomState(seed)
    Xs = torch.cat([x for x, _ in groups] + [N]); mu, sd = Xs.mean(0), Xs.std(0).clamp_min(1e-2)
    w = np.array([1.0 if by_group else float(c) for _, c in groups]); w /= w.sum()
    d = N.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4); bce = torch.nn.BCEWithLogitsLoss()
    y = torch.cat([torch.ones(256), torch.zeros(256)]).to(DEV)
    for _ in range(steps):
        gi = np.bincount(g.choice(len(groups), 256, p=w), minlength=len(groups))
        p = torch.cat([groups[i][0][torch.randint(len(groups[i][0]), (int(c),), device=DEV)] for i, c in enumerate(gi) if c])
        nn_ = N[torch.randint(len(N), (256,), device=DEV)]
        loss = bce(net((torch.cat([p, nn_]) - mu) / sd).squeeze(1), y); opt.zero_grad(); loss.backward(); opt.step()

    def f(Q):
        with torch.no_grad():
            return torch.cat([net((q - mu) / sd).squeeze(1) for q in Q.split(8192)]).cpu().numpy()
    return f


def run(name, loader):
    fed = loader(); pool = Pool(fed); n = fed.n; G = lambda ix: pool.X[torch.from_numpy(np.asarray(ix)).to(DEV)]
    src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
    bte = fed.yte == fed.benign; Xte = torch.from_numpy(fed.Xte).to(DEV)
    tgs = [t if isinstance(t, list) else [fed.class_names.index(t)] for t in TG[name]]; ally = pool.y
    out = []; t0 = time.time()
    outf = ROOT / f"results/explore/aoe_{name}{'' if SEEDS == [0, 1, 2] else '_s' + ''.join(map(str, SEEDS))}.json"
    for seed in SEEDS:
        rs = np.random.RandomState(seed); btr, sh = {}, {}
        for i in range(n):
            ix = pool.client(i)
            btr[i] = rs.permutation(ix[ally[ix] == fed.benign])[:CAPB]; sh[i] = rs.permutation(ix[ally[ix] != fed.benign])[:CAPS]
        sources = [j for j in range(n) if len(sh[j]) >= MINH]
        for ti, tg in enumerate(tgs):
            holders = [j for j in sources if np.isin(ally[sh[j]], tg).sum() >= MINH]
            if not holders:
                continue
            E, Rv = {}, []
            for j in holders:
                t = sh[j][np.isin(ally[sh[j]], tg)]; E[j] = t[: len(t) // 2]; Rv.append(t[len(t) // 2:])
            kp = rs.permutation(np.concatenate(Rv)); kc, kh = kp[: len(kp) // 2], kp[len(kp) // 2:][:KCAP]
            h1 = int(rs.choice(holders))
            victims = [v for v in range(n) if v not in holders and len(btr[v]) >= MINB and (bte & (src == v)).sum() >= 20]
            for v in victims:
                b = btr[v]; Ntr, Cal = b[:300], b[300:600]; T1, T2 = b[:100], b[100:200]
                nl = min(LPOOL, (len(b) - 600) // 3); lp = b[600:600 + nl]; base = b[600 + nl:]   # as in aw.py
                k = math.ceil(EPS / (1 - EPS) * len(base)); cont = rs.choice(kc, k, replace=k > len(kc)); D = np.concatenate([base, cont])
                ob = bte & (src == v); nB = int(ob.sum())
                Q = torch.cat([Xte[torch.from_numpy(ob).to(DEV)], G(kh)])
                XT1 = G(T1); mu, sd = XT1.mean(0), XT1.std(0).clamp_min(1e-3); Z = lambda X: (X - mu) / sd
                pLA = {}
                sK = nn_dist(Z(torch.cat([G(T2), Q])), Z(G(D))).cpu().numpy(); pLA["kNN"] = pvals(sK[:100], sK[100:])
                fa = ae(G(D), mu, sd, 100 * seed + v); sA = fa(torch.cat([G(T2), Q])); pLA["AE"] = pvals(sA[:100], sA[100:])
                Xlp = G(lp); dev = (Xlp - Xlp.mean(0)).norm(dim=1).cpu().numpy(); lab10 = kmeans(Xlp, 10, seed)
                benign_other = np.concatenate([btr[i] for i in range(n) if i != v])
                for cond in CONDS:
                    groups, gneg, honest_counts = [], [G(D), G(benign_other)], []
                    for j in sources:
                        if j == v:
                            continue
                        rows = sh[j][~np.isin(ally[sh[j]], tg)]
                        if j in holders:
                            if cond == "hon" or (cond != "flipall" and j == h1):
                                rows = np.concatenate([rows, E[j]])
                            else:
                                gneg.append(G(E[j]))          # flipper reports its target rows as benign
                        if len(rows) >= MINH:
                            groups.append((G(rows), len(rows))); honest_counts.append(len(rows))
                    liars = []
                    if cond == "mimic10":
                        liars = [(Xlp[torch.from_numpy(lab10 == c).to(DEV)], int((lab10 == c).sum())) for c in range(10) if (lab10 == c).sum() >= 5]
                    elif cond == "tail":
                        m = dev >= np.quantile(dev, 0.7); liars = [(Xlp[torch.from_numpy(m).to(DEV)], int(m.sum()))]
                    elif cond == "flood":
                        liars = [(Xlp, int(10 * np.median(honest_counts)))] * 10
                    groups = groups + liars
                    r = {"seed": seed, "target": ti, "v": int(v), "cond": cond, "n_src": len(groups), "n_liar": len(liars)}
                    sc = {}
                    for tag, byg in (("AOE-row", False), ("AOE-peer", True)):
                        f = train_clf(groups, G(Ntr), 1000 * seed + 10 * v + ti, byg)
                        s = f(torch.cat([G(Cal), Q])); sc[tag] = pvals(s[:300], s[300:])
                    # G: pooled supervised classifier, labels as reported, threshold 0.5
                    fg = train_clf(groups, torch.cat(gneg), 7 + seed, False)
                    sg = fg(Q) > 0
                    r["G"] = [float(sg[:nB].mean() * 100), float((~sg[nB:]).mean() * 100)]
                    for a in ALPHAS:
                        for tag, al in (("LA-kNN", pLA["kNN"] <= a), ("LA-AE", pLA["AE"] <= a), ("AOE-row", sc["AOE-row"] <= a),
                                        ("AOE-peer", sc["AOE-peer"] <= a), ("FUSE", (sc["AOE-peer"] <= a / 2) | (pLA["kNN"] <= a / 2))):
                            r[f"{tag}@{a}"] = [float(al[:nB].mean() * 100), float((~al[nB:]).mean() * 100)]
                    out.append(r)
                print(name, seed, ti, v, f"{time.time() - t0:.0f}s",
                      {x["cond"]: (x["LA-kNN@0.05"], x["AOE-peer@0.05"], x["G"]) for x in out[-len(CONDS):]}, flush=True)
            json.dump(out, open(outf, "w"))


if __name__ == "__main__":
    for nm, ld in [("nbaiot", load_nbaiot), ("toniot", load_toniot), ("iot23", load_iot23)]:
        if nm in sys.argv[1:]:
            run(nm, ld)
