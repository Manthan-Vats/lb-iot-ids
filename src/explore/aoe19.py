"""E-AOE19 (RESEARCH_PLAN): AdaDetect-style peer-free baseline (Marandon et al.): MLP trained to separate the anchor
Ntr from an unlabelled mixture (Cal + benign test + attacks at 5% prevalence); conformal p-values from Cal. Cal and
benign test rows enter the mixture symmetrically, so the FA guarantee holds. Same victims/split as E-AOE15."""
import sys, json, time
import numpy as np, torch
from aw import DEV, load_toniot, load_iot23, load_nbaiot, Pool, ROOT, nbaiot_src, pvals, TG, MINH, CAPB, CAPS, KCAP
from aoe import train_clf
SEEDS = [int(a[5:]) for a in sys.argv[1:] if a.startswith("seed=")] or [0, 1, 2]
NTR, NCAL, MINV, AL, NTG, PREV = 300, 1000, 2300, (0.001, 0.01), 2, 0.05


def run(name, loader):
    fed = loader(); pool = Pool(fed); n = fed.n; G = lambda ix: pool.X[torch.from_numpy(np.asarray(ix)).to(DEV)]
    src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
    bte = fed.yte == fed.benign; Xte = torch.from_numpy(fed.Xte).to(DEV)
    tgs = [t if isinstance(t, list) else [fed.class_names.index(t)] for t in TG[name]][:NTG]; ally = pool.y
    out = []
    for seed in SEEDS:
        rs = np.random.RandomState(seed); sh, bfull = {}, {}
        for i in range(n):
            ix = pool.client(i)
            bfull[i] = rs.permutation(ix[ally[ix] == fed.benign]); sh[i] = rs.permutation(ix[ally[ix] != fed.benign])[:CAPS]
        sources = [j for j in range(n) if len(sh[j]) >= MINH]
        for ti, tg in enumerate(tgs):
            holders = [j for j in sources if np.isin(ally[sh[j]], tg).sum() >= MINH]
            if not holders:
                continue
            Rv = []
            for j in holders:
                t = sh[j][np.isin(ally[sh[j]], tg)]; Rv.append(t[len(t) // 2:])
            kh = rs.permutation(np.concatenate(Rv))[:KCAP]; rs.choice(holders)
            victims = [v for v in range(n) if v not in holders and len(bfull[v]) >= MINV and (bte & (src == v)).sum() >= 20]
            for v in victims:
                b = bfull[v]; Ntr, Cal = b[:NTR], b[NTR:NTR + NCAL]
                ob = bte & (src == v); nB = int(ob.sum()); XB = Xte[torch.from_numpy(ob).to(DEV)]
                QC = torch.cat([G(Cal), XB, G(kh)])
                na = max(1, int(round(PREV / (1 - PREV) * (NCAL + nB))))
                ar = np.random.RandomState(7 + 1000 * seed + v).choice(len(kh), min(na, len(kh)), replace=False)
                mix = torch.cat([G(Cal), XB, G(kh[ar])])
                s = train_clf([(mix, len(mix))], G(Ntr), 1000 * seed + 10 * v + ti, True)(QC)
                p = pvals(s[:NCAL], s[NCAL:])
                r = {"seed": seed, "target": ti, "v": int(v), "nB": nB, "na": int(len(ar))}
                for a in AL:
                    al = p <= a
                    r[f"AdaDetect@{a}"] = [int(al[:nB].sum()), float(al[:nB].mean() * 100), float((~al[nB:]).mean() * 100)]
                out.append(r); print(name, seed, ti, v, nB, len(ar), r["AdaDetect@0.001"][1:], r["AdaDetect@0.01"][1:], flush=True)
    json.dump(out, open(ROOT / f"results/explore/aoe19_{name}.json", "w"))


if __name__ == "__main__":
    for nm, ld in [("nbaiot", load_nbaiot), ("toniot", load_toniot), ("iot23", load_iot23)]:
        if nm in sys.argv[1:]:
            run(nm, ld)
