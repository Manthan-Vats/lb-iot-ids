"""E-AOE15 (RESEARCH_PLAN): local budget rule (LB). s(x) = d_A(x) + BIG * [d_S(x) < d_A(x)], conformal on Cal.
For x outside B the p-value exceeds OC-kNN's by at most #Cal-in-B / (n + 1), so liars cost at most the budget they
occupy. LB-v drops sources (largest trusted-Sel footprint first) until the Sel footprint <= floor(a/2 * |Sel|)."""
import sys, json, time
import numpy as np, torch
from aw import (DEV, load_toniot, load_iot23, load_nbaiot, Pool, ROOT, nbaiot_src, nn_dist, pvals, TG,
                MINH, CAPB, CAPS, KCAP, kmeans)
from aoe import train_clf
DEDUP = "dedup" in sys.argv[1:]
SEEDS = [int(a[5:]) for a in sys.argv[1:] if a.startswith("seed=")] or [0, 1, 2]
NTR, NCAL, NSEL, MINV, AL, NTG, BIG, KAPPA = 300, 1000, 500, 2300, (0.001, 0.01), 2, 1e6, 0.5
for a_ in sys.argv[1:]:
    if a_.startswith("nsel="):
        NSEL = int(a_[5:]); MINV = NTR + NCAL + NSEL + 100
CONDS = ("hon", "flip", "cw3", "cw10", "mimic10")


def dmin(Q, R):
    return torch.cat([nn_dist(q, R) for q in Q.split(2048)])


def run(name, loader):
    fed = loader(); pool = Pool(fed); n = fed.n; G = lambda ix: pool.X[torch.from_numpy(np.asarray(ix)).to(DEV)]
    src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
    bte = fed.yte == fed.benign; Xte = torch.from_numpy(fed.Xte).to(DEV)
    tgs = [t if isinstance(t, list) else [fed.class_names.index(t)] for t in TG[name]][:NTG]; ally = pool.y
    out = []; t0 = time.time()
    outf = ROOT / f"results/explore/aoe15_{name}{'_dedup' if DEDUP else ''}{'' if NSEL == 500 else f'_sel{NSEL}'}{'' if SEEDS == [0, 1, 2] else '_s' + ''.join(map(str, SEEDS))}.json"
    for seed in SEEDS:
        rs = np.random.RandomState(seed); btr, sh, bfull = {}, {}, {}
        for i in range(n):
            ix = pool.client(i)
            bfull[i] = rs.permutation(ix[ally[ix] == fed.benign]); btr[i] = bfull[i][:CAPB]
            sh[i] = rs.permutation(ix[ally[ix] != fed.benign])[:CAPS]
        sources = [j for j in range(n) if len(sh[j]) >= MINH]
        for ti, tg in enumerate(tgs):
            holders = [j for j in sources if np.isin(ally[sh[j]], tg).sum() >= MINH]
            if not holders:
                continue
            E, Rv = {}, []
            for j in holders:
                t = sh[j][np.isin(ally[sh[j]], tg)]; E[j] = t[: len(t) // 2]; Rv.append(t[len(t) // 2:])
            kh = rs.permutation(np.concatenate(Rv))[:KCAP]; h1 = int(rs.choice(holders))
            if DEDUP:   # B7: keep only test attacks with no exact duplicate among rows any peer can share
                shr = np.setdiff1d(np.concatenate([sh[j] for j in range(n)]), np.concatenate(Rv))
                dd = torch.cat([nn_dist(q, G(shr)) for q in G(kh).split(1024)]).cpu().numpy()
                print(name, seed, ti, "dedup kept", int((dd > 1e-6).sum()), "of", len(kh), flush=True)
                kh = kh[dd > 1e-6]
                if len(kh) < 20:
                    continue
            victims = [v for v in range(n) if v not in holders and len(bfull[v]) >= MINV and (bte & (src == v)).sum() >= 20]
            for v in victims:
                b = bfull[v]; Ntr, Cal = b[:NTR], b[NTR:NTR + NCAL]
                Sel = b[NTR + NCAL:NTR + NCAL + NSEL]; lp = b[NTR + NCAL + NSEL:]
                ob = bte & (src == v); nB = int(ob.sum()); Q = torch.cat([Xte[torch.from_numpy(ob).to(DEV)], G(kh)])
                QC = torch.cat([G(Cal), Q])
                XT1 = G(Ntr[:100]); mu, sd = XT1.mean(0), XT1.std(0).clamp_min(1e-3); Z = lambda X: (X - mu) / sd
                ZA = Z(G(Ntr)); dA = dmin(Z(QC), ZA); dAS = dmin(Z(G(Sel)), ZA)
                p0 = {"OC-kNN": pvals(dA[:NCAL].cpu().numpy(), dA[NCAL:].cpu().numpy())}
                Xlp = G(lp); lab10 = kmeans(Xlp, 10, seed)
                cwr = np.random.RandomState(100 * seed + v)
                cwc = [i for i in cwr.permutation(n) if i != v and len(btr[i]) >= MINH][:10]
                sd_ = 1000 * seed + 10 * v + ti
                for cond in CONDS:
                    groups, liar = [], []
                    for j in sources:
                        if j == v:
                            continue
                        rows = sh[j][~np.isin(ally[sh[j]], tg)]
                        if j in holders and (cond == "hon" or j == h1):
                            rows = np.concatenate([rows, E[j]])
                        if len(rows) >= MINH:
                            groups.append((G(rows), len(rows))); liar.append(False)
                    if cond == "mimic10":
                        for c in range(10):
                            if (lab10 == c).sum() >= 5:
                                groups.append((Xlp[torch.from_numpy(lab10 == c).to(DEV)], int((lab10 == c).sum()))); liar.append(True)
                    elif cond.startswith("cw"):
                        for i in cwc[:int(cond[2:])]:
                            groups.append((G(btr[i]), len(btr[i]))); liar.append(True)
                    p = dict(p0)
                    s = train_clf(groups, G(Ntr), sd_, True)(QC); p["AOE-peer"] = pvals(s[:NCAL], s[NCAL:])
                    # per-source boost regions on QC and on Sel
                    inQ = torch.stack([dmin(Z(QC), Z(X_)) < dA for X_, _ in groups])          # (K, |QC|)
                    inS = torch.stack([dmin(Z(G(Sel)), Z(X_)) < dAS for X_, _ in groups])     # (K, |Sel|)
                    fp = inS.float().mean(1).cpu().numpy(); isl = np.array(liar)

                    def lb(keep):
                        bq = inQ[torch.from_numpy(keep).to(DEV)].any(0) if keep.any() else torch.zeros_like(dA, dtype=torch.bool)
                        sc = (dA + BIG * bq.float()).cpu().numpy()
                        return pvals(sc[:NCAL], sc[NCAL:]), float(bq[:NCAL].float().mean())
                    p["LB"], cB = lb(np.ones(len(groups), bool))
                    r = {"seed": seed, "target": ti, "v": int(v), "cond": cond, "nB": nB, "nK": len(kh), "cB_LB": cB,
                         "fp_liar": fp[isl].tolist(), "fp_hon": fp[~isl].tolist()}
                    for a in AL:
                        cap = int(np.floor(KAPPA * a * NSEL)); keep = np.ones(len(groups), bool)
                        for k in np.argsort(-fp):   # drop the largest footprints first until the union fits the cap
                            if not keep.any() or int(inS[torch.from_numpy(keep).to(DEV)].any(0).sum()) <= cap:
                                break
                            keep[k] = False
                        p[f"LB-v@{a}"], r[f"cB_v@{a}"] = lb(keep)
                        r[f"kept_liar@{a}"] = float(keep[isl].mean()) if isl.any() else -1.0; r[f"kept_hon@{a}"] = float(keep[~isl].mean())
                    for a in AL:
                        for d, pp in p.items():
                            if d.startswith("LB-v@") and d != f"LB-v@{a}":
                                continue
                            al = pp <= a; key = "LB-v" if d.startswith("LB-v@") else d
                            r[f"{key}@{a}"] = [int(al[:nB].sum()), float(al[:nB].mean() * 100), float((~al[nB:]).mean() * 100)]
                    out.append(r)
                print(name, seed, ti, v, nB, f"{time.time() - t0:.0f}s",
                      {x["cond"]: [round(x[f"{d}@0.01"][2], 1) for d in ("OC-kNN", "AOE-peer", "LB", "LB-v")]
                       + [round(x["cB_LB"], 3), x["kept_liar@0.01"], round(x["kept_hon@0.01"], 2)] for x in out[-len(CONDS):]}, flush=True)
            json.dump(out, open(outf, "w"))


if __name__ == "__main__":
    for nm, ld in [("nbaiot", load_nbaiot), ("toniot", load_toniot), ("iot23", load_iot23)]:
        if nm in sys.argv[1:]:
            run(nm, ld)
