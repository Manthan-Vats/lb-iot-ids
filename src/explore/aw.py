"""E-AW1 (RESEARCH_PLAN): anchored witnesses. Benign evidence comes only from the victim's anchor-calibrated
local detector; attack evidence may come from any single peer, thresholded on the victim's anchor (A1) and
certified as a union on a held-out anchor part (A2), so FA is bounded whatever the number of lying peers."""
import sys, json, math, time
from pathlib import Path
import numpy as np, torch
from scipy.stats import binom
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedpart import DEV, load_toniot, load_iot23, load_nbaiot, Pool, ROOT
from selfwit import nbaiot_src
from conf import nn_dist, pvals
from anytrust import TG, ALPHAS, MINH
from abl import CAPB, CAPS
from absorb4 import ae
NT1, NT2, NA1, NA2 = 100, 100, 100, 300
ANC = NT1 + NT2 + NA1 + NA2
GAMMA, DELTA, PCAP, LPOOL, MINB, KCAP = 0.02, 0.05, 2000, 1000, 1100, 5000
EPS = (0.0, 0.02, 0.1)
CONDS = ("hon", "hide", "flip", "flipall", "mimic1", "mimic3", "mimic10", "tail")
KAPPA = 0.5
LIARS = {"mimic1": 1, "mimic3": 3, "mimic10": 10, "tail": 1}
SEEDS = [int(a[5:]) for a in sys.argv[1:] if a.startswith("seed=")] or [0, 1, 2]


def kmeans(X, k, seed, it=25):
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g, device=X.device)[:k]].clone()
    for _ in range(it):
        a = torch.cdist(X, C).argmin(1)
        for c in range(k):
            if (a == c).any():
                C[c] = X[a == c].mean(0)
    return torch.cdist(X, C).argmin(1).cpu().numpy()


def knn_vote(Q, R, lab, k=5):
    out = []
    for s in range(0, len(Q), 512):
        idx = torch.cdist(Q[s:s + 512], R).topk(k, largest=False).indices
        out.append(lab[idx].float().mean(1) > 0.5)
    return torch.cat(out).cpu().numpy()


def ladder(w):
    s, out = w, []
    while s >= 1:
        out.append(s); s //= 2
    return out


def cert_c(n_cand):
    """Largest A2 count c with BinomCDF(c; NA2, GAMMA) <= DELTA / n_cand (-1 if none)."""
    c = -1
    while binom.cdf(c + 1, NA2, GAMMA) <= DELTA / n_cand:
        c += 1
    return c


def run(name, loader):
    fed = loader(); pool = Pool(fed); n = fed.n; G = lambda ix: pool.X[torch.from_numpy(np.asarray(ix)).to(DEV)]
    src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
    bte = fed.yte == fed.benign; Xte = torch.from_numpy(fed.Xte).to(DEV)
    tgs = [t if isinstance(t, list) else [fed.class_names.index(t)] for t in TG[name]]
    ally = pool.y; out = []; t0 = time.time()
    outf = ROOT / f"results/explore/aw_{name}{'' if SEEDS == [0, 1, 2] else '_s' + ''.join(map(str, SEEDS))}.json"
    for seed in SEEDS:
        rs = np.random.RandomState(seed); btr, sh = {}, {}
        for i in range(n):
            ix = pool.client(i)
            btr[i] = rs.permutation(ix[ally[ix] == fed.benign])[:CAPB]
            sh[i] = rs.permutation(ix[ally[ix] != fed.benign])[:CAPS]
        sources = [j for j in range(n) if len(sh[j]) >= MINH]
        for ti, tg in enumerate(tgs):
            holders = [j for j in sources if np.isin(ally[sh[j]], tg).sum() >= MINH]
            if not holders:
                continue
            kp = rs.permutation(np.concatenate([sh[j][np.isin(ally[sh[j]], tg)] for j in holders]))
            kc, kh = kp[: len(kp) // 2], kp[len(kp) // 2:][:KCAP]
            h1 = int(rs.choice(holders))                 # the one honest holder under flip / mimic
            victims = [v for v in range(n) if v not in holders and len(btr[v]) >= MINB and (bte & (src == v)).sum() >= 20]
            for v in victims:
                b = btr[v]; T1, T2, A1, A2 = b[:NT1], b[NT1:NT1 + NT2], b[NT1 + NT2:NT1 + NT2 + NA1], b[NT1 + NT2 + NA1:ANC]
                rest = b[ANC:]; nl = min(LPOOL, len(rest) // 3); lp, base = rest[:nl], rest[nl:]
                XT1 = G(T1); mu, sd = XT1.mean(0), XT1.std(0).clamp_min(1e-3); Z = lambda X: (X - mu) / sd
                cont = {}
                for e in EPS:
                    k = math.ceil(e / (1 - e) * len(base)) if e else 0
                    cont[e] = rs.choice(kc, k, replace=k > len(kc)) if k else np.array([], int)
                ob = bte & (src == v)
                # query set: base, all contamination, A1, A2, T2, own benign test, K'
                parts = [base, np.concatenate([cont[e] for e in EPS]), A1, A2, T2]
                Xq = torch.cat([G(p) if len(p) else G(base[:0]) for p in parts] + [Xte[torch.from_numpy(ob).to(DEV)], G(kh)])
                off = np.cumsum([0] + [len(p) for p in parts] + [int(ob.sum()), len(kh)])
                sl = lambda i: slice(off[i], off[i + 1])
                Zq = Z(Xq); ZT1 = Z(XT1)
                                # witness pool: honest (peer, family) evidence + liar clusters
                wit = {}                                 # id -> (scores on Xq)
                for j in sources:
                    if j == v:
                        continue
                    lab = ally[sh[j]]
                    for f in np.unique(lab):
                        rows = sh[j][lab == f][:PCAP]
                        if len(rows) >= MINH:
                            wit[(j, int(f))] = nn_dist(Xq, G(rows)).cpu().numpy()
                Xlp = G(lp); liar_rows = {}
                for cn, K in LIARS.items():
                    if cn == "tail":
                        dev = (Xlp - Xlp.mean(0)).norm(dim=1).cpu().numpy()
                        lab = np.where(dev >= np.quantile(dev, 0.7), 0, -1)
                    else:
                        lab = kmeans(Xlp, K, seed) if K > 1 else np.zeros(len(lp), int)
                    for c in range(K):
                        if (lab == c).sum() >= 5:
                            liar_rows[(cn, c)] = lp[lab == c]
                            wit[(cn, c)] = nn_dist(Xq, Xlp[torch.from_numpy(lab == c).to(DEV)]).cpu().numpy()
                rad = {w: d[sl(2)].min() for w, d in wit.items()}     # r_w = min distance from A1 to P_w
                flg = {w: d < KAPPA * rad[w] for w, d in wit.items()}
                mA1 = {w: -rad[w] for w, d in wit.items()}             # order: r_w descending
                is_t = lambda w: isinstance(w[0], int) and w[1] in set(np.atleast_1d(tg).tolist())
                LA = {}                                  # local detectors on dirty D (Paper B anchor), per eps
                cache = {}

                def local(e, keep, tag):
                    D = np.concatenate([base, cont[e]])[keep]
                    key = (e, tag, keep.tobytes())
                    if key not in cache:
                        XD = G(D); sc = {}
                        fk = nn_dist(Zq, Z(XD)).cpu().numpy(); sc["kNN"] = fk
                        fa = ae(XD, mu, sd, 100 * seed + v)(Xq); sc["AE"] = fa
                        cache[key] = {d: pvals(s_[sl(4)], s_) for d, s_ in sc.items()}
                    return cache[key]
                for e in EPS:
                    nD = len(base) + len(cont[e]); iD = np.r_[np.arange(len(base)), len(base) + np.arange(len(cont[e]))]
                    # positions of this eps's D rows in the query set
                    ce = np.cumsum([0] + [len(cont[x]) for x in EPS]); cpos = off[1] + ce[EPS.index(e)] + np.arange(len(cont[e]))
                    qD = np.r_[np.arange(len(base)), cpos]
                    isK = np.r_[np.zeros(len(base), bool), np.ones(len(cont[e]), bool)]
                    pLA = local(e, np.ones(nD, bool), "all")
                    for cond in CONDS:
                        W = []
                        for w in wit:
                            if isinstance(w[0], str):
                                if w[0] == cond:
                                    W.append(w)
                                continue
                            j, f = w
                            if is_t(w) and j in holders:
                                if cond in ("hide", "flipall") or (cond != "hon" and j != h1):
                                    continue
                            W.append(w)
                        order = sorted(W, key=lambda w: mA1[w])
                        cands = ladder(len(order)); c = cert_c(max(1, len(cands)))
                        acc = []
                        for s_ in cands:                 # largest ladder prefix certified on A2
                            U = np.zeros(len(Xq), bool)
                            for w in order[:s_]:
                                U |= flg[w]
                            if c >= 0 and U[sl(3)].sum() <= c:
                                acc = order[:s_]; break
                        UA = np.zeros(len(Xq), bool); UN = np.zeros(len(Xq), bool)
                        for w in acc:
                            UA |= flg[w]
                        for w in W:
                            UN |= flg[w]
                        r = {"seed": seed, "target": ti, "v": int(v), "eps": e, "cond": cond, "n_hold": len(holders),
                             "n_wit": len(W), "n_acc": len(acc), "acc_target": int(sum(is_t(w) for w in acc)),
                             "acc_liar": int(sum(isinstance(w[0], str) for w in acc)), "c": c,
                             "rmK_AW": float(UA[qD][isK].mean() * 100) if isK.any() else None,
                             "rmB_AW": float(UA[qD][~isK].mean() * 100)}
                        pAW = local(e, ~UA[qD], "aw"); pNW = local(e, ~UN[qD], "nw")
                        # G: pooled classifier proxy, labels as reported (flippers report target as benign)
                        R, L = [np.concatenate([base, cont[e]])], [np.zeros(nD, bool)]
                        for j in sources:
                            if j == v:
                                continue
                            rows = sh[j]; t = np.isin(ally[rows], tg)
                            if j in holders and cond == "hide":
                                rows, t = rows[~t], t[~t]
                            rep = np.ones(len(rows), bool)
                            if j in holders and (cond == "flipall" or (cond not in ("hon", "hide") and j != h1)):
                                rep = ~t
                            R.append(rows); L.append(rep)
                        for (cn, _), rows in liar_rows.items():
                            if cn == cond:
                                R.append(rows); L.append(np.ones(len(rows), bool))
                        gal = knn_vote(Zq[off[5]:], Z(G(np.concatenate(R))), torch.from_numpy(np.concatenate(L)).to(DEV))
                        ib, ik = slice(0, int(ob.sum())), slice(int(ob.sum()), None)
                        r["G"] = [float(gal[ib].mean() * 100), float((~gal[ik]).mean() * 100)]
                        for a in ALPHAS:
                            for d in ("kNN", "AE"):
                                for tag, pp, U in (("LA", pLA, None), ("AW", pAW, UA), ("NW", pNW, UN)):
                                    al = pp[d] <= a
                                    if U is not None:
                                        al = al | U
                                    r[f"{tag}-{d}@{a}"] = [float(al[sl(5)].mean() * 100), float((~al[sl(6)]).mean() * 100)]
                        out.append(r)
                    print(name, seed, ti, v, e, f"{time.time() - t0:.0f}s",
                          {x["cond"]: (x["n_acc"], x["acc_target"], x["acc_liar"], [round(y, 1) for y in x["AW-kNN@0.05"]],
                                       [round(y, 1) for y in x["NW-kNN@0.05"]], [round(y, 1) for y in x["G"]])
                           for x in out if x["seed"] == seed and x["target"] == ti and x["v"] == v and x["eps"] == e}, flush=True)
            json.dump(out, open(outf, "w"))


if __name__ == "__main__":
    for name, loader in [("nbaiot", load_nbaiot), ("toniot", load_toniot), ("iot23", load_iot23)]:
        if name in sys.argv[1:]:
            run(name, loader)
