"""E-SW20 (RESEARCH_PLAN): universality of calibration absorption. A victim's dirty baseline holds a
fraction eps of one attack family; every calibrated detector should then release >= 1 - a/eps of that
family (Theorem U), whatever the model."""
import sys, json, math
from pathlib import Path
import numpy as np, torch
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedpart import DEV, load_toniot, load_iot23, load_nbaiot, Pool, ROOT
from selfwit import nbaiot_src
from conf import local_score, pvals
from anytrust import TG, ALPHAS, MINH
from abl import CAPB, CAPS
EPS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def ae_score(X, seed, steps=1500):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3); Z = (X - mu) / sd; d = Z.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16), torch.nn.ReLU(),
                              torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, d)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(steps):
        b = Z[torch.randint(len(Z), (256,), device=DEV)]
        loss = ((net(b) - b) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    def f(Q):
        with torch.no_grad():
            Qz = (Q - mu) / sd
            return torch.cat([((net(q) - q) ** 2).mean(1) for q in Qz.split(8192)]).cpu().numpy()
    return f


def sk_scores(Xtr, seed):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3; z = lambda Q: (Q - mu) / sd; Z = z(Xtr)
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(Z)
    pca = PCA(0.95, random_state=seed).fit(Z)
    gmm = GaussianMixture(min(8, max(1, len(Z) // 50)), covariance_type="diag", reg_covar=1e-3, random_state=seed).fit(Z)
    rec = lambda Q: ((pca.inverse_transform(pca.transform(Q)) - Q) ** 2).sum(1)
    return {"IF": lambda Q: -iso.score_samples(z(Q)), "PCA": lambda Q: rec(z(Q)), "GMM": lambda Q: -gmm.score_samples(z(Q))}


def run(name, loader):
    fed = loader(); pool = Pool(fed); Xte = torch.from_numpy(fed.Xte).to(DEV); n = fed.n
    src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
    bte = fed.yte == fed.benign; Xte_np = fed.Xte
    tgs = [t if isinstance(t, list) else [fed.class_names.index(t)] for t in TG[name]]
    ally = pool.y; out = []; Xall = pool.X.cpu().numpy()
    for seed in range(3):
        rs = np.random.RandomState(seed); btr, sh = {}, {}
        for i in range(n):
            ix = pool.client(i)
            btr[i] = rs.permutation(ix[ally[ix] == fed.benign])[:CAPB]
            sh[i] = rs.permutation(ix[ally[ix] != fed.benign])[:CAPS]
        for ti, tg in enumerate(tgs):
            holders = [j for j in range(n) if np.isin(ally[sh[j]], tg).sum() >= MINH]
            if not holders:
                continue
            kp = rs.permutation(np.concatenate([sh[j][np.isin(ally[sh[j]], tg)] for j in holders]))
            kc, kh = kp[: len(kp) // 2], kp[len(kp) // 2:]           # contamination pool / held-out K'
            Xkh = Xall[kh]; tte = np.isin(fed.yte, tg)
            victims = [v for v in range(n) if v not in holders and len(btr[v]) >= 200 and (bte & (src == v)).sum() >= 20]
            for v in victims:
                ob = bte & (src == v)
                for e in EPS:
                    k = math.ceil(e / (1 - e) * len(btr[v])) if e else 0
                    cr = rs.choice(kc, k, replace=k > len(kc)) if k else np.array([], int)
                    rows = rs.permutation(np.concatenate([btr[v], cr])); a = int(0.6 * len(rows))
                    tr, cc = rows[:a], rows[a:]
                    Xtr, Xcc = Xall[tr], Xall[cc]
                    S = sk_scores(Xtr, seed)
                    fk = local_score(torch.from_numpy(Xtr).to(DEV)); S["kNN"] = lambda Q, fk=fk: fk(torch.from_numpy(Q).to(DEV))
                    fa = ae_score(torch.from_numpy(Xtr).to(DEV), 100 * seed + v); S["AE"] = lambda Q, fa=fa: fa(torch.from_numpy(Q).to(DEV))
                    r = {"seed": seed, "target": ti, "v": int(v), "eps": e, "ncal": len(cc), "ntr": len(tr)}
                    for dn, f in S.items():
                        sc, st, sk = f(Xcc), f(Xte_np), f(Xkh)
                        for al in ALPHAS:
                            pt, pk = pvals(sc, st), pvals(sc, sk)
                            r[f"{dn}@{al}"] = [float((pt[ob] <= al).mean() * 100), float((pt[tte] > al).mean() * 100),
                                               float((pk > al).mean() * 100)]
                        if dn == "AE":                                # Kitsune rule: threshold = max train score
                            thr = f(Xtr).max()
                            r["KIT"] = [float((st[ob] > thr).mean() * 100), float((st[tte] <= thr).mean() * 100),
                                        float((sk <= thr).mean() * 100)]
                    out.append(r)
                print(name, seed, ti, v, {k: [round(x, 1) for x in r[k]] for k in r if "@0.01" in k or k == "KIT"}, flush=True)
            json.dump(out, open(ROOT / f"results/explore/absorb_{name}.json", "w"))


if __name__ == "__main__":
    for name, loader in [("toniot", load_toniot), ("nbaiot", load_nbaiot), ("iot23", load_iot23)]:
        if name in sys.argv[1:]:
            run(name, loader)
