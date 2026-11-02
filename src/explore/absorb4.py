"""E-SW23 (RESEARCH_PLAN): scaler absorption. Dirty training rows, clean calibration; the scaler is fit on the
dirty rows (DZ), on the clean benign rows (OZ, oracle) or robustly on the dirty rows (RS)."""
import sys, json, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedpart import DEV, load_toniot, load_iot23, load_nbaiot, Pool, ROOT
from selfwit import nbaiot_src
from conf import nn_dist, pvals
from anytrust import TG, ALPHAS, MINH
from abl import CAPB, CAPS
from absorb import EPS


def scaler(X, kind):
    if kind == "RS":
        q = torch.quantile(X, torch.tensor([0.25, 0.5, 0.75], device=X.device), dim=0)
        return q[1], ((q[2] - q[0]) / 1.349).clamp_min(1e-3)
    return X.mean(0), X.std(0).clamp_min(1e-3)


def knn(Xtr, mu, sd):
    R = (Xtr - mu) / sd
    return lambda Q: nn_dist((Q - mu) / sd, R).cpu().numpy()


def ae(Xtr, mu, sd, seed, steps=1500):
    torch.manual_seed(seed); Z = (Xtr - mu) / sd; d = Z.shape[1]
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


def run(name, loader):
    fed = loader(); pool = Pool(fed); n = fed.n; G = lambda ix: pool.X[torch.from_numpy(np.asarray(ix)).to(DEV)]
    src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
    bte = fed.yte == fed.benign; Xte = torch.from_numpy(fed.Xte).to(DEV)
    tgs = [t if isinstance(t, list) else [fed.class_names.index(t)] for t in TG[name]]
    ally = pool.y; out = []
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
            kc, kh = kp[: len(kp) // 2], kp[len(kp) // 2:]; Xkh = G(kh)
            victims = [v for v in range(n) if v not in holders and len(btr[v]) >= 200 and (bte & (src == v)).sum() >= 20]
            for v in victims:
                ob = torch.from_numpy(bte & (src == v)).to(DEV); Xob = Xte[ob]
                b = btr[v]; a6 = int(0.6 * len(b)); trb, ccb = b[:a6], b[a6:]; Xcb = G(ccb)
                oz = scaler(G(trb), "DZ")
                for e in EPS:
                    k = math.ceil(e / (1 - e) * len(b)) if e else 0
                    cr = rs.choice(kc, k, replace=k > len(kc)) if k else np.array([], int)
                    Xtr = G(np.concatenate([trb, cr[rs.rand(k) < 0.6]]))
                    r = {"seed": seed, "target": ti, "v": int(v), "eps": e}
                    for sk, (mu, sd) in (("DZ", scaler(Xtr, "DZ")), ("OZ", oz), ("RS", scaler(Xtr, "RS"))):
                        for dn, f in (("kNN", knn(Xtr, mu, sd)), ("AE", ae(Xtr, mu, sd, 100 * seed + v))):
                            cal, so, sk_ = f(Xcb), f(Xob), f(Xkh)
                            for al in ALPHAS:
                                r[f"{dn}@{al}@{sk}"] = [float((pvals(cal, so) <= al).mean() * 100), float((pvals(cal, sk_) > al).mean() * 100)]
                    out.append(r)
                print(name, seed, ti, v, {k_: [round(x, 1) for x in r[k_]] for k_ in r if "@0.01@" in k_}, flush=True)
            json.dump(out, open(ROOT / f"results/explore/absorb4_{name}.json", "w"))


if __name__ == "__main__":
    for name, loader in [("nbaiot", load_nbaiot), ("toniot", load_toniot), ("iot23", load_iot23)]:
        if name in sys.argv[1:]:
            run(name, loader)
