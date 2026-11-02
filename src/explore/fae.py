"""E-SW9 (RESEARCH_PLAN): local witness vs federated benign-only autoencoder (DIoT-style) under
PDR-style data poisoning. Model-free LW exposure (own support) vs AE threshold release."""
import sys, json, math, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fedlloyd
from fedpart import DEV, load_toniot, load_iot23, load_nbaiot, Pool, ROOT
from own_pull import own_matrix
from selfwit import nbaiot_src
from datasets import MIRAI
TG = {"nbaiot": None, "toniot": ["ddos", "scanning"], "iot23": ["DDoS", "PortScan"]}
QS, ROUNDS, CAPR = (0.95, 0.99), 20, 20000


def ae(d):
    return nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(),
                         nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d)).to(DEV)


def local_step(model, X, g):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    if len(X) > CAPR:
        X = X[torch.randperm(len(X), generator=g, device=DEV)[:CAPR]]
    else:
        X = X[torch.randperm(len(X), generator=g, device=DEV)]
    for s in range(0, len(X), 512):
        xb = X[s:s + 512]; loss = ((model(xb) - xb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def scaler(data):
    """Benign-only standardisation (as N-BaIoT's authors); under attack the statistics come from the
    reported training data, so injected rows influence them (federated sums via secure aggregation)."""
    A = torch.cat(data); mu, sd = A.mean(0), A.std(0).clamp_min(1e-3)
    return lambda Z: ((Z - mu) / sd).clamp(-100, 100)


def score(model, X):
    f = getattr(model, "_f", lambda Z: Z)
    with torch.no_grad():
        out = []
        for s in range(0, len(X), 8192):
            Z = f(X[s:s + 8192]); out.append(((model(Z) - Z) ** 2).mean(1))
        return torch.cat(out).cpu().numpy()


def train_fed(data, agg, d, seed):
    f = scaler(data); data = [f(x) for x in data]
    torch.manual_seed(seed); g = torch.Generator(device=DEV).manual_seed(seed)
    glob = ae(d); n = [len(x) for x in data]
    for r in range(ROUNDS):
        states = []
        for X in data:
            m = copy.deepcopy(glob); local_step(m, X, g); states.append(m.state_dict())
        new = {}
        for k in states[0]:
            S = torch.stack([s[k].float() for s in states])
            if agg == "avg":
                w = torch.tensor(n, device=DEV, dtype=torch.float32); w = w / w.sum()
                new[k] = (S * w.view(-1, *[1] * (S.dim() - 1))).sum(0)
            else:
                new[k] = S.median(0).values
        glob.load_state_dict(new)
    glob._f = f
    return glob


def train_local(X, d, seed):
    f = scaler([X]); X = f(X)
    torch.manual_seed(seed); g = torch.Generator(device=DEV).manual_seed(seed); m = ae(d)
    for r in range(ROUNDS):
        local_step(m, X, g)
    m._f = f
    return m


def run(name, loader):
    fed = loader(); pool = Pool(fed); Xte = torch.from_numpy(fed.Xte).to(DEV); d = Xte.shape[1]
    src = nbaiot_src(fed) if name == "nbaiot" else fed.meta["test_src"].astype(int)
    b = fed.yte == fed.benign; n = fed.n; allc = list(range(n))
    tgs = [list(MIRAI)] if name == "nbaiot" else [[fed.class_names.index(c)] for c in TG[name]]
    tr, ho = {}, {}                                  # own benign train / held-out split per client
    for i in allc:
        ix = pool.client(i); ix = ix[pool.y[ix] == fed.benign]
        ix = np.random.RandomState(i).permutation(ix); cut = int(len(ix) * 0.8)
        tr[i], ho[i] = ix[:cut], ix[cut:]

    def ae_metrics(models, hon, rowsets):
        out = {}
        for q in QS:
            fa, ex = [], [[] for _ in rowsets]
            for v in hon:
                if len(ho[v]) < 10:
                    continue
                t = np.quantile(score(models[v], pool.X[ho[v]]), q); st = score(models[v], Xte)
                ob = b & (src == v)
                if ob.sum():
                    fa.append((st[ob] > t).mean())
                for j, rows in enumerate(rowsets):
                    ex[j].append((st[rows] <= t).mean())
            out[f"q{q}"] = {"FA": round(float(np.mean(fa) * 100), 2),
                            "EXP": [round(float(np.mean(e) * 100), 2) for e in ex]}
        return out

    def lw_metrics(C, hon, rowsets):
        own = own_matrix(pool, C, Xte, 512, hon); out = {}
        for vr, o in own.items():
            fa = [(~o[v][b & (src == v)]).mean() for v in hon if (b & (src == v)).sum()]
            out[vr] = {"FA": round(float(np.mean(fa) * 100), 2),
                       "EXP": [round(float(np.mean([o[v][rows].mean() for v in hon]) * 100), 2)
                               for rows in rowsets]}
        return out

    fedlloyd.B = 512
    rowsets = [np.isin(fed.yte, tg) for tg in tgs]
    res = {"targets": [[fed.class_names[c] for c in tg] for tg in tgs], "clean": {}, "attack": []}
    benign_data = [pool.X[tr[i]] for i in allc]
    for agg in ("avg", "med"):
        m = train_fed(benign_data, agg, d, 0)
        res["clean"][f"FAE-{agg}"] = ae_metrics({v: m for v in allc}, allc, rowsets)
    res["clean"]["LAE"] = ae_metrics({v: train_local(benign_data[v], d, v) for v in allc}, allc, rowsets)
    res["clean"]["LW"] = lw_metrics(fedlloyd.lloyd(pool.X), allc, rowsets)
    print(name, "clean", json.dumps(res["clean"]), flush=True)
    json.dump(res, open(ROOT / f"results/explore/fae_{name}.json", "w"), indent=1)
    sizes = sorted({1, math.ceil(n / 4), n // 2})
    for j, tg in enumerate(tgs):
        pool_t = np.nonzero(np.isin(pool.y, tg))[0]
        for bsz in sizes:
            for seed in range(3):
                byz = sorted(np.random.RandomState(1000 * seed + bsz).choice(n, bsz, replace=False).tolist())
                hon = [i for i in allc if i not in byz]
                for ratio in (0.1, 1.0):
                    g = np.random.RandomState(seed); data = list(benign_data); inj = []
                    for i in byz:
                        k = max(1, int(round(ratio * len(tr[i]))))
                        sel = pool_t[g.randint(0, len(pool_t), k)]
                        data[i] = torch.cat([benign_data[i], pool.X[sel]]); inj.append(pool.X[sel])
                    row = {"target": j, "b": bsz, "seed": seed, "ratio": ratio, "byz": byz}
                    for agg in ("avg", "med"):
                        m = train_fed(data, agg, d, seed)
                        r = ae_metrics({v: m for v in hon}, hon, [rowsets[j]])
                        row[f"FAE-{agg}"] = {q: x["EXP"][0] for q, x in r.items()}
                    r = lw_metrics(fedlloyd.lloyd(torch.cat([pool.X] + inj)), hon, [rowsets[j]])
                    row["LW"] = {vr: x["EXP"][0] for vr, x in r.items()}
                    res["attack"].append(row); print(name, json.dumps(row), flush=True)
                    json.dump(res, open(ROOT / f"results/explore/fae_{name}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    for name, loader in [(a, l) for a, l in [("nbaiot", load_nbaiot), ("toniot", load_toniot),
                                             ("iot23", load_iot23)] if a in sys.argv[1:]]:
        run(name, loader)
