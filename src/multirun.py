"""Train MANY federated runs at once on one GPU.

Why: one 9-client run of a 23k-parameter MLP cannot occupy the GPU (measured
16% utilisation, 75 s/run, kernel-launch bound). Here every client of every
run is one slice of a single batched model, and the feature matrix is stored
ONCE: each (run, client) holds an index array into a shared pool, so attacks
that change data (flip, inject, volume) only change indices and labels.

Semantics per client are identical to runner.run (same Adam, same class
weighting from the labels actually trained on, same per-client shuffle seeds);
aggregation, update-space attacks and evaluation are applied per run.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from flcore import (CFG, DEVICE, aggregate, class_weights, evaluate, forward_flat, init_flat,
                    shapes, update_attack)

UPDATE_ATTACKS = {"signflip", "gauss", "ipm", "alie", "scale"}


class Pool:
    def __init__(self, fed):
        self.fed = fed
        self.off = np.cumsum([0] + [len(y) for y in fed.ys])
        self.X = torch.from_numpy(np.concatenate(fed.Xs)).to(DEVICE)
        self.y = np.concatenate(fed.ys)

    def client(self, i):
        return np.arange(self.off[i], self.off[i + 1])

    def cells(self, B, Xte):
        """k-corroboration gate partition: k-means(B) on pooled TRAINING INPUTS only (label-free,
        so label flipping cannot move it; seed fixed). Returns cell ids of pool rows and test rows."""
        if not hasattr(self, "_cells"):
            self._cells = {}
        if B not in self._cells:
            # GPU k-means differs across devices, so the centroids are computed once and shipped
            # as a cache file (cells_<tag>_B<B>.npy); every machine then uses the same partition.
            from datasets import CACHE
            fp = CACHE / f"cells_{self.fed.meta['tag']}_B{B}.npy"
            if fp.exists():
                C = torch.from_numpy(np.load(fp)).to(self.X.device)
            else:
                g = torch.Generator(device=self.X.device).manual_seed(2027)
                sub = self.X[torch.randperm(len(self.X), generator=g, device=self.X.device)[:150000]]
                C = sub[torch.randperm(len(sub), generator=g, device=self.X.device)[:B]].clone()
                for _ in range(30):
                    a = torch.cat([torch.cdist(sub[s:s + 16384], C).argmin(1)
                                   for s in range(0, len(sub), 16384)])
                    S = torch.zeros_like(C).index_add_(0, a, sub)
                    n = torch.bincount(a, minlength=B).float()
                    m = n > 0
                    C[m] = S[m] / n[m, None]
                np.save(fp, C.cpu().numpy())
            Cd = C.double()
            if not hasattr(self, "_cent"):
                self._cent = {}
            self._cent[B] = Cd
            asg = lambda X: torch.cat([torch.cdist(X[s:s + 16384].double(), Cd).argmin(1)
                                       for s in range(0, len(X), 16384)])
            self._cells[B] = (asg(self.X).cpu().numpy(), asg(Xte).cpu().numpy())
        return self._cells[B]


def gate_metrics(pool, idx, ys, byz, pred, yt, src, targets, C, benign, B, ks, m=5):
    """k-corroboration gate on top of a trained detector. N(b) = number of clients with >= m
    REPORTED-benign training rows in cell b. A benign prediction on a test row in a cell with
    N < k becomes an alert (index C). Modes: 'nat' = Byzantine clients flag their own reported
    data; 'worst' = Byzantine clients flag every cell (strongest flag-lying adversary)."""
    ctr, cte = pool._cells[B]
    flags = np.zeros((len(idx), B), dtype=bool)
    for i, (ix, y) in enumerate(zip(idx, ys)):
        flags[i] = np.bincount(ctr[ix[y == benign]], minlength=B) >= m
    out = {}
    for mode in ("nat", "worst"):
        f = flags.copy()
        if mode == "worst":
            for i in byz:
                f[i] = True
        N = f.sum(0)[cte]
        for k in ks:
            g = pred.copy()
            g[(g == benign) & (N < k)] = C
            sup = np.bincount(yt, minlength=C)
            tb = [float(((yt == c) & (g == benign)).sum() / sup[c] * 100) if sup[c] else None
                  for c in range(C)]
            tm = np.isin(yt, targets)
            r = {"asr": float((g[tm] == benign).mean() * 100) if tm.any() else None,
                 "fpr": float((g[yt == benign] != benign).mean() * 100), "to_benign": tb}
            if src is not None:
                r["per_src_tb"] = {f"{c}:{i}": float((g[(yt == c) & (src == i)] == benign).mean() * 100)
                                   for c in targets for i in np.unique(src[yt == c])}
            out[f"B{B}k{k}{mode}"] = r
    return out


def stacked_src(yte, n):
    """Source client of each pooled test row when the loader did not store it (N-BaIoT): rows are
    stacked per client with classes ascending (datasets._split_scale), so a new client starts
    where the label decreases."""
    src = np.zeros(len(yte), np.int64)
    src[1:] = np.cumsum(yte[1:] < yte[:-1])
    assert src[-1] == n - 1
    return src


def lw_metrics(pool, idx, ys, byz, pred, yt, src, targets, benign, B, Xte, m=5, contam=()):
    """Local-witness rule (RESEARCH_PLAN, E-SW): at victim v a benign prediction stands only if x's
    cell holds >= m of v's OWN benign training rows ('cell'), and for 'r100' also dist(x, centroid)
    <= the max distance of those rows. Byzantine clients are never victims. FPR: benign test rows at
    their own (honest) client. ASR: target rows released, averaged over honest victims.
    contam: honest clients whose own benign data was contaminated (their ys carry the injected
    rows); they are victims too, reported separately as asr_contam (asr = clean victims)."""
    ctr, cte = pool._cells[B]; Cd = pool._cent[B]
    dist = lambda X, a: torch.cat([(X[s:s + 65536].double() - Cd[torch.from_numpy(a[s:s + 65536]).to(X.device)])
                                   .norm(dim=1) for s in range(0, len(X), 65536)]).cpu().numpy()
    dtr, dte = dist(pool.X, ctr), dist(Xte, cte)
    hon = [i for i in range(len(idx)) if i not in set(byz) and i not in set(contam)]
    own = {"cell": {}, "r100": {}}
    for i in hon + list(contam):
        ix = idx[i][ys[i] == benign]; c, d = ctr[ix], dtr[ix]
        cnt = np.bincount(c, minlength=B); r = np.full(B, -1.0)
        np.maximum.at(r, c, d); r[cnt < m] = -1.0
        own["cell"][i] = (cnt >= m)[cte]; own["r100"][i] = r[cte] >= dte
    pb = pred == benign; tm = np.isin(yt, targets); out = {}
    for v, o in own.items():
        bm = (yt == benign) & np.isin(src, hon)
        stand = np.zeros(len(yt), bool)
        for i in hon:
            r_ = src == i; stand[r_] = pb[r_] & o[i][r_]
        res = {"fpr": float((~stand[bm]).mean() * 100),
               "asr": float(np.mean([(pb & o[i])[tm].mean() for i in hon]) * 100) if tm.any() else None}
        bt = tm & np.isin(src, list(byz))
        if bt.any():
            res["asr_byzsrc"] = float(np.mean([(pb & o[i])[bt].mean() for i in hon]) * 100)
        if contam:
            res["asr_contam"] = float(np.mean([(pb & o[i])[tm].mean() for i in contam]) * 100)
        out[f"B{B}{v}"] = res
    return out


def client_views(pool, spec, rng):
    """Index-based equivalent of runner.build_client_data."""
    fed = pool.fed
    idx = [pool.client(i) for i in range(fed.n)]
    ys = [pool.y[ix].copy() for ix in idx]
    att = spec.get("attack", "none")
    byz = set(spec.get("byz", []))
    tg = list(spec.get("targets", []))
    b = fed.benign
    q = spec.get("byz_target_frac")
    if q is not None and q < 1:
        for i in byz:
            ix_t = np.nonzero(np.isin(ys[i], tg))[0]
            drop = rng.choice(ix_t, size=len(ix_t) - int(round(q * len(ix_t))),
                              replace=False) if len(ix_t) else ix_t
            keep = np.setdiff1d(np.arange(len(ys[i])), drop)
            idx[i], ys[i] = idx[i][keep], ys[i][keep]
    if att in ("flip", "flip_scale", "flip_fakeown", "flip_forgerow", "flip_forgepos", "flip_forgeall"):
        for i in byz:
            ys[i][np.isin(ys[i], tg)] = b
    elif att == "flip_partial":
        # Adaptive vs OSA: keep a fraction kappa of target rows correctly
        # labelled so the client still looks like an owner; flip the rest.
        kappa = float(spec["kappa"])
        for i in byz:
            ix_t = np.nonzero(np.isin(ys[i], tg))[0]
            n_keep = int(round(kappa * len(ix_t)))
            keep = rng.choice(ix_t, size=n_keep, replace=False) if n_keep else ix_t[:0]
            ys[i][np.setdiff1d(ix_t, keep)] = b
    elif att == "relabel_rel":
        rel = list(spec["rel_classes"])
        for i in byz:
            ys[i][np.isin(ys[i], tg + rel)] = b
    elif att == "inject":
        k = int(spec["inject_n"])
        for i in byz:
            pool_ix = np.concatenate(
                [pool.client(j)[np.isin(fed.ys[j], tg)]
                 for j in range(fed.n) if j != i and j not in byz])
            take = rng.choice(len(pool_ix), size=min(k, len(pool_ix)), replace=False)
            ys[i][np.isin(ys[i], tg)] = b
            idx[i] = np.concatenate([idx[i], pool_ix[take]])
            ys[i] = np.concatenate([ys[i], np.full(len(take), b, dtype=ys[i].dtype)])
    elif att == "none" or att in UPDATE_ATTACKS:
        pass
    else:
        raise ValueError(att)
    return idx, ys


def run_many(fed, specs, cfg=None, test=None, pool=None):
    """All specs must share ROUNDS, LOCAL_EPOCHS, LR, HIDDEN (checked)."""
    cfg = dict(CFG, **(cfg or {}))
    R, n, d, C, H = len(specs), fed.n, fed.d, fed.C, cfg["HIDDEN"]
    for k in ("ROUNDS", "LOCAL_EPOCHS", "LR"):
        vals = {s.get(k.lower(), cfg[k]) for s in specs}
        assert len(vals) == 1, f"batch mixes {k}: {vals}"
        cfg[k] = vals.pop()
    t0 = time.time()
    pool = pool or Pool(fed)
    M = R * n
    IDX, YS, CW = [], [], []
    for s in specs:
        rng = np.random.RandomState(int(s["seed"]) + 31337)
        idx, ys = client_views(pool, s, rng)
        w = s.get("weighting", cfg["WEIGHTING"])
        for ix, y in zip(idx, ys):
            IDX.append(ix)
            YS.append(y)
            CW.append(class_weights(torch.from_numpy(y).to(DEVICE), C, w, cfg["W_CAP"]))
    lengths = torch.tensor([len(y) for y in YS], device=DEVICE)
    L = int(lengths.max())
    IX = torch.zeros(M, L, dtype=torch.long, device=DEVICE)
    Y = torch.zeros(M, L, dtype=torch.long, device=DEVICE)
    for m_, (ix, y) in enumerate(zip(IDX, YS)):
        IX[m_, :len(y)] = torch.from_numpy(ix).to(DEVICE)
        Y[m_, :len(y)] = torch.from_numpy(y).to(DEVICE)
    CWt = torch.stack(CW)
    gens = [torch.Generator().manual_seed(int(s["seed"]) * 1000 + 7919 * i)
            for s in specs for i in range(n)]
    flats = torch.stack([init_flat(d, C, H, int(s["seed"])) for s in specs])  # (R,p)
    sh = shapes(d, C, H)
    ctxs = []
    for s in specs:
        byz = sorted(s.get("byz", []))
        ctxs.append({"sizes": None, "f": int(s.get("f", len(byz))), "prev": None,
                     "d": d, "C": C, "hidden": H, "byz": byz,
                     "targets": list(s.get("targets", [])), "benign": fed.benign})
    for r_, s in enumerate(specs):
        ctxs[r_]["sizes"] = [len(YS[r_ * n + i]) for i in range(n)]
    arngs = [np.random.RandomState(int(s["seed"]) + 4242) for s in specs]
    sel_logs = [[] for _ in specs]
    bs, lr = cfg["BATCH_SIZE"], cfg["LR"]
    b1c, b2c, eps = 0.9, 0.999, 1e-8
    steps = (lengths + bs - 1) // bs
    ar = torch.arange(L, device=DEVICE)
    for rnd in range(cfg["ROUNDS"]):
        # unpack per-run globals into per-client params
        params, off = [], 0
        for s_ in sh:
            k = int(np.prod(s_))
            blk = flats[:, off:off + k]                      # (R,k)
            if len(s_) == 2:
                p = blk.view(R, *s_).transpose(1, 2)         # (R,in,out)
                p = p.repeat_interleave(n, 0)
            else:
                p = blk.view(R, 1, -1).repeat_interleave(n, 0)
            params.append(p.contiguous().clone().requires_grad_(True))
            off += k
        mom = [torch.zeros_like(p) for p in params]
        vel = [torch.zeros_like(p) for p in params]
        t = torch.zeros(M, device=DEVICE)
        P = torch.full((M, L), L - 1, dtype=torch.long)
        for m_ in range(M):
            k = int(lengths[m_])
            P[m_, :k] = torch.randperm(k, generator=gens[m_])
        P = P.to(DEVICE)
        for _ in range(cfg["LOCAL_EPOCHS"]):
            for st in range(int(steps.max())):
                lo, hi = st * bs, min((st + 1) * bs, L)
                if lo >= hi:
                    break
                pi = P[:, lo:hi]
                valid = (pi < lengths[:, None]) & (ar[lo:hi][None, :] < lengths[:, None])
                active = (st < steps)[:, None]
                valid = valid & active
                safe = torch.where(valid, pi, torch.zeros_like(pi))
                rows = torch.gather(IX, 1, safe)              # pool rows
                xb = pool.X[rows]                             # (M,B,d)
                yb = torch.gather(Y, 1, safe)
                W1, bb1, W2, bb2, W3, bb3 = params
                h = torch.relu(torch.baddbmm(bb1, xb, W1))
                h = torch.relu(torch.baddbmm(bb2, h, W2))
                lg = torch.baddbmm(bb3, h, W3)
                ce = F.cross_entropy(lg.reshape(-1, C), yb.reshape(-1),
                                     reduction="none").view(yb.shape)
                w = torch.gather(CWt, 1, yb) * valid
                loss = ((ce * w).sum(1) / w.sum(1).clamp(min=1e-12)).sum()
                for p in params:
                    p.grad = None
                loss.backward()
                with torch.no_grad():
                    act = active.view(-1).float()
                    t += act
                    c1 = 1 - torch.pow(b1c, t.clamp(min=1))
                    c2 = 1 - torch.pow(b2c, t.clamp(min=1))
                    for i_, p in enumerate(params):
                        g = p.grad
                        a = act.view(-1, 1, 1)
                        mn = mom[i_] * b1c + (1 - b1c) * g
                        vn = vel[i_] * b2c + (1 - b2c) * g * g
                        mom[i_] = torch.where(a > 0, mn, mom[i_])
                        vel[i_] = torch.where(a > 0, vn, vel[i_])
                        upd = (mom[i_] / c1.view(-1, 1, 1)) / \
                              ((vel[i_] / c2.view(-1, 1, 1)).sqrt() + eps)
                        p.add_(-lr * upd * a)
        with torch.no_grad():
            cols = [(p.transpose(1, 2) if len(s_) == 2 else p).reshape(M, -1)
                    for p, s_ in zip(params, sh)]
            new = torch.cat(cols, 1).view(R, n, -1)
            for r_, s in enumerate(specs):
                D = new[r_] - flats[r_][None, :]
                att = s.get("attack", "none")
                byz = sorted(s.get("byz", []))
                if att in UPDATE_ATTACKS:
                    D = update_attack(D, byz, att, arngs[r_], s)
                if att == "flip_scale":
                    D = update_attack(D, byz, "scale", arngs[r_], s)
                if att == "flip_fakeown" and byz:
                    # Adaptive vs OSA (omniscient): overwrite the Byzantine
                    # clients' target-class bias deltas with the honest
                    # clients' per-class median of positive deltas, so OSA's
                    # ownership test admits them to the target rows.
                    from flcore import slices as _sl
                    ba = _sl(d, C, H)[5][0]
                    hon = [i for i in range(n) if i not in byz]
                    D = D.clone()
                    for c in s.get("targets", []):
                        hv = D[hon, ba + c]
                        pos = hv[hv > 0]
                        val = pos.median() if len(pos) else hv.abs().median()
                        D[byz, ba + c] = val
                if att in ("flip_forgerow", "flip_forgepos") and byz:
                    # Adaptive vs certified OSA (omniscient). forgerow: the
                    # Byzantine target rows+biases are replaced by the honest
                    # owners' median (admitted; poison only via the rest of
                    # the model). forgepos: keep the poisoned target rows but
                    # pass the certificate at minimal cost -- bias set to the
                    # honest positive median and the smallest-|.| row entry
                    # set to +1e-6.
                    from flcore import slices as _sl
                    _s = _sl(d, C, H)
                    wa, ba, h2 = _s[4][0], _s[5][0], H[1]
                    hon = [i for i in range(n) if i not in byz]
                    D = D.clone()
                    for c in s.get("targets", []):
                        cols = list(range(wa + c * h2, wa + (c + 1) * h2)) + [ba + c]
                        Dh = D[hon][:, cols]
                        ho = [j for j in range(len(hon)) if (Dh[j] > 0).any()]
                        if att == "flip_forgerow":
                            ref = Dh[ho].median(0).values if ho else Dh.median(0).values
                            for i in byz:
                                D[i, cols] = ref
                        else:
                            hv = D[hon, ba + c]
                            pos = hv[hv > 0]
                            D[byz, ba + c] = pos.median() if len(pos) else hv.abs().median()
                            for i in byz:
                                row = D[i, wa + c * h2: wa + (c + 1) * h2]
                                row[row.abs().argmin()] = 1e-6
                if att == "flip_forgeall" and byz:
                    # Adaptive vs evidence weighting: claim EVERY class at
                    # minimal cost (one +1e-6 row entry and a positive bias per
                    # class), keeping the poisoned update otherwise.
                    from flcore import slices as _sl
                    _s = _sl(d, C, H)
                    wa, ba, h2 = _s[4][0], _s[5][0], H[1]
                    D = D.clone()
                    for i in byz:
                        for c in range(C):
                            row = D[i, wa + c * h2: wa + (c + 1) * h2]
                            if not (row > 0).any():
                                row[row.abs().argmin()] = 1e-6
                            if D[i, ba + c] <= 0:
                                D[i, ba + c] = 1e-6
                agg, info = aggregate(D, s["method"], ctxs[r_])
                flats[r_] = flats[r_] + agg
                ctxs[r_]["prev"] = agg.detach().clone()
                if "kept" in info or "selected" in info:
                    sel_logs[r_].append(info.get("kept", info.get("selected")))
                if "owners_est" in info and rnd in (0, 4, 14, cfg["ROUNDS"] - 1):
                    sel_logs[r_].append({"round": rnd + 1, "owners_est": info["owners_est"]})
    if test is None:
        test = (torch.from_numpy(fed.Xte).to(DEVICE), torch.from_numpy(fed.yte).to(DEVICE))
    rows = []
    el = time.time() - t0
    for r_, s in enumerate(specs):
        m = evaluate(flats[r_], test[0], test[1], d, C, H, fed.benign, s.get("targets", [1]))
        row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
               for k, v in s.items() if not k.startswith("_")}
        row.update({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in m.items()})
        src = fed.meta.get("test_src") if fed.meta else None
        if src is not None and s.get("per_src"):
            # per source client: predicted-class distribution of its target-class test rows
            pred = forward_flat(flats[r_], test[0], d, C, H).argmax(1).cpu().numpy()
            yt = test[1].cpu().numpy()
            per = {}
            for c in s.get("targets", []):
                for i in np.unique(src[yt == c]):
                    p = pred[(yt == c) & (src == i)]
                    per[f"{c}:{i}"] = np.bincount(p, minlength=C).tolist()
            row["per_src"] = json.dumps(per)
        if s.get("gate"):
            pred = torch.cat([forward_flat(flats[r_], test[0][i:i + 65536], d, C, H).argmax(1)
                              for i in range(0, test[0].shape[0], 65536)]).cpu().numpy()
            yt = test[1].cpu().numpy()
            gm = {}
            for B, ks in s["gate"]:
                pool.cells(B, test[0])
                gm.update(gate_metrics(pool, IDX[r_ * n:(r_ + 1) * n], YS[r_ * n:(r_ + 1) * n],
                                       s.get("byz", []), pred, yt, src if s.get("per_src") else None,
                                       s.get("targets", [1]), C, fed.benign, B, ks))
            row["gate"] = json.dumps(gm)
        if s.get("lw"):
            pred = torch.cat([forward_flat(flats[r_], test[0][i:i + 65536], d, C, H).argmax(1)
                              for i in range(0, test[0].shape[0], 65536)]).cpu().numpy()
            yt = test[1].cpu().numpy()
            srcl = src if src is not None else stacked_src(yt, fed.n)
            lm = {}
            for B in s["lw"]:
                pool.cells(B, test[0])
                ct = s.get("byz", []) if s.get("contam") else []
                lm.update(lw_metrics(pool, IDX[r_ * n:(r_ + 1) * n], YS[r_ * n:(r_ + 1) * n],
                                     [] if ct else s.get("byz", []), pred, yt, srcl,
                                     s.get("targets", [1]), fed.benign, B, test[0], contam=ct))
            row["lw"] = json.dumps(lm)
        row["elapsed_s"] = round(el / R, 2)
        row["sel_log"] = json.dumps(sel_logs[r_])     # always present: fixed CSV schema
        rows.append(row)
    return rows
