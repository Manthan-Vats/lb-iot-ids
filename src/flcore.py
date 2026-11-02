"""v2 federated-learning core: data container, batched local training,
aggregation rules, attacks, evaluation.

Design rules for this file
  * Every client follows the protocol on the labels it actually trains on.
    In particular class weights are computed from the (possibly poisoned)
    training labels of that client -- v1 computed them from clean labels,
    which no protocol-following client could do.
  * Nothing reads ground truth except the evaluation and the attack itself.
  * Determinism: every random draw comes from a generator seeded from
    (run seed, purpose, client), so paired comparisons are exact.

The batched trainer is ported from v1 (rebuild/fedguard_v13_core.py,
BatchedClientTrainer), whose equivalence to a sequential per-client Adam loop
was verified there; v2 only changes how class weights are supplied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    "ROUNDS": 30,
    "LOCAL_EPOCHS": 2,
    "BATCH_SIZE": 512,
    "LR": 1e-3,
    "HIDDEN": (128, 64),
    "WEIGHTING": "balanced",   # 'balanced' | 'none'
    "W_CAP": 50.0,
}


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────── data ────────────────────────────────────

@dataclass
class Fed:
    """A federation: per-client training data (numpy) + pooled test set."""
    name: str
    Xs: list            # list of (n_i, d) float32 arrays, already scaled
    ys: list            # list of (n_i,) int64 arrays (clean labels)
    Xte: np.ndarray
    yte: np.ndarray
    class_names: list
    client_names: list
    benign: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.Xs)

    @property
    def C(self):
        return len(self.class_names)

    @property
    def d(self):
        return self.Xte.shape[1]

    def counts(self) -> np.ndarray:
        """(n, C) clean per-client class counts."""
        return np.stack([np.bincount(y, minlength=self.C) for y in self.ys])

    def owners(self, classes) -> list:
        cnt = self.counts()[:, list(classes)].sum(1)
        return [i for i in range(self.n) if cnt[i] > 0]


def class_weights(y: torch.Tensor, C: int, mode: str, cap: float) -> torch.Tensor:
    """Per-client inverse-frequency weights from the labels the client trains on."""
    if mode == "none":
        return torch.ones(C, device=y.device)
    counts = torch.bincount(y, minlength=C).float()
    present = (counts > 0).sum().clamp(min=1)
    w = torch.where(counts > 0, counts.sum() / (counts.clamp(min=1) * present),
                    torch.zeros_like(counts))
    return w.clamp(max=cap)


class Padded:
    """GPU padded view of a list of per-client (X, y) arrays."""

    def __init__(self, Xs, ys, C, cfg, device=DEVICE):
        n = len(Xs)
        self.n, self.C = n, C
        self.lengths = torch.tensor([len(y) for y in ys], device=device)
        L = int(self.lengths.max())
        d = Xs[0].shape[1]
        self.X = torch.zeros(n, L, d, device=device)
        self.y = torch.zeros(n, L, dtype=torch.long, device=device)
        self.cw = torch.zeros(n, C, device=device)
        for i, (X, y) in enumerate(zip(Xs, ys)):
            k = len(y)
            self.X[i, :k] = torch.from_numpy(np.ascontiguousarray(X)).to(device)
            yt = torch.from_numpy(np.ascontiguousarray(y)).to(device)
            self.y[i, :k] = yt
            self.cw[i] = class_weights(yt, C, cfg["WEIGHTING"], cfg["W_CAP"])
        self.max_len = L

    def perms(self, gens):
        P = torch.full((self.n, self.max_len), self.max_len - 1, dtype=torch.long)
        for i in range(self.n):
            k = int(self.lengths[i])
            P[i, :k] = torch.randperm(k, generator=gens[i])
        return P.to(self.X.device)


# ─────────────────────────────── model ───────────────────────────────────

def shapes(d, C, hidden):
    h1, h2 = hidden
    return [(h1, d), (h1,), (h2, h1), (h2,), (C, h2), (C,)]


def slices(d, C, hidden):
    out, off = [], 0
    for s in shapes(d, C, hidden):
        k = int(np.prod(s))
        out.append((off, off + k))
        off += k
    return out


def init_flat(d, C, hidden, seed):
    seed_all(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(d, hidden[0]), torch.nn.ReLU(),
        torch.nn.Linear(hidden[0], hidden[1]), torch.nn.ReLU(),
        torch.nn.Linear(hidden[1], C)).to(DEVICE)
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def forward_flat(flat, X, d, C, hidden):
    W1, b1, W2, b2, W3, b3 = [flat[a:b].view(s) for (a, b), s in
                              zip(slices(d, C, hidden), shapes(d, C, hidden))]
    h = torch.relu(X @ W1.t() + b1)
    h = torch.relu(h @ W2.t() + b2)
    return h @ W3.t() + b3


class Trainer:
    """All clients trained at once as one batched MLP (ported from v1)."""

    def __init__(self, n, d, C, hidden):
        self.n, self.d, self.C = n, d, C
        self.h = hidden
        self.sh = shapes(d, C, hidden)

    def _unpack(self, flat):
        out, off = [], 0
        for s in self.sh:
            k = int(np.prod(s))
            blk = flat[off:off + k]
            if len(s) == 2:
                p = blk.view(*s).t().contiguous().unsqueeze(0).repeat(self.n, 1, 1)
            else:
                p = blk.view(1, 1, -1).repeat(self.n, 1, 1)
            out.append(p.clone().requires_grad_(True))
            off += k
        return out

    def _repack(self, params):
        cols = []
        for p, s in zip(params, self.sh):
            cols.append((p.transpose(1, 2) if len(s) == 2 else p).reshape(self.n, -1))
        return torch.cat(cols, 1)

    def round(self, flat, pad: Padded, cfg, perms):
        params = self._unpack(flat)
        m = [torch.zeros_like(p) for p in params]
        v = [torch.zeros_like(p) for p in params]
        lr, bs = cfg["LR"], cfg["BATCH_SIZE"]
        b1c, b2c, eps = 0.9, 0.999, 1e-8
        t = torch.zeros(self.n, dtype=flat.dtype, device=flat.device)
        L = pad.max_len
        steps = (pad.lengths + bs - 1) // bs
        ar = torch.arange(L, device=flat.device)
        for _ in range(cfg["LOCAL_EPOCHS"]):
            for s in range(int(steps.max())):
                lo, hi = s * bs, min((s + 1) * bs, L)
                if lo >= hi:
                    break
                idx = perms[:, lo:hi]
                valid = (idx < pad.lengths.unsqueeze(1)) & \
                        (ar[lo:hi].unsqueeze(0) < pad.lengths.unsqueeze(1))
                active = (s < steps).unsqueeze(1)
                valid = valid & active
                if not bool(valid.any()):
                    continue
                safe = torch.where(valid, idx, torch.zeros_like(idx))
                xb = torch.gather(pad.X, 1, safe.unsqueeze(-1).expand(-1, -1, self.d))
                yb = torch.gather(pad.y, 1, safe)
                W1, b1, W2, b2, W3, b3 = params
                h = torch.relu(torch.baddbmm(b1, xb, W1))
                h = torch.relu(torch.baddbmm(b2, h, W2))
                logits = torch.baddbmm(b3, h, W3)
                ce = F.cross_entropy(logits.reshape(-1, self.C), yb.reshape(-1),
                                     reduction="none").view(yb.shape)
                w = torch.gather(pad.cw, 1, yb) * valid
                loss = ((ce * w).sum(1) / w.sum(1).clamp(min=1e-12)).sum()
                for p in params:
                    p.grad = None
                loss.backward()
                with torch.no_grad():
                    act = active.view(-1).to(flat.dtype)
                    t += act
                    c1 = 1 - torch.pow(b1c, t.clamp(min=1))
                    c2 = 1 - torch.pow(b2c, t.clamp(min=1))
                    for i, p in enumerate(params):
                        g = p.grad
                        sh = (-1,) + (1,) * (p.dim() - 1)
                        a = act.view(sh)
                        mn = m[i] * b1c + (1 - b1c) * g
                        vn = v[i] * b2c + (1 - b2c) * g * g
                        m[i] = torch.where(a > 0, mn, m[i])
                        v[i] = torch.where(a > 0, vn, v[i])
                        upd = (m[i] / c1.view(sh)) / ((v[i] / c2.view(sh)).sqrt() + eps)
                        p.add_(-lr * upd * a)
        with torch.no_grad():
            return self._repack(params).detach()


# ───────────────────────────── evaluation ────────────────────────────────

@torch.no_grad()
def evaluate(flat, Xte, yte, d, C, hidden, benign, targets):
    pred = torch.cat([forward_flat(flat, Xte[i:i + 65536], d, C, hidden).argmax(1)
                      for i in range(0, Xte.shape[0], 65536)])
    cm = torch.zeros(C * C, dtype=torch.long, device=Xte.device)
    cm.scatter_add_(0, yte * C + pred, torch.ones_like(pred))
    cm = cm.view(C, C).cpu().numpy()
    sup = cm.sum(1)
    rec = np.where(sup > 0, np.diag(cm) / np.maximum(sup, 1), np.nan)
    prd = cm.sum(0)
    prec = np.where(prd > 0, np.diag(cm) / np.maximum(prd, 1), 0.0)
    f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-12), 0.0)
    pres = sup > 0
    tg = list(targets)
    t_tot = cm[tg].sum()
    out = {
        "acc": float(np.trace(cm) / cm.sum() * 100),
        "macro_f1": float(np.nanmean(np.where(pres, f1, np.nan)) * 100),
        "bal_acc": float(np.nanmean(rec[pres]) * 100),
        "asr": float(cm[tg, benign].sum() / t_tot * 100) if t_tot else float("nan"),
        "fpr": float((sup[benign] - cm[benign, benign]) / max(sup[benign], 1) * 100),
        "recall": [None if np.isnan(r) else float(r * 100) for r in rec],
        # fraction of each class predicted benign (the attack's view per class)
        "to_benign": [float(cm[c, benign] / sup[c] * 100) if sup[c] else None
                      for c in range(C)],
    }
    return out


# ───────────────────────────── aggregation ───────────────────────────────

def _gm(D, w=None, iters=100, tol=1e-6, nu=1e-6):
    """Weighted geometric median (smoothed Weiszfeld)."""
    n = D.shape[0]
    w = torch.ones(n, device=D.device, dtype=D.dtype) if w is None else w
    z = (w[:, None] * D).sum(0) / w.sum()
    for _ in range(iters):
        dist = (D - z).norm(dim=1).clamp(min=nu)
        beta = w / dist
        zn = (beta[:, None] * D).sum(0) / beta.sum()
        if (zn - z).norm() <= tol * max(z.norm().item(), 1.0):
            z = zn
            break
        z = zn
    return z


def _krum_scores(D, f):
    n = D.shape[0]
    dist = torch.cdist(D, D) ** 2
    k = max(n - f - 2, 1)
    srt, _ = torch.sort(dist, dim=1)
    return srt[:, 1:k + 1].sum(1)        # skip self-distance


class LFighter:
    """Port of Jebreel et al. (Neural Networks 2024), reference code LFD.aggregate.

    dw = global - local last-layer weights; pick the two output classes with
    the largest summed row norm + |bias delta|; k-means (k=2) on those rows;
    keep the cluster with the HIGHER weighted dissimilarity (honest clients
    are the more diverse group); unweighted mean of the kept clients.
    """

    @staticmethod
    def select(D, d, C, hidden):
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity
        sl = slices(d, C, hidden)
        (a, b), (c, e) = sl[4], sl[5]
        W = (-D[:, a:b]).view(D.shape[0], C, hidden[1]).cpu().numpy()  # global-local
        B = (-D[:, c:e]).cpu().numpy()
        mem = np.linalg.norm(W, axis=-1).sum(0) + np.abs(B).sum(0)
        top2 = mem.argsort()[-2:]
        data = W[:, top2].reshape(D.shape[0], -1)
        lab = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data).labels_
        cl = [data[lab == 0], data[lab == 1]]
        if min(len(cl[0]), len(cl[1])) == 0:
            return np.ones(D.shape[0], dtype=bool), top2
        m = len(data)
        ds = []
        for c_ in cl:
            k = len(c_)
            if k == 1:
                ds.append(k / m * 1.0)       # single point: min-cos undefined
                continue
            cs = cosine_similarity(c_) - np.eye(k)
            ds.append(k / m * (1 - np.mean(np.min(cs, axis=1))))
        good = 0 if ds[0] >= ds[1] else 1
        # reference: good_cl=0; if cs0 < cs1: good_cl=1  (same rule)
        return lab == good, top2


def weighted_median(X, w):
    """Coordinate-wise weighted median of rows of X (n, p) with weights w (n,)."""
    v, idx = X.sort(0)
    cw = w[idx].cumsum(0)
    k = (cw < 0.5 * w.sum()).sum(0, keepdim=True)
    return v.gather(0, k.clamp(max=X.shape[0] - 1)).squeeze(0)


def aggregate(D, rule, ctx):
    """D: (n, p) client deltas. ctx carries sizes, f, prev, server_delta, owners..."""
    n = D.shape[0]
    if rule == "mean":
        return D.mean(0), {}
    if rule == "wmean":
        w = torch.as_tensor(ctx["sizes"], device=D.device, dtype=D.dtype)
        return (w[:, None] * D).sum(0) / w.sum(), {}
    if rule == "median":
        return D.median(0).values, {}
    if rule == "trmean":
        b = ctx.get("trim", max(1, int(math.floor(0.2 * n))))
        b = min(b, (n - 1) // 2)
        s, _ = torch.sort(D, 0)
        return s[b:n - b].mean(0), {}
    if rule == "krum":
        sc = _krum_scores(D, ctx["f"])
        return D[int(torch.argmin(sc))], {"selected": [int(torch.argmin(sc))]}
    if rule == "mkrum":
        sc = _krum_scores(D, ctx["f"])
        m = max(n - ctx["f"], 1)
        idx = torch.argsort(sc)[:m]
        return D[idx].mean(0), {"selected": idx.tolist()}
    if rule == "gm":
        return _gm(D), {}
    if rule == "cclip":
        # Karimireddy et al. 2021: centre carried across rounds (prev aggregate),
        # tau = median norm of (D - centre), L=1 iteration (reference default 1).
        v = ctx.get("prev")
        v = torch.zeros(D.shape[1], device=D.device) if v is None else v
        diff = D - v
        tau = diff.norm(dim=1).median()
        scale = torch.clamp(tau / diff.norm(dim=1).clamp(min=1e-12), max=1.0)
        return v + (diff * scale[:, None]).mean(0), {}
    if rule == "fltrust":
        g0 = ctx["server_delta"]
        cos = F.cosine_similarity(D, g0[None, :], dim=1)
        ts = torch.relu(cos)
        if ts.sum() <= 0:
            return torch.zeros_like(g0), {"ts": ts.tolist()}
        Dn = D * (g0.norm() / D.norm(dim=1).clamp(min=1e-12))[:, None]
        return (ts[:, None] * Dn).sum(0) / ts.sum(), {"ts": ts.tolist()}
    if rule.startswith("osa"):
        # Ownership-Scoped Aggregation (proposed, v2). rule = 'osa_<inner>'.
        # Class c's output row and bias are aggregated only over clients whose
        # bias delta for c is positive (their local data pushed logit c up,
        # i.e. they trained on class-c labels). Everything else -- hidden
        # layers and rows with no estimated owner -- uses <inner> over all n.
        inner = rule.split("_", 1)[1] if "_" in rule else "median"
        d, C, H = ctx["d"], ctx["C"], ctx["hidden"]
        sl = slices(d, C, H)
        (wa, wb), (ba, bb) = sl[4], sl[5]
        h2 = H[1]
        out, _ = aggregate(D, inner, ctx)
        out = out.clone()
        bias = D[:, ba:bb]                                  # (n, C)
        own = ctx["own_sub"] if "own_sub" in ctx else (bias > ctx.get("osa_tau", 0.0))
        est = []
        for c in range(C):
            rows = torch.nonzero(own[:, c], as_tuple=True)[0]
            est.append(rows.tolist())
            if len(rows) == 0:
                continue
            cols = torch.cat([torch.arange(wa + c * h2, wa + (c + 1) * h2, device=D.device),
                              torch.tensor([ba + c], device=D.device)])
            sub = D[rows][:, cols]
            sctx = dict(ctx, f=min(ctx.get("f", 0), max(len(rows) - 3, 0)) if len(rows) > 2 else 0)
            out[cols] = aggregate(sub, inner, sctx)[0] if len(rows) > 2 else sub.mean(0)
        return out, {"owners_est": est}
    if rule.startswith("oats"):
        # Owner-Anchored Trust Scoring (proposed, v2). rule = 'oats_<inner>',
        # <inner> may itself be e.g. 'osa_mean'. For every class c, the label
        # owners O_c (bias delta > 0) contain no label flipper (T1), so the
        # median of their shared-layer updates is a flipper-free anchor g_c.
        # A client holding class-c INPUTS under a wrong label pushes those
        # inputs the opposite way (cos < 0); a client with no class-c inputs
        # is ~orthogonal. Score s_i = min over classes i does not own of
        # cos(D_i[shared], g_c); clients with s_i < -tau are dropped.
        inner = rule.split("_", 1)[1] if "_" in rule else "mean"
        d, C, H = ctx["d"], ctx["C"], ctx["hidden"]
        sl = slices(d, C, H)
        shared = slice(0, sl[3][1])
        own = D[:, sl[5][0]:sl[5][1]] > 0
        S = D[:, shared]
        s_ = torch.zeros(n, device=D.device)
        for c in range(C):
            O = torch.nonzero(own[:, c], as_tuple=True)[0]
            if len(O) == 0 or len(O) == n:
                continue
            g = S[O].median(0).values if len(O) > 1 else S[O[0]]
            cs = torch.nn.functional.cosine_similarity(S, g[None], dim=1)
            cs = torch.where(own[:, c], torch.ones_like(cs), cs)
            s_ = torch.minimum(s_, cs)
        keep = s_ > -ctx.get("oats_tau", 0.1)
        if keep.sum() < 1:
            keep[:] = True
        kidx = torch.nonzero(keep, as_tuple=True)[0]
        kctx = dict(ctx, f=max(0, min(ctx.get("f", 0), len(kidx) - 3)),
                    sizes=[ctx["sizes"][i] for i in kidx.tolist()] if ctx.get("sizes") else None)
        out, info = aggregate(D[kidx], inner, kctx)
        return out, dict(info, kept=kidx.tolist())
    if rule.startswith("swcosa"):
        # EXPLORATORY: direction screen, then wcosa. Clients whose shared-layer
        # update has negative cosine with the certificate-weighted median are dropped.
        sl = slices(ctx["d"], ctx["C"], ctx["hidden"])
        sh = slice(0, sl[3][1])
        (wa, wb), (ba, bb) = sl[4], sl[5]
        C_, H_ = ctx["C"], ctx["hidden"]
        cert = (D[:, wa:wb].view(n, C_, H_[1]) > 0).any(-1) | (D[:, ba:bb] > 0)
        ref = weighted_median(D[:, sh], cert.sum(1).to(D.dtype))
        cs = torch.nn.functional.cosine_similarity(D[:, sh], ref[None], dim=1)
        keep = torch.nonzero(cs > 0, as_tuple=True)[0]
        if len(keep) < 3:
            keep = torch.arange(n, device=D.device)
        kctx = dict(ctx, f=max(0, min(ctx.get("f", 0), len(keep) - 3)),
                    sizes=[ctx["sizes"][i] for i in keep.tolist()] if ctx.get("sizes") else None)
        out, info = aggregate(D[keep], rule[1:], kctx)
        return out, dict(info, kept=keep.tolist())
    if rule.startswith("wcosa"):
        # EXPLORATORY: cOSA + shared layers as a weighted mean, weight = number
        # of classes the client certifiably owns (positive-evidence prior).
        inner = rule.split("_", 1)[1] if "_" in rule else "mean"
        d, C, H = ctx["d"], ctx["C"], ctx["hidden"]
        sl = slices(d, C, H)
        (wa, wb), (ba, bb) = sl[4], sl[5]
        cert = (D[:, wa:wb].view(n, C, H[1]) > 0).any(-1) | (D[:, ba:bb] > 0)
        inner = inner.replace("_wmed", "") or "mean"
        out, info = aggregate(D, "osa_" + inner, dict(ctx, own_sub=cert))
        w = cert.sum(1).to(D.dtype)
        sh = slice(0, sl[3][1])
        out = out.clone()
        if ctx.get("wcosa_shared", "wmean") == "wmed" or rule.endswith("_wmed"):
            out[sh] = weighted_median(D[:, sh], w)
        else:
            out[sh] = (w[:, None] * D[:, sh]).sum(0) / w.sum()
        return out, dict(info, cert_w=w.tolist())
    if rule.startswith("cosa") or rule.startswith("scosa"):
        # Certified OSA (proposed, v2). Lemma T1': with ReLU features (h >= 0)
        # and softmax CE, a client with no class-c labels has EVERY entry of
        # its class-c output-row delta <= 0 and bias delta < 0 (SGD or Adam).
        # So i is admitted as an owner of c iff any entry of (row c, bias c)
        # is > 0. 'scosa' additionally keeps a sticky registry across rounds.
        inner = rule.split("_", 1)[1] if "_" in rule else "mean"
        d, C, H = ctx["d"], ctx["C"], ctx["hidden"]
        sl = slices(d, C, H)
        (wa, wb), (ba, bb) = sl[4], sl[5]
        W = D[:, wa:wb].view(n, C, H[1])
        cert = (W > 0).any(-1) | (D[:, ba:bb] > 0)
        if rule.startswith("scosa"):
            cert = cert if ctx.get("own_reg") is None else (ctx["own_reg"] | cert)
            ctx["own_reg"] = cert
        out, info = aggregate(D, "osa_" + inner, dict(ctx, own_sub=cert))
        return out, info
    if rule.startswith("sosa"):
        # Sticky OSA: OSA with the sticky ownership registry (see oap).
        inner = rule.split("_", 1)[1] if "_" in rule else "mean"
        sl = slices(ctx["d"], ctx["C"], ctx["hidden"])
        pos = D[:, sl[5][0]:sl[5][1]] > 0
        reg = pos if ctx.get("own_reg") is None else (ctx["own_reg"] | pos)
        ctx["own_reg"] = reg
        out, info = aggregate(D, "osa_" + inner, dict(ctx, own_sub=reg))
        return out, info
    if rule.startswith("oap"):
        # Owner-Anchored Projection (proposed, v2). rule = 'oap_<inner>'.
        # Sticky ownership registry: i owns c once its bias delta for c has
        # been > 0 in any round (non-owners never are, T1). For each class c
        # with 0 < |O_c| < n, anchor g_c = median of owners' updates; every
        # non-owner's update has its NEGATIVE component along g_c removed
        # ("non-owners cannot veto the owners of c"). Then <inner> aggregates;
        # an osa_* inner uses the registry instead of the per-round sign.
        inner = rule.split("_", 1)[1] if "_" in rule else "mean"
        d, C, H = ctx["d"], ctx["C"], ctx["hidden"]
        sl = slices(d, C, H)
        pos = D[:, sl[5][0]:sl[5][1]] > 0
        reg = ctx.get("own_reg")
        reg = pos if reg is None else (reg | pos)
        ctx["own_reg"] = reg
        scope = slice(0, sl[3][1]) if ctx.get("oap_scope", "all") == "shared" else slice(0, D.shape[1])
        Dp = D.clone()
        clipped = 0
        for c in range(C):
            O = torch.nonzero(reg[:, c], as_tuple=True)[0]
            if len(O) == 0 or len(O) == n:
                continue
            g = D[O, scope].median(0).values if len(O) > 1 else D[O[0], scope]
            g = g / g.norm().clamp(min=1e-12)
            proj = Dp[:, scope] @ g
            neg = (~reg[:, c]) & (proj < 0)
            clipped += int(neg.sum())
            Dp[:, scope] -= (torch.where(neg, proj, torch.zeros_like(proj)))[:, None] * g[None]
        out, info = aggregate(Dp, inner, dict(ctx, own_sub=reg))
        return out, dict(info, clipped=clipped)
    if rule.startswith("oracle_"):
        # DIAGNOSTIC ONLY (uses the true Byzantine set): mean over all clients,
        # except the named parameter blocks, which use the honest-only mean.
        # Blocks: L1 (W1,b1), L2 (W2,b2), Rb (benign output row+bias),
        # Rt (target output rows+biases), R (whole output layer).
        d, C, H = ctx["d"], ctx["C"], ctx["hidden"]
        sl = slices(d, C, H)
        (wa, wb), (ba, bb) = sl[4], sl[5]
        hon = [i for i in range(n) if i not in set(ctx["byz"])]
        out = D.mean(0).clone()
        hm = D[hon].mean(0)
        idx = []
        for blk in rule.split("_", 1)[1].split("+"):
            if blk == "L1":
                idx.append(torch.arange(sl[0][0], sl[1][1]))
            elif blk == "L2":
                idx.append(torch.arange(sl[2][0], sl[3][1]))
            elif blk == "R":
                idx.append(torch.arange(wa, bb))
            elif blk in ("Rb", "Rt"):
                cls = [ctx.get("benign", 0)] if blk == "Rb" else list(ctx["targets"])
                for c in cls:
                    idx.append(torch.arange(wa + c * H[1], wa + (c + 1) * H[1]))
                    idx.append(torch.tensor([ba + c]))
        idx = torch.cat(idx).to(D.device)
        out[idx] = hm[idx]
        return out, {}
    if rule == "fldefender":
        # Port of Jebreel et al. (KBS 2023) reference code FLDefender.score +
        # update_score_history: cosine matrix of last-layer weight deltas,
        # standardise + PCA(2), cosine to the median centroid, accumulated
        # over rounds; trust = (hist - q25) / max, clipped at 0; weighted mean.
        from sklearn.decomposition import PCA
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import StandardScaler
        sl = slices(ctx["d"], ctx["C"], ctx["hidden"])
        G = (-D[:, sl[4][0]:sl[4][1]]).cpu().numpy()
        cs = cosine_similarity(G) - np.eye(n)
        cs = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(cs))
        sc = cosine_similarity(np.median(cs, axis=0)[None], cs)[0]
        hist = ctx.get("fld_hist")
        hist = sc if hist is None else hist + sc
        ctx["fld_hist"] = hist
        tr = hist - np.quantile(hist, 0.25)
        tr = tr / tr.max() if tr.max() > 0 else np.ones(n)
        tr[tr < 0] = 0
        w = torch.as_tensor(tr, device=D.device, dtype=D.dtype)
        return (w[:, None] * D).sum(0) / w.sum(), {"kept": np.nonzero(tr > 0)[0].tolist()}
    if rule == "lfighter":
        keep, top2 = LFighter.select(D, ctx["d"], ctx["C"], ctx["hidden"])
        return D[torch.as_tensor(keep, device=D.device)].mean(0), \
            {"kept": np.nonzero(keep)[0].tolist(), "top2": top2.tolist()}
    raise ValueError(rule)


# ─────────────────────────────── attacks ─────────────────────────────────

def flip_labels(y: np.ndarray, targets, benign) -> np.ndarray:
    out = y.copy()
    out[np.isin(y, list(targets))] = benign
    return out


def update_attack(D, byz_rows, kind, rng, ctx):
    """Update-space attacks applied after honest training (in place on a copy)."""
    if kind is None or not byz_rows:
        return D
    D = D.clone()
    hon = [i for i in range(D.shape[0]) if i not in byz_rows]
    H = D[hon]
    mu, sd = H.mean(0), H.std(0, unbiased=True) if len(hon) > 1 else torch.zeros_like(H[0])
    if kind == "signflip":
        D[byz_rows] = -D[byz_rows]
    elif kind == "gauss":
        D[byz_rows] = torch.randn_like(D[byz_rows]) * ctx.get("sigma", 1.0) * mu.norm() / math.sqrt(D.shape[1])
    elif kind == "ipm":
        D[byz_rows] = -ctx.get("eps", 0.5) * mu
    elif kind == "alie":
        nn_, f = D.shape[0], len(byz_rows)
        s = nn_ // 2 + 1 - f
        from scipy.stats import norm
        z = float(norm.ppf((nn_ - s) / nn_)) if s > 0 else 1.0
        D[byz_rows] = mu - z * sd
    elif kind == "scale":       # model replacement of the poisoned update
        D[byz_rows] = D[byz_rows] * ctx.get("boost", D.shape[0])
    else:
        raise ValueError(kind)
    return D
