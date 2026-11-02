"""IoT-23 with a TEMPORAL split: same captures, classes and per-class cap as datasets.load_iot23, but
inside every (capture, class) the last 20% of flows by Zeek timestamp form the test set. Used to
check whether own-support checks stay cheap when test traffic is later than training traffic.
Cached as <CACHE>/iot23time_*.npz with test_src."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets
from datasets import (DATA_ROOT, CACHE, IOT23_USED, Fed, iot23_identify, iot23_coarse,
                      _iot23_features)


def load_iot23_time(split_seed=42, test_ratio=0.2, per_class_cap=20000, min_rows=50):
    root = DATA_ROOT / "iot23"
    nfiles = len(list(root.glob("*.counts.json")))
    tag = f"iot23time_s{split_seed}_t{test_ratio}_c{per_class_cap}_m{min_rows}_f{nfiles}"
    fp = CACHE / f"{tag}.npz"
    if not fp.exists():
        frames, names = [], []
        for cj in sorted(root.glob("*.counts.json")):
            cap = iot23_identify(json.loads(cj.read_text()))
            df = pd.read_parquet(cj.with_name(cj.name.replace(".counts.json", ".parquet")))
            df["coarse"] = df.detailed_label.map(iot23_coarse)
            df = df[df.coarse.isin(IOT23_USED)].copy()
            df["cls"] = df.coarse.map(IOT23_USED.index)
            rs = np.random.RandomState(split_seed)
            df = (df.groupby("cls", group_keys=False)
                    .apply(lambda g: g.sample(n=min(len(g), per_class_cap), random_state=rs)))
            if len(df) < min_rows:
                continue
            label = cap if cap != "honeypot" else f"honeypot-{cj.name.split('.')[0]}"
            frames.append((len(frames), _iot23_features(df), df["cls"].values.astype(np.int64),
                           pd.to_numeric(df["ts"], errors="coerce").values))
            names.append(label)
        tr, te, src = [], [], []
        for cid, X, y, ts in frames:
            Xc, yc = [], []
            for c in np.unique(y):
                ix = np.nonzero(y == c)[0]
                ix = ix[np.argsort(ts[ix], kind="stable")]          # oldest first
                k = len(ix)
                n_te = min(max(int(round(k * test_ratio)), 1 if k >= 3 else 0), k - 1)
                te.append((X[ix[k - n_te:]], y[ix[k - n_te:]])); src += [cid] * n_te
                Xc.append(X[ix[:k - n_te]]); yc.append(y[ix[:k - n_te]])
            tr.append((np.concatenate(Xc), np.concatenate(yc)))
        allX = np.concatenate([x for x, _ in tr])
        mu, sd = allX.mean(0), allX.std(0); sd[sd < 1e-8] = 1.0
        Xs = [((x - mu) / sd).astype(np.float32) for x, _ in tr]
        ys = [y.astype(np.int64) for _, y in tr]
        Xte = ((np.concatenate([x for x, _ in te]) - mu) / sd).astype(np.float32)
        yte = np.concatenate([y for _, y in te]).astype(np.int64)
        np.savez_compressed(fp, Xte=Xte, yte=yte, names=np.array(names), test_src=np.array(src),
                            **{f"X{i}": x for i, x in enumerate(Xs)},
                            **{f"y{i}": y for i, y in enumerate(ys)})
    z = np.load(fp)
    n = sum(1 for k in z.files if k.startswith("y") and k != "yte")
    return Fed("iot23", [z[f"X{i}"] for i in range(n)], [z[f"y{i}"] for i in range(n)],
               z["Xte"], z["yte"], IOT23_USED, list(z["names"]), benign=0,
               meta={"tag": tag, "test_src": z["test_src"]})


if __name__ == "__main__":
    fed = load_iot23_time()
    print(fed.n, len(fed.yte), fed.client_names)
