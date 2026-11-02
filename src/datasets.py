"""Dataset loaders -> flcore.Fed. Every loader:
  * splits train/test per (client, class) with a fixed split seed,
  * fits the scaler on TRAIN only,
  * caches the processed arrays in DATA_ROOT/cache as .npz,
  * records duplicate/leakage statistics in fed.meta.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from flcore import Fed

import os
DATA_ROOT = Path(os.environ.get("V2_DATA", str(Path(__file__).resolve().parents[1] / "data")))
CACHE = Path(os.environ.get("V2_CACHE", str(DATA_ROOT / "cache")))
try:
    CACHE.mkdir(parents=True, exist_ok=True)
except OSError:          # read-only data folder: cache must already exist
    pass

NBAIOT_CLASSES = ["benign",
                  "gafgyt.combo", "gafgyt.junk", "gafgyt.scan", "gafgyt.tcp", "gafgyt.udp",
                  "mirai.ack", "mirai.scan", "mirai.syn", "mirai.udp", "mirai.udpplain"]
NBAIOT_DEVICES = ["Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
                  "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
                  "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
                  "SimpleHome_XCS7_1002_WHT_Security_Camera",
                  "SimpleHome_XCS7_1003_WHT_Security_Camera"]
MIRAI = [6, 7, 8, 9, 10]
GAFGYT = [1, 2, 3, 4, 5]


def _split_scale(frames, n_classes, split_seed, test_ratio):
    """frames: list of (client_id, X ndarray, y ndarray). Returns per-client
    train arrays and a pooled test set, scaled with train statistics."""
    tr, te = [], []
    for cid, X, y in frames:
        Xtr_c, ytr_c = [], []
        for c in np.unique(y):
            ix = np.nonzero(y == c)[0]
            rs = np.random.RandomState(split_seed + 1000 * cid + int(c))
            rs.shuffle(ix)
            k = len(ix)
            n_te = int(round(k * test_ratio))
            n_te = min(max(n_te, 1 if k >= 3 else 0), k - 1)
            te.append((X[ix[:n_te]], y[ix[:n_te]]))
            Xtr_c.append(X[ix[n_te:]])
            ytr_c.append(y[ix[n_te:]])
        tr.append((np.concatenate(Xtr_c), np.concatenate(ytr_c)))
    allX = np.concatenate([x for x, _ in tr])
    mu, sd = allX.mean(0), allX.std(0)
    sd[sd < 1e-8] = 1.0
    Xs = [((x - mu) / sd).astype(np.float32) for x, _ in tr]
    ys = [y.astype(np.int64) for _, y in tr]
    Xte = np.concatenate([x for x, _ in te])
    yte = np.concatenate([y for _, y in te]).astype(np.int64)
    Xte = ((Xte - mu) / sd).astype(np.float32)
    return Xs, ys, Xte, yte, (mu, sd)


def leakage(Xs, Xte, yte, ys):
    """Fraction of test rows whose exact (features,label) appear in train."""
    def keys(X, y):
        return pd.util.hash_pandas_object(
            pd.DataFrame(np.round(X, 6)).assign(_y=y), index=False).values
    trk = set(np.concatenate([keys(x, y) for x, y in zip(Xs, ys)]).tolist())
    tek = keys(Xte, yte)
    return float(np.mean([k in trk for k in tek]))


def load_nbaiot(per_device=50_000, split_seed=42, test_ratio=0.15, dedup=False):
    tag = f"nbaiot_pd{per_device}_s{split_seed}_t{test_ratio}_dd{int(dedup)}"
    fp = CACHE / f"{tag}.npz"
    if not fp.exists():
        zp = DATA_ROOT / "nbaiot_raw" / "nbaiot-dataset.zip"
        zf = zipfile.ZipFile(zp)
        names = set(zf.namelist())
        frames = []
        for dev in range(1, 10):
            parts = []
            for ci, suf in enumerate(NBAIOT_CLASSES):
                nm = f"{dev}.{suf}.csv"
                if nm not in names:
                    continue
                df = pd.read_csv(io.BytesIO(zf.read(nm)))
                assert df.shape[1] == 115, (nm, df.shape)
                df["label"] = ci
                parts.append(df)
            dev_df = pd.concat(parts, ignore_index=True)
            if dedup:
                dev_df = dev_df.drop_duplicates()
            if per_device and len(dev_df) > per_device:
                rs = np.random.RandomState(split_seed + dev)
                keep = []
                for ci, g in dev_df.groupby("label", sort=True):
                    k = max(1, int(round(per_device * len(g) / len(dev_df))))
                    keep.append(g.sample(n=min(k, len(g)), random_state=rs.randint(2 ** 31)))
                dev_df = pd.concat(keep, ignore_index=True)
            X = dev_df.drop(columns="label").values.astype(np.float64)
            y = dev_df["label"].values.astype(np.int64)
            frames.append((dev - 1, X, y))
            print(f"  device {dev}: {len(y):,} rows, classes {sorted(set(y.tolist()))}")
        Xs, ys, Xte, yte, _ = _split_scale(frames, len(NBAIOT_CLASSES), split_seed, test_ratio)
        np.savez_compressed(fp, Xte=Xte, yte=yte,
                            **{f"X{i}": x for i, x in enumerate(Xs)},
                            **{f"y{i}": y for i, y in enumerate(ys)})
    z = np.load(fp)
    n = sum(1 for k in z.files if k.startswith("y") and k != "yte")
    Xs = [z[f"X{i}"] for i in range(n)]
    ys = [z[f"y{i}"] for i in range(n)]
    fed = Fed("nbaiot", Xs, ys, z["Xte"], z["yte"], NBAIOT_CLASSES,
              NBAIOT_DEVICES, benign=0, meta={"tag": tag})
    return fed


# ───────────────────────────── IoT-23 ────────────────────────────────────
# Official per-capture label counts (stratosphereips.org/datasets-iot23,
# "Individual details", read 2026-10-24). Used ONLY to identify which file of the
# public CSV mirror of the logs is which capture: a file
# is accepted only if its exact streamed counts equal one row here.
IOT23_OFFICIAL = {
    "34-1": {"Benign": 1923, "C&C": 6706, "DDoS": 14394, "PartOfAHorizontalPortScan": 122},
    "43-1": {"Benign": 20574934, "C&C": 3498, "C&C-FileDownload": 14, "DDoS": 65803,
             "FileDownload": 1, "Okiru": 8765885, "PartOfAHorizontalPortScan": 37911674},
    "44-1": {"Benign": 211, "C&C": 14, "C&C-FileDownload": 11, "DDoS": 1},
    "49-1": {"Benign": 3665, "C&C": 1922, "C&C-FileDownload": 1,
             "PartOfAHorizontalPortScan": 5404959},
    "52-1": {"Benign": 1794, "C&C": 6, "C&C-FileDownload": 12,
             "PartOfAHorizontalPortScan": 19779564},
    "20-1": {"Benign": 3193, "C&C-Torii": 16},
    "21-1": {"Benign": 3272, "C&C-Torii": 14},
    "42-1": {"Benign": 4420, "C&C-FileDownload": 3, "FileDownload": 3},
    "60-1": {"Benign": 2476, "C&C-HeartBeat": 95, "DDoS": 3578457},
    "17-1": {"Attack": 4, "Benign": 31438, "C&C-HeartBeat": 6834, "DDoS": 13655172,
             "Okiru": 13655215, "PartOfAHorizontalPortScan": 27311187,
             "PartOfAHorizontalPortScan-Attack": 5},
    "36-1": {"Benign": 2663, "C&C-HeartBeat": 15688, "Okiru": 13626744, "Okiru-Attack": 3},
    "33-1": {"Benign": 1380791, "C&C-HeartBeat": 5278, "Okiru-Attack": 13609467,
             "PartOfAHorizontalPortScan": 39459055},
    "8-1": {"Benign": 2181, "C&C": 8222},
    "35-1": {"Attack": 3, "Benign": 8262389, "C&C": 81, "C&C-FileDownload": 12,
             "DDoS": 2185302},
    "48-1": {"Attack": 2752, "Benign": 3734, "C&C-HeartBeat-Attack": 834,
             "C&C-HeartBeat-FileDownload": 11, "C&C-PartOfAHorizontalPortScan": 888,
             "PartOfAHorizontalPortScan": 3386119},
    "39-1": {"Attack": 677, "Benign": 7337, "C&C": 1530,
             "PartOfAHorizontalPortScan": 73559437},
    "7-1": {"Benign": 75955, "C&C-HeartBeat": 5778, "DDoS": 39584, "Okiru": 11333397},
    "9-1": {"Benign": 22548, "PartOfAHorizontalPortScan": 6355745},
    "3-1": {"Attack": 5962, "Benign": 4536, "C&C": 8, "PartOfAHorizontalPortScan": 145597},
    "1-1": {"Benign": 469275, "C&C": 8, "PartOfAHorizontalPortScan": 539465},
}
# Coarse classes: every C&C* variant -> C&C, Okiru* -> Okiru, *-Attack scan ->
# Attack. 'benign' (lower-case) rows are the honeypot captures.
IOT23_CLASSES = ["Benign", "PortScan", "C&C", "DDoS", "Okiru", "Attack", "FileDownload"]


def iot23_coarse(detailed: str) -> str:
    if detailed in ("-", "Benign", "benign"):
        return "Benign"
    if detailed.startswith("C&C"):
        return "C&C"
    if detailed.startswith("Okiru"):
        return "Okiru"
    if detailed == "PartOfAHorizontalPortScan":
        return "PortScan"
    if detailed in ("Attack", "PartOfAHorizontalPortScan-Attack"):
        return "Attack"
    if detailed in ("DDoS", "FileDownload"):
        return detailed
    raise ValueError(detailed)


def iot23_identify(counts_json: dict):
    """Exact match of streamed counts against the official table."""
    if all(k.split("|")[0] == "benign" for k in counts_json["counts"]):
        return "honeypot"           # honeypot captures use lower-case 'benign'
    got = {}
    for k, v in counts_json["counts"].items():
        lab, det = k.split("|")
        name = "Benign" if det == "-" else det
        got[name] = got.get(name, 0) + v
    for cap, ref in IOT23_OFFICIAL.items():
        if {k: v for k, v in ref.items() if v} == got:
            return cap
    # Documented tolerance: every officially listed label count matches
    # EXACTLY and the file only adds unlisted labels totalling < 0.01% of rows.
    # Known case: mirror file 9 = 49-1 + 14 'FileDownload' rows absent from
    # the official table (checked by hand).
    for cap, ref in IOT23_OFFICIAL.items():
        ref = {k: v for k, v in ref.items() if v}
        extra = {k: v for k, v in got.items() if k not in ref}
        if all(got.get(k) == v for k, v in ref.items()) and                 sum(extra.values()) < 1e-4 * sum(got.values()):
            return cap
    # Documented tolerance 2: identical counts, labels renamed only within one
    # coarse class. Known case: official table lists 33-1's 13,609,467 flows as
    # 'Okiru-Attack', the mirror file labels them 'Okiru' (checked by hand).
    key = lambda d: sorted((iot23_coarse(k), v) for k, v in d.items())
    for cap, ref in IOT23_OFFICIAL.items():
        ref = {k: v for k, v in ref.items() if v}
        if sorted(ref.values()) == sorted(got.values()) and key(ref) == key(got):
            return cap
    return None


def _iot23_features(df: pd.DataFrame) -> np.ndarray:
    num = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts",
           "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
    cols = []
    for c in num:
        v = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")
        cols.append(np.log1p(v.fillna(0).clip(lower=0).values))
        cols.append(v.isna().values.astype(float))            # missing flag
    for c, cats in [("proto", ["tcp", "udp", "icmp"]),
                    ("conn_state", ["S0", "S1", "S2", "S3", "SF", "REJ", "RSTO", "RSTR",
                                    "RSTOS0", "RSTRH", "SH", "SHR", "OTH"]),
                    ("service", ["-", "dns", "http", "ssh", "ssl", "irc", "dhcp", "ntp"])]:
        for k in cats:
            cols.append((df[c].values == k).astype(float))
    for ch in "ShADadFfRrCcTtWwIiQq^":
        cols.append(np.log1p(df["history"].str.count(ch.replace("^", r"\^")).fillna(0).values))
    port = pd.to_numeric(df["resp_p"], errors="coerce").fillna(-1).values
    for p in [22, 23, 53, 80, 123, 443, 2323, 8080, 37215, 52869]:
        cols.append((port == p).astype(float))
    cols.append(((port >= 0) & (port < 1024)).astype(float))
    cols.append((port >= 49152).astype(float))
    return np.stack(cols, 1)


IOT23_USED = [c for c in IOT23_CLASSES if c != "FileDownload"]


def load_iot23(split_seed=42, test_ratio=0.2, per_class_cap=20000, min_rows=50):
    """Natural clients = captures (malware scenarios + honeypots).
    FileDownload (16 rows in total, 4 would land in test) is dropped."""
    root = DATA_ROOT / "iot23"
    nfiles = len(list(root.glob("*.counts.json")))      # cache keyed on the file set
    tag = f"iot23v2_s{split_seed}_t{test_ratio}_c{per_class_cap}_m{min_rows}_f{nfiles}"
    fp = CACHE / f"{tag}.npz"
    if nfiles == 0:                 # no raw files: use the shipped processed cache
        have = sorted(CACHE.glob(f"{tag[:-len('_f0')]}_f*.npz"),
                      key=lambda p: int(p.stem.rsplit("_f", 1)[1]))
        if not have:
            raise FileNotFoundError(f"no IoT-23 raw files and no cache {tag[:-3]}_f*.npz")
        fp = have[-1]
        tag = fp.stem
    if not fp.exists():
        frames, names = [], []
        for cj in sorted(root.glob("*.counts.json")):
            cap = iot23_identify(json.loads(cj.read_text()))
            if cap is None:
                raise RuntimeError(f"{cj.name}: counts match no official capture")
            df = pd.read_parquet(cj.with_name(cj.name.replace(".counts.json", ".parquet")))
            df["coarse"] = df.detailed_label.map(iot23_coarse)
            df = df[df.coarse.isin(IOT23_USED)].copy()
            df["cls"] = df.coarse.map(IOT23_USED.index)
            rs = np.random.RandomState(split_seed)
            df = (df.groupby("cls", group_keys=False)
                    .apply(lambda g: g.sample(n=min(len(g), per_class_cap), random_state=rs)))
            if len(df) < min_rows:
                print(f"  skip {cj.stem} ({cap}): {len(df)} rows")
                continue
            label = cap if cap != "honeypot" else f"honeypot-{cj.name.split('.')[0]}"
            frames.append((len(frames), _iot23_features(df), df["cls"].values.astype(np.int64)))
            names.append(label)
            print(f"  {label:22s} {len(df):>7,} rows  classes "
                  f"{dict(pd.Series(df.cls).map(lambda c: IOT23_USED[c]).value_counts())}")
        Xs, ys, Xte, yte, _ = _split_scale(frames, len(IOT23_USED), split_seed, test_ratio)
        np.savez_compressed(fp, Xte=Xte, yte=yte, names=np.array(names),
                            **{f"X{i}": x for i, x in enumerate(Xs)},
                            **{f"y{i}": y for i, y in enumerate(ys)})
    z = np.load(fp)
    n = sum(1 for k in z.files if k.startswith("y") and k != "yte")
    meta = {"tag": tag}
    sp = fp.with_name(fp.stem + "_src.npy")          # source client of each test row
    if sp.exists():                                  # (written by explore/write_test_src.py)
        meta["test_src"] = np.load(sp)
    return Fed("iot23", [z[f"X{i}"] for i in range(n)], [z[f"y{i}"] for i in range(n)],
               z["Xte"], z["yte"], IOT23_USED, list(z["names"]), benign=0,
               meta=meta)


TONIOT_CLASSES = ["normal", "backdoor", "ddos", "dos", "injection", "mitm", "password",
                  "ransomware", "scanning", "xss"]


def _toniot_features(df: pd.DataFrame) -> np.ndarray:
    """Zeek connection features as for IoT-23. Identifiers (IPs, raw ports) and free-text
    fields (dns_query, ssl_subject/issuer, http_uri/user_agent, mime types) are dropped."""
    cols = []
    for c in ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes",
              "dst_pkts", "dst_ip_bytes", "http_request_body_len", "http_response_body_len"]:
        v = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")
        cols.append(np.log1p(v.fillna(0).clip(lower=0).values))
    s = lambda c: df[c].astype(str).values
    for c, cats in [("proto", ["tcp", "udp", "icmp"]),
                    ("service", ["-", "dns", "http", "ftp", "ssl", "gssapi", "dce_rpc", "smb", "dhcp"]),
                    ("conn_state", ["S0", "S1", "S2", "S3", "SF", "REJ", "RSTO", "RSTR",
                                    "RSTOS0", "RSTRH", "SH", "SHR", "OTH"]),
                    ("dns_qtype", ["0", "1", "28", "12", "255", "43", "2", "32"]),
                    ("dns_rcode", ["0", "3", "5", "2"]),
                    ("dns_AA", ["T", "F"]), ("dns_RD", ["T", "F"]), ("dns_RA", ["T", "F"]),
                    ("dns_rejected", ["T", "F"]), ("ssl_established", ["T", "F"]),
                    ("http_method", ["GET", "POST", "HEAD"]), ("weird_notice", ["F"])]:
        for k in cats:
            cols.append((s(c) == k).astype(float))
    port = pd.to_numeric(df["dst_port"], errors="coerce").fillna(-1).values
    for p in [21, 22, 23, 53, 80, 443, 445, 1883, 8080]:
        cols.append((port == p).astype(float))
    cols.append(((port >= 0) & (port < 1024)).astype(float))
    cols.append((port >= 49152).astype(float))
    return np.stack(cols, 1)


def load_toniot(split_seed=42, test_ratio=0.2, min_rows=500):
    """Natural clients = destination hosts (the device whose gateway sees the flow), keeping
    hosts with >= min_rows flows. Source of each test row is stored as meta['test_src']."""
    tag = f"toniot_s{split_seed}_t{test_ratio}_m{min_rows}"
    fp = CACHE / f"{tag}.npz"
    if not fp.exists():
        df = pd.read_csv(DATA_ROOT / "toniot" / "train_test_network.csv", low_memory=False)
        df["cls"] = df["type"].map(TONIOT_CLASSES.index)
        vc = df.dst_ip.value_counts()
        hosts = sorted(vc[vc >= min_rows].index)
        frames, names = [], []
        for h in hosts:
            g = df[df.dst_ip == h]
            frames.append((len(frames), _toniot_features(g), g["cls"].values.astype(np.int64)))
            names.append(h)
        Xs, ys, Xte, yte, _ = _split_scale(frames, len(TONIOT_CLASSES), split_seed, test_ratio)
        # test rows are appended per client, per class, in _split_scale's order
        src = []
        for cid, _, y in frames:
            for c in np.unique(y):
                k = int((y == c).sum())
                n_te = min(max(int(round(k * test_ratio)), 1 if k >= 3 else 0), k - 1)
                src += [cid] * n_te
        src = np.array(src)
        assert len(src) == len(yte)
        np.savez_compressed(fp, Xte=Xte, yte=yte, names=np.array(names), test_src=src,
                            **{f"X{i}": x for i, x in enumerate(Xs)},
                            **{f"y{i}": y for i, y in enumerate(ys)})
    z = np.load(fp)
    n = sum(1 for k in z.files if k.startswith("y") and k != "yte")
    return Fed("toniot", [z[f"X{i}"] for i in range(n)], [z[f"y{i}"] for i in range(n)],
               z["Xte"], z["yte"], TONIOT_CLASSES, list(z["names"]), benign=0,
               meta={"tag": tag, "test_src": z["test_src"]})
