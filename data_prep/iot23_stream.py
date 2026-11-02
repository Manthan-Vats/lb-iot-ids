"""Stream one IoT-23 Zeek conn.log.labeled from stdin and keep a seeded
reservoir sample per detailed label, plus the EXACT count of every label.

Usage:  curl -s URL | python -I iot23_stream.py CAPTURE_NAME OUT_DIR [CAP]

Why streaming: several captures are 1-11 GB; disk is limited. Reservoir
sampling (Algorithm R) gives a uniform sample of each label's rows without
storing the file, and the exact per-label counts are kept so the true class
ownership of every capture is known, not estimated.
"""
import sys, json, random
from pathlib import Path
import pandas as pd

FIELDS = ["ts", "uid", "orig_h", "orig_p", "resp_h", "resp_p", "proto",
          "service", "duration", "orig_bytes", "resp_bytes", "conn_state",
          "local_orig", "local_resp", "missed_bytes", "history", "orig_pkts",
          "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents",
          "label", "detailed_label"]

def main():
    cap_name, out_dir = sys.argv[1], Path(sys.argv[2])
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 60_000
    rng = random.Random(f"iot23-{cap_name}")      # seeded per capture
    res, seen, bad = {}, {}, 0
    for raw in sys.stdin.buffer:
        if raw.startswith(b"#") or raw.startswith(b"ts,"):
            continue
        # Zeek logs are tab-separated; the public CSV mirror is comma-separated
        # with the last three fields space-separated. Zeek never emits empty
        # fields ('-' instead) and no field contains a space or comma, so
        # splitting on both is lossless for both formats.
        parts = raw.decode("utf-8", "replace").replace(",", " ").split()
        if len(parts) != len(FIELDS):
            bad += 1
            continue
        key = f"{parts[21]}|{parts[22]}"
        n = seen.get(key, 0) + 1
        seen[key] = n
        bucket = res.setdefault(key, [])
        if n <= cap:
            bucket.append(parts)
        else:
            j = rng.randrange(n)
            if j < cap:
                bucket[j] = parts
    rows = [r for b in res.values() for r in b]
    df = pd.DataFrame(rows, columns=FIELDS)
    df.insert(0, "capture", cap_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{cap_name}.parquet", index=False)
    (out_dir / f"{cap_name}.counts.json").write_text(
        json.dumps({"capture": cap_name, "counts": seen, "malformed": bad,
                    "cap": cap}, indent=1))
    print(cap_name, "rows kept", len(df), "malformed", bad, seen)

if __name__ == "__main__":
    main()
