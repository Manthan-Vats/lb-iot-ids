# Sharing attack evidence between IoT intrusion detectors without trusting the sharers

Code, processed data, raw result files and the pre-registration log for the paper of the same title
(Manthan Vats, Manipal University Jaipur). The paper is under review; the citation will be added here.

The method in one paragraph: a site keeps a small set of its own benign traffic rows private. Peers' labelled attack
rows may only *raise* the anomaly score of points that lie closer to some peer's attack rows than to the site's own
normal traffic, and the final detector is calibrated with a conformal p-value on the private rows. The false-alarm
rate then stays at the chosen level whatever the peers send, and a lying peer can cost at most the share of the
false-alarm budget its rows occupy. A vetting step on further private rows drops peers whose share is too large.

## Contents

| Path | What it is |
|---|---|
| `src/explore/aoe15.py` ... `aoe28.py` | The experiments of the paper (one script per pre-registered experiment family, see below) |
| `src/explore/aw.py`, `aoe.py` and the other modules in `src/` | Shared code the experiments import (data loading, conformal p-values, nearest-neighbour distances, baselines) |
| `src/analysis/make_paper_figs.py` | Draws Figures 3-5 from the result files and writes every plotted value to `fig_numbers.txt` |
| `src/analysis/boot_worst.py` | Seed-clustered bootstrap of the worst-case miss rate |
| `data/cache/` | Processed N-BaIoT and IoT-23 arrays exactly as used in the paper (see `DATA_LICENSES.md`) |
| `data_prep/iot23_stream.py` | Builds the IoT-23 sample from the official Zeek `conn.log.labeled` files |
| `results/explore/*.json` | Raw per-cell results of every experiment in the paper's pre-registration table |
| `PREREGISTRATION.md` | The dated, append-only plan: each prediction was written before its run, and each outcome after it |

## Setup

Python 3.12 with the packages in `requirements.txt`. A CUDA GPU is used when present (the paper used an NVIDIA RTX
3050 laptop GPU with 6 GB; one cell of an experiment takes about 14 s); the code also runs on CPU, more slowly.

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The processed arrays in `data/cache/` are found automatically. To use another folder, set `V2_DATA` to a folder that
contains `cache/`.

## Reproducing the paper

Run every command from the repository root. Each writes `results/explore/<script>_<dataset>[_dedup][_sel2000][_s<seeds>].json`;
the shipped files of the same name are the ones the paper reports, so move them aside first if you want to compare.

| Paper | Pre-registered experiment | Command |
|---|---|---|
| Figure 3 (main result, N-BaIoT seeds 6-11) | E-AOE25 | `python src/explore/aoe21.py nbaiot iot23 nsel=2000 seed=6 seed=7 seed=8 seed=9 seed=10 seed=11` |
| Figure 3 (IoT-23 de-duplicated) | E-AOE25 | `python src/explore/aoe21.py iot23 dedup nsel=2000 seed=6 seed=7 seed=8 seed=9 seed=10 seed=11` |
| Figure 3 (N-BaIoT seeds 15-29) | E-AOE27 | `python src/explore/aoe21.py nbaiot nsel=2000 seed=15 seed=16 ... seed=29` (all 15 seeds) |
| Figure 4 (vetting-set size) | E-AOE18, E-AOE24 | `python src/explore/aoe18.py nbaiot iot23` and `python src/explore/aoe18.py nbaiot iot23 seed=6 seed=7 seed=8` |
| Figure 5 (rule-aware liars) | E-AOE26, E-AOE28 | `python src/explore/aoe28.py nbaiot iot23 nsel=2000 seed=12 seed=13 seed=14` (and `seed=30 seed=31 seed=32`; add `dedup` with `iot23` for the de-duplicated runs) |
| Appendix table, earlier checks | E-AOE15-E-AOE23, B7 | `aoe15.py` (E-AOE15/16/17, B7 with `dedup`), `aoe19.py` (AdaDetect), `aoe20.py` (E-AOE20/22), `aoe21.py` (E-AOE21/23) with the seeds named in `PREREGISTRATION.md` |

Then `python src/analysis/make_paper_figs.py` redraws Figures 3-5 into `manuscript/figs/`.

Notes on exact reproduction:
* The shipped arrays are the ones the paper used, so splits, cells and peers are identical. Checked before release on
  a fresh copy of this repository: re-running seed 6 of the main experiment reproduced all 720 (IoT-23) and 160
  (N-BaIoT) false-alarm and miss-rate values of the shipped files exactly, and `make_paper_figs.py` reproduced every
  plotted value. Neural baselines are trained on GPU; on different hardware or library versions their miss rates can
  differ in the last digits.
* E-AOE17 (TON\_IoT) needs the TON\_IoT network dataset from its authors (UNSW Canberra), which is not redistributed
  here; its result file is included.
* To rebuild the IoT-23 sample from the official logs instead of using the shipped arrays, stream each capture's
  `conn.log.labeled` through `data_prep/iot23_stream.py` into `data/iot23/` and delete `data/cache/iot23v2_*`. The
  sample is a seeded reservoir sample per capture, so a rebuild gives a different random sample of the same captures
  and close but not identical numbers.

## Vocabulary of the code and the log

The code and the pre-registration log use internal names; the paper uses these:

| Code / log | Paper |
|---|---|
| release, released | miss rate: share of attack rows not flagged |
| OC-kNN | OC-kNN (distance to the anchor set, conformal) |
| AOE-peer | learned classifier on peer rows |
| Ada.05 | AdaDetect at 5% attack prevalence |
| LB-v, LB-vc | greedy and certified vetting |
| LBA.05-v | LBA-v (the wrapper on the AdaDetect base) |
| Sel, nsel | vetting set and its size m |
| k* | f* (largest certified count) |
| cB | share of calibration rows inside the boost region |

## Licence

Code: MIT (see `LICENSE`). Processed data: CC BY 4.0, inherited from the original datasets (see `DATA_LICENSES.md`).
