"""Paper figures 3-5, recomputed from the logged result JSONs (no number is typed by hand).
Worst case = max over conditions of the mean release over cells (cell = seed, target, victim), as in RESEARCH_PLAN.
CIs: seed-clustered percentile bootstrap (2000 resamples), as in boot_worst.py.
Writes manuscript/figs/fig{3,4,5}.pdf (+ 300 dpi .png) and manuscript/figs/fig_numbers.txt (every plotted value)."""
import json, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]; RES = ROOT / "results/explore"; OUT = ROOT / "manuscript/figs"
OUT.mkdir(parents=True, exist_ok=True); LOG = open(OUT / "fig_numbers.txt", "w")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"            # dataviz reference slots 1-3 (validated on white)
G1, G2, G3, INK, MUTED = "#d3d1c7", "#a3a198", "#5f5e5a", "#0b0b0b", "#52514e"
MM = 72 / 25.4                                                    # points per mm (Sadhana: axes .3 mm, curves .6 mm)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": .3 * MM, "axes.edgecolor": MUTED,
                     "xtick.major.width": .3 * MM, "ytick.major.width": .3 * MM, "lines.linewidth": .6 * MM,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "savefig.dpi": 300,
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType, not Type 3 (IEEE PDF checks flag Type 3)
rng = np.random.default_rng(0)


def load(*names):
    return [r for n in names for r in json.load(open(RES / n))]


def table(R, method, a):
    """(cells x conditions) release matrix and the seed of each cell."""
    conds = sorted({r["cond"] for r in R}); idx = {(r["seed"], r["target"], r["v"], r["cond"]): r for r in R}
    cells = sorted({(r["seed"], r["target"], r["v"]) for r in R})
    M = np.array([[idx[c + (k,)][f"{method}@{a}"][2] for k in conds] for c in cells])
    return M, np.array([c[0] for c in cells]), conds


def worst_ci(R, method, a, B=2000):
    M, s, _ = table(R, method, a); seeds = np.unique(s); rows = [np.where(s == x)[0] for x in seeds]
    bs = [M[np.concatenate([rows[i] for i in rng.integers(0, len(seeds), len(seeds))])].mean(0).max() for _ in range(B)]
    return M.mean(0).max(), *np.percentile(bs, [2.5, 97.5])


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------- Figure 3: worst-case miss rate under standard liars (E-AOE25 + E-AOE27) ----------
RUNS3 = [("N-BaIoT", ["aoe21_nbaiot_sel2000_s67891011.json", "aoe21_nbaiot_sel2000_s151617181920212223242526272829.json"]),
         ("IoT-23", ["aoe21_iot23_sel2000_s67891011.json"]),
         ("IoT-23 dedup.", ["aoe21_iot23_dedup_sel2000_s67891011.json"])]
METH3 = [("OC-kNN", "OC-kNN", G1, ""), ("AOE-peer", "learned, peer rows", G3, "////"), ("Ada0.05", "AdaDetect", G2, ""),
         ("LB-v", "LB-v (ours)", BLUE, ""), ("LBA0.05-v", "LBA-v (ours)", ORANGE, "")]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=True)
print("FIG3 worst-case release % [95% seed-bootstrap CI]; conditions hon/flip/cw3/cw10/mimic10; |Sel| 2000", file=LOG)
for ax, a in zip(axs, ("0.001", "0.01")):
    w = .16
    for i, (name, files) in enumerate(RUNS3):
        R = load(*files); ns = len({r["seed"] for r in R}); nc = len({(r["seed"], r["target"], r["v"]) for r in R})
        for j, (m, lab, col, hat) in enumerate(METH3):
            v, lo, hi = worst_ci(R, m, a); x = i + (j - 2) * w
            ax.bar(x, v, w * .88, color=col, hatch=hat, edgecolor="white" if not hat else "#ffffff", linewidth=0,
                   label=lab if i == 0 else None)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK, lw=.3 * MM, capsize=1.5, capthick=.3 * MM)
            print(f"  a={a} {name:14s} ({ns} seeds, {nc} cells) {m:10s} {v:5.1f} [{lo:5.1f}, {hi:5.1f}]", file=LOG)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for n, _ in RUNS3]); ax.set_ylim(0, 105)
    ax.set_title(f"false-alarm level {float(a) * 100:g}%", fontsize=8, color=INK); ax.yaxis.grid(True, color="#e6e5e0", lw=.4); ax.set_axisbelow(True)
axs[0].set_ylabel("worst-case miss rate (%)")
axs[0].legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.22), fontsize=7, handlelength=1.2, columnspacing=1)
save(fig, "fig3_main")

# ---------- Figure 4: vetting-set size sweep (E-AOE18 seeds 0-2 + E-AOE24 seeds 6-8) ----------
MS = [200, 500, 1000, 2000]
RUNS4 = [("N-BaIoT", ["aoe18_nbaiot.json", "aoe18_nbaiot_s678.json"], BLUE, "o"),
         ("IoT-23", ["aoe18_iot23.json", "aoe18_iot23_s678.json"], ORANGE, "s")]
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3))
print("\nFIG4 |Sel| sweep, seeds 0-2 + 6-8 pooled; conditions hon/flip/cw3/cw10/mimic10", file=LOG)
for name, files, col, mk in RUNS4:
    R = load(*files)
    for ax, a in zip(axs[:2], ("0.001", "0.01")):
        ys = [table(R, f"LB-v{m}", a)[0].mean(0).max() for m in MS]; oc = table(R, "OC-kNN", a)[0].mean(0).max()
        ax.plot(MS, ys, color=col, marker=mk, ms=4, label=name)
        ax.axhline(oc, color=col, ls=(0, (3, 2)), lw=.35 * MM)
        print(f"  a={a} {name:8s} LB-v(m) {[round(y, 1) for y in ys]}  OC-kNN {oc:.1f}", file=LOG)
    cb = [np.mean([r[f"cB_v{m}@0.001"] for r in R]) for m in MS]
    axs[2].plot(MS, cb, color=col, marker=mk, ms=4, label=name)
    print(f"  a=0.001 {name:8s} mean Cal share in B {[round(c, 5) for c in cb]}", file=LOG)
for ax, t in zip(axs, ("miss rate, level 0.1%", "miss rate, level 1%", "budget used by B, level 0.1%")):
    ax.set_xscale("log"); ax.set_xticks(MS); ax.set_xticklabels(MS); ax.minorticks_off()
    ax.set_xlabel("vetting rows $m$"); ax.set_title(t, fontsize=8, color=INK)
axs[0].set_ylabel("worst-case miss rate (%)"); axs[0].set_ylim(0, 100); axs[1].set_ylim(0, 40)
axs[2].axhline(.0005, color=INK, ls=(0, (1, 1.5)), lw=.35 * MM); axs[2].text(200, .00022, "target", ha="left", fontsize=7)
axs[2].set_ylabel("Cal share inside B"); axs[2].set_ylim(0, None)
axs[0].legend(loc="lower left", fontsize=7); axs[1].text(210, 1, "dashed: no peers (OC-kNN)", fontsize=6.5, color=MUTED)
fig.tight_layout(w_pad=1.2); save(fig, "fig4_sel")

# ---------- Figure 5: rule-aware liars, LB-v vs certified LB-vc (E-AOE28, seeds 12-14 + 30-32) ----------
RUNS5 = [("N-BaIoT", "nbaiot"), ("IoT-23", "iot23"), ("IoT-23 dedup.", "iot23_dedup")]
CO = ["hon", "mimic10", "split100", "tail0.002", "tail0.005", "tail0.01", "tail0.02"]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
print("\nFIG5 rule-aware liars, a = 0.01, seeds 12-14 + 30-32 pooled", file=LOG)
allR = {k: load(f"aoe28_{k}_sel2000_s121314.json", f"aoe28_{k}_sel2000_s303132.json") for _, k in RUNS5}
R = [r for v in allR.values() for r in v]; w = .38
for j, (key, lab, col) in enumerate([("cB_v@0.01", "LB-v", BLUE), ("cB_vc@0.01", "LB-vc (certified)", AQUA)]):
    ys = [np.mean([r[key] for r in R if r["cond"] == c]) for c in CO]
    mx = [np.max([r[key] for r in R if r["cond"] == c]) for c in CO]
    axs[0].bar(np.arange(len(CO)) + (j - .5) * w, ys, w * .9, color=col, label=lab)
    print(f"  {lab:18s} mean Cal share in B per cond {dict(zip(CO, [round(y, 4) for y in ys]))}; max {dict(zip(CO, [round(y, 3) for y in mx]))}", file=LOG)
axs[0].axhline(.005, color=INK, ls=(0, (1, 1.5)), lw=.35 * MM); axs[0].text(6.6, .0052, "target: half the budget", ha="right", fontsize=7)
axs[0].set_xticks(range(len(CO))); axs[0].set_xticklabels(["honest", "mimic10", "split100", "tail .002", "tail .005", "tail .01", "tail .02"], rotation=30, ha="right")
axs[0].set_ylabel("Cal share inside B"); axs[0].set_title("budget taken by the kept peers (level 1%)", fontsize=8, color=INK)
axs[0].legend(loc="upper left", fontsize=7)
w = .25
for j, (m, lab, col) in enumerate([("OC-kNN", "OC-kNN", G2), ("LB-v", "LB-v", BLUE), ("LB-vc", "LB-vc", AQUA)]):
    for i, (name, k) in enumerate(RUNS5):
        v, lo, hi = worst_ci(allR[k], m, "0.01"); x = i + (j - 1) * w
        axs[1].bar(x, v, w * .88, color=col, label=lab if i == 0 else None)
        axs[1].errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK, lw=.3 * MM, capsize=1.5, capthick=.3 * MM)
        print(f"  {name:14s} {m:7s} worst-case release {v:5.1f} [{lo:5.1f}, {hi:5.1f}]", file=LOG)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels([n for n, _ in RUNS5]); axs[1].set_ylabel("worst-case miss rate (%)")
axs[1].set_title("worst case over the 7 conditions", fontsize=8, color=INK); axs[1].legend(fontsize=7, loc="upper right")
for ax in axs:
    ax.yaxis.grid(True, color="#e6e5e0", lw=.4); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.5); save(fig, "fig5_ruleaware")
LOG.close(); print(open(OUT / "fig_numbers.txt").read())
