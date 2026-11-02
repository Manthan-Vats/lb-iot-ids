# Pre-registration log

This is the part of the project's dated, append-only research plan that covers the paper (experiments E-AOE15 to
E-AOE28 and B7), copied verbatim. Each "Pre-registration" entry was written before its run and fixed the code, seeds,
conditions and numeric pass thresholds; each "OUTCOME" entry was appended after the run. Earlier entries of the plan
concern a different research direction and are not part of the paper.

Reading guide: "release" in this log means the miss rate (the share of attack rows that were not flagged); see the
vocabulary table in `README.md` for the other internal names. Notes it refers to (for example `notes/THEORY_vetting.md`)
are working notes whose content is in the paper's theory section and appendix.

---

## Pre-registration E-AOE15: local budget rule LB (2026-11-01; theory in notes/THEORY_vetting.md; nothing run)
Score s(x) = d_A(x) + BIG * 1[x in B], B = union over kept sources of {x : d_S(x) < d_A(x)} (Z space, d_A = NN distance
to Ntr 300, d_S = NN distance to the source's rows). Conformal on Cal. Proven (by hand): for x outside B,
p_LB(x) <= p_OC(x) + #Cal-in-B / (n + 1), for ANY peer behaviour. LB = all sources kept. LB-v (per level a): drop
sources, largest Sel footprint first, until #Sel rows in the union B <= floor(.5 a |Sel|) (|Sel| = 500, disjoint
from Ntr and Cal: 2 rows at a .01, 0 at a .001). Split Ntr b[:300], Cal b[300:1300], Sel b[1300:1800], lp b[1800:];
MINV 2300 uncapped; first 2 targets; seeds 0-2; conditions hon, flip, cw3, cw10, mimic10. Methods OC-kNN, AOE-peer,
LB, LB-v. Datasets with eligible victims (logged).
B1 FA pooled <= a + .2 / a + 1 for all methods.
B2 (main, robustness floor) worst case over conditions of LB-v <= OC-kNN release + 3 at a .01, every dataset run.
B3 (honest gain) LB-v hon release <= OC-kNN - 5 at a .01 on >= 1 dataset, and <= OC-kNN + 1 at a .001 on all.
B4 LB-v worst case < AOE-peer worst case at both a, every dataset run.
B5 footprints: median Sel footprint of cw10 / mimic10 liar sources > median of honest sources, every dataset run.
B6 (bound check, deterministic) in every cell LB release on attacks outside B <= OC-kNN release at level a - cB;
   not tested numerically here (follows from the proof); instead log cB and report mean cB per condition.
- E-AOE15 addendum (written BEFORE any full-run result was read; smoke run on N-BaIoT seed 0 was seen): exact-duplicate
  audit (seed 0, raw features): share of held-out test attacks that exactly equal some shared target row = .004
  N-BaIoT; .932 and .816 IoT-23 (targets 0, 1). IoT-23 conn-log flows repeat verbatim, so every peer-using method
  (AOE-peer included, all earlier E-AOE runs) is helped by verbatim repeats there. Added analysis B7: rerun IoT-23
  with test attacks restricted to rows with NO exact duplicate among any peer-shared row; B2-B4 rescored on it.
  If B2/B4 hold on full data but fail on de-duplicated data, the IoT-23 claim is limited to "verbatim-repeat" attacks.
- E-AOE15 OUTCOME (2026-11-01; no tracebacks; TON no eligible victim -> empty; N-BaIoT 6 pairs, IoT-23 45 pairs).
  Release OC-kNN | AOE-peer | LB | LB-v:
   N-BaIoT a .01: hon 20.2|0.3|16.7|3.5 flip 20.2|16.8|16.7|3.5 cw3 20.2|29.6|16.7|3.5 cw10 20.2|30.0|16.7|3.5
     mimic10 20.2|46.7|16.7|3.5 -> WORST 20.2|46.7|16.7|3.5. a .001 WORST 68.4|90.7|68.4|68.4 (hon LB-v 51.7).
   IoT-23 a .01: hon 17.2|2.7|11.6|0.6 flip 17.2|15.3|11.0|3.0 cw3 17.2|13.8|17.4|3.4 cw10 17.2|12.9|18.0|4.1
     mimic10 17.2|26.8|19.5|3.2 -> WORST 17.2|26.8|19.5|4.1. a .001 WORST 70.1|62.9|71.2|53.4 (hon LB-v 50.2,
     AOE-peer hon 33.3).
  FA pooled max .16% (a .001), 1.11% (a .01, LB on N-BaIoT). Cal share inside B after vetting (a .01) .0025-.0060
  (target .005; Sel-estimate noise). Kept at a .01: liars cw10 .17 / .41, mimic .07 / .33; honest .48-.54 / .86-.90.
  Median Sel footprint liar vs honest: cw10 .020 vs .003, mimic .082 vs .003 (N-BaIoT); .006 vs 0, .011 vs 0 (IoT-23).
  B1 S. B2 S (worst 3.5 vs OC 20.2; 4.1 vs 17.2). B3 S. B4 S 4/4. B5 S. ALL PREDICTIONS SUPPORTED.
  Reading: at a .01 the budget-accounted local rule turns liars from a 2.3x / 1.6x power loss (AOE-peer worst vs OC)
  into a 5.8x / 4.2x power GAIN over anchor-only in the worst case. At a .001 the gain is smaller (cap = 0 Sel rows,
  resolution 1/500 > a) and AOE-peer is better when all peers are honest (IoT-23 33.3 vs 50.2): the 1/a law again.
  Pending before any claim: B7 de-duplicated IoT-23; fresh-seed confirmation; TON (needs a smaller-Sel variant).

## Pre-registration E-AOE16: fresh-seed confirmation of LB-v (2026-11-01; before B7 results)
Identical code (aoe15.py, no dedup), seeds 3, 4, 5, N-BaIoT + IoT-23. Same B1-B5 criteria as E-AOE15, scored afresh.
Claim rule: LB-v is claimed as the defence only if B2 and B4 hold here AND in B7 (de-duplicated IoT-23).

## Pre-registration E-AOE17: LB-v on TON_IoT with a smaller trusted set (2026-11-01; nothing run)
TON has no victim with >= 2300 uncapped benign rows outside the holders, so use |Sel| = 200 (aoe15.py nsel=200,
MINV 1600; cap at a .01 = floor(.5 * .01 * 200) = 1 Sel row; a .001 = 0). Seeds 0-5 (6 seeds, because TON has few
victims). Same conditions/methods. Predictions: B1 FA; B2' worst-case LB-v <= OC-kNN + 3 at a .01; B4' LB-v worst
case < AOE-peer worst case at a .01. The theory predicts a WEAKER floor here (cap resolution 1/200 = .005 = a/2).

## Pre-registration E-AOE18: trusted-set size sweep for LB-v (tests Theorem 3 / the 1/a law; 2026-11-01; nothing run)
Theory: notes/THEORY_LB.md. Code aoe18.py (= aoe15 with Sel 2000, MINV 3400 uncapped, seeds 0-2, first 2 targets,
N-BaIoT + IoT-23). LB-v(m) for m in {200, 500, 1000, 2000} (nested Sel prefixes; cap floor(.5 a m)); LB-vs = split
variant (source order from Sel[:1000], cap floor(.5 a 1000) checked on Sel[1000:2000]; the version Theorem 3 covers).
R1 FA pooled <= a + .2 / a + 1, all methods.
R2 (law, a .001) worst case over conditions of LB-v2000 <= LB-v500's - 5 on >= 1 dataset, and not worse by > 3 on any.
R3 (law, budget control) mean post-vetting Cal share in B at a .001 decreases from m = 200 to m = 2000 on both
   datasets, and at m = 2000 is <= .001 (= a) on both.
R4 (a .01) LB-v2000 worst case <= OC-kNN + 3 on both (floor holds at larger m too).
R5 LB-vs worst case within 3 points of LB-v2000 at both a on both (the provable variant costs little).
- Novelty check for LB (2026-11-01, web, abstracts only): no prior work found combining untrusted/Byzantine labelled
  outlier contributors with a power-damage guarantee. Closest: Liang, Sesia, Sun, arXiv 2208.11111 (integrative
  conformal p-values with labelled outliers; abstract does not address mislabelled/adversarial outliers; journal
  venue JRSSB 2024 from an earlier session: VERIFY before citing); Bashari, Sesia, Romano arXiv 2502.04807
  (contaminated REFERENCE inliers, non-adversarial, labelling budget). Neither has the budget-accounting bound.

## Pre-registration E-AOE19: AdaDetect-style peer-free baseline (2026-11-01; nothing run)
aoe19.py: MLP separating Ntr from an unlabelled mixture (Cal + victim's benign test + held-out attacks at 5%
prevalence), conformal p-values from Cal (Marandon et al.'s transductive construction; FA valid because Cal and
benign test rows are treated symmetrically). Same seeds 0-2, victims and split rule as E-AOE15; peer-free, so one
number per pair. A1 FA pooled <= a + .2 / a + 1. A2 LB-v (E-AOE15) worst case at a .01 < AdaDetect release at a .01 on
both datasets. A3 (reported, no prediction) a .001 comparison. Note: the 5% prevalence is a design choice fixed
here; AdaDetect's power grows with prevalence, so this is one operating point, not its best case.

## OUTCOME E-AOE14 (2026-11-01; aoe13_{nbaiot,iot23}_s345.json, 30 + 225 rows, 0 Tracebacks)
Release % (mean per condition; worst = max over hon/flip/cw3/cw10/mimic10). FA pooled = alarms / benign test rows.
| data a | OC-kNN | AOE-T | AOE-peer hon / worst | AOE-peer+T worst | max FA |
| N-BaIoT .001 | 68.2 | 50.8 | 99.9 / 100.0 | 100.0 | .125% |
| N-BaIoT .01  | 16.7 | 32.8 | 12.5 / 57.4 | 62.7 | 1.21% |
| IoT-23 .001  | 80.1 | 35.7 | 34.7 / 67.7 | 68.6 | .058% |
| IoT-23 .01   | 10.0 | 12.4 | 3.4 / 19.0 | 19.2 | .74% |
S1 S. S2 S (35.7 <= 37.7 and <= 60.1). S3 S (.001: 35.7 < 67.7, 68.6; .01: 12.4 < 19.0, 19.2). S4 S (32.8 > 16.7,
predicted failure replicated). AOE-T is CONFIRMED on IoT-23 as a peer-free detector that matches honest peers at a .001.
Not a universal win: worse than OC-kNN at a .01 on both datasets; better than OC at a .001 on both (post hoc on N-BaIoT,
not pre-registered there). Next: AOE-T as the base score inside LB (Theorem 2 holds for ANY base score).

## Pre-registration E-AOE20: LB with a trusted-attack base score (2026-11-01; nothing run)
Theorem 2 (notes/THEORY_LB.md) holds for ANY base score fixed before Cal: for x outside B, p(x) <= p_base(x) + c/(n+1).
So the budget accounting can wrap a stronger peer-free base. Code aoe20.py (derived from aoe15; same B, same LB-v
dropping rule and caps, kappa .5, |Sel| 500). Bases: d_A (= LB-v), s_T = AOE-T MLP on 200 trusted non-target attack
rows vs Ntr (as in aoe13; rows never shared by any peer, not from the victim) (= LBT-v), and COMB = max of the two
scores' empirical CDF ranks computed on Sel (disjoint from Cal, so calibration stays valid) (= LBC-v; peer-free
version without B = COMB). Runs: N-BaIoT and IoT-23 seeds 0-5; IoT-23 dedup seeds 0-2.
T1 FA pooled <= a + .2 / a + 1 for every method.
T2 LBC-v worst-case release <= min(LB-v worst, AOE-T) + 2 at both a, both datasets, both IoT-23 variants.
T3 LBC-v worst-case release at a .001 <= LB-v worst-case - 5 on IoT-23 (non-dedup) AND on N-BaIoT.
T4 LBT-v worse than LB-v at a .01 on N-BaIoT (predicted from E-AOE14 S4).
LBC-v replaces LB-v as the headline method only if T1, T2, T3 all hold; otherwise LB-v stays and this is reported.

## OUTCOME B7 (de-duplicated IoT-23; aoe15_iot23_dedup.json, 225 rows, 0 Tracebacks, 2026-11-01)
Test attacks kept after removing every row with an exact copy among rows any peer can share: 282-947 of 5000 per cell.
Release % worst case over conditions (full IoT-23 from E-AOE15 in brackets):
| a | OC-kNN | AOE-peer worst (hon) | LB worst | LB-v worst (hon) |
| .001 | 66.5 [70.1] | 61.5 (33.5) [62.9] | 70.1 [71.2] | 51.8 (50.2) [53.4] |
| .01  | 15.9 [17.2] | 27.7 (3.0) [26.8]  | 21.7 [19.5] | 6.2 (0.2) [4.1] |
FA identical to full run (benign rows unchanged; max .085% / .75%).
B2 S (6.2 <= 18.9). B3 S (0.2 <= 10.9; 50.2 <= 67.5). B4 S (51.8 < 61.5; 6.2 < 27.7).
Reading: LB-v's gain does NOT come from verbatim repeats; removing them moves its worst case by under 2.1 points.

## OUTCOME E-AOE16 (fresh seeds 3-5; aoe15_{nbaiot,iot23}_s345.json, 30 + 225 rows, 0 Tracebacks, 2026-11-01)
Release % worst case over hon/flip/cw3/cw10/mimic10 (hon in brackets):
| data a | OC-kNN | AOE-peer worst (hon) | LB worst | LB-v worst (hon) | max FA |
| N-BaIoT .001 | 68.2 | 100.0 (99.9) | 68.2 | 68.2 (51.6) | .125% |
| N-BaIoT .01  | 16.7 | 58.6 (12.5)  | 16.7 | 8.0 (0.0)   | 1.28% |
| IoT-23 .001  | 80.1 | 72.4 (34.7)  | 80.2 | 69.4 (62.1) | .058% |
| IoT-23 .01   | 10.0 | 20.3 (3.4)   | 13.7 | 3.9 (1.1)   | .77% |
B1 S. B2 S (8.0 <= 19.7; 3.9 <= 13.0). B3 S (.01 hon 0.0 <= 11.7; .001 hon 51.6 <= 69.2, 62.1 <= 81.1).
B4 S (N-BaIoT 68.2 < 100, 8.0 < 58.6; IoT-23 69.4 < 72.4 (narrow), 3.9 < 20.3).
B5 S (median Sel footprint liar vs honest: N-BaIoT cw10 .023 vs .002, mimic10 .086 vs .002; IoT-23 .006 vs 0, .011 vs 0).
CLAIM RULE MET (B2, B4 hold here and in B7): LB-v is the defence.
Honest caveats: (1) post-vetting Cal share in B reached .013 (N-BaIoT) and .017 (IoT-23) in single cells at a .01,
above the a/2 target: one shared Sel for ordering and testing (Remark 2) plus |Sel| = 500; E-AOE18 tests the split
variant and larger Sel. (2) At a .001 on IoT-23 LB-v's honest-peer gain is small (62.1 vs 80.1 no-peers) while the
learned detector with honest peers reaches 34.7: the strict level is where LB-v leaves most on the table (E-AOE20).

## OUTCOME E-AOE17 (TON, |Sel| 200, seeds 0-5; aoe15_toniot_sel200_s012345.json, 60 rows, 0 Tracebacks, 2026-11-01)
| a | OC-kNN | AOE-peer worst (hon) | LB worst | LB-v worst (hon) | max FA |
| .001 | 23.6 | 37.4 (4.9) | 23.6 | 11.9 (5.4) | .031% |
| .01  | 0.0  | 0.1 (0.0)  | 0.0  | 0.0 (0.0)  | .41% |
B1 S. B2' S (0.0 <= 3.0). B4' S only formally (0.0 < 0.1): at a .01 TON is saturated (no-peers already misses 0%),
so this run is UNINFORMATIVE for the pre-registered a .01 claims. Post hoc (not pre-registered), a .001: LB-v worst
11.9 vs 23.6 no-peers vs 37.4 learned detector; same direction as the other datasets. Reported as post hoc only.

## OUTCOME E-AOE18 (Sel size sweep; aoe18_{nbaiot,iot23}.json, 30 + 135 rows, 0 Tracebacks, 2026-11-01)
Worst-case release % over conditions:
| data a | OC-kNN | AOE-peer | LB-v200 | LB-v500 | LB-v1000 | LB-v2000 | LB-vs |
| N-BaIoT .001 | 68.4 | 92.7 | 68.4 | 68.4 | 51.7 | 51.7 | 51.7 |
| N-BaIoT .01  | 20.2 | 49.5 | 20.2 | 3.5  | 3.5  | 3.5  | 3.5 |
| IoT-23 .001  | 82.7 | 71.6 | 67.9 | 58.2 | 51.7 | 55.2 | 63.8 |
| IoT-23 .01   | 23.9 | 28.7 | 6.2  | 4.7  | 4.0  | 3.7  | 3.9 |
Mean post-vetting Cal share in B at a .001 (m = 200 / 500 / 1000 / 2000 / split): N-BaIoT .0059 / .0009 / 0 / 0 / 0;
IoT-23 .0045 / .0019 / .0010 / .00102 / .00078. At a .01: N-BaIoT .0095 -> .0040; IoT-23 .0062 -> .0041 (max cell .012).
R1 S (max FA .161% / 1.10%). R2 S (N-BaIoT 51.7 vs 68.4; IoT-23 55.2 vs 58.2, not worse).
R3 R on the strict letter: decreases on both, but IoT-23 at m 2000 is .00102 > .001 (N-BaIoT 0). The split variant
meets it (.00078). R4 S (3.5 <= 23.2; 3.7 <= 26.9). R5 R: LB-vs is 8.6 points worse than LB-v2000 on IoT-23 at a .001
(63.8 vs 55.2; within 0.2 elsewhere); the provable variant pays at the strict level on IoT-23.
Reading: the 1/a law shows as predicted: at a .01 the floor appears once m reaches 500 on N-BaIoT (m = 200 gives cap
floor(.5 * .01 * 200) = 1 row and no gain); at a .001 the N-BaIoT gain appears only at m >= 1000. Gains past m = 1000 are
nil or slightly negative (IoT-23 51.7 -> 55.2 at .001).

## OUTCOME E-AOE19 (AdaDetect-style, 5% prevalence; aoe19_{nbaiot,iot23}.json, 6 + 45 rows, 0 Tracebacks, 2026-11-01)
| data | a .001 release | a .01 release | FA .001 / .01 | LB-v worst (E-AOE15) .001 / .01 |
| N-BaIoT | 55.4 | 5.5 | .107% / .96% | 68.4 / 3.5 |
| IoT-23  | 56.5 | 4.9 | .071% / .68% | 53.4 / 4.1 |
A1 S. A2 S, but by small margins (3.5 < 5.5; 4.1 < 4.9). A3 (reported): at a .001 AdaDetect beats LB-v's worst case on
N-BaIoT by 13.0 points and loses on IoT-23 by 3.1.
Reading (important, negative for the "peers help a lot" story): a strong PEER-FREE transductive baseline closes most
of the gap. It needs the attack to be present in the unlabelled traffic it trains on (here 5%); LB-v does not. They
use different information, and Theorem 2 allows ANY base score that treats Cal and benign test rows symmetrically, so
the natural next step is LB-v on top of the AdaDetect score (E-AOE21).

## Pre-registration E-AOE21: LB-v on top of AdaDetect, two prevalences (2026-11-01; nothing run)
aoe21.py (derived from aoe15; same seeds 0-2, victims, conditions, B, LB-v keep sets). Bases: AdaDetect score
(Ntr vs mixture Cal + benign test + attack subsample at prevalence pv in {.05, .005}; Cal and benign test symmetric, B
does not use Cal, so Theorem 1 and Theorem 2 still apply). LBA{pv}-v = AdaDetect score + BIG * 1[B_kept].
D1 FA pooled <= a + .2 / a + 1 for every method.
D2 LBA{pv}-v worst case <= min(LB-v worst, Ada{pv}) + 2 at both a, both pv, both datasets.
D3 (peers matter when the attack is locally rare) at pv .005, LBA-v worst case <= Ada.005 - 5 at a .01 on both datasets.
D4 (reported) Ada.005 vs Ada.05 release.
If D2 and D3 hold, LBA-v is the headline method; if only D2, it is reported as "never worse than the better of the two".

## OUTCOME E-AOE20 (aoe20_{nbaiot,iot23}_s012345.json + aoe20_iot23_dedup.json; 60 + 450 + 225 rows; 0 Tracebacks)
Worst-case release % over conditions:
| data a | OC-kNN | AOE-T | COMB | LB-v | LBT-v | LBC-v | max FA |
| N-BaIoT .001 | 68.3 | 62.8 | 100.0 | 68.3 | 58.3 | 91.7 | .134% |
| N-BaIoT .01  | 18.5 | 37.6 | 14.3  | 5.8  | 14.3 | 4.4  | 1.15% |
| IoT-23 .001  | 75.1 | 35.3 | 91.2  | 61.4 | 34.2 | 67.7 | .070% |
| IoT-23 .01   | 13.6 | 12.2 | 22.0  | 3.7  | 5.5  | 5.8  | .76% |
| IoT-23 dedup .001 | 66.4 | 51.0 | 88.2 | 51.7 | 42.4 | 65.3 | .085% |
| IoT-23 dedup .01  | 15.9 | 16.0 | 24.0 | 6.3  | 6.8  | 7.8  | .75% |
T1 S. T2 R (LBC-v at a .001 is far worse than both parts, e.g. 91.7 vs 62.8 on N-BaIoT). T3 R. T4 S (14.3 > 5.8).
LBC-v is NOT adopted; LB-v stays. Cause of the COMB failure (design error, found after the run): the Sel-based CDF ranks
have resolution 1/|Sel| = .002 > a = .001, so every row past the Sel maximum ties at rank 1 and COMB cannot rank the top
.1%. Should have been caught before running; recorded as a lesson (rank transforms need |Sel| >> 1/a).
Post hoc (NOT claimed): LBT-v at a .001 beats LB-v on every run (58.3 vs 68.3; 34.2 vs 61.4; dedup 42.4 vs 51.7) and is
<= AOE-T + 1 everywhere; at a .01 it is worse than LB-v. Needs fresh-seed confirmation (E-AOE22).

## Pre-registration E-AOE22: confirm LBT-v at the strict level on fresh seeds (2026-11-01; nothing run)
Same code aoe20.py, seeds 6, 7, 8 (never used by any LB run), N-BaIoT + IoT-23 + IoT-23 dedup. Rule fixed in advance:
LB* = LBT-v at a .001 and LB-v at a .01 (base chosen by level, before seeing these seeds).
U1 FA pooled <= a + .2 / a + 1, all methods.
U2 a .001: LBT-v worst case <= LB-v worst case - 5 on N-BaIoT, IoT-23 and IoT-23 dedup.
U3 a .001: LBT-v worst case <= AOE-T + 2 on all three (liars cannot push it below the peer-free trusted library).
U4 a .001: LBT-v worst case <= OC-kNN - 5 on all three.
LB* is claimed only if U1-U4 all hold; otherwise LB-v alone is the method and LBT-v is reported as seed-dependent.

## OUTCOME E-AOE21 (aoe21_{nbaiot,iot23}.json, 30 + 225 rows, 0 Tracebacks, 2026-11-01)
Worst-case release % over conditions (AdaDetect is peer-free, so one number):
| data a | OC-kNN | LB-v | Ada.05 | LBA.05-v | Ada.005 | LBA.005-v | max FA |
| N-BaIoT .001 | 68.4 | 68.4 | 55.4 | 47.5 | 51.4 | 47.6 | .161% |
| N-BaIoT .01  | 20.2 | 3.5  | 5.5  | 0.9  | 3.7  | 0.8  | 1.02% |
| IoT-23 .001  | 70.1 | 53.4 | 56.5 | 45.5 | 58.2 | 47.9 | .085% |
| IoT-23 .01   | 17.2 | 4.1  | 4.9  | 3.2  | 10.6 | 4.6  | .75% |
D1 S. D2 S in all 8 cells; in fact LBA-v's WORST case is below the better of LB-v and AdaDetect in every cell.
D3 R: IoT-23 S (4.6 <= 5.6) but N-BaIoT R (0.8 vs 3.7 - 5 = -1.3; AdaDetect at .005 is already at 3.7, so the 5-point
margin was unreachable: a badly set threshold, but it is the pre-registered one).
Per the rule: LBA-v is reported as "never worse than the better of LB-v and AdaDetect" (supported), not as the
headline on D3. Fresh-seed confirmation: E-AOE23.
Surprise (reported, no prediction): AdaDetect at 0.5% prevalence is not uniformly weaker than at 5% (N-BaIoT better,
IoT-23 worse at a .01).

## Pre-registration E-AOE23: confirm LBA-v on fresh seeds and de-duplicated IoT-23 (2026-11-01; nothing run)
aoe21.py unchanged, seeds 6, 7, 8; N-BaIoT, IoT-23, IoT-23 dedup.
V1 FA pooled <= a + .2 / a + 1, all methods.
V2 (= D2) LBA{pv}-v worst case <= min(LB-v worst, Ada{pv}) + 2 at both a, both pv, all three runs.
V3 a .001: LBA.05-v worst case <= LB-v worst case - 5 on all three runs.
V4 a .001: LBA.05-v worst case < Ada.05 on all three runs (peers add something even over a strong peer-free base).
LBA-v becomes the headline method if V1-V4 hold.

## OUTCOME E-AOE22 (aoe20 seeds 6-8; aoe20_{nbaiot,iot23,iot23_dedup}_s678.json, 30 + 225 + 225 rows, 0 Tracebacks)
Worst-case release %:
| run a | OC-kNN | AOE-T | LB-v | LBT-v | max FA |
| N-BaIoT .001 | 66.7 | 73.0 | 66.7 | 62.3 | .196% |
| N-BaIoT .01  | 33.4 | 47.4 | 43.3 | 25.4 | 1.11% |
| IoT-23 .001  | 72.1 | 32.8 | 69.2 | 32.5 | .073% |
| IoT-23 .01   | 9.9  | 13.0 | 4.0  | 5.7  | .73% |
| dedup .001   | 69.7 | 45.5 | 67.3 | 44.8 | .073% |
| dedup .01    | 7.2  | 18.7 | 2.5  | 5.9  | .73% |
U1 S. U2 R (N-BaIoT 62.3 vs 66.7 - 5 = 61.7, misses by 0.6; IoT-23 and dedup S by 36.7 / 22.5). U3 S. U4 R (N-BaIoT
62.3 > 61.7). LB* (level-dependent base) is NOT claimed. LBT-v at a .001 on IoT-23 replicates strongly (now 2 seed sets).
IMPORTANT NEGATIVE for LB-v itself (not a pre-registered check of this run, reported because it bears on B2): on N-BaIoT
seeds 6-8 at a .01 LB-v's worst case (43.3) is 9.9 points WORSE than OC-kNN (33.4), breaking the B2 floor (+3).
Cause, from the cells: seed 7 victim 6 mimic10: one mimic cluster kept; its B holds 8 of 1000 Cal rows (.008 against
target .005) while only <= 2 of 500 Sel rows (cap) - Sel under-sampled it; OC-kNN flags the whole family at p in
(.002, .01], so losing .008 of budget drops it from 0% to 100% released (N-BaIoT families are all-or-nothing).
This is exactly what Theorem 3 allows at m = 500: Pr[Bin(500, .008) <= 2] ~ .24 per liar cluster. It is a predicted
failure of a too-small vetting set, not a bug; E-AOE18 (m = 1000-2000) shows no such breach on seeds 0-2. Consequence for
the method: the default |Sel| must be >= 1000 at a .01 (cap 5 rows), or kappa smaller. Pre-register E-AOE24 to test.

## Pre-registration E-AOE24: does a larger vetting set remove the N-BaIoT floor breach? (2026-11-01; nothing run)
aoe18.py unchanged (Sel 2000, nested LB-v{200,500,1000,2000}, LB-vs), seeds 6, 7, 8, N-BaIoT and IoT-23. Victim set can
differ from E-AOE22 (MINV 3400), so the breached cell may not be present; W3 is scored only if it is.
W1 FA pooled <= a + .2 / a + 1.
W2 a .01: LB-v1000 AND LB-v2000 worst case <= OC-kNN + 3 on both datasets.
W3 (theory check) if seed 7 victim 6 target 0 appears on N-BaIoT: LB-v500 mimic10 release > OC-kNN + 3 there and
   LB-v2000 <= OC-kNN + 3 there.
If W2 fails, the floor claim of LB-v is withdrawn and the paper states the floor only as Theorem 2 + Theorem 3 (in
probability), not as an empirical worst-case guarantee.

## OUTCOME E-AOE23 (aoe21 seeds 6-8; aoe21_{nbaiot,iot23,iot23_dedup}_s678.json, 30 + 225 + 225 rows, 0 Tracebacks)
Worst-case release %:
| run a | OC-kNN | LB-v | Ada.05 | LBA.05-v | Ada.005 | LBA.005-v |
| N-BaIoT .001 | 66.7 | 66.7 | 66.3 | 61.2 | 16.0 | 15.7 |
| N-BaIoT .01  | 33.4 | 43.3 | 4.1  | 14.0 | 0.2  | 1.6  |
| IoT-23 .001  | 72.1 | 70.5 | 51.4 | 50.7 | 50.4 | 49.3 |
| IoT-23 .01   | 9.9  | 4.0  | 2.5  | 1.2  | 6.8  | 2.5  |
| dedup .001   | 69.7 | 69.4 | 51.0 | 51.9 | 54.4 | 56.3 |
| dedup .01    | 7.3  | 2.5  | 1.8  | 1.0  | 3.7  | 1.6  |
max FA .196% / 1.15%. V1 S. V2 R: N-BaIoT a .01 pv .05, LBA worst 14.0 > min(43.3, 4.1) + 2 = 6.1 (all other 11 cells S;
dedup .001 pv .005 only just: 56.3 <= 56.4). V3 S (61.2 <= 61.7; 50.7 <= 65.5; 51.9 <= 64.4). V4 R (dedup 51.9 > 51.0;
IoT-23 50.7 < 51.4 only just). LBA-v is NOT the headline.
Reading: the V2 breach is the SAME cell as E-AOE22's LB-v breach (seed 7, victim 6, mimic10): a liar cluster passes a
500-row vetting set and takes ~.008 of the .01 budget; the wrapper then loses whatever the base detected in the p range
it stole. Theorem 2 says this is the price; Theorem 3 says the price is controlled only when |Sel| is large. Second
reading (negative): over a strong peer-free base, the strict-level gain from peers on IoT-23 is ~1 point (V4).
Both failures point at one design parameter (|Sel|). Next: E-AOE24 (running) and a decisive E-AOE25 with |Sel| 2000.

## Pre-registration E-AOE25: decisive run, |Sel| 2000 (2026-11-01; nothing run)
aoe21.py nsel=2000 (MINV 3400), seeds 6-11 (6-8 contain the E-AOE22/23 breach; 9-11 never used), N-BaIoT, IoT-23,
IoT-23 dedup. Methods OC-kNN, AOE-peer, LB, LB-v, Ada{.05,.005}, LBA{.05,.005}-v. Cap floor(.5 a 2000) = 1 / 10 rows.
X1 FA pooled <= a + .2 / a + 1, all methods, all runs.
X2 (floor) a .01: LB-v worst <= OC-kNN + 3, and LBA{pv}-v worst <= Ada{pv} + 2 for both pv, all three runs.
X3 (robustness) LB-v worst < AOE-peer worst at both a, all three runs.
X4 (wrapper) LBA{pv}-v worst <= min(LB-v worst, Ada{pv}) + 2 at both a, both pv, all three runs.
Decision: if X1-X4 hold, the method is "LB-v wrapper with |Sel| >= 2000" (base d_A, or AdaDetect when unlabelled
traffic is available) and the paper's empirical floor claim stands with that requirement. If X2 fails, the floor is
claimed only in probability (Theorems 2-3) and the paper says so.

## OUTCOME E-AOE24 (aoe18 seeds 6-8; aoe18_{nbaiot,iot23}_s678.json, 30 + 135 rows, 0 Tracebacks)
Worst-case release %:
| data a | OC-kNN | AOE-peer | LB-v200 | LB-v500 | LB-v1000 | LB-v2000 | LB-vs |
| N-BaIoT .001 | 66.7 | 100.0 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| N-BaIoT .01  | 33.4 | 55.4  | 27.7 | 27.7 | 11.0 | 16.8 | 16.8 |
| IoT-23 .001  | 87.1 | 69.0  | 87.8 | 86.2 | 77.5 | 73.3 | 75.3 |
| IoT-23 .01   | 16.5 | 18.7  | 9.6  | 6.6  | 1.8  | 1.8  | 2.3 |
W1 S (max FA .214% / 1.23%). W2 S (N-BaIoT 11.0, 16.8 <= 36.4; IoT-23 1.8 <= 19.5).
W3 R / not reproducible: the cell (seed 7, victim 6, target 0) is present but LB-v500 releases 0% there in every
condition. With |Sel| 2000 the mimic liar's clusters are built from a different leftover pool (lp starts after Sel),
so this run cannot recreate the E-AOE22 liar; the breach is neither confirmed nor refuted here. E-AOE25 tests the floor
on those seeds with the wrapper code directly.
Also reported: on N-BaIoT a .01 LB-v2000 (16.8) is worse than LB-v1000 (11.0) under ALL conditions including hon, i.e.
a larger vetting set can drop honest sources too (cap 10 rows at m 2000 vs 5 at 1000 is the same .5 a share but more
honest sources reach it). IoT-23 at a .001: LB-v's gain over no-peers is modest (73.3 vs 87.1) vs AOE-peer hon 50.9.

## OUTCOME E-AOE25 (decisive, |Sel| 2000, seeds 6-11; aoe21_{nbaiot,iot23,iot23_dedup}_sel2000_s67891011.json,
## 60 + 270 + 270 rows, 0 Tracebacks, 2026-11-01)
Worst-case release % over hon/flip/cw3/cw10/mimic10:
| run a | OC-kNN | AOE-peer | LB-v | Ada.05 | LBA.05-v | Ada.005 | LBA.005-v | max FA |
| N-BaIoT .001 | 60.5 | 100.0 | 52.2 | 52.1 | 49.4 | 24.2 | 20.4 | .170% |
| N-BaIoT .01  | 16.7 | 59.3  | 8.4  | 2.8  | 1.8  | 1.2  | 0.1  | 1.08% |
| IoT-23 .001  | 83.1 | 69.1  | 64.1 | 63.5 | 52.4 | 61.7 | 51.7 | .066% |
| IoT-23 .01   | 17.0 | 21.2  | 2.8  | 5.1  | 2.2  | 10.2 | 3.8  | .69% |
| dedup .001   | 81.7 | 68.0  | 62.3 | 59.3 | 47.9 | 59.9 | 51.1 | .066% |
| dedup .01    | 11.7 | 21.2  | 2.2  | 2.7  | 1.8  | 6.3  | 3.1  | .69% |
X1 S. X2 S (LB-v 8.4 / 2.8 / 2.2 vs OC+3 19.7 / 20.0 / 14.7; LBA <= Ada + 2 in all 6 cells).
X3 S (all 6 cells). X4 S (all 12 cells; tightest IoT-23 a .01 pv .005: 3.8 <= 4.8).
DECISION (pre-registered): the method is the LB-v wrapper with |Sel| >= 2000; base d_A when no unlabelled traffic is
available, AdaDetect when it is. LBA.05-v's worst case under every liar type is BELOW peer-free AdaDetect in all 6
cells (strict level: 49.4 vs 52.1; 52.4 vs 63.5; 47.9 vs 59.3), i.e. lying peers cannot drag the wrapper below the
best peer-free detector, while honest peer evidence adds up to 11 points at a .001 on IoT-23.
Caveats kept: (1) N-BaIoT has 2 eligible victims per seed at MINV 3400 (60 rows): small; (2) TON excluded (no victim
has 3400 benign rows); (3) the 500-row breach (E-AOE22/23) stands as a documented failure mode of small vetting sets.

## Pre-registration E-AOE26: rule-aware liars against the final method (2026-11-01; nothing run)
Purpose: solidify (a reviewer's first question). The liar knows the LB-v rule and the cap, and (stronger than the
threat model) uses the victim's own held-out benign rows ranked by rarity under the victim's anchor. Code aoe26.py
(= aoe21 + new conditions), nsel=2000, seeds 12, 13, 14 (never used), N-BaIoT, IoT-23, IoT-23 dedup.
Conditions: hon; mimic10 (old); split100 (100 small k-means clusters as 100 sources, to slip under per-source checks);
tail{q} for q in {.002, .005, .01, .02}: the victim's rarest q share of held-out benign rows, split into <= 10
sources (aimed at the alarm tail, sized below / at / above the a .01 cap of .5 a = .005).
Y1 FA pooled <= a + .2 / a + 1, all methods, all conditions.
Y2 (floor under rule-aware liars) a .01: LB-v worst over all conditions <= OC-kNN + 3; LBA.05-v worst <= Ada.05 + 2;
   all three runs.
Y3 (budget) mean post-vetting Cal share in B at a .01 over the tail and split conditions <= .0075 (1.5 x target .005)
   on every run; max cell reported.
Y4 (reported, no prediction) which condition is worst, and the worst-case release vs E-AOE25.
If Y2 fails, the paper reports the attack that breaks the floor, the size of the damage, and Theorem 2's bound for it.

## Uncertainty for E-AOE25 (analysis added after the run; bootstrap over cells, 2000 resamples, 2026-11-01)
Statistic: worst case (max over conditions of the cell-mean release) of method A minus that of method B; 95%
percentile CI. Cells = (seed, target, victim); cells share peers within a seed, so these CIs are approximate (likely
too narrow). Script: scratchpad boot.py (to be copied into src/analysis before the paper).
| comparison | N-BaIoT (12 cells) .001 / .01 | IoT-23 (54) .001 / .01 | IoT-23 dedup (54) .001 / .01 |
| LBA.05-v - Ada.05 | -2.7 [-14.4,+7.4] / -1.0 [-2.0,-0.3] | -11.1 [-18.7,-2.0] / -3.0 [-5.6,-0.5] | -11.4 [-17.0,-2.7] / -0.9 [-3.0,+0.8] |
| LBA.05-v - LB-v   | -2.8 [-24.7,+16.7] / -6.6 [-21.3,+1.3] | -11.7 [-19.5,-5.7] / -0.6 [-1.9,0.0] | -14.3 [-21.4,-8.2] / -0.4 [-1.9,+0.7] |
| LB-v - OC-kNN     | -8.3 [-24.9,0.0] / -8.3 [-25.0,0.0] | -18.9 [-28.4,-6.9] / -14.2 [-22.4,-6.6] | -19.4 [-27.9,-8.6] / -9.5 [-15.8,-3.7] |
| LB-v - AOE-peer   | -47.8 [-73.9,-22.8] / -50.9 [-73.8,-27.2] | -5.0 [-15.4,+6.6] / -18.4 [-23.7,-12.6] | -5.7 [-14.6,+5.8] / -19.0 [-24.7,-13.3] |
Reading: on IoT-23 (both versions) the strict-level gain of the wrapper over peer-free AdaDetect (~11 points) and over
LB-v is clearly nonzero; at a .01 the gains are 0-3 points and partly within noise. N-BaIoT (12 cells) is too small
for any claim except "LB-v far better than the learned peer detector under liars". The paper must state the N-BaIoT
results as consistent but statistically weak.

## Pre-registration E-AOE27: more N-BaIoT cells for the final method (2026-11-01; nothing run)
aoe21.py nbaiot nsel=2000, seeds 15-29 (never used), same conditions/methods as E-AOE25. Criteria X1-X4 of E-AOE25
re-scored on these seeds alone, plus:
X5 pooled with E-AOE25 N-BaIoT (seeds 6-29): seed-clustered bootstrap 95% CI of worst(LB-v) - worst(OC-kNN) at a .01
   lies entirely below 0, and of worst(LBA.05-v) - worst(Ada.05) at a .01 has upper end <= +2.

## OUTCOME E-AOE26 (rule-aware liars; aoe26_{nbaiot,iot23,iot23_dedup}_sel2000_s121314.json, 42 + 189 + 189 rows,
## 0 Tracebacks; aoe26_nbaiot_sel2000_s12.json was the smoke test and was deleted)
Worst-case release % over hon, mimic10, split100, tail{.002,.005,.01,.02}:
| run a | OC-kNN | LB-v (worst cond) | Ada.05 | LBA.05-v (worst cond) | max FA |
| N-BaIoT .001 | 17.5 | 33.8 (tail.02) | 33.4 | 36.8 (tail.02) | .133% |
| N-BaIoT .01  | 0.0  | 16.9 (split100) | 1.4 | 20.0 (split100) | 1.38% (Ada) |
| IoT-23 .001  | 85.2 | 67.8 (tail.002) | 75.7 | 61.8 (split100) | .081% |
| IoT-23 .01   | 15.4 | 7.0 (tail.02) | 10.1 | 6.4 (tail.02) | .78% |
| dedup .001   | 82.4 | 68.0 (split100) | 75.2 | 65.6 (split100) | .076% |
| dedup .01    | 12.6 | 3.8 (tail.02) | 10.6 | 4.2 (tail.02) | .78% |
Mean post-vetting Cal share in B at a .01 (split100 / tail.002 / .005 / .01 / .02): N-BaIoT .0083 / .0053 / .0063 /
.0050 / .0065 (max cell .011); IoT-23 and dedup .0056 / .0039 / .0046 / .0051 / .0047 (max .015).
Y1 S. Y2 R: N-BaIoT breaks the floor (LB-v 16.9 > 0.0 + 3; LBA 20.0 > 1.4 + 2), IoT-23 and dedup hold.
Y3 S on the pooled mean (N-BaIoT .0063, IoT-23 .0048 <= .0075); per condition N-BaIoT split100 is .0083 (> .0075).
Y4: rule-aware liars are the worst conditions everywhere (split100 or tail), worse than mimic10.
Diagnosis (cells): both N-BaIoT breaches are seed 14, victim 6 (split100: 100% released vs 0.18%; tail.02: 59.1%).
The kept liar sources hold <= 10 of 2000 Sel rows (cap) but 10-11 of 1000 Cal rows: twice the target. Cause: the
greedy rule keeps the sources whose Sel footprint happened to be smallest (selection on the same sample it tests:
winner's curse), which Remark 2 / Theorem 3 warned about. Many small sources (split100) maximise this selection
effect. Victim 6's attack family sits just inside the a .01 tail, so N-BaIoT's all-or-nothing families turn a 1%
budget theft into 100% release.

## Pre-registration E-AOE28: certified split vetting against rule-aware liars (2026-11-01; nothing run)
Rule LB-vc (fixes the selection effect with the theory's own construction):
 - order sources by footprint on Sel half 1 (first 1000 rows);
 - drop in that order until the kept union's count k on Sel half 2 (other 1000 rows) is <= k*(a), where k*(a) is the
   largest k whose one-sided 95% Clopper-Pearson upper bound on the union's true benign mass is <= .5 a;
 - if no k qualifies (k* < 0), keep no peer (fall back to the base detector).
Guarantee (new, written in notes/THEORY_LB.md before the run): the visited sets are nested and both the true mass and
the half-2 count decrease along the order, so if any set with true mass > .5 a passes, the smallest such set passes;
hence Pr[kept mass > .5 a] <= .05 with NO union bound over sources. At m2 = 1000: k*(.01) = 1, k*(.001) < 0 (no
certification possible at a .001 with 1000 rows: the 1/a law). LBA-vc = same keep set on the AdaDetect base.
Code aoe28.py (= aoe26 + LB-vc, LBA-vc; LB-v kept for comparison), nsel=2000; seeds 12-14 (where the breach was found)
AND seeds 30-32 (fresh, so the fix is not tuned to the breach seeds); N-BaIoT, IoT-23, IoT-23 dedup.
Z1 FA pooled <= a + .2 / a + 1.
Z2 (floor, a .01, all 6 runs = 3 datasets x 2 seed sets) LB-vc worst <= OC-kNN + 3 and LBA.05-vc worst <= Ada.05 + 2.
Z3 (budget) mean post-vetting Cal share in B at a .01 <= .005 in every condition, every run.
Z4 (cost, reported with a threshold) IoT-23 a .01: LB-vc worst <= LB-v worst + 3 (the certified rule keeps most of the
   peer gain at the normal level). At a .001 LB-vc = base detector by construction (reported, not tested).
If Z2 and Z3 hold, LB-vc replaces LB-v as the method and the paper states the floor as Theorem 3' (probability >= .95).

## OUTCOME E-AOE27 (N-BaIoT seeds 15-29, |Sel| 2000; aoe21_nbaiot_sel2000_s15...29.json, 150 rows = 30 cells,
## 0 Tracebacks)
Worst-case release % over hon/flip/cw3/cw10/mimic10:
| a | OC-kNN | AOE-peer | LB-v | Ada.05 | LBA.05-v | Ada.005 | LBA.005-v | max FA |
| .001 | 65.3 | 98.4 | 58.6 | 47.0 | 34.8 | 31.5 | 24.8 | .161% |
| .01  | 14.2 | 49.1 | 7.5  | 4.8  | 0.3  | 4.9  | 0.2  | 1.29% |
X1 S. X2 S (7.5 <= 17.2; LBA .3 <= 6.8, .2 <= 6.9). X3 S. X4 S (all 4 cells).
X5 S (seed-clustered bootstrap, seeds 6-29 pooled, 21 seeds with eligible victims, 42 cells):
  worst(LB-v) - worst(OC-kNN): .001 -7.1 [-14.3, -0.0]; .01 -7.1 [-14.2, -1.5]
  worst(LBA.05-v) - worst(Ada.05): .001 -9.5 [-16.4, -3.1]; .01 -3.5 [-7.0, -1.2]
  worst(LB-v) - worst(AOE-peer): .001 -42.1 [-55.6, -28.5]; .01 -44.2 [-55.7, -32.0]
N-BaIoT now supports the E-AOE25 conclusions with intervals excluding 0 (under the standard liar conditions; the
rule-aware liars of E-AOE26 are a separate, failed check, being addressed by E-AOE28).

## OUTCOME E-AOE28 (certified split vetting LB-vc; aoe28_{nbaiot,iot23,iot23_dedup}_sel2000_s{121314,303132}.json,
## 42/42 + 189/189 + 189/189 rows, 0 Tracebacks; k*(.01) = 1 in every cell, as computed before the run)
Worst-case release % at a .01 over hon, mimic10, split100, tail{.002,.005,.01,.02}:
| run (seeds) | OC-kNN | LB-v | LB-vc | Ada.05 | LBA.05-v | LBA.05-vc |
| N-BaIoT 12-14 | 0.0  | 16.9 | 3.1  | 1.4  | 20.0 | 1.4  |
| N-BaIoT 30-32 | 0.1  | 18.1 | 0.1  | 16.7 | 16.9 | 17.1 |
| IoT-23 12-14  | 15.4 | 7.0  | 13.1 | 10.1 | 6.4  | 11.5 |
| IoT-23 30-32  | 19.4 | 7.0  | 11.3 | 15.0 | 4.6  | 8.6  |
| dedup 12-14   | 12.6 | 3.8  | 13.0 | 10.6 | 4.2  | 10.9 |
| dedup 30-32   | 17.7 | 5.4  | 11.0 | 15.6 | 4.0  | 9.4  |
Max pooled FA: .268% at .001 (N-BaIoT 12-14), 1.45% at .01. Mean post-vetting Cal share in B for LB-vc at .01:
.0000-.0018 per condition (max cell .005); honest sources kept: LB-vc 13-46% vs LB-v 59-81%.
Z1 S. Z2 R by 0.1 point: N-BaIoT 12-14 LB-vc worst 3.1 > 0.0 + 3 (seed 14, victim 6, tail.02: 18.3% released while the
kept liars hold only .002 of the budget). All other 11 Z2 comparisons S. Z3 S (all means <= .0018 << .005).
Z4 R (IoT-23 LB-vc 13.1 vs LB-v 7.0 + 3; 11.3 vs 10.0).
Per the pre-registered rule LB-vc does NOT replace LB-v.
What this run establishes:
 (1) Certified split vetting does what Theorem 3' says: the stolen budget is controlled under rule-aware liars (Z3;
     the 2x selection excess of E-AOE26 is gone, means .0000-.0018 vs .005-.0083).
 (2) Budget control does NOT imply a release floor for concentrated attack families: on N-BaIoT victim 6, a theft of
     .002 of the budget (below the .005 certificate) still released 18% of one family, because that family's
     p-values sit just inside a. The floor that can be guaranteed is in budget units (Theorem 2), not in release.
 (3) Certification costs power at m2 = 1000 (cap of 1 row): it drops most honest sources, losing ~4-9 points of
     worst-case release on IoT-23 vs LB-v. The trade-off between certified safety and power is set by the number of
     trusted rows (the 1/a law), and it is the paper's honest central message.
EXPERIMENT PHASE CLOSED for the paper (user instruction 2026-11-01: solidify, then write). Final claims are fixed in
notes/DECISION.md.
