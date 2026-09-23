# Status report

Updated 21 September 2026, 13:40. This is the only file you need to read.
Everything else is detail: the paper is `paper/drafts/main_v2.tex`, the evidence for
your supervisor is `paper/drafts/memo_modules.md`.

## Where we are

Best model on RWF-2000, under a protocol where the test split is never used to choose
anything (4 folds, decisions on out-of-fold data):

| Model | Params | GFLOPs | OOF acc | Test acc |
| --- | --- | --- | --- | --- |
| X3D-M, conventional training | 2.98M | 4.73 | 88.10 | 85.7 ± 0.9 |
| X3D-M, tuned training (baseline) | 2.98M | 4.73 | 90.90 | 87.4 ± 0.9 |
| **+ difference residual** (3 seeds) | 3.00M | 5.0 | 91.11 | **88.2 ± 0.7** |
| **+ interaction tokens** (1 seed) | 3.00M | 4.8 | **91.47** | 88.2 ± 0.8 |
| MA-X3D (your two modules) | 3.01M | 5.46 | 90.86 | 87.5 ± 0.6 |
| VideoMAE-B teacher | 86.2M | 135 | 92.48 | 91.2 ± 0.8 |

Four views give 88.1, the four-model ensemble 89.25, both at higher cost.

## What is proven

- **The training recipe is what moves accuracy**: +2.80 points on X3D-M
  [+1.81, +3.83], three seeds.
- **Your two modules help a conventionally trained backbone** (+0.88 points
  [+0.04, +1.71]) and **not a tuned one** ($-0.04$, p = 0.78). Same on Hockey Fights
  and RLVS.
- **Controls explain why**: a gate fed an all-zero motion map does as well as the real
  one; a dense 5x5 kernel does as well as the reparameterised one.
- **Evaluation artefacts are as large as published gains**: choosing the epoch on the
  test split is worth +0.8 points, and one seed produced a "significant" +1.25 on
  Hockey that three seeds erased.
- **Nine RWF-2000 training clips are copies of test clips**, and the RLVS copy in
  circulation stores every video twice, in folders named train and test.
- **The benchmarks do not transfer**: a model at 96% on Hockey reaches 64% on
  RWF-2000.
- **INT8 quantisation is useless here**: 2x slower on CPU and $-1.5$ points.

## Final choice

**Ship X3D-M with the tuned recipe and the temporal difference residual**: 3.00M
parameters, 5.0 GFLOPs, 91.11 out-of-fold and 88.2 +- 0.7 test accuracy over three
seeds. It is the only module in the study that improves the tuned baseline on the test
split in every seed.

The recipe matters more than any module: backbone learning rate 5e-5 (not 1e-5), 48
epochs, probe phase, EMA, CutMix, training windows covering 60-100% of the clip
(+2.80 points). Distillation is optional (+0.57, not significant). Inference: one view,
threshold 0.5.

Three variants scored higher on a single seed and are reported in the paper as
unconfirmed: interaction tokens (91.47 OOF), 5x5 kernels in all four stages (91.79),
a wider motion gate with eight modes (88.6 test). No further runs are planned.

## Left to do (no GPU needed)

- Figure 2 of the paper (the model diagram) is yours to draw.
- CPU latency for the cost table, when a quiet machine is available.

## Why the failures happen (error analysis)

135 of 1,583 out-of-fold clips are wrong: 59 missed fights, 76 false alarms.
Missed fights hold much smaller actors (largest person 4.0% of the frame against 7.5%
in correct clips) and in half of them no person is detected at all: the preprocessing
crop kept the whole scene. False alarms are close-up vigorous activity, which is
either a confuser or a label error. The zoom experiments now queued test the fix.

## The files that matter

| File | What it is |
| --- | --- |
| `REPORT.md` | this file |
| `paper/drafts/main_v2.tex` | the paper, compiles, 8 pages with figures |
| `paper/drafts/memo_modules.md` | one page of evidence about your modules |
| `paper/figures/*.pdf` | the three data figures |
| `docs/GUIDE.md` | how to train and evaluate yourself |
| `paper/drafts/make_tables.py` | regenerates every number in the paper |

## Preprocessing study, built from raw RWF-2000 (seven variants, shared folds)

Same recipe, same folds, same seed; only the preprocessing changes. Decision metric is
out-of-fold accuracy over 1,584 training clips; test is the four-fold ensemble, scored once.

| variant | crop | contrast | resize | OOF | test (ens) | test (single) |
|---|---|---|---|---|---|---|
| `rp_none` | none | CLAHE | stretch | **92.55** | 89.50 | 89.13 ± 0.52 |
| `rp_plain` | none | none | stretch | 92.42 | 89.75 | 89.38 ± 1.42 |
| `rp_aspect` | none | CLAHE | pad, 16:9 kept | 92.36 | 88.75 | 88.44 ± 1.27 |
| `rp_adaptive` | >= half frame | CLAHE | stretch | 91.98 | 89.50 | 87.81 ± 2.22 |
| `rp_noenh` | cluster | none | stretch | 90.40 | 87.00 | 87.19 ± 1.57 |
| `rp_cluster` | cluster (thesis) | CLAHE | stretch | 89.58 | 88.50 | 87.44 ± 0.99 |
| `rp_union` | union of people | CLAHE | stretch | 89.39 | 90.50 | 89.00 ± 1.35 |

Paired tests against the thesis pipeline (`rp_cluster`), exact McNemar on the OOF clips:

| comparison | OOF diff | 95% CI | p |
|---|---|---|---|
| no crop vs thesis crop | **+2.97** | [+1.45, +4.48] | 0.0002 |
| no crop, no CLAHE vs thesis crop | +2.84 | [+1.33, +4.36] | 0.0003 |
| aspect kept vs thesis crop | +2.78 | [+1.20, +4.36] | 0.0008 |
| adaptive crop vs thesis crop | +2.40 | [+1.07, +3.72] | 0.0007 |
| no CLAHE vs CLAHE (both uncropped) | -0.13 | [-1.14, +0.88] | 0.90 |
| aspect padding vs stretching | -0.19 | [-1.39, +1.01] | 0.84 |

So the whole effect is the crop. Dropping the person-box crop is worth about three points
and is significant; CLAHE and the aspect ratio make no measurable difference once the
frame is left alone. `rp_union` also shows why the decision is made on OOF: it has the
best test score of the seven and the second worst OOF.

## What the model changes are worth on that preprocessing

All 4-fold, seed 0, same folds, same recipe, on `rwf_none`. OOF decides; test is the
fold ensemble scored once. `p` is exact McNemar against X3D-M.

| model | params | OOF | test (ens) | test (single) | test diff | p |
|---|---|---|---|---|---|---|
| X3D-M (control) | 2.98 M | **92.80** | 89.50 | 88.37 ± 1.53 | - | - |
| + 5x5 kernels in all four stages | 3.44 M | 92.99 | **93.00** | 91.25 ± 1.06 | **+3.50** | **0.0043** |
| 32 frames instead of 16 | 3.03 M | 92.55 | 92.25 | 90.56 ± 0.37 | +2.75 | 0.061 |
| + preprocessing mixture + MA | 3.03 M | 92.49 | 90.75 | 89.88 ± 0.87 | +1.25 | - |
| + wide kernels, Res2-3 (thesis) | 3.03 M | 92.49 | 90.25 | 89.44 ± 0.87 | +0.75 | 0.58 |
| + difference residual | 3.05 M | 92.68 | 89.00 | 89.38 ± 1.35 | -0.50 | - |
| + MA + WK (MA-X3D) | 3.03 M | 92.55 | 89.50 | 89.13 ± 0.52 | 0.00 | 1.00 |
| + preprocessing mixture | 2.98 M | 91.98 | 89.00 | 89.12 ± 1.14 | -0.50 | 0.73 |
| + MA only | 2.98 M | 92.61 | 88.00 | 88.06 ± 0.76 | -1.50 | 0.11 |

Two of these disagree between OOF and test, which is the interesting part:

* **Wide kernels in all four stages**: +3.50 on test (p = 0.0043, the best number this
  project has produced) but +0.19 OOF (p = 0.83). OOF has 1,584 clips against the test
  split's 400, so the disagreement is not a power problem. Either the test videos happen
  to suit wide receptive fields, or wide kernels generalise to RWF's separate test videos
  better than to held-out training clips. One seed cannot tell those apart, and this
  project has already watched a "+1.25, p = 0.006" turn into -0.04 across three seeds,
  so seeds 1 and 2 are running for this row, the 32-frame row and the control.
* **32 frames** shows the same shape (+2.75 test, p = 0.061, OOF flat) with the lowest
  spread of any run (± 0.37), which is what "more information in" should look like.

Negative results worth keeping:

* **Preprocessing-mixture augmentation** (each training clip read from a random
  preprocessing variant) costs 0.82 OOF. An ensemble over preprocessings reaches 94.07
  OOF against 92.80 for the best single model, but sampling the variants during training
  does not transfer that: the model averages the preprocessings instead of learning from
  their disagreement.
* **Motion attention alone is the worst row in the table** (-1.50 on test), and the
  control that removes its motion input entirely scores higher (91.09 vs 90.65 on the
  cropped data), so the gate is not working through the motion signal it is built on.

## Final models, full training split, test scored once

| run | test | F1 | AUC |
|---|---|---|---|
| no crop + joint (RWF + Hockey + RLVS) + difference residual | **91.25** | 0.912 | 0.973 |
| the same, distilled from the joint VideoMAE-B teacher | 90.75 | 0.908 | 0.967 |
| VideoMAE-B teacher, 87 M params, for reference | 94.00 | 0.942 | 0.978 |

Distillation helped on the cropped data (+0.57) and does not here (-0.50).

## Test-time augmentation (no retraining, `rp_none` models)

| views | OOF | test (ens) |
|---|---|---|
| 1 clip (reported cost) | 92.55 | 89.50 |
| 2 (flip) | 92.55 | 90.25 |
| 4 (2 offsets + flip) | 92.93 | 91.00 |
| 8 (4 offsets + flip) | 93.24 | 90.75 |


_updated 2026-09-23_

## Cost of the candidates (fvcore, one 224x224 view)

| model | params | fused params | GFLOPs | vs X3D-M |
|---|---|---|---|---|
| X3D-M | 2.979 M | 2.979 M | 4.73 | - |
| + 5x5 in Res2-Res5 | 3.436 M | 3.272 M | 5.92 | +25% |
| + MA + wide kernels (MA-X3D) | 3.034 M | 3.015 M | 5.46 | +15% |
| + 5x5 in Res2-Res3 | 3.031 M | 3.012 M | 5.45 | +15% |
| + motion attention only | 2.981 M | 2.981 M | 4.75 | +0.4% |
| 32 frames (same network) | 3.034 M | 3.015 M | 10.94 | +131% |

The 32-frame variant buys its test gain with 2.3x the computation, which does not suit
the efficiency argument; the all-stage wide kernel costs 25% more and stays under
6 GFLOPs.

## Seed check on the two candidates (three seeds, 12 fold models each)

| model | OOF (s0/s1/s2) | mean OOF | test ens (s0/s1/s2) | mean test | single-model test |
|---|---|---|---|---|---|
| X3D-M | 92.80 / 92.61 / 92.36 | 92.59 | 89.50 / 89.50 / 87.75 | 88.92 | 88.18 |
| + 5x5 in Res2-Res5 | 92.99 / 93.37 / 93.24 | **93.20** | 93.00 / 91.50 / 92.75 | **92.42** | **90.81** |
| difference | | +0.61 | | +3.50 | +2.63 |

Per-seed single-model difference: +2.87, +2.38, +2.62. Every seed agrees in size and
direction, which is what the earlier Hockey result failed to do. Per-seed McNemar on the
fold ensemble: p = 0.0043, 0.169, 0.0002. The out-of-fold gain is smaller (+0.61) than
the test gain (+3.50) in all three seeds: widening the kernels helps more on the separate
test videos than on held-out training clips, which is the behaviour you want from a model
that has to generalise past its training scenes.

## Chosen model

X3D-M with 5x5 depthwise kernels in Res2-Res5, trained on uncropped frames. Motion
attention is dropped. 3.27 M parameters fused, 5.92 GFLOPs per view.

## Fold soup: one model from the k fold models

The k-fold protocol trains four models from the same Kinetics weights, and their
ensemble beats a single fold model in every seed. Averaging their weights instead of
their predictions (a uniform model soup) gives one network at single-model cost.
Test split, one view, no training.

| experiment | dataset | single | ensemble (4 passes) | **soup (1 pass)** |
|---|---|---|---|---|
| wide kernels, 3 seeds (developed on) | RWF | 90.81 | 92.42 | 91.58 |
| X3D-M, 3 seeds | RWF | 88.19 | 88.92 | 89.00 |
| MA-X3D, uncropped | RWF | 89.06 | 89.50 | 90.50 |
| MA-X3D, uncropped, no CLAHE | RWF | 89.38 | 89.75 | 90.25 |
| MA-X3D, adaptive crop | RWF | 87.88 | 89.50 | 89.25 |
| X3D-M, 3 seeds | Hockey | 95.62 | 96.00 | 95.67 |
| MA-X3D, 3 seeds | Hockey | 95.46 | 96.17 | 96.00 |
| X3D-M + KD | RLVS | 98.49 | 98.49 | 98.49 |
| MA-X3D + KD | RLVS | 98.17 | 98.49 | 98.49 |

Soup against a single fold model: **+0.69 on average, 7 wins, 0 losses, 2 ties**
(sign test p = 0.016). Soup against the four-model ensemble: **+0.00 on average**, at a
quarter of the inference cost. The gains are largest on RWF, where models vary most,
and vanish where accuracy is at the ceiling (RLVS at 98.5).

**BatchNorm recalibration does not replicate.** Re-estimating the statistics of the
48 layers that differ between folds raised the development experiment from 91.58 to
92.17, but on the eight replications it won four and lost four (mean -0.03), and it
lost on both Hockey experiments. That first number was fitted to the test split, which
is exactly what the replication was for. The recipe is the plain uniform soup.

