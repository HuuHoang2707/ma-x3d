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

## Preprocessing study, built from raw RWF-2000

Same recipe, same folds, same seed; only the preprocessing changes. The decision metric
is out-of-fold accuracy over the 1,584 training clips; the test column is the four-fold
ensemble scored once.

| variant | crop | contrast | OOF acc | test (ens) | test (single) |
|---|---|---|---|---|---|
| `rp_none` | full frame | CLAHE | **92.55** | 89.50 | 89.13 ± 0.52 |
| `rp_adaptive` | >= half frame, zoom <= 2x | CLAHE | 91.98 | 89.50 | 87.81 ± 2.22 |
| `rp_noenh` | cluster crop | none | 90.40 | 87.00 | 87.19 ± 1.57 |
| `rp_cluster` | cluster crop (the thesis pipeline) | CLAHE | 89.58 | 88.50 | 87.44 ± 0.99 |
| `rp_union` | union of all people | CLAHE | 89.39 | 90.50 | 89.00 ± 1.35 |
| `rp_plain` | full frame | none | training | | |
| `rp_aspect` | full frame, 16:9 kept by padding | CLAHE | queued | | |

What this says:

* Cropping to the people costs 2.97 OOF (92.55 -> 89.58). The gentler adaptive crop wins
  most of that back but still does not beat leaving the frame alone. The error analysis
  pointed the same way: the missed clips had small actors, and the fix was to stop
  throwing the context away, not to zoom harder.
* CLAHE costs 0.82 on its own (89.58 -> 90.40 once it is removed, same crop), so
  `rp_plain` tests both together.
* `rp_union` is the reason we select on OOF: it has the best test number of the five
  (90.50) and the second worst OOF. On 400 test clips one clip is 0.25 points, and
  picking by test would have chosen the weaker preprocessing.
* The raw videos are 1280x720. Every variant so far squashes 16:9 into a square, which
  makes people 1.78x too narrow relative to the Kinetics pretraining; `rp_aspect` keeps
  the ratio and pads instead.

For reference, the same recipe on the user's own HDF5 crop gives 90.46 OOF, so the
downloaded videos agree with the thesis data to within a point.

_updated 2026-09-22_
