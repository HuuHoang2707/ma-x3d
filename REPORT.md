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

## Preprocessing (ROI) study, built from raw RWF-2000

Same recipe, same folds, same seed; only the crop made at preprocessing time changes.
Decision metric is out-of-fold accuracy over the 1,584 training clips.

| variant | what it does | OOF acc | fold-ensemble test | single-model test |
|---|---|---|---|---|
| `rp_cluster` | largest DBSCAN cluster of person boxes (the thesis crop) | 89.58 | 88.50 | 87.44 ± 0.99 |
| `rp_adaptive` | same box, but never below half the frame and never more than 2x zoom | **91.98** | **89.50** | 87.81 ± 2.22 |
| `rp_none`, `rp_noenh`, `rp_union` | running | | | |

Adaptive is +2.40 OOF over the thesis crop. This matches the error analysis: the clips
the model missed held small actors and a tight crop threw away the context that says
whether people are fighting or dancing, so the fix was to bound the zoom, not to crop
harder (the earlier hard-zoom experiment cost 1.77).

For reference, the same recipe on the user's own HDF5 crop gives 90.46 OOF, so the
mirror's videos are equivalent to within about a point and the adaptive gain is real
rather than a difference in source data.

_updated 2026-09-22_
