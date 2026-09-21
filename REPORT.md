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

## What is still open

- **Interaction tokens** (the new module): 91.47 out-of-fold on one seed, the best
  result of the project. Seeds 1-2 are running and decide whether it goes in the paper.
- Screening runs queued: in-model zoom, X3D-S and X3D-XS with the tuned recipe.
- CPU latency for the cost table, to be measured when the GPUs go idle.
- Figure 2 of the paper (the model diagram) is yours to draw.

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
