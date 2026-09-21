# Progress note

State on 21 September 2026. 64 finished experiments (each = 4 folds, 4 trained models).
Every number here can be regenerated with `paper/drafts/make_tables.py`.

## 1. What was built

**Evaluation that cannot leak.** Duplicate and same-scene audit of every dataset,
grouped four-fold cross-validation, all decisions on out-of-fold (OOF) predictions, the
test split scored once, paired McNemar tests and bootstrap intervals
(`ma-x3d cv --compare`), per-epoch test logging to measure selection bias
(`ma-x3d cv --selection-bias`).

**Datasets.** RWF-2000 (audited, 17 duplicated clips removed) plus two built from raw
video with the same pipeline: Hockey Fights (797/200) and RLVS (1,588/397), each with a
grouped 80/20 split.

**Reproduced baselines.** X3D-XS, X3D-S, X3D-M, S3D, MC3-18, R(2+1)D-18 and a
VideoMAE-B teacher, all trained by us on the same folds with the same recipe.

**Training.** Two-phase discriminative fine-tuning, distillation from a per-fold teacher
with a split guard, low-data subsampling, Triton depthwise kernels (2.3x faster steps),
DDP on 4 GPUs.

**New modules (this week).** Temporal feature-difference residual, burst pooling and
motion-peak interaction tokens; all identity-initialised, all separately ablatable.

## 2. What was measured

| Question | Answer |
| --- | --- |
| Do Motion Attention and the wide kernel help a tuned backbone? | No. RWF-2000 $-0.04$ (p = 0.78, 3 seeds), Hockey $-0.04$, RLVS $-0.19$, 25% data $0.00$ |
| Do they help a conventionally fine-tuned backbone? | Yes, +0.88 points [+0.04, +1.71] on RWF-2000 (3 seeds) |
| Does the gate need the motion signal? | No: an all-zero motion map performs the same |
| Does the wide kernel need reparameterisation or its own learning rate? | No: dense 5x5 and ring-at-backbone-rate perform the same |
| What does improve accuracy? | The training recipe: +2.80 points on X3D-M [+1.81, +3.83] |
| Does distillation transfer? | Little: +0.57 over the same schedule without a teacher (n.s.) |
| How much does test-set epoch selection inflate results? | +0.8 points on average (0.38-1.38 over six models) |
| How much does the seed matter? | One seed showed a "significant" +1.25 on Hockey that three seeds removed |
| Do the benchmarks transfer to each other? | Poorly: Hockey-trained models reach 63-65% on RWF-2000, RLVS-trained 58-60% on Hockey |
| Is INT8 quantisation useful here? | No: 2x slower on CPU and -1.5 points |

Headline numbers on RWF-2000 (single model, one view, no test-set selection):
X3D-M tuned **90.90 OOF / 87.4 test**, MA-X3D **90.86 / 87.5**, VideoMAE-B teacher
**92.48 / 91.2**. Four views give 88.1, the four-fold ensemble 89.25 at 16x the cost.

**New, still at one seed:** the temporal difference residual reaches **91.47 OOF /
88.1 test** and with burst pooling **88.4 test**, the best a 3M-parameter model has
reached here. Seeds 1-2 are running and decide whether this enters the paper as a
result.

## 3. What changed in the paper

`paper/drafts/main_v2.tex` (7-8 pages, compiles, IEEE conference format).

| Before (thesis draft) | Now |
| --- | --- |
| 90.5% headline, epoch chosen on the test split | 87.4-87.5% single model under a protocol where the test split is used once |
| "+5.75 points from the modules" | Measured module effect: +0.88 under conventional fine-tuning, $-0.04$ once tuned, with controls |
| Ablation on one split, one seed | Three datasets, three seeds, paired tests, matched controls for every claim |
| No baselines of our own | Six Kinetics backbones retrained under the same protocol |
| No statement about evaluation | A measured selection bias (+0.8) and a duplicate audit; a section on what the protocol is worth |
| Method section argued for the modules | Method describes what is studied; results decide |
| Three empty figure slots | Three data figures generated from the results (cost, effect sizes, selection bias) plus one drawing left to do |

New sections: evaluation protocol (datasets, audit, statistics, reproduced baselines),
"what the modules contribute", "what the training recipe contributes", "what the
protocol is worth", transfer between datasets.

## 4. Open items

- Seeds 1-2 for the difference residual (running); then the interaction tokens and the
  combined model.
- Figure 2 (model diagram) to draw.
- Re-measure CPU latency on an idle machine.
- Choose the venue and trim to its page limit (AVSS, ICPR or MAPR/RIVF/KSE).
