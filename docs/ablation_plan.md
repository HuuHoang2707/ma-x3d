# MA-X3D ablation plan (RWF-2000)

Each row is one config file in `configs/`. Rows are trained with 3 seeds and reported
as mean ± std on the official 400-clip test set.

## Protocol (same for every row)

- **Data.** Official RWF-2000 split: 1600 training clips, 400 test clips, person-centred
  crop as in the thesis.
- **Validation.** 160 training clips (80 Fight + 80 NonFight) are held out for choosing
  the checkpoint. Clips cut from the same YouTube video sit next to each other in the
  data files, so the held-out clips are taken as contiguous blocks and a source video
  does not end up on both sides. The split is fixed (`split_seed=0`) for all rows.
- **Model selection.** The epoch with the best validation Fight-F1. The test set is used
  once per run, after training.
- **Seeds.** 0, 1, 2. Tables report test accuracy, Fight-F1 and ROC-AUC as mean ± std.
- **Recipe** (`configs/base.yaml`): 16 frames, batch 12, at most 24 epochs (4 probe
  epochs with the backbone frozen), early stopping after 6 epochs without improvement,
  AdamW (weight decay 5e-4), learning rates 1e-3 (probe) / 1e-5 (pre-trained weights) /
  5e-4 (new modules), cosine schedule, label smoothing 0.1, bf16. In fine-tuning, Res2,
  Res3, Res5 and the head are trained; the stem and Res4 stay frozen.
- **Hardware.** One MI250 GCD per run. A full MA-X3D run (24 epochs + evaluation) takes
  about 24 min, X3D-M about 16 min; early stopping often ends runs sooner.

## What changed since the thesis

The numbers will not match the thesis exactly, for these reasons:

1. **Checkpoint selection.** The Kaggle notebook chose the best epoch on the 400 test
   clips and reported that epoch's test score. Here a validation split is carved out of
   the training data, so the test score is not selected on. Row L1 re-runs the old
   protocol so the two can be compared.
2. **Frozen layers.** The notebook chose trainable parameters by substring match.
   `"blocks.5"` also matched Res4 block 5, and `"blocks.1"` matched Res4 blocks 1 and 10,
   so parts of Res4 and Res5 trained during the "frozen" probe phase, some of them at
   the 5e-4 rate. The new code matches exact stage names.
3. **Same trainable stages in every row.** In the notebook, only the wide-kernel model
   unfroze Res2 and Res3, so the "+ Wide-Kernel" gain mixed two changes. Here every row
   trains the same stages, and only the component under test changes.
4. **Input normalisation.** The notebook fed [0, 1] frames to a backbone pre-trained on
   Kinetics-normalised input. The new code applies the Kinetics mean and std (row E4
   tests the old behaviour).
5. **Frozen BatchNorm.** BatchNorm layers of frozen stages stay in eval mode, so
   "frozen" also covers their running statistics (row D4 tests the alternative).

## Scenarios

Priority: **P1** = needed for the paper's ablation table, **P2** = supports design
choices argued in the method section, **P3** = optional.

### A. Contribution of each component (main ablation table)

| ID | Question | Change | Config | Priority |
| --- | --- | --- | --- | --- |
| A0 | Baseline | X3D-M, no new modules | `ablation/a0_x3d_m` | P1 |
| A1 | Does Motion Attention help on its own? | + MA | `ablation/a1_ma` | P1 |
| A2 | Does the wide kernel help on its own? | + WK | `ablation/a2_wk` | P1 |
| A3 | Full model | + MA + WK | `ablation/a3_ma_wk` | P1 |

The thesis added the components one after another (cumulative). A1 and A2 also give
each module alone, so the result does not depend on the order in which they are added.

### B. Wide-kernel design

| ID | Question | Change vs. A3 | Config | Priority |
| --- | --- | --- | --- | --- |
| B1 | Is reparameterisation better than a plain 5×5 kernel? | dense 5×5, zero ring | `ablation/b1_wk_dense` | P1 |
| B2 | Is the gain from the separate learning rate or the parameterisation? | ring trained at 1e-5 | `ablation/b2_ring_lr_backbone` | P1 |
| B3 | Does the trained model use the ring? | ring set to zero at test time, same weights | reported for every A2/A3 run | P1 (no extra training) |
| B4 | Are early stages the right place? | 5×5 in Res2–Res5 | `ablation/b4_wk_all_stages` | P3 |

### C. Motion Attention design

| ID | Question | Change vs. A3 | Config | Priority |
| --- | --- | --- | --- | --- |
| C1 | Is the two-step motion map needed? | frame step 1 only | `ablation/c1_ma_single_scale` | P2 |
| C2 | Are the learnable temporal modes needed? | raw motion map into the gate | `ablation/c2_ma_raw_map` | P2 |
| C3 | Placement: earlier | gate after Res3 | `ablation/c3_ma_after_res3` | P2 |
| C4 | Placement: later | gate after Res5 | `ablation/c4_ma_after_res5` | P2 |

### D. Training recipe

| ID | Question | Change vs. A3 | Config | Priority |
| --- | --- | --- | --- | --- |
| D1 | Is the frozen-backbone probe phase needed? | no probe phase | `ablation/d1_no_probe` | P2 |
| D2 | Is the discriminative learning rate needed? | one rate (1e-5) for all | `ablation/d2_single_lr` | P2 |
| D3 | Should Res4 stay frozen? | Res4 fine-tuned too | `ablation/d3_unfreeze_res4` | P2 |
| D4 | Frozen BN statistics or updated ones? | BN of frozen stages in train mode | `ablation/d4_bn_update` | P2 |

### E. Input and augmentation

| ID | Question | Change vs. A3 | Config | Priority |
| --- | --- | --- | --- | --- |
| E1 | Accuracy vs. cost of more frames | 32 frames | `ablation/e1_frames32` | P3 |
| E2 | Reduced rotation (thesis choice) | ±25° as in SepConvLSTM | `ablation/e2_rotation25` | P3 |
| E3 | No temporal reversal (thesis choice) | reversal with p = 0.15 | `ablation/e3_temporal_inverse` | P3 |
| E4 | Effect of the missing normalisation | no Kinetics mean/std | `ablation/e4_no_normalize` | P3 |

E2 and E3 are the two augmentation choices the thesis lists as "not ablated".

### F. Additions tried in the last notebook (not in the paper)

| ID | Question | Change vs. A3 | Config | Priority |
| --- | --- | --- | --- | --- |
| F1 | Global token mixing after Res5 | + efficient additive attention | `ablation/f1_eaa` | P3 |
| F2 | Weight averaging and CutMix | + EMA 0.999, CutMix p = 0.5 | `ablation/f2_ema_cutmix` | P3 |

### L. Link to the thesis numbers

| ID | Question | Setup | Config | Priority |
| --- | --- | --- | --- | --- |
| L1 | How much of 90.5% comes from choosing the epoch on the test set? | thesis recipe with notebook behaviour (items 1, 2, 4, 5 above) | `legacy/thesis` | P1 |
| L2 | The last notebook as it was | notebook v5 settings | `legacy/notebook_v5` | P3 |

### Blocked: needs data that is not on the server

| ID | Question | What is needed |
| --- | --- | --- |
| R1 | Gain from the person-centred crop (thesis row "+ ROI") | raw RWF-2000 videos; `ma-x3d build-dataset --roi none` |
| R2 | Largest-cluster crop vs. crop around all people (CUE-Net) | raw videos; `--roi union` |
| H1 | Hockey Fight result | the Hockey Fight videos and a fixed split |

## Budget

| Priority | Rows | Runs (× 3 seeds) | GPU-hours (upper bound) | Wall time on 4 GCDs |
| --- | --- | --- | --- | --- |
| P1 | A0–A3, B1, B2, L1 (B3 comes free) | 21 | ~8 | ~2 h |
| P2 | C1–C4, D1–D4 | 24 | ~10 | ~2.5 h |
| P3 | B4, E1–E4, F1, F2, L2 | 24 | ~11 | ~3 h |

Upper bounds assume no early stopping. E1 (32 frames) costs about twice a normal run.
Measured on MI250 with the Triton depthwise kernels (training step 233 ms at batch 12,
2.3× faster than MIOpen).

## Questions for review

1. Are 3 seeds enough, or should P1 rows get 5?
2. Is a validation split from the training set acceptable, or should model selection
   use the last epoch (no selection at all)?
3. Which P2/P3 rows should go into the paper, and which only into the thesis appendix?
4. Can the raw RWF-2000 and Hockey Fight videos be copied to the server for R1, R2, H1?

## Commands

```bash
make ablation-p1 GPUS="4 5 6 7"      # P1 rows, 3 seeds, one job per free GPU
make ablation-all GPUS="4 5 6 7"     # everything in configs/ablation/
make legacy GPUS="4 5 6 7"           # L1
make report                          # reports/summary.md + LaTeX rows
```
