# MA-X3D paper: revised outline (v2)

Focus: a lightweight model with competitive accuracy. The deployed model is one MA-X3D
(3.01M parameters, 5.46 GFLOPs per view); the teacher and the fold ensemble are training
and analysis tools, not the method being sold.

Target: IEEE conference format, 6 pages + references (about 4,500 words).

- **[s0]** = seed 0 only, to be replaced by the 3-seed mean (runs are queued).
- **[TODO]** = not measured yet; see the experiment list at the end.
- Numbers of other methods are as reported in their papers; their selection protocol is
  mostly not stated, so they are not strictly comparable.

---

## 0. The numbers the paper rests on (best run: `e11_kd_long`) [s0]

Protocol: grouped 4-fold CV on 1,583 training clips; every choice (recipe, views,
threshold) made on out-of-fold (OOF) predictions; the 400 test clips scored once.

### Main result: one deployed model

Test metrics are the mean ± std over the 4 fold models (each trained on 75% of the
training clips). Threshold chosen on OOF.

| Setting | Views | GFLOPs / clip | Test Acc | Test P | Test R | Test F1 | Test AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MA-X3D, standard fine-tuning (E00) | 1 | 5.46 | 86.1 ± 1.0 | 0.892 | 0.823 | 0.856 | 0.923 |
| MA-X3D, final recipe (E11) | 1 | 5.46 | 87.6 ± 0.7 | 0.883 | 0.867 | 0.875 | 0.949 |
| MA-X3D, final recipe (E11) | 4 | 21.9 | **88.1 ± 0.6** | 0.902 | 0.854 | **0.877** | 0.953 |
| VideoMAE-B teacher (T01) | 1 | 135 | 91.2 ± 0.8 | 0.902 | 0.925 | 0.913 | 0.967 |

OOF (1,583 clips, one prediction per clip):

| Setting | Views | OOF Acc | OOF P | OOF R | OOF F1 | OOF AUC |
| --- | --- | --- | --- | --- | --- | --- |
| E00 standard fine-tuning | 1 | 88.44 | 0.882 | 0.888 | 0.885 | 0.947 |
| E11 final recipe | 1 | 91.03 | 0.902 | 0.922 | 0.912 | 0.962 |
| E11 final recipe | 4 | **91.28** | 0.913 | 0.913 | **0.913** | 0.966 |
| T01 teacher | 1 | 92.48 | 0.918 | 0.933 | 0.926 | 0.965 |

### Secondary result: 4-model ensemble (state the cost next to it)

E11, 4 fold models x 4 views = 87.4 GFLOPs per clip: test 89.25%, P 0.915, R 0.865,
F1 0.889, AUC 0.960. Use it only in the analysis, never as the headline of a
"lightweight" paper.

### Cost of one model

| | MA-X3D | VideoMAE-B teacher |
| --- | --- | --- |
| Parameters (inference, ring fused) | 3.01M | 86.2M |
| GFLOPs per view (16x224x224) | 5.46 | 135 |
| GPU latency, batch 1, fp32 (MI250, shared) | 30 ms | 24 ms |
| GPU throughput, batch 16, fp16 | 118 clips/s | 136 clips/s |
| CPU latency, batch 1, 16 threads (PyTorch) | 208 ms | 400 ms |
| Training per fold (one MI250 GCD) | about 48 min, 5.8 GB | about 12 min, 9.4 GB |

### What the evidence supports, and what it does not

- Supported: the final recipe raises OOF accuracy from 88.4% to 91.3% and single-model
  test accuracy from 86.1% to 88.1%, with the same network.
- Supported: most of the test gain comes from the backbone learning rate (E05 single-model
  test 88.0 ± 0.9). Distillation adds +1.2 OOF points (91.03 vs 89.83, 1 view) but no
  measurable single-model test gain at seed 0. Say this; the 3-seed runs decide whether
  the OOF gain holds.
- Supported: MA-X3D is 29x smaller and 25x cheaper in FLOPs than the teacher and 2x
  faster on CPU; it is not faster on the MI250 GPU (depthwise 3D convolutions are
  memory-bound). Report both.
- Not supported: "state of the art", "outperforms lightweight methods" (reported
  lightweight results are 89.7-90.25% under unstated protocols; ours is 88.1% single
  model under a stricter one), any ROI or Hockey claim until re-run.

---

## Title

*MA-X3D: A Lightweight Motion-Attentive 3D CNN for Violence Detection in Surveillance Video*

(Keep. Distillation is part of the training recipe, not the title.)

---

## Abstract (about 180 words), draft

> Violence detection in surveillance video has to run on many camera streams at once, yet
> the most accurate models on RWF-2000 use tens to hundreds of millions of parameters.
> We study how far a compact 3D CNN can go under this constraint. MA-X3D extends X3D-M
> with two identity-initialised modules: a Motion Attention gate driven by a multi-scale
> frame difference, and a reparameterised 5x5 depthwise kernel in the first two residual
> stages that is fused into a single kernel after training. The network is trained with a
> discriminative fine-tuning schedule and distilled from a VideoMAE-B teacher trained on
> the same data. Because RWF-2000 has no validation split and contains duplicated clips,
> we remove duplicates, group near-identical clips, and make every design decision by
> grouped 4-fold cross-validation, using the test set once. Under this protocol a single
> MA-X3D model with 3.0M parameters and 5.5 GFLOPs per view reaches 88.1% test accuracy
> and 0.88 F1 (91.3% out-of-fold), against 86.1% for standard fine-tuning of the same
> network and 91.2% for the 86M-parameter teacher. [Add one sentence on the ablation
> once Table II is complete.]

Style notes for the whole paper: report numbers with their protocol; prefer "reaches",
"is within", "we observe"; no "significantly" without a test; no "novel"; no adjectives
on our own results.

---

## I. Introduction (about 750 words)

1. **Motivation** (keep the current text, cut by a third): camera numbers, operator
   vigilance, a detector as a filter for the operator.
2. **Difficulty** (keep): short events, small actors, viewpoint and lighting variation,
   look-alike non-violent motion.
3. **Cost** (keep, update): CUE-Net 94.0% at 354M parameters and about 5.8 TFLOPs;
   VideoMAE-class 91.8-92.4% at 67-305M. A per-stream detector has a budget of a few
   GFLOPs. This paper works inside that budget.
4. **Gaps** (three):
   - motion is implicit in a single 3D backbone; the frame difference is cheap, but
     existing uses add a second stream;
   - the early receptive field is narrow (3x3 depthwise at 56x56 and 28x28);
   - with 1,600 training clips, fine-tuning a small backbone leaves a gap to large
     pre-trained transformers, and reported numbers are hard to compare because the
     benchmark has no validation split and selection on the test split is common.
5. **Approach** (one paragraph): MA-X3D; identity initialisation; fusion at inference;
   training by discriminative fine-tuning and distillation from a per-fold VideoMAE-B
   teacher that sees the same augmented clip.
6. **Contributions**, draft wording:
   - "A Motion Attention module that adds a frame-difference cue to a single 3D backbone
     with about 2.5K parameters and no second stream." [+ Table II number]
   - "A wide-kernel reparameterisation that enlarges the depthwise kernels of Res2 and
     Res3 to 5x5 and fuses into a single kernel for inference; the added ring is
     trained in its own learning-rate group." [+ Table II and ring-energy numbers]
   - "A training recipe for small video backbones on small datasets: a higher backbone
     learning rate and long, view-consistent distillation from a video transformer. It
     raises single-model test accuracy from 86.1% to 88.1% without changing the network."
   - "A leakage-free evaluation of RWF-2000: duplicate removal, grouped 4-fold
     cross-validation with all decisions on out-of-fold predictions, and a single use of
     the test set. Split files and code are released."
7. Organisation (one sentence).

**Fig. 1** (teaser): test accuracy vs GFLOPs per clip (log x-axis). Points: lightweight
methods and large transformers as reported (hollow markers: protocol not stated), X3D-M,
MA-X3D single model (1 and 4 views), teacher (filled markers: our protocol).

---

## II. Related Work (about 650 words)

A. **Violence detection** (shorten the current text). End with the two groups:
lightweight (Flow Gated 87.25%, SepConvLSTM 89.75%, MDCN 89.70%, BiLTMA 90.25%) and
large (VideoMAE 91.8-92.4%, CUE-Net 94.0%). Say that most do not report how the
checkpoint was chosen.

B. **Efficient video backbones** (keep, shorter): X3D; why X3D-M.

C. **Motion cues, gating, reparameterisation** (keep).

D. **Knowledge distillation (new, one paragraph)**: Hinton et al. (2015); Beyer et al.
(CVPR 2022): the teacher and student should see the same view, and the schedule should be
long. Our setting: a video transformer teacher, a 3D CNN student, 1,600 clips.

---

## III. Method (about 1,300 words)

A. **Overview and principles**: P1 lightweight, P2 identity initialisation, P3 no
inference overhead, P4 the teacher is used only in training. Pipeline figure with the
teacher branch dashed.

B. **Input pipeline** (merge the ROI section into one paragraph, no gain claim):
- static person-centred crop (the camera is fixed, so the frame difference stays valid);
- 16 frames: evenly spread at test time; during training a random window covering
  60-100% of the stored clip, so both see the same time scale;
- clip-consistent augmentation (list), CutMix with p = 0.5.

C. **Motion Attention** (keep equations and the placement argument).

D. **Wide-kernel reparameterisation** (keep the equation; the ring learning rate is
5e-4, 10x the backbone rate of 5e-5; remove "fifty times").

E. **Training**:
1. Two phases: probe (4 epochs, backbone frozen, lr 1e-3), then fine-tuning of Res2,
   Res3, Res5 and the head (Res4 frozen); AdamW, backbone 5e-5, new parameters 5e-4,
   cosine schedule, label smoothing 0.1, weight EMA 0.999; 48 epochs, early stopping
   after 16 epochs without a gain in validation F1.
2. Distillation:
   L = (1 - a) CE(y, s) + a T^2 KL(softmax(t / T) || softmax(s / T)), a = 0.5, T = 2.
   Teacher: VideoMAE-B (Kinetics-400), fine-tuned for at most 15 epochs on the same
   fold's training clips; it runs on the same augmented, CutMix-ed clip as the student.
   No validation or test clip is seen by the teacher.

F. **Inference**: fuse the ring; one model; 1 or 4 temporal views; threshold 0.5.

---

## IV. Experimental Protocol (about 500 words)

A. **Data audit**: RWF-2000, 1,600 train / 400 test clips. 9 training clips duplicate
test clips (all NonFight) and 9 duplicate pairs occur within training; 17 training
clips are removed, leaving 1,583. Near-identical clips (descriptor cosine >= 0.9) form
1,572 groups.

B. **Cross-validation**: stratified group 4-fold (about 1,187 / 396 clips); a group
never spans two folds; the checkpoint is chosen on the fold's validation F1.

C. **Decisions and test use**: every choice uses the OOF predictions. Test results are
the mean ± std of the 4 fold models (main text) and, where stated, their ensemble.

D. **Seeds and metrics**: 3 seeds for the main rows; accuracy, precision, recall and F1
of the Fight class, ROC-AUC. Cost: parameters, GFLOPs (fvcore), GPU and CPU latency,
throughput, training time and memory.

E. **Implementation**: PyTorch 2.14, ROCm 7.2, one MI250 GCD per fold, Triton kernels for
depthwise 3D convolution (one sentence).

---

## V. Results (about 1,300 words)

A. **Comparison** (Table I): prior methods as reported (params, GFLOPs, accuracy,
protocol column), then ours under the protocol above: X3D-M [TODO: p0], MA-X3D standard
fine-tuning (86.1%), MA-X3D final recipe (87.6% at 1 view, 88.1% at 4 views), teacher
(91.2%). Draft wording:
> "A single MA-X3D model reaches 88.1% with 3.0M parameters. Reported lightweight
> methods reach 89.7-90.25%, but their selection protocol is not stated, so the
> difference cannot be read as a ranking. The teacher, 29 times larger, is 3.1 points
> more accurate."

B. **Component ablation** (Table II, final recipe, 3 seeds) [TODO, queued]:
X3D-M | + MA | + WK | + MA + WK | dense 5x5 | ring at the backbone rate.
Columns: OOF acc, OOF F1, single-model test acc and F1, params, GFLOPs, ring energy.

C. **Training recipe** (Table III, one change at a time; decided on OOF) [s0]:

| Row | Change | OOF Acc | OOF F1 | Test Acc (single) | Test F1 (single) |
| --- | --- | --- | --- | --- | --- |
| R0 | standard fine-tuning (backbone lr 1e-5, 24 ep) | 88.44 | 0.885 | 86.1 ± 1.0 | 0.856 |
| R1 | backbone lr 5e-5 | 89.83 | 0.900 | 87.8 ± 0.8 | 0.875 |
| R2 | R1 + distillation, 24 epochs | 89.83 | 0.900 | 87.3 ± 0.8 | 0.870 |
| R3 | R1 + distillation, 48 epochs (final) | 91.03 | 0.912 | 87.6 ± 0.7 | 0.875 |
| - | 32 frames (vs R0) | 88.76 | 0.881 | [compute] | |
| - | X3D-L backbone (vs R1) | 90.02 | 0.900 | [compute] | |

Wording: "The backbone learning rate accounts for most of the test gain. Long
distillation adds 1.2 OOF points; on the test set the single-model difference is within
one standard deviation." Remove or keep R2-R3 after the 3-seed runs.

D. **Teacher and student**: OOF gap closed (88.44 -> 91.03 of 92.48 = 64%); agreement
rate; clips the teacher gets right and the student does not [TODO: from OOF predictions].

E. **Inference options**: 4 views +0.25 OOF, +0.4 single-model test, at 4x cost; flip TTA
and threshold tuning: no gain on OOF. The ensemble of the 4 fold models reaches 89.25% at
87 GFLOPs (one sentence, for completeness).

F. **Qualitative**: Grad-CAM and gate maps from the final model; error modes (crowds,
distant actors, sport and horseplay) [TODO: `ma-x3d visualize`].

G. **Cost** (Table IV, from section 0 above). Wording:
> "On CPU the student is twice as fast as the teacher (208 vs 400 ms per clip). On the
> MI250 GPU the two are close, because depthwise 3D convolutions are memory-bound on this
> hardware; the gain of the student there is in memory and parameter count."

---

## VI. Limitations (about 150 words)

- One benchmark; 400 test clips, so one point is four clips.
- Test accuracy is 1-3 points below OOF for every model, the teacher included; the
  test clips seem harder than the training clips.
- The static crop assumes a fixed camera.
- Training needs a large teacher (about 15 min per fold on one GPU).
- Clip-level labels only.

## VII. Conclusion (about 120 words)

A compact 3D CNN with two small modules and a better training recipe reaches 88.1% on
RWF-2000 under a protocol that does not use the test set for any decision, with 3.0M
parameters. Future work: localisation in untrimmed streams, an edge-device study,
cross-dataset evaluation under the same protocol.

---

## Experiments still needed

Queued now (`runs/queue_H.sh`, about 11 h on 4 GCDs):

1. Seeds 1 and 2 of E11 (final), E00 (standard fine-tuning), p0 (plain X3D-M, final recipe).
2. Table II rows at seed 0: p0, p1 (+MA), p2 (+WK), p4 (dense 5x5), p5 (ring at backbone lr).
3. E17 (teacher weight 0.9), E15 (no CutMix), E16 (96 epochs).

Then:

4. Seeds 1 and 2 for the Table II rows that stay in the paper.
5. Single-model test numbers for the 32-frame and X3D-L rows.
6. Qualitative figures from the final model; one edge-device latency number.
7. Hockey Fight / RLVS and the ROI ablation (need data not on the server).
