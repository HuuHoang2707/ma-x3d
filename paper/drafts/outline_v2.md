# MA-X3D paper: revised outline (v2)

Target: IEEE conference format, 6 pages + references (about 4,500 words).
Numbers marked **[s0]** are seed 0 only and must be replaced by 3-seed means.
Numbers marked **[TODO]** do not exist yet; the experiment list at the end says how to get them.

---

## What changes from the current draft, and why

| Current draft | Problem | New version |
| --- | --- | --- |
| 90.5% test accuracy, +5.75 over X3D-M | Epoch chosen on the test set; 9 train clips duplicate test clips | Leakage-free protocol: grouped 4-fold CV, decisions on out-of-fold (OOF), test scored once by the fold ensemble |
| Two contributions (MA, WK) | The largest measured gain now comes from the training recipe (backbone lr, patient distillation) | Three contributions: the model, the transformer-to-3D-CNN distillation recipe, and the evaluation protocol |
| Ring lr = 50x backbone lr | Final recipe uses backbone lr 5e-5, ring 5e-4 | 10x; B2 ablation re-run at the new rate |
| 5.16 GFLOPs, 3.02M params | Measured (fvcore, 16x224x224, fused): 5.46 GFLOPs, 3.01M | Use the measured values |
| Hockey Fight 97.0% | Old protocol, data not on the server | Re-run with the new protocol, or move to a footnote as a thesis result |
| "+ROI" gain | Cannot be re-run (raw videos not available) | Describe ROI as preprocessing, no ablation claim, or re-run when the videos are available |

Headline for the paper (replace with 3-seed means):

- MA-X3D (final recipe): OOF 91.0% (1 view), 91.3% (4 views); RWF-2000 test 89.25%,
  F1 0.889, AUC 0.960 (4-fold ensemble, 4 views) **[s0]**; 3.01M params, 5.46 GFLOPs per view.
- Same backbone, standard fine-tuning: OOF 88.4%, test 86.25% **[s0]**.
- VideoMAE-B teacher: OOF 92.5%, test 90.75%, 86M params, 135 GFLOPs **[s0]**.
- The student recovers about 64% of the baseline-to-teacher OOF gap
  ((91.03 - 88.44) / (92.48 - 88.44)) at 1/29 of the parameters.

---

## Title

Keep: *MA-X3D: A Lightweight Motion-Attentive 3D CNN for Violence Detection in Surveillance Video*

Alternative if the distillation result stays the main gain:
*Motion-Attentive X3D with Transformer Distillation for Lightweight Violence Detection*

---

## Abstract (about 180 words)

1. Problem and cost gap: the best RWF-2000 results use 10^7 to 10^8+ parameter transformers; per-camera deployment needs a small model.
2. What we do: X3D-M plus two identity-initialised modules (Motion Attention on a multi-scale frame difference; a reparameterised 5x5 ring in Res2/Res3 that fuses away), trained by distillation from a VideoMAE-B teacher of the same fold.
3. Protocol sentence: prior RWF-2000 numbers are usually selected on the test set; we audit duplicates, use grouped 4-fold CV with all decisions on out-of-fold predictions, and score the test set once.
4. Results: test accuracy / F1 / AUC, parameter and FLOP counts, gap to the teacher, the effect of the protocol (how much test selection inflates accuracy).
5. One sentence on what the ablation shows.

Do not write "state of the art". Write "within X points of models 29x larger".

---

## I. Introduction (about 750 words)

Paragraph plan:

1. **Motivation** (keep the current text, shorten by a third): camera counts, operator vigilance, a detector as a filter.
2. **Why the task is hard** (keep): short events, small actors, viewpoint and lighting variation, look-alike non-violent motion.
3. **Cost gap** (keep, update numbers): CUE-Net 94.0% at 354M params and 5.8 TFLOPs; VideoMAE-class 91.8-92.4% at 67-305M. Per-stream deployment needs about 5 GFLOPs.
4. **Three gaps** (was two):
   - motion is implicit in a single 3D backbone; the frame difference is cheap but existing uses need a second stream;
   - the early receptive field is narrow (3x3 depthwise at 56x56 and 28x28);
   - **new:** with 1,600 training clips, a small model cannot reach the accuracy of large pre-trained transformers by fine-tuning alone, and reported numbers on this benchmark are hard to compare because checkpoints are often picked on the test set and the split contains duplicated clips.
5. **Approach in one paragraph**: MA-X3D (two modules, identity init, fused at inference), trained with patient distillation from a per-fold VideoMAE-B teacher that sees the same augmented clip.
6. **Contributions** (4 bullets, each with one number):
   - Motion Attention: explicit frame-difference cue inside one backbone, about 2.5K parameters. [ablation number TODO]
   - Wide-kernel reparameterisation with a separate learning-rate group; dense inflation leaves the ring near zero. [ring energy, B1/B2 numbers TODO]
   - A distillation recipe from a video transformer into a 3M-parameter 3D CNN, with a consistent teacher view and a long schedule; +2.6 OOF points over standard fine-tuning, about 64% of the gap to the teacher closed. [s0]
   - A leakage-free RWF-2000 protocol (duplicate audit, grouped 4-fold CV, OOF decisions, fold-ensemble test) and a measurement of how much test-set selection inflates accuracy. Release the split files and code.
7. Paper organisation (one sentence).

Figure 1 (teaser, top of page 2 column 1): accuracy vs GFLOPs (log scale) on RWF-2000 for prior work, the teacher, and MA-X3D; mark which points use test-set selection or an unknown protocol.

---

## II. Related Work (about 650 words)

A. **Violence detection** (shorten the current text): hand-crafted; two-stream with flow; 3D CNNs; transformers. End with the lightweight line (SepConvLSTM, MDCN, BiLTMA) and the heavy line (VideoMAE, CUE-Net).

B. **Efficient video backbones** (keep, shorter): X3D family; why X3D-M.

C. **Motion cues, gating, reparameterisation** (keep): frame difference, SE/CBAM gating with identity init, RepVGG/ACNet, our difference: fusing into a larger kernel.

D. **Knowledge distillation for video (new)**: Hinton et al. 2015; "A good teacher is patient and consistent" (Beyer et al., CVPR 2022) for consistent views and long schedules; video distillation from transformers to CNNs (cite 1-2 video KD papers). Position: we use the same augmented clip for teacher and student and a per-fold teacher so no validation clip is seen by either.

E. **Evaluation practice on RWF-2000 (new, short)**: the benchmark has no validation split; many works report the best test epoch; duplicated and same-scene clips across the split. Keep the tone factual: "the selection procedure is not reported in [..]" rather than accusing.

Table I (comparison, moves here or to Section V): method, backbone type, params, GFLOPs, RWF-2000 accuracy, **selection protocol** (test / validation / not reported).

---

## III. Method (about 1,300 words)

A. **Overview and design principles** (keep P1-P3; add P4: the teacher is used only in training).
   Update the pipeline figure: add the teacher branch drawn dashed, "training only".

B. **Input pipeline** (merge ROI + sampling + augmentation):
   - person-centred static crop (keep, cut to one paragraph; no gain claim unless re-run);
   - temporal sampling: 16 frames spread over the clip at test time; during training a random window covering 60-100% of the stored clip, so train and test see the same time scale (this fixed early overfitting; one sentence and one number from the thesis-to-code comparison);
   - clip-consistent augmentation (keep).

C. **Motion Attention** (keep equations 1-2 and the placement argument). Add the parameter count and the measured gate statistics if the gate analysis is kept.

D. **Wide-kernel reparameterised convolution** (keep equation 3). Change "fifty times" to the actual ratio (10x at the final recipe) and point to the B2 ablation. Keep the receptive-field arithmetic.

E. **Training** (rewrite):
   1. Two-phase discriminative fine-tuning (probe 4 epochs, then fine-tuning; stages trained; lr groups 5e-5 backbone / 5e-4 new; AdamW, cosine, label smoothing 0.1, EMA 0.999, CutMix p=0.5).
   2. **Distillation** (new, with an equation):
      L = (1 - a) L_CE(y, s) + a T^2 KL(softmax(t/T) || softmax(s/T)), a = 0.5, T = 2.
      The teacher (VideoMAE-B, Kinetics-400, fine-tuned on the same fold's training clips, 15 epochs) runs on the same augmented and CutMix-ed clip as the student. Schedule 48 epochs, patience 16.
      State explicitly: the teacher never sees validation or test clips; the code refuses a teacher trained on different clips.

F. **Inference**: fuse the ring, single model or fold ensemble, 1 or 4 temporal views, threshold 0.5 (chosen on OOF; tuning gained nothing).

Figures: pipeline (with teacher), Motion Attention, wide kernel (all three exist in the thesis; update the pipeline).

---

## IV. Experimental Protocol (about 600 words, new section)

A. **Data and audit**: RWF-2000 official split, 1,600 train / 400 test, 64 stored frames. Descriptor-based audit: 9 train clips are exact duplicates of test clips (all NonFight) and 9 duplicate pairs inside train; 17 train clips are removed in total (one clip is in both sets), leaving 1,583. Same-scene grouping (cosine similarity >= 0.9) gives 1,572 groups.

B. **Cross-validation**: stratified group 4-fold on the remaining 1,583 training clips (about 1,187 train / 396 validation per fold); a group never spans two folds. Checkpoint chosen on the fold's validation F1.

C. **Decisions on OOF only**: every choice (recipe, number of views, flip TTA, threshold on 0.30-0.70) uses the 1,583 OOF predictions. The test set is scored once per configuration with the average of the 4 fold models. Also report single-model test mean +- std.

D. **Seeds**: 3 seeds per row for the main table and the key ablations; mean +- std.

E. **Metrics**: accuracy, precision, recall, F1 for the Fight class, ROC-AUC. Cost: params (fused), GFLOPs per view (fvcore, 16x224x224), GPU latency and throughput (MI250, fp32 batch 1 and fp16 batch 16), CPU latency (16 threads), peak training memory.

F. **Implementation**: PyTorch 2.14 on ROCm 7.2, one MI250 GCD per fold; Triton depthwise 3D kernels (2.3x faster training step; one sentence). Code and split files released.

---

## V. Results (about 1,300 words)

A. **Main comparison** (Table I): prior work as reported (with protocol column), X3D-M, MA-X3D standard fine-tuning, MA-X3D final, VideoMAE-B teacher. Text: where MA-X3D sits in accuracy vs cost; say plainly that prior numbers with test selection are not directly comparable.

B. **Effect of the evaluation protocol** (small table or 3 sentences): the thesis recipe with test-set epoch selection vs the same recipe under the new protocol (was 90.5 vs about 86-88). One run of each recipe on the same folds. This is what justifies Section IV.

C. **Component ablation** (Table II, final recipe, 3 seeds):
   X3D-M | + MA | + WK | + MA + WK | + MA + WK, dense 5x5 (B1) | ring at backbone lr (B2).
   Columns: OOF acc, OOF F1, test acc, test F1, params, GFLOPs, ring energy.
   **[TODO: none of these exist under the final recipe yet.]**

D. **Training recipe ablation** (Table III, one change at a time, OOF decides) **[s0, to extend to 3 seeds for the rows kept]**:

   | Row | Change | OOF acc | Test acc |
   | --- | --- | --- | --- |
   | R0 | standard fine-tuning, backbone lr 1e-5 | 88.44 | 86.25 |
   | R1 | backbone lr 5e-5 | 89.83 | 88.50 |
   | R2 | R1 + distillation, 24 epochs | 89.83 | 88.00 |
   | R3 | R1 + distillation, 48 epochs | 91.03 | 87.75 (89.25, 4 views) |
   | R4 | R2 + teacher weight 0.9 | 90.46 | 88.25 |
   | -- | 32 frames (vs R0) | 88.76 | 86.75 |
   | -- | X3D-L backbone (vs R1) | 90.02 | 88.50 |

   Text: the gain comes from letting the backbone adapt and from a long distillation schedule; short distillation adds nothing; more frames and a deeper backbone do not pay for their cost.

E. **Teacher-student analysis** (short): gap closed; agreement rate between teacher and student on OOF; where the student still fails (clips the teacher gets right). Optional figure: OOF reliability or per-clip confidence scatter.

F. **Inference options** (2-3 sentences or a small table): 1 vs 2 vs 4 temporal views, flip TTA, threshold tuning, all on OOF: +0.25 points for 4 views, 0 for flip and threshold. Deployment uses 1 view.

G. **Qualitative analysis**: Grad-CAM with and without MA; gate maps on Fight vs NonFight; ring energy per layer. Error modes: crowded scenes, distant actors, sports/horseplay. (Figure: 2 rows x 4 clips.)

H. **Cost** (Table IV): params, GFLOPs, GPU latency / throughput, CPU latency, ONNX CPU latency, training time. Report honestly that on MI250 the teacher is not slower at batch 16 (136 vs 118 clips/s) because depthwise 3D convolutions are memory-bound; the advantage is 29x fewer parameters, 25x fewer FLOPs, and 2x lower CPU latency (208 vs 400 ms). If possible, add one edge device measurement (Jetson or a laptop CPU).

I. **Cross-dataset** (optional): Hockey Fight or RLVS under the same protocol, only if re-run.

---

## VI. Limitations (about 150 words)

- One benchmark (plus Hockey if re-run); RWF-2000 test is 400 clips, so 1 point = 4 clips.
- Test accuracy is 1.5-2 points below OOF for every model, including the teacher: the test clips appear harder than the training distribution.
- The static person crop assumes a fixed camera.
- Distillation needs a large teacher at training time (about 15 min per fold on one GPU).
- Clip-level labels only; no temporal localisation.

## VII. Conclusion (about 120 words)

Two modules + distillation + protocol; headline numbers; future work: temporal localisation on untrimmed streams, an edge-device study, a stronger or multi-teacher distillation, a public leakage-free split for RWF-2000.

---

## Tables and figures (budget for 6 pages)

| # | Content | Status |
| --- | --- | --- |
| Fig. 1 | Accuracy vs GFLOPs teaser | needs data from Table I |
| Fig. 2 | Pipeline with teacher (training only) | update thesis figure |
| Fig. 3 | Motion Attention | thesis figure |
| Fig. 4 | Wide kernel | thesis figure |
| Fig. 5 | Grad-CAM / gate maps | re-generate with the final model (`ma-x3d visualize`) |
| Tab. I | Comparison with prior work (protocol column) | fill |
| Tab. II | Component ablation | **run** |
| Tab. III | Training recipe ablation | have [s0] |
| Tab. IV | Cost | have |

---

## Experiments still needed before submission

Priority 1 (the paper cannot go without these):

1. Final recipe, 3 seeds: `e11_kd_long` seeds 0, 1, 2 (seeds 1-2 running).
2. Baseline, 3 seeds: `e00_baseline` seeds 0, 1, 2 (seeds 1-2 running).
3. Component ablation under the final recipe (Table II), 4 folds each, 3 seeds for X3D-M and the full model, at least 1 seed for the rest:
   X3D-M + distillation (no MA, no WK); + MA only; + WK only; dense 5x5 (B1); ring at backbone lr (B2).
   Configs to add: `configs/exp/a0..a3_*`, `b1`, `b2`, each `_base_: e11_kd_long.yaml` with one model change.
   Cost: about 50 min per fold on one GCD; 6 rows x 4 folds x 1-3 seeds = 24-72 fold runs, 5-15 h on 4 GCDs.
4. Protocol effect (Section V-B): the thesis recipe with test-set selection vs the same recipe on the folds (have: legacy runs and `e03_notebook_v5`).

Priority 2:

5. E16 (96 epochs) and E17 (48 epochs, teacher weight 0.9) are queued; keep the better one only if it wins on OOF over 3 seeds.
6. Qualitative figures from the final model.
7. One edge-device latency number.

Priority 3:

8. Hockey Fight / RLVS under the same protocol (needs the data on the server).
9. ROI ablation (needs the raw RWF-2000 videos).
