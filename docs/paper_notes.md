# Notes for the paper

Things found while porting the code that affect what `paper/main.tex` may claim.

## Must change

1. **Checkpoint selection.** The thesis numbers (90.5%, the ablation table, 97.0% on
   Hockey Fight) come from runs that kept the epoch with the best F1 on the 400 test
   clips (`notebooks/x3dv5_kaggle.ipynb`, last cell: the test loader is passed as the
   validation loader). The thesis text says a separate validation split was used. Either
   re-run with the held-out protocol (`data.protocol: holdout`, the default) or describe
   the selection as it was. Row L1 of the ablation plan measures the difference.
2. **GFLOPs.** MA-X3D needs **5.46** GFLOPs per 16-frame clip, not 5.16. The same counter
   gives 4.73 for X3D-M, the published value. By hand, the eight widened depthwise
   convolutions add 54·16·56·56·3·16·3 ≈ 0.39 G (Res2) + 108·16·28·28·3·16·5 ≈ 0.33 G
   (Res3). Parameters are right (3.015M deployed, 2.98M for X3D-M).
3. **"Res4 stays frozen" / "entire backbone frozen in the probe phase."** Not true of the
   notebook. Its substring matching trained block 5 of Res4 and of Res5 already in the
   probe phase, and for the wide-kernel model also Res4 blocks 1, 2, 4 and 10 during
   fine-tuning; the two block-5s ran at the 5e-4 rate of the new modules. The new code
   does what the text says; the old runs did not (`tests/test_model.py` shows both).
4. **Input normalisation.** Thesis 3.2.2 says normalisation happens at load time. The
   notebook never normalised; frames went in as [0, 1].
5. **Temporal sampling (Eq. 3.1–3.2).** The training window is `T·r` stored frames with
   T = 16 and r ∈ [0.7, 1.5], i.e. 11–24 of the 64 stored frames (about 1–2 s), while
   evaluation spreads the 16 frames over all 64 (5 s). Training and test clips therefore
   have different time scales, and the frame differences that feed Motion Attention are
   several times larger at test time. With the window set to 60–100% of the stored clip
   (`data.train_span: [0.6, 1.0]`), validation loss and F1 improve and the wide-kernel
   ring starts to matter (seed 0: zeroing it costs 5.5 test points, against 0 before).
   If the paper keeps Eq. 3.1–3.2, it describes a sampler that hurts the model.

## Inconsistencies inside the thesis

- Fig. 3.1 shows 64 → 32 frames and a 3×32×224² input; the text and Table 4.1 say 16.
  The last notebook used 32. The 5.46 GFLOPs above are for 16 frames (32 frames: 10.9).
- Sec. 4.4.2 gives the outer-ring energy as "1696"; Table 4.7 says 0.1696.
- Sec. 4.4.2 says the ring-off evaluation gives "a gain of 1%", which equals the
  cumulative-table difference (89.5 → 90.5) from a separately trained model. The code
  now reports the ring-off accuracy of the same checkpoint (`test_ring_off`).
- In the thesis ablation only the wide-kernel model also fine-tuned Res2/Res3, so its
  +1.0 mixes the wider kernel with unfreezing two stages.
- Sec. 2.2 says X3D-M has "about 3.8 million parameters"; Sec. 2.3.3 and Table 2.3 use
  3.76M. Both are the 400-class model; with 2 classes it is 2.98M.

## References

`paper/drafts/refs.bib` has every key cited in `main.tex`, built from the thesis list
with these fixes: SlowFast is ICCV 2019, C3D is ICCV 2015, Video Swin is CVPR 2022,
SE-Net CVPR 2018 has three authors (Albanie and Wu are on the TPAMI version), Grad-CAM
is ICCV 2017, Hockey Fight's first author is Bermejo Nievas.
