# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MA-X3D: a violence detector (RWF-2000, binary Fight/NonFight) built on Kinetics-pretrained
X3D-M (pytorchvideo) with two modules, Motion Attention after Res4 and reparameterised 5×5
depthwise kernels in Res2/Res3. It started as a Kaggle notebook (`notebooks/x3dv5_kaggle.ipynb`,
kept for reference) and a graduation thesis. It is now a package (`src/ma_x3d`) plus a
conference paper in `paper/main.tex`. The user's thesis numbers came from that notebook.

## Commands

```bash
make env                     # .venv with ROCm 7.2 PyTorch + package (uv)
make data                    # dataset/*.h5 -> dataset/rwf2000/*.npy (needed before training)
make test                    # fast tests; `make test-all` adds full-network + CPU training tests
.venv/bin/python -m pytest tests/test_model.py -k exact_phases -q   # single test
make lint                    # ruff
HIP_VISIBLE_DEVICES=4 .venv/bin/python -m ma_x3d.cli train configs/ma_x3d.yaml --seed 0 --set train.epochs=2
.venv/bin/python -m ma_x3d.cli sweep --configs configs/ablation/*.yaml --seeds 0 1 2 --gpus 4 5 6 7
.venv/bin/python -m ma_x3d.cli report        # reports/summary.md from runs/*/seed*/results.json
.venv/bin/python -m ma_x3d.cli cv runs/<exp>/seed0   # k-fold: OOF decisions + fold-ensemble test
```

`docs/GUIDE.md` has the full workflow; `docs/ablation_plan.md` lists every experiment row.

## Hardware

The host is a shared node with 8 MI250 GCDs (ROCm, gfx90a). Other users occupy some GCDs:
use `ma-x3d gpus` (HIP ids differ from rocm-smi ids on this node: HIP 0-7 = rocm-smi 2,3,0,1,6,7,4,5) and pin jobs with `HIP_VISIBLE_DEVICES`; `sweep --gpus auto` picks idle GPUs. PyTorch uses the
`cuda` API on ROCm. The system `python3` has a CUDA build of torch and cannot use the GPUs,
so always use `.venv/bin/python`.

## Architecture (things that span files)

- **Config** (`config.py`): dataclasses, YAML with `_base_` inheritance, `--set a.b=c`
  overrides. Unknown keys raise. Every experiment is a YAML in `configs/`, and the run
  directory is `runs/<name>/seed<seed>/`, so `name` must be unique per config (a test
  enforces this).
- **Model input contract**: `MAX3D.forward(clip)` takes RGB in [0, 1], shape [B,3,T,224,224].
  It computes the motion map itself (on [0,1] frames) and then applies Kinetics
  normalisation. Datasets must not normalise. The X3D head pools with a fixed 7×7 kernel,
  so inputs must be 224×224 and T ≥ 16.
- **Stage indices**: `blocks[0..5]` = stem, res2, res3, res4, res5, head (`STAGES` in
  `models/ma_x3d.py`). Module names (`blocks`, `motion_attn`, `eaa`, `base_weight`,
  `delta_weight`, `ring_mask`) match the Kaggle notebook so its checkpoints load; don't rename.
- **Two-phase training** (`train/engine.py` + `train/params.py`): the probe phase trains the
  head and new modules at `lr_probe`; fine-tuning trains `train.unfreeze` stages at
  `lr_backbone` and new modules plus `delta_weight` at `lr_new`. The optimizer is rebuilt at
  the phase switch. `param_match: legacy` deliberately reproduces the notebook's
  substring-matching bug ("blocks.5" matches "res_blocks.5"); keep it for `configs/legacy/`.
- **Protocol**: `data.protocol: holdout` picks the checkpoint on 160 training clips held
  out in contiguous index blocks (same-source clips are adjacent), then tests once.
  `test_as_val` is the notebook's selection on the test set; only legacy configs use it.
- **Wide kernel**: train in reparam form, fuse for inference (`fuse_wide_kernels`).
  `results.json` records ring energy, test accuracy with the ring zeroed, and a
  fused-vs-unfused check.
- **Final recipe** (`configs/exp/e11_kd_long.yaml`): grouped 4-fold CV, backbone lr 5e-5,
  48 epochs, distillation from a per-fold VideoMAE-B teacher (`configs/exp/t01_videomae.yaml`,
  `models/teacher.py`). The student checks the teacher trained on the same clips. GUIDE 8c.

## Data

`dataset/rwf2000_{train,val}.h5`: `Fight`/`NonFight` uint8 `[N,64,3,224,224]`, RGB,
person-cropped. Gzip chunks span 25–50 clips (≈4 s per random clip read), which is why
training reads `dataset/rwf2000/*.npy` memmaps made by `ma-x3d prepare`. Label 1 = Fight;
Fight clips come first. The raw videos and Hockey Fight are not on this machine, so the
ROI ablation and the Hockey result cannot be re-run here.

## Paper

`paper/main.tex` (IEEEtran) is written and compiled by the user. Do not edit or
compile it unless asked. `paper/drafts/` holds an earlier suggested Experiments
section and a `refs.bib` built from the thesis references, for the user to reuse.
