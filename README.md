# MA-X3D

A lightweight violence detector for surveillance video: X3D-M with two small modules.

- **Motion Attention** gates the Res4 features with a frame-difference motion map
  (4 learnable temporal modes, tanh gate, starts as an identity).
- **Wide-Kernel** widens the depthwise kernels of Res2 and Res3 from 3×3 to 5×5 by
  structural reparameterisation; the new outer ring has its own learning rate and is
  fused into one kernel after training.

About 3.02M parameters and 5.16 GFLOPs per 16-frame clip. Paper draft in `paper/`,
graduation thesis results in the table at the bottom.

## Quick start

```bash
make env          # .venv with ROCm PyTorch (MI250) + this package
make data         # dataset/*.h5 -> dataset/rwf2000/*.npy  (once, ~3 min)
make test         # unit tests, ~20 s
make smoke GPU=4  # 2 short epochs on the real data
make train GPU=4  # MA-X3D, 1 seed
```

See [docs/GUIDE.md](docs/GUIDE.md) for the full workflow (training, resuming,
evaluation, ablations, reports, figures, export), and
[docs/ablation_plan.md](docs/ablation_plan.md) for the experiment plan.

## Layout

```text
configs/            experiment configs (YAML, inherit from base.yaml)
  ablation/         one file per ablation row (docs/ablation_plan.md)
  legacy/           Kaggle-notebook behaviour, for comparison with the thesis
src/ma_x3d/
  data/             prepare (HDF5 -> npy), dataset, augmentation, splits, raw-video builder
  models/           X3D-M wrapper, Motion Attention, wide kernel, EAA
  train/            two-phase training loop, parameter groups
  eval/             metrics, prediction, benchmark, Grad-CAM / gate maps
  export.py         ONNX export, INT8 quantisation, CPU timing
  report.py         result tables and curves from runs/
  cli.py            `ma-x3d <command>`
tests/              pytest suite
docs/               guide, ablation plan, edge inference notes
paper/              conference paper (LaTeX)
notebooks/          the original Kaggle notebook
dataset/, runs/     data and outputs (not in git)
```

## Commands

| Command | What it does |
| --- | --- |
| `ma-x3d prepare` | convert the HDF5 files to memory-mapped `.npy` arrays |
| `ma-x3d train CONFIG [--seed N] [--set key=value ...]` | train one run into `runs/<name>/seed<N>/` |
| `ma-x3d sweep --configs ... --seeds 0 1 2 --gpus 4 5 6 7` | many runs, one per GPU at a time |
| `ma-x3d eval RUN [--ring-off] [--fuse]` | re-evaluate the best checkpoint |
| `ma-x3d bench [CONFIG \| --run RUN]` | parameters, GFLOPs, latency, throughput |
| `ma-x3d report` | `reports/summary.md` (mean ± std over seeds), LaTeX rows, curves |
| `ma-x3d visualize RUN --index 0 5 9` | gate and Grad-CAM figures |
| `ma-x3d export RUN [--int8]` | ONNX export, accuracy check and CPU timing |
| `ma-x3d build-dataset --videos RWF-2000/` | rebuild the HDF5 files from raw videos |

## Differences from the Kaggle notebook

The code in `notebooks/x3dv5_kaggle.ipynb` was ported with these changes. Each one is a
config option, and `configs/legacy/` switches them back.

| | Notebook | Here | Option |
| --- | --- | --- | --- |
| Checkpoint selection | best epoch on the 400 test clips | best epoch on 160 held-out training clips | `data.protocol` |
| Trainable parameters | substring match; parts of Res4/Res5 trained while "frozen" | exact stage prefixes | `train.param_match` |
| Stages fine-tuned | depended on the model variant | same list for every model | `train.unfreeze` |
| Input | raw [0, 1] frames | Kinetics mean/std | `model.normalize_input` |
| BN in frozen stages | running stats updated (thesis) | frozen | `train.freeze_bn` |
| Data loading | gzip HDF5, ~4 s per clip | `.npy` memmap, ~2 ms per clip | `ma-x3d prepare` |
| Precision | fp16 (T4) | bf16 (MI250) | `train.amp` |
| Depthwise 3D convs | MIOpen (im2col / naive kernels on ROCm) | Triton kernels, same result, 2.3× faster training step | `train.fast_depthwise` |

## Thesis results (to be replaced)

From the thesis (Kaggle, 2× T4, one run, checkpoint chosen on the test set). The
re-runs with the protocol above go in `reports/summary.md`.

| Model | Params | GFLOPs | Acc. (%) | Fight F1 |
| --- | --- | --- | --- | --- |
| X3D-M, full frame | 2.98M | 4.73 | 84.75 | – |
| X3D-M, person crop | 2.98M | 4.73 | 86.75 | – |
| + Motion Attention | 2.98M | – | 89.50 | – |
| + Wide-Kernel (MA-X3D) | 3.02M | 5.16 | 90.50 | 0.91 |
