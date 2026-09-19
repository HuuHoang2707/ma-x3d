# Working with the code: step by step

Everything runs from the repository root. Commands use `.venv/bin/python -m ma_x3d.cli`;
after `source .venv/bin/activate` you can type `ma-x3d` instead.

## 1. Environment (once)

```bash
make env
```

This creates `.venv` with PyTorch for ROCm 7.2 (MI250) and installs the package with
the test tools. For ONNX export also run `uv pip install --python .venv/bin/python -e ".[edge]"`.
On an NVIDIA machine use `make env TORCH_INDEX=https://download.pytorch.org/whl/cu128`.

Check that the GPUs are visible:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

In VS Code, select `.venv/bin/python` as the interpreter so imports resolve.

## 2. Pick a free GPU

The MI250 node is shared. **On this node `HIP_VISIBLE_DEVICES` and `rocm-smi` number the
GPUs differently** (HIP 0-7 = rocm-smi 2, 3, 0, 1, 6, 7, 4, 5), so do not read an id off
`rocm-smi` and pass it to `HIP_VISIBLE_DEVICES`. Use:

```bash
.venv/bin/python -m ma_x3d.cli gpus
```

It lists every GPU with its HIP id (the one to use), its rocm-smi id, memory and use.
Pick GPUs with ~0% memory. `sweep --gpus auto` does this by itself and only starts a
job on a GPU nobody is using.

## 3. Data (once)

```bash
make data            # = ma-x3d prepare --h5-dir dataset --out dataset/rwf2000
```

This unpacks `dataset/rwf2000_train.h5` and `dataset/rwf2000_val.h5` into
`dataset/rwf2000/{train,test}_{x,y}.npy` (about 19 GB). The RWF-2000 folder called
`val` is the official 400-clip test set, so it is saved as `test`.

## 4. Tests

```bash
make test        # fast unit tests (~20 s)
make test-all    # plus the full-network tests and a tiny CPU training run (~2 min)
```

Run `make test` after every code change. A single test file or test:

```bash
.venv/bin/python -m pytest tests/test_modules.py -q
.venv/bin/python -m pytest tests/test_model.py -k exact_phases -q
```

## 5. Train one model

```bash
make smoke GPU=4                                  # 2 short epochs, checks everything works
make train GPU=4                                  # MA-X3D, seed 0
make train GPU=5 CONFIG=configs/x3d_m.yaml        # baseline
```

or directly, with a seed and config overrides:

```bash
HIP_VISIBLE_DEVICES=4 .venv/bin/python -m ma_x3d.cli train configs/ma_x3d.yaml \
    --seed 1 --set train.batch_size=16 data.num_workers=16
```

In a terminal you see a progress bar per epoch (loss, accuracy, ETA) and one summary
line per epoch:

```text
ep 07/24 finetune | train 0.3121/0.912 | val 0.3650/0.881 f1 0.8859 * | 58s | eta 16 min
```

`*` marks a new best validation F1 (that checkpoint is saved as `best.pt`).

To keep a run alive after you log out, use `nohup` or `tmux`:

```bash
HIP_VISIBLE_DEVICES=4 nohup .venv/bin/python -m ma_x3d.cli train configs/ma_x3d.yaml \
    > runs/ma_x3d_seed0.out 2>&1 &
tail -f runs/ma_x3d_seed0.out
```

Without a terminal the bar turns into a line every 25% of an epoch.
`MA_X3D_PROGRESS=0` silences it, `MA_X3D_PROGRESS=1` forces the bar.

The lines `(null): No such file or directory` at start-up come from libdrm on this node
(it cannot find its GPU name table) and are harmless.

### Speed

Depthwise 3D convolutions run through Triton kernels (`src/ma_x3d/ops/depthwise3d.py`)
because MIOpen handles them with slow generic kernels. On one MI250 GCD a training
step (batch 12, bf16) takes 233 ms for MA-X3D instead of 531 ms, and 154 ms for X3D-M
instead of 298 ms. `--set train.fast_depthwise=false` switches back to MIOpen. The
block sizes can be re-tuned with `MA_X3D_DW_BLOCK` / `MA_X3D_DW_BLOCK_W`; the GPU
tests (`pytest tests/test_gpu.py`) check the kernels against `F.conv3d`.

### One run on several GPUs

```bash
make train-ddp DDP_GPUS=4,5,6,7 CONFIG=configs/legacy/notebook_v5.yaml
# = HIP_VISIBLE_DEVICES=4,5,6,7 .venv/bin/torchrun --standalone --nproc_per_node=4 \
#     -m ma_x3d.cli train configs/legacy/notebook_v5.yaml
```

`train.batch_size` stays the total batch: 20 on 4 GPUs is 5 clips per GPU, with
BatchNorm synchronised across GPUs, so the training is the same as on one GPU. The batch
size must be divisible by the number of GPUs. Measured at 32 frames, batch 20: 91 s per
epoch on 4 GPUs against 253 s on one. At 16 frames with batch 12 (3 clips per GPU) the
gain is small; there, running different seeds on different GPUs (`sweep`) is the
better use of the GPUs.

### What a run writes

```text
runs/ma_x3d/seed0/
  config.yaml          full config actually used
  env.json             torch/ROCm versions, GPU, git commit
  split.json           clip indices of train / val / test
  metrics.jsonl        one line per epoch
  train.log            console output
  best.pt              weights of the best validation epoch
  results.json         test metrics of best.pt, ring energy, ring-off test, fused check
  test_predictions.npz probabilities per test clip
```

### Resume, rerun

```bash
... train configs/ma_x3d.yaml --resume      # continue an interrupted run
... train configs/ma_x3d.yaml --overwrite   # delete the run and start again
```

A finished run (with `results.json`) is skipped, so re-launching a sweep only runs
what is missing.

## 6. Evaluate

`results.json` already holds the test metrics. To evaluate again, on another split, or
with the ring switched off:

```bash
.venv/bin/python -m ma_x3d.cli eval runs/ma_x3d/seed0
.venv/bin/python -m ma_x3d.cli eval runs/ma_x3d/seed0 --split val
.venv/bin/python -m ma_x3d.cli eval runs/ma_x3d/seed0 --ring-off
.venv/bin/python -m ma_x3d.cli eval runs/ma_x3d/seed0 --fuse    # the deployed form
```

## 7. Ablations and seeds

The plan is in [ablation_plan.md](ablation_plan.md). One config per row in
`configs/ablation/`. The sweep runs every (config, seed) pair, one job per GPU:

```bash
make ablation-p1 GPUS="4 5 6 7"        # priority-1 rows, seeds 0 1 2
make ablation-all GPUS="4 5 6 7"       # every row
.venv/bin/python -m ma_x3d.cli sweep --gpus 4 5 --seeds 0 1 2 3 4 \
    --configs configs/ablation/a1_ma.yaml configs/ablation/a3_ma_wk.yaml
```

Logs go to `runs/logs/`. Stopping the sweep (Ctrl-C) leaves partial runs; start it again
with `--resume`.

To add an experiment, copy a file in `configs/ablation/`, give it a new `name` and
`description`, and change only the keys you are testing. Every key is listed in
`configs/base.yaml` and documented in `src/ma_x3d/config.py`.

## 8. Report

```bash
make report
```

writes

- `reports/summary.md`: one row per experiment, mean ± std over seeds of test accuracy,
  Fight-F1, AUC, validation F1, ring-off accuracy and ring energy;
- `reports/ablation_rows.tex`: LaTeX rows for the paper's tables;
- `runs/<name>/seed<N>/curves.png`: loss, accuracy and F1 per epoch.

## 8b. How to improve the model

### Rules

1. **Decide on validation, never on test.** Keep `data.protocol: holdout`. Compare
   runs by validation F1 and validation loss. Look at the test score only for the final
   configuration. Choosing on test is how the Kaggle 90.5% happened.
2. **Change one thing at a time**, so you know what caused a difference.
3. **Screen with 1 seed, confirm with 3.** On 400 test clips, one clip is 0.25 points,
   and seeds differ by 1-3 points. A gap smaller than the std over seeds is noise.
4. **Commit before a sweep.** Each run records the commit in `env.json`.

### The loop

```bash
# 1. make an experiment: copy a config, give it a new name, change one key
cp configs/ma_x3d.yaml configs/tune/t01_lr_new_2e4.yaml   # edit: name + the one change
# or without a file:
... train configs/ma_x3d.yaml --set name=t01_lr_new_2e4 train.lr_new=2e-4

# 2. screen: 1 seed (16 frames: one run per GPU)
.venv/bin/python -m ma_x3d.cli sweep --seeds 0 --gpus 4 5 6 7 \
    --configs configs/tune/t01_*.yaml configs/tune/t02_*.yaml configs/tune/t03_*.yaml

# 3. look at the curves and the table
make report          # reports/summary.md; curves in runs/<name>/seed0/curves.png

# 4. confirm the best one or two with seeds 1 and 2, then compare mean +- std
```

### Reading the curves (`curves.png`)

| What you see | Meaning | Try |
| --- | --- | --- |
| val loss rises while train loss falls; train acc >> val acc | overfitting | `ema_decay: 0.999`, `cutmix_prob: 0.5`, `head_new: proj`, lower `lr_new`, higher `weight_decay` |
| val loss rises from epoch 1-2 even in the probe | train and val data differ | check sampling (`train_span`), frames, augmentation |
| train and val both flat, train acc low | underfitting | more epochs, higher `lr_new`, add `res4` to `unfreeze`, less augmentation |
| val F1 jumps up and down by several points | small val set, noise | trust val loss more, use 3 seeds |
| best epoch is in the probe phase | fine-tuning hurts | lower `lr_new` / `lr_backbone`, more regularisation |

### Knobs that mattered so far (seed 0, RWF-2000)

| Change | Effect seen |
| --- | --- |
| 32 frames instead of 16 (notebook v5 setup) | best result so far: 88.75% test, val loss 0.40 and still falling |
| `train_span: [0.6, 1.0]` (training window matches evaluation) | removed the early val-loss rise; the wide kernel started to be used |
| EMA 0.999 + CutMix 0.5 | val loss flat instead of rising; alone, each helped less |
| `head_new: proj` | small gain on top of EMA + CutMix |
| weight decay 0.05, lower head LR, BN updates, no normalisation | no clear gain |

Good next experiments: the new recipe at 32 frames
(`configs/ablation/e1_frames32.yaml`), and the notebook-v5 setup with
`--set data.protocol=holdout` so its number is valid for the paper.

### Making runs faster

- 16 frames: one run per GPU with `sweep`. 4 GPUs = 4 experiments at once.
- 32 frames: one run on 4 GPUs with `make train-ddp` (about 3x faster per run).
- Keep `train.fast_depthwise: true` (Triton kernels, 2.3x faster steps).
- A run is data-bound if the GPU use in `rocm-smi --showuse` stays low: raise
  `data.num_workers`.
- `data.augment_multiplier: 1` halves the epoch time for quick screening.

### Final numbers for the paper

Pick the configuration by validation, then run it with 3 (or 5) seeds and report
test mean +- std from `reports/summary.md`. Run the baseline and every ablation row
the same way.

## 9. Cost and figures

```bash
make bench GPU=4                                           # config, untrained weights
.venv/bin/python -m ma_x3d.cli bench --run runs/ma_x3d/seed0 --batch 16 --amp fp16
.venv/bin/python -m ma_x3d.cli visualize runs/ma_x3d/seed0 --index 0 3 17 --out reports/figures
```

`bench` fuses the wide kernels first, so parameters and GFLOPs are those of the deployed
model. GFLOPs count one multiply-add as one operation (fvcore / X3D convention).

## 10. Edge export

```bash
uv pip install --python .venv/bin/python -e ".[edge]"
.venv/bin/python -m ma_x3d.cli export runs/ma_x3d/seed0 --out exports/ma_x3d --int8
```

See [edge_inference.md](edge_inference.md).

## 11. Numbers for the paper

`reports/summary.md` has every experiment as mean ± std over seeds, and
`reports/ablation_rows.tex` has the same numbers as LaTeX table rows.

## 12. Git

```bash
git status
git add -A && git commit -m "..."     # dataset/, runs/, *.pt are ignored
git push origin main
```

Commit configs and code before a sweep; every run records the commit it was started
from (`env.json`), and whether the tree had uncommitted changes.
