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

The MI250 node is shared. Look at memory use and pick GCDs that are empty:

```bash
rocm-smi --showmemuse | grep VRAM
```

Each run then gets one GCD through `HIP_VISIBLE_DEVICES` (the Makefile does this with
`GPU=`). Inside the process the chosen GCD is always `cuda:0`.

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
