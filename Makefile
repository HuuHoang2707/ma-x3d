# Common commands. Pick free GPUs with `rocm-smi --showmemuse`, then e.g. `make train GPU=4`.
PY          ?= .venv/bin/python
CLI         := $(PY) -m ma_x3d.cli
TORCH_INDEX ?= https://download.pytorch.org/whl/rocm7.2
GPU         ?= 4
GPUS        ?= 4 5 6 7
SEEDS       ?= 0 1 2
CONFIG      ?= configs/ma_x3d.yaml
RUN         ?= runs/ma_x3d/seed0
DDP_GPUS    ?= 4,5,6,7
comma       := ,

.PHONY: env data test test-all lint format smoke train train-ddp eval bench report ablation-p1 ablation-all legacy export

env:            ## create .venv with ROCm PyTorch and the package
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) --index-url $(TORCH_INDEX) torch==2.14.0 torchvision==0.29.0
	uv pip install --python $(PY) -e ".[dev,teacher]"

data:           ## HDF5 -> memory-mapped .npy (once, ~3 min)
	$(CLI) prepare --h5-dir dataset --out dataset/rwf2000

test:           ## fast unit tests (no full network)
	$(PY) -m pytest -m "not slow and not gpu" -q

test-all:       ## all tests, including the CPU end-to-end run (~2 min)
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests

format:
	.venv/bin/ruff check --fix src tests
	.venv/bin/ruff format src tests

smoke:          ## 2 short epochs on the real data to check the GPU setup
	HIP_VISIBLE_DEVICES=$(GPU) $(CLI) train $(CONFIG) --overwrite --set name=smoke \
	  train.epochs=2 train.probe_epochs=1 data.augment_multiplier=1

train:          ## one run: make train CONFIG=configs/x3d_m.yaml GPU=5
	HIP_VISIBLE_DEVICES=$(GPU) $(CLI) train $(CONFIG)

train-ddp:      ## one run on several GPUs: make train-ddp DDP_GPUS=4,5,6,7 CONFIG=...
	HIP_VISIBLE_DEVICES=$(DDP_GPUS) .venv/bin/torchrun --standalone \
	  --nproc_per_node=$(words $(subst $(comma), ,$(DDP_GPUS))) -m ma_x3d.cli train $(CONFIG)

eval:           ## re-evaluate a finished run on the test split
	HIP_VISIBLE_DEVICES=$(GPU) $(CLI) eval $(RUN)

bench:          ## params / GFLOPs / latency of a config
	HIP_VISIBLE_DEVICES=$(GPU) $(CLI) bench $(CONFIG)

report:         ## tables + curves from everything in runs/
	$(CLI) report --runs runs --out reports

ablation-p1:    ## priority-1 ablations (docs/ablation_plan.md), 3 seeds each
	$(CLI) sweep --gpus $(GPUS) --seeds $(SEEDS) --configs \
	  configs/ablation/a0_x3d_m.yaml configs/ablation/a1_ma.yaml configs/ablation/a2_wk.yaml \
	  configs/ablation/a3_ma_wk.yaml configs/ablation/b1_wk_dense.yaml \
	  configs/ablation/b2_ring_lr_backbone.yaml configs/legacy/thesis.yaml

ablation-all:   ## every ablation config, 3 seeds each
	$(CLI) sweep --gpus $(GPUS) --seeds $(SEEDS) --configs configs/ablation/*.yaml

legacy:         ## reproduce the thesis setup (checkpoint picked on the test set)
	$(CLI) sweep --gpus $(GPUS) --seeds $(SEEDS) --configs configs/legacy/thesis.yaml

export:         ## ONNX export of a trained run for edge inference
	$(CLI) export $(RUN) --out exports/$(notdir $(patsubst %/,%,$(dir $(RUN))))
