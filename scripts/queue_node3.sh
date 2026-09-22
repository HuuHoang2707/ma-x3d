#!/bin/bash
# Node 3 shares its GPUs with another user's vLLM service: pin the four least-loaded
# GCDs instead of asking for idle ones, and run the three variants that are still missing.
cd /remote/vast0/hoangnguyenhuu/ma-x3d
.venv/bin/python -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
  --set data.num_workers=5 \
  --configs configs/prep/rp_none.yaml configs/prep/rp_noenh.yaml configs/prep/rp_union.yaml
