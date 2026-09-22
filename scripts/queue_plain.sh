#!/bin/bash
# The sixth variant, once the dataset is built and the union folds have the GPUs free.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
until [ -f dataset/rwf_plain/train_groups.npy ]; do sleep 60; done
until ! pgrep -f "ma_x3d.cli train" > /dev/null; do sleep 120; done
.venv/bin/python -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
  --set data.num_workers=5 --configs configs/prep/rp_plain.yaml
echo "rp_plain done $(date +%H:%M)"
