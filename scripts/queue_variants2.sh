#!/bin/bash
# The last two variants (plain, aspect), then tell the final chain the study is complete.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
until [ -f dataset/rwf_aspect/train_groups.npy ]; do sleep 60; done
until [ "$(ls runs/rp_plain/seed0/fold*/results.json 2>/dev/null | wc -l)" -eq 4 ]; do sleep 120; done
until ! pgrep -f "ma_x3d.cli train" > /dev/null; do sleep 120; done
.venv/bin/python -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
  --set data.num_workers=5 --configs configs/prep/rp_aspect.yaml
touch runs/variants_done.flag
echo "all seven variants done $(date +%H:%M)"
