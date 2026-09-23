#!/bin/bash
# The second module: motion as features (TDM) alone, with the wide kernels, and the
# gating alternative as its control. Uncropped pipeline, four folds each.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
until ! pgrep -f "ma_x3d.cli sweep" > /dev/null && ! pgrep -f "ma_x3d.cli train" > /dev/null; do
  sleep 300
done
.venv/bin/python -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
  --set data.num_workers=5 \
  --configs configs/mod/m1_tdm.yaml configs/mod/m2_tdm_wkall.yaml configs/mod/m3_tdm_ma.yaml
for d in runs/m1_tdm runs/m2_tdm_wkall runs/m3_ma_control; do
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -m ma_x3d.cli cv "$d/seed0" --variants 1clip \
    > /dev/null 2>&1
done
echo "second module done $(date +%H:%M)"
