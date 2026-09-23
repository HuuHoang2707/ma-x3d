#!/bin/bash
# The two new modules: motion as features (TDM), and a SlowFast-style fast pathway
# that sees every frame at a fraction of the cost. Uncropped pipeline, four folds.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
until ! pgrep -f "ma_x3d.cli sweep" > /dev/null && ! pgrep -f "ma_x3d.cli train" > /dev/null; do
  sleep 300
done
.venv/bin/python -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
  --set data.num_workers=5 \
  --configs configs/mod/m1_tdm.yaml configs/mod/m2_tdm_wkall.yaml \
            configs/mod/m3_tdm_ma.yaml configs/mod/s1_slowfast.yaml \
            configs/mod/s2_slowfast64.yaml configs/mod/s3_slowfast_wkall.yaml
for d in runs/m1_tdm runs/m2_tdm_wkall runs/m3_ma_control \
         runs/s1_slowfast runs/s2_slowfast64 runs/s3_slowfast_wkall; do
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -m ma_x3d.cli cv "$d/seed0" --variants 1clip \
    > /dev/null 2>&1
done
echo "second module done $(date +%H:%M)"
