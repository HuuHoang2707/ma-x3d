#!/bin/bash
# After the full-split DDP runs finish, train the five preprocessing variants,
# one job per GPU (20 runs), then evaluate them.
cd /remote/vast0/hoangnguyenhuu/ma-x3d
while pgrep -f "torchrun --standalone" > /dev/null; do sleep 120; done
.venv/bin/python -m ma_x3d.cli sweep --gpus auto --seeds 0 --folds 0 1 2 3 \
  --configs configs/prep/rp_cluster.yaml configs/prep/rp_adaptive.yaml \
            configs/prep/rp_none.yaml configs/prep/rp_noenh.yaml configs/prep/rp_union.yaml
for d in runs/rp_*/seed0; do
  [ -f $d/cv_eval.json ] || HIP_VISIBLE_DEVICES=0 .venv/bin/python -m ma_x3d.cli cv $d --variants 1clip > /dev/null 2>&1
done
echo "variants trained and evaluated $(date +%H:%M)"
