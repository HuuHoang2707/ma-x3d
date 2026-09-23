#!/bin/bash
# Kernel study: which axis, how wide, which stages, and whether the reparameterisation
# matters. Five configurations, four folds each, on the uncropped pipeline.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
until ! pgrep -f "ma_x3d.cli sweep" > /dev/null && ! pgrep -f "ma_x3d.cli train" > /dev/null; do
  sleep 300
done
.venv/bin/python -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
  --set data.num_workers=5 \
  --configs configs/wk/wk_t.yaml configs/wk/wk_st.yaml configs/wk/wk_s7.yaml \
            configs/wk/wk_late.yaml configs/wk/wk_dense_all.yaml
for d in runs/wk_t runs/wk_st runs/wk_s7 runs/wk_late runs/wk_dense_all; do
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -m ma_x3d.cli cv "$d/seed0" --variants 1clip \
    > /dev/null 2>&1
done
echo "kernel study done $(date +%H:%M)"
