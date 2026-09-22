#!/bin/bash
# Separate the two modules on the preprocessing we ship: MA alone, WK alone, and the
# all-stage WK variant that looked best on the cropped data.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
until ! pgrep -f "ma_x3d.cli sweep" > /dev/null && ! pgrep -f "ma_x3d.cli train" > /dev/null \
      && [ -f runs/f5_best_kd_none/seed0/results.json -o -f runs/final_done.flag ]; do sleep 300; done
.venv/bin/python -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
  --set data.num_workers=5 \
  --configs configs/win/wa_ma.yaml configs/win/wa_wk.yaml configs/win/wa_wk_all.yaml
for d in runs/wa_ma_none/seed0 runs/wa_wk_none/seed0 runs/wa_wk_all_none/seed0; do
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -m ma_x3d.cli cv "$d" --variants 1clip > /dev/null 2>&1
done
echo "module separation done $(date +%H:%M)"
