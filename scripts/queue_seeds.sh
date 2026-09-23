#!/bin/bash
# The all-stage wide kernel and the 32-frame clip are the only changes with a
# significant test gain, and both are on one seed. Two more seeds each, with the
# X3D-M control, so the claim survives the seed check that sank earlier results.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
until ! pgrep -f "ma_x3d.cli sweep" > /dev/null && ! pgrep -f "ma_x3d.cli train" > /dev/null; do
  sleep 300
done
.venv/bin/python -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 1 2 --folds 0 1 2 3 \
  --set data.num_workers=5 \
  --configs configs/win/wa_x3d.yaml configs/win/wa_wk_all.yaml configs/win/wa_t32.yaml
for s in 1 2; do
  for d in runs/wa_x3d_none/seed$s runs/wa_wk_all_none/seed$s runs/wa_t32_none/seed$s; do
    HIP_VISIBLE_DEVICES=0 .venv/bin/python -m ma_x3d.cli cv "$d" --variants 1clip > /dev/null 2>&1
  done
done
echo "seed check done $(date +%H:%M)"
