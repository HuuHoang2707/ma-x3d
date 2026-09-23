#!/bin/bash
# Headline model, four GPUs, once the seed sweep is done.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
until ! pgrep -f "ma_x3d.cli sweep" > /dev/null && ! pgrep -f "ma_x3d.cli train" > /dev/null; do
  sleep 300
done
G=$(.venv/bin/python -m ma_x3d.cli gpus | awk 'NR>1 && $5<=35 {print $1}' | head -4 | paste -sd,)
HIP_VISIBLE_DEVICES=$G .venv/bin/torchrun --standalone --nproc_per_node=4 --master_port=29690 \
  -m ma_x3d.cli train configs/final/f6_wkall_joint.yaml --seed 0 > runs/logs/f6_ddp.log 2>&1
.venv/bin/python -c "
import json
r = json.load(open('runs/f6_wkall_joint/seed0/results.json'))['test']
print('f6 test %.2f  F1 %.3f  AUC %.3f' % (100*r['accuracy'], r['f1'], r['auc']))"
