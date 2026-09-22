#!/bin/bash
# Once the final chain is done: re-measure what the modules are worth on the winning
# preprocessing (the old ablation used the cropped data), and test a longer clip.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python

until [ -f runs/winner.txt ]; do sleep 120; done
W=$(cat runs/winner.txt)
until ! pgrep -f "final_chain.sh" > /dev/null; do sleep 300; done
until ! pgrep -f "ma_x3d.cli train" > /dev/null; do sleep 120; done

cat > configs/win/wa_x3d.yaml <<CFG
# Control: no MA-X3D modules, so the module delta is measured on the data we ship.
_base_: ../prep/rp_$W.yaml
name: wa_x3d_$W
description: plain X3D-M, winning preprocessing
model:
  motion_attention: false
  wide_kernel: none
CFG
cat > configs/win/wa_diff.yaml <<CFG
# MA-X3D plus the difference residual, the one added module that paid off.
_base_: ../prep/rp_$W.yaml
name: wa_diff_$W
description: MA-X3D + difference residual, winning preprocessing
model:
  diff_residual: [res3, res4]
CFG
cat > configs/win/wa_t32.yaml <<CFG
# Twice the temporal context; batch halved to fit.
_base_: ../prep/rp_$W.yaml
name: wa_t32_$W
description: MA-X3D, 32 frames, winning preprocessing
data:
  frames: 32
train:
  batch_size: 6
CFG

$P -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 --set data.num_workers=5 \
  --configs configs/win/wa_x3d.yaml configs/win/wa_diff.yaml configs/win/wa_t32.yaml
for d in runs/wa_*/seed0; do
  HIP_VISIBLE_DEVICES=0 $P -m ma_x3d.cli cv "$d" --variants 1clip > /dev/null 2>&1
done
echo "module ablation on the winning preprocessing done $(date +%H:%M)"
