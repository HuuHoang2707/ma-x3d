#!/bin/bash
# Is it the reparameterisation or just the wider kernel? Dense 5x5 everywhere (Res4 then
# stays frozen, ring and all), and the reparameterised ring at the backbone rate.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python
until ! pgrep -f "ma_x3d.cli train" > /dev/null; do sleep 120; done
$P -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 --set data.num_workers=5 \
  --configs configs/wk/wk_dense_all.yaml configs/wk/wk_all_ringlr.yaml
for d in wk_dense_all wk_all_ringlr; do
  HIP_VISIBLE_DEVICES=0 $P -m ma_x3d.cli cv "runs/$d/seed0" --variants 1clip > /dev/null 2>&1
done
echo "reparam controls done $(date +%H:%M)"
