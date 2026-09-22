#!/bin/bash
# One supervisor for node 3: the last two preprocessing variants, then the final chain
# (winner -> rebuild the other datasets -> joint model -> teacher -> distilled student),
# then the module ablation on the winning preprocessing. Node 3 shares its GPUs with
# another user, so training is pinned to the four least loaded GCDs.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python
sweep () { $P -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
             --set data.num_workers=5 --configs "$@"; }

if [ "$(ls runs/rp_plain/seed0/fold*/results.json 2>/dev/null | wc -l)" -lt 4 ]; then
  sweep configs/prep/rp_plain.yaml
  echo "rp_plain done $(date +%H:%M)"
fi
until [ -f dataset/rwf_aspect/train_groups.npy ]; do sleep 60; done
if [ "$(ls runs/rp_aspect/seed0/fold*/results.json 2>/dev/null | wc -l)" -lt 4 ]; then
  sweep configs/prep/rp_aspect.yaml
  echo "rp_aspect done $(date +%H:%M)"
fi
touch runs/variants_done.flag
echo "all seven variants done $(date +%H:%M)"

MAX3D_MAX_MEM=35 bash scripts/final_chain.sh
bash scripts/queue_after_chain.sh
