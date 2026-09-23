#!/bin/bash
# Final queue: finish the joint k-fold run and soup it, then the kernel-design table.
# TDM and the fast pathway are dropped (built and tested, not needed for the paper).
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python
sweep () { $P -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
             --set data.num_workers=5 --configs "$@"; }
score () { for d in "$@"; do
             [ -f "runs/$d/seed0/cv_eval.json" ] || HIP_VISIBLE_DEVICES=0 $P -m ma_x3d.cli cv \
               "runs/$d/seed0" --variants 1clip > /dev/null 2>&1
           done; }

# the j1 sweep keeps running after the old queue script is stopped
until [ "$(ls runs/j1_wkall_joint_kfold/seed0/fold*/results.json 2>/dev/null | wc -l)" -eq 4 ] \
      && ! pgrep -f "ma_x3d.cli train" > /dev/null; do sleep 120; done
echo "joint k-fold done $(date +%H:%M)"
HIP_VISIBLE_DEVICES=0 $P paper/drafts/fold_soup.py runs/j1_wkall_joint_kfold 2>&1 | grep -E "^seed|^mean"
score r2_lr1e4 j1_wkall_joint_kfold

sweep configs/tune/d1_graded.yaml configs/tune/d2_dilated7.yaml \
      configs/wk/wk_t.yaml configs/wk/wk_st.yaml configs/wk/wk_late.yaml
score d1_graded d2_dilated7 wk_t wk_st wk_late
touch runs/experiments_frozen.flag
echo "kernel design done, experiments frozen $(date +%H:%M)"
