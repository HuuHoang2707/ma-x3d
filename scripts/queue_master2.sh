#!/bin/bash
# Reordered queue. Unfreezing Res4 cost 2 test points, so the two runs built on it
# (r3, r4) are dropped; the joint k-fold run for the fold soup goes first instead.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python
sweep () { $P -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
             --set data.num_workers=5 --configs "$@"; }
score () { for d in "$@"; do
             [ -f "runs/$d/seed0/cv_eval.json" ] || HIP_VISIBLE_DEVICES=0 $P -m ma_x3d.cli cv \
               "runs/$d/seed0" --variants 1clip > /dev/null 2>&1
           done; }

# let r2 finish, then stop the old queue before it starts r3
until [ "$(ls runs/r2_lr1e4/seed0/fold*/results.json 2>/dev/null | wc -l)" -eq 4 ]; do sleep 60; done
for pat in "queue_master\\.sh" "r1_unfreeze_all\\.yaml" "r3_unfreeze_lr1e4" "r4_long"; do
  for pid in $(pgrep -f "$pat"); do [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null; done
done
sleep 20
rm -rf runs/r3_unfreeze_lr1e4 runs/r4_long
until ! pgrep -f "ma_x3d.cli train" > /dev/null; do sleep 60; done
score r2_lr1e4

sweep configs/tune/j1_wkall_joint_kfold.yaml
echo "joint k-fold done $(date +%H:%M)"
HIP_VISIBLE_DEVICES=0 $P paper/drafts/fold_soup.py runs/j1_wkall_joint_kfold 2>&1 | grep -E "^seed|^mean"
score j1_wkall_joint_kfold

sweep configs/tune/d1_graded.yaml configs/tune/d2_dilated7.yaml \
      configs/wk/wk_t.yaml configs/wk/wk_st.yaml configs/wk/wk_late.yaml
score d1_graded d2_dilated7 wk_t wk_st wk_late
echo "kernel design done $(date +%H:%M)"

sweep configs/mod/m1_tdm.yaml configs/mod/m2_tdm_wkall.yaml \
      configs/mod/s1_slowfast.yaml configs/mod/s3_slowfast_wkall.yaml
score m1_tdm m2_tdm_wkall s1_slowfast s3_slowfast_wkall
echo "modules done $(date +%H:%M)"
