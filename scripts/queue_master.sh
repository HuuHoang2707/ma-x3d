#!/bin/bash
# One ordered queue, single seed, four folds each. Recipe first: it was tuned for a
# different model on different data, so everything after it depends on the answer.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python
sweep () { $P -m ma_x3d.cli sweep --gpus 0 1 2 3 --seeds 0 --folds 0 1 2 3 \
             --set data.num_workers=5 --configs "$@"; }
score () { for d in "$@"; do
             [ -f "runs/$d/seed0/cv_eval.json" ] || HIP_VISIBLE_DEVICES=0 $P -m ma_x3d.cli cv \
               "runs/$d/seed0" --variants 1clip > /dev/null 2>&1
           done; }

until ! pgrep -f "ma_x3d.cli train" > /dev/null; do sleep 180; done

# 1. training recipe on the confirmed model
sweep configs/tune/r1_unfreeze_all.yaml configs/tune/r2_lr1e4.yaml \
      configs/tune/r3_unfreeze_lr1e4.yaml configs/tune/r4_long.yaml
score r1_unfreeze_all r2_lr1e4 r3_unfreeze_lr1e4 r4_long
echo "recipe retune done $(date +%H:%M)"

# 2. kernel design: which axis, how wide, how parameterised
sweep configs/tune/d1_graded.yaml configs/tune/d2_dilated7.yaml \
      configs/wk/wk_t.yaml configs/wk/wk_st.yaml configs/wk/wk_late.yaml
score d1_graded d2_dilated7 wk_t wk_st wk_late
echo "kernel design done $(date +%H:%M)"

# 3. the other two modules
sweep configs/mod/m1_tdm.yaml configs/mod/m2_tdm_wkall.yaml \
      configs/mod/s1_slowfast.yaml configs/mod/s3_slowfast_wkall.yaml
score m1_tdm m2_tdm_wkall s1_slowfast s3_slowfast_wkall
echo "modules done $(date +%H:%M)"
