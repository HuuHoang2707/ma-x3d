#!/bin/bash
# The soup recipe is now fixed (uniform over every fold, BN recalibrated only where the
# folds differ). Replicate it on experiments it was never tuned on: other models, other
# pipelines, and two other datasets.
cd /remote/vast0/hoangnguyenhuu/ma-x3d
for e in wa_x3d_none rp_none rp_plain rp_adaptive hk_b3_x3d_m hk_e05_ma_x3d \
         rl_p0_x3d_m_kd rl_e11_ma_x3d_kd; do
  echo "== $e"
  HIP_VISIBLE_DEVICES=${GPU:-2} .venv/bin/python paper/drafts/fold_soup.py runs/$e 2>&1 \
    | grep -E "^seed|^mean"
done
