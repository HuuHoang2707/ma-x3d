#!/bin/bash
# The final runs failed on a checkpoint bug (fixed). Re-run them once the ablation
# sweep has freed the GPUs; the chain skips the variants and the dataset rebuilds.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
until ! pgrep -f "ma_x3d.cli sweep" > /dev/null && ! pgrep -f "ma_x3d.cli train" > /dev/null; do
  sleep 300
done
MAX3D_MAX_MEM=35 bash scripts/final_chain.sh
