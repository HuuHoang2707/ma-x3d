#!/bin/bash
# The three full-split runs, each on four GPUs (global batch 12 = 4 x 3 per GPU).
# One job per group of GPUs is faster here than one job per GPU, because only a few
# runs are left and the rest of the machine would otherwise sit idle.
cd /remote/vast0/hoangnguyenhuu/ma-x3d
mkdir -p runs/logs

run () {  # gpus, config
  local tag
  tag=$(basename "$2" .yaml)
  HIP_VISIBLE_DEVICES=$1 .venv/bin/torchrun --standalone --nproc_per_node=4 \
    --master_port=$((29500 + RANDOM % 500)) -m ma_x3d.cli train "$2" --seed 0 --overwrite \
    > "runs/logs/${tag}_ddp.log" 2>&1
  echo "done $tag $(date +%H:%M)"
}

run 0,1,2,3 configs/final/f1_full_maxd.yaml &
run 4,5,6,7 configs/final/f2_full_joint.yaml &
wait
run 0,1,2,3 configs/final/f3_full_joint_diff.yaml
echo "full-split runs finished $(date +%H:%M)"
