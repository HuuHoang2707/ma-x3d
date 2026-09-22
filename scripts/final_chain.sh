#!/bin/bash
# After the preprocessing variants finish: pick the best, rebuild Hockey and RLVS the
# same way, and train the final model (best preprocessing + joint + difference residual).
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python

echo "waiting for the five preprocessing variants $(date +%H:%M)"
while [ "$(ls runs/rp_*/seed0/fold*/results.json 2>/dev/null | wc -l)" -lt 20 ]; do sleep 240; done
for d in runs/rp_*/seed0; do
  [ -f "$d/cv_eval.json" ] || HIP_VISIBLE_DEVICES=0 $P -m ma_x3d.cli cv "$d" --variants 1clip \
    > /dev/null 2>&1
done

ROI=$($P - <<'PY'
import json, glob, pathlib
rows = []
for f in glob.glob("runs/rp_*/seed0/cv_eval.json"):
    acc = json.load(open(f))["variants"]["1clip"]["oof@0.5"]["accuracy"]
    rows.append((acc, pathlib.Path(f).parts[1].replace("rp_", "")))
for acc, tag in sorted(rows, reverse=True):
    print(f"{tag:10s} OOF {100 * acc:.2f}")
print(sorted(rows, reverse=True)[0][1], file=open("runs/winner.txt", "w"))
PY
)
echo "$ROI"
W=$(cat runs/winner.txt)
echo "best preprocessing: $W"

case $W in
  none)     ARGS="--roi none";;
  adaptive) ARGS="--roi adaptive";;
  union)    ARGS="--roi union";;
  noenh)    ARGS="--roi cluster --no-enhance";;
  *)        ARGS="--roi cluster";;
esac

# the other two datasets, built the same way, so the joint set is consistent
for ds in hockey rlvs; do
  [ -f "dataset/${ds}_$W/train_x.npy" ] && continue
  case $ds in
    hockey) F=dataset/hockey_raw/Fight; N=dataset/hockey_raw/NonFight;;
    rlvs)   F="dataset/rlvs_raw/archive/train/Violence"; N="dataset/rlvs_raw/archive/train/NonViolence";;
  esac
  $P -m ma_x3d.cli build-clips --fight "$F" --nonfight "$N" \
     --all-dir "dataset/${ds}_${W}_all" --out "dataset/${ds}_$W" --workers 12 $ARGS \
    && $P -m ma_x3d.cli audit --root "dataset/${ds}_$W" && echo "built ${ds}_$W $(date +%H:%M)"
done

cat > configs/final/f4_best.yaml <<CFG
# The winning preprocessing, joint training on all three datasets, difference residual.
_base_: ../prep/rp_$W.yaml
name: f4_best_$W
description: best preprocessing ($W) + joint training + difference residual
data:
  protocol: full
  extra_roots: [dataset/hockey_$W, dataset/rlvs_$W]
model:
  diff_residual: [res3, res4]
train:
  epochs: 20
  probe_epochs: 2
CFG
mkdir -p runs/logs
HIP_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --master_port=29677 -m ma_x3d.cli train configs/final/f4_best.yaml --seed 0 \
  > runs/logs/f4_best_ddp.log 2>&1
echo "final model trained $(date +%H:%M)"
$P -c "
import json
r = json.load(open('runs/f4_best_$W/seed0/results.json'))['test']
print('FINAL test accuracy %.2f  F1 %.3f  AUC %.3f' % (100*r['accuracy'], r['f1'], r['auc']))"

# Distillation on the final model: one VideoMAE-B teacher on the same joint training
# clips (the split guard checks that), then the student. KD was worth +0.57 OOF.
cat > configs/final/t02_full_joint.yaml <<CFG
# Teacher for the final model: VideoMAE-B on the whole joint training set.
_base_: ../exp/t01_videomae.yaml
name: t02_full_joint_$W
description: VideoMAE-B teacher, winning preprocessing, joint training set
data:
  root: dataset/rwf_$W
  protocol: full
  extra_roots: [dataset/hockey_$W, dataset/rlvs_$W]
CFG
cat > configs/final/f5_best_kd.yaml <<CFG
# The final model: winning preprocessing + joint training + difference residual + KD.
_base_: f4_best.yaml
name: f5_best_kd_$W
description: f4 + distillation from the joint VideoMAE-B teacher
train:
  distill: runs/t02_full_joint_$W/seed0
  distill_alpha: 0.5
  distill_temp: 2.0
  epochs: 48
  patience: 16
CFG
HIP_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --master_port=29678 -m ma_x3d.cli train configs/final/t02_full_joint.yaml --seed 0 \
  > runs/logs/t02_full_joint_ddp.log 2>&1
if [ ! -f "runs/t02_full_joint_$W/seed0/results.json" ]; then
  # the student waits for the teacher for ever, so stop here instead
  echo "teacher failed, see runs/logs/t02_full_joint_ddp.log; f4 stands as the final model"
  exit 1
fi
echo "teacher trained $(date +%H:%M)"
HIP_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --master_port=29679 -m ma_x3d.cli train configs/final/f5_best_kd.yaml --seed 0 \
  > runs/logs/f5_best_kd_ddp.log 2>&1
echo "final KD model trained $(date +%H:%M)"
$P -c "
import json
for tag in ['f4_best_$W', 'f5_best_kd_$W']:
    r = json.load(open(f'runs/{tag}/seed0/results.json'))['test']
    print('%-20s test %.2f  F1 %.3f  AUC %.3f' % (tag, 100*r['accuracy'], r['f1'], r['auc']))"
