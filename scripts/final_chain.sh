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
