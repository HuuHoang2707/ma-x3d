#!/bin/bash
# 1. wait for the preprocessing variants to be trained, 2. pick the best by OOF,
# 3. rebuild Hockey and RLVS with that same preprocessing, 4. joint-train on all three.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python

echo "waiting for the five preprocessing runs $(date +%H:%M)"
while true; do
  n=$(ls runs/rp_*/seed0/fold*/results.json 2>/dev/null | wc -l)
  [ "$n" -ge 20 ] && break
  sleep 300
done
for d in runs/rp_*/seed0; do
  [ -f $d/cv_eval.json ] || HIP_VISIBLE_DEVICES=0 $P -m ma_x3d.cli cv $d --variants 1clip > /dev/null 2>&1
done

WINNER=$($P - <<'PY'
import json, glob, pathlib
best, name = -1, "cluster"
for f in glob.glob("runs/rp_*/seed0/cv_eval.json"):
    acc = json.load(open(f))["variants"]["1clip"]["oof@0.5"]["accuracy"]
    tag = pathlib.Path(f).parts[1].replace("rp_", "")
    print(f"{tag:10s} OOF {100*acc:.2f}", flush=True)
    if acc > best:
        best, name = acc, tag
print("WINNER=" + name)
PY
)
echo "$WINNER"
ROI=$(echo "$WINNER" | sed -n 's/^WINNER=//p')
echo "best preprocessing: $ROI"

# 3. the other two datasets, built the same way, so the joint set is consistent
ARGS="--roi cluster"; SUF=$ROI
case $ROI in
  none)     ARGS="--roi none";;
  adaptive) ARGS="--roi adaptive";;
  union)    ARGS="--roi union";;
  noenh)    ARGS="--roi cluster --no-enhance";;
esac
for ds in hockey rlvs; do
  [ -f dataset/${ds}_$SUF/train_x.npy ] && continue
  case $ds in
    hockey) F=dataset/hockey_raw/Fight; N=dataset/hockey_raw/NonFight;;
    rlvs)   F="dataset/rlvs_raw/archive/train/Violence"; N="dataset/rlvs_raw/archive/train/NonViolence";;
  esac
  $P -m ma_x3d.cli build-clips --fight "$F" --nonfight "$N" --all-dir dataset/${ds}_${SUF}_all \
     --out dataset/${ds}_$SUF --workers 12 $ARGS \
    && $P -m ma_x3d.cli audit --root dataset/${ds}_$SUF && echo "built ${ds}_$SUF $(date +%H:%M)"
done

# 4. joint training on the three consistently preprocessed datasets
cat > configs/prep/rj_joint.yaml <<CFG
# The best preprocessing, joint with Hockey and RLVS built the same way.
_base_: rp_$ROI.yaml
name: rj_joint_$ROI
description: joint training on three datasets, $ROI preprocessing
data:
  extra_roots: [dataset/hockey_$SUF, dataset/rlvs_$SUF]
train:
  epochs: 20
  probe_epochs: 2
  patience: 8
CFG
cat > configs/prep/rj_joint_diff.yaml <<CFG
_base_: rj_joint.yaml
name: rj_joint_${ROI}_diff
description: joint training, $ROI preprocessing, difference residual
model:
  diff_residual: [res3, res4]
CFG
$P -m ma_x3d.cli sweep --gpus auto --seeds 0 --folds 0 1 2 3 \
  --configs configs/prep/rj_joint.yaml configs/prep/rj_joint_diff.yaml
echo "final joint training done $(date +%H:%M)"
