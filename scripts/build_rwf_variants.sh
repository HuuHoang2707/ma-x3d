#!/bin/bash
# Waits for the raw download, then builds one dataset per preprocessing variant and
# trains X3D-M (tuned recipe) on each. Folds are shared so the comparison is paired.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python

until [ -f dataset/rwf_raw/done.flag ]; do sleep 120; done
echo "download finished $(date +%H:%M)"
# files arrive already extracted from the file-level mirror
ROOT=$(find dataset/rwf_raw -type d -name "train" | head -1 | xargs dirname)
echo "raw root: $ROOT"
ls "$ROOT"/train "$ROOT"/val | head
for split in train val; do
  for cls in Fight NonFight; do
    echo "  $split/$cls: $(ls "$ROOT/$split/$cls" 2>/dev/null | wc -l) videos"
  done
done

build () {  # name, extra args
  name=$1; shift
  [ -f dataset/$name/train_x.npy ] && { echo "$name exists"; return; }
  $P -m ma_x3d.cli build-rwf --videos "$ROOT" --out dataset/$name --workers 12 "$@" \
    && echo "built $name $(date +%H:%M)"
}
build rwf_cluster  --roi cluster
build rwf_none     --roi none
build rwf_adaptive --roi adaptive
build rwf_union    --roi union
build rwf_noenh    --roi cluster --no-enhance

# one audit, shared by every variant, so the folds and excluded clips are identical
$P -m ma_x3d.cli audit --root dataset/rwf_cluster
for v in rwf_none rwf_adaptive rwf_union rwf_noenh; do
  cp dataset/rwf_cluster/train_groups.npy dataset/rwf_cluster/train_exclude.npy dataset/$v/ 2>/dev/null
done
echo "datasets ready $(date +%H:%M)"
