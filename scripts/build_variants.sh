#!/bin/bash
# Build one dataset per preprocessing variant from the raw clips, then one shared audit.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python
ROOT="dataset/rwf_raw/data/RWF-2000 Sliced"
build () { name=$1; shift
  [ -f dataset/$name/train_x.npy ] && { echo "$name exists"; return; }
  $P -m ma_x3d.cli build-rwf --videos "$ROOT" --out dataset/$name --workers 12 "$@" \
    && echo "built $name $(date +%H:%M)"; }
build rwf_cluster  --roi cluster
build rwf_adaptive --roi adaptive
build rwf_none     --roi none
build rwf_noenh    --roi cluster --no-enhance
build rwf_union    --roi union
$P -m ma_x3d.cli audit --root dataset/rwf_cluster
for v in rwf_adaptive rwf_none rwf_noenh rwf_union; do
  cp dataset/rwf_cluster/train_groups.npy dataset/rwf_cluster/train_exclude.npy dataset/$v/
done
echo "all variants ready $(date +%H:%M)"
