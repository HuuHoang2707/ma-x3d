#!/bin/bash
# Sixth variant: full frame and no CLAHE, since each of those helped on its own.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python
ROOT="dataset/rwf_raw/data/RWF-2000 Sliced"   # the extracted mirror
[ -f dataset/rwf_plain/train_x.npy ] || \
  $P -m ma_x3d.cli build-rwf --videos "$ROOT" --out dataset/rwf_plain --workers 24 \
     --roi none --no-enhance
# same audit as the other variants, so every variant is scored on the same folds
cp dataset/rwf_cluster/train_groups.npy dataset/rwf_cluster/train_exclude.npy dataset/rwf_plain/
echo "rwf_plain ready $(date +%H:%M)"
