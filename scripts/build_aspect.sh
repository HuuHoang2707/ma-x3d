#!/bin/bash
# Seventh variant: full frame with the 16:9 ratio kept (padded), not stretched.
set -u
cd /remote/vast0/hoangnguyenhuu/ma-x3d
P=.venv/bin/python
ROOT="dataset/rwf_raw/data/RWF-2000 Sliced"
[ -f dataset/rwf_aspect/train_x.npy ] || \
  $P -m ma_x3d.cli build-rwf --videos "$ROOT" --out dataset/rwf_aspect --workers 24 \
     --roi none --keep-aspect
cp dataset/rwf_cluster/train_groups.npy dataset/rwf_cluster/train_exclude.npy dataset/rwf_aspect/
echo "rwf_aspect ready $(date +%H:%M)"
