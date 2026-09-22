#!/bin/bash
# Resumable, throttled download: the hub allows 1000 API calls / 5 min and the repo
# has 2000 files, so fetch in batches and wait out the limit.
cd /remote/vast0/hoangnguyenhuu/ma-x3d
for attempt in $(seq 1 12); do
  n=$(find dataset/rwf_raw -name "*.avi" | wc -l)
  echo "attempt $attempt: $n/2000 clips $(date +%H:%M)"
  [ "$n" -ge 2000 ] && { echo ok > dataset/rwf_raw/done.flag; echo "complete"; break; }
  .venv/bin/python -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download('A1mal/RWF-2000-Dataset', repo_type='dataset',
                      local_dir='dataset/rwf_raw', allow_patterns=['*.avi'], max_workers=4)
except Exception as e:
    print('stopped:', type(e).__name__, str(e)[:120])
" 2>&1 | tail -2
  sleep 300
done
