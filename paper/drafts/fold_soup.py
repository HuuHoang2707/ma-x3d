"""Does averaging the fold models' weights keep the fold ensemble's gain?

For each seed of a k-fold experiment: the mean single fold model, the fold ensemble
(average of probabilities, k forward passes), and the fold soup (average of weights,
one forward pass), with and without BatchNorm recalibration. Test split, one view.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ma_x3d.cli import _load_run  # noqa: E402
from ma_x3d.data.dataset import array_paths  # noqa: E402
from ma_x3d.eval.cv import VARIANTS, predict_variant  # noqa: E402
from ma_x3d.eval.soup import average_states, differing_bn, recalibrate  # noqa: E402

device = torch.device("cuda")
exp = sys.argv[1] if len(sys.argv) > 1 else "runs/wa_wk_all_none"
seeds = sorted(Path(exp).glob("seed*"))
acc = lambda p, y: 100 * float(((p >= 0.5) == y).mean())  # noqa: E731

rows = []
for sd in seeds:
    folds = sorted(sd.glob("fold*/best.pt"))
    if len(folds) < 2:
        continue
    cfg, model = _load_run(folds[0].parent, device)
    xp, yp = array_paths(cfg.data.root, "test")
    y = np.load(yp)
    idx = np.arange(len(y))
    view = VARIANTS["1clip"]

    singles = []
    for f in folds:
        _, m = _load_run(f.parent, device)
        singles.append(predict_variant(m, str(xp), idx, cfg.data.frames, view, device,
                                       torch.bfloat16))
    ens = np.mean(singles, axis=0)

    model.load_state_dict(average_states(folds))
    soup = predict_variant(model, str(xp), idx, cfg.data.frames, view, device, torch.bfloat16)

    # BN recalibration on the training clips every fold model saw at least once
    xtr, _ = array_paths(cfg.data.root, "train")
    ntr = np.load(xtr, mmap_mode="r").shape[0]
    only = differing_bn(folds)
    n_bn = recalibrate(model, str(xtr), np.arange(ntr), cfg.data.frames, device, only)
    soup_bn = predict_variant(model, str(xp), idx, cfg.data.frames, view, device,
                              torch.bfloat16)

    rows.append((sd.name, np.mean([acc(s, y) for s in singles]),
                 np.std([acc(s, y) for s in singles]), acc(ens, y), acc(soup, y),
                 acc(soup_bn, y)))
    print(f"{sd.name}: single {rows[-1][1]:.2f} ± {rows[-1][2]:.2f} | ensemble "
          f"{rows[-1][3]:.2f} | soup {rows[-1][4]:.2f} | soup + BN recal {rows[-1][5]:.2f}"
          f" ({n_bn} BN layers)",
          flush=True)

if rows:
    r = np.array([row[1:] for row in rows])
    print(f"mean over {len(rows)} seeds: single {r[:, 0].mean():.2f} | ensemble "
          f"{r[:, 2].mean():.2f} | soup {r[:, 3].mean():.2f} | soup + BN {r[:, 4].mean():.2f}")
