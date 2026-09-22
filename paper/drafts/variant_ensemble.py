"""Do models trained on different preprocessings make different mistakes?

The folds are shared, so their out-of-fold predictions line up clip by clip and an
ensemble can be scored without training anything.
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, "src")
from ma_x3d.eval.cv import _oof  # noqa: E402

RUNS = ["rp_none", "rp_plain", "rp_aspect", "rp_adaptive", "rp_cluster", "rp_union", "rp_noenh",
        "wa_x3d_none", "wa_diff_none"]

oof, test, y_ref = {}, {}, None
for r in RUNS:
    try:
        p, y, te = _oof(f"runs/{r}/seed0", "1clip")
    except FileNotFoundError:
        continue
    oof[r], y_ref = p, y
    test[r] = (np.mean([t[1] for t in te], axis=0), te[0][0])

acc = lambda p, y: float(((p >= 0.5) == y).mean())  # noqa: E731
print("single models")
for r, p in sorted(oof.items(), key=lambda kv: -acc(kv[1], y_ref)):
    print(f"  {r:14s} OOF {100 * acc(p, y_ref):6.2f}  test {100 * acc(*test[r]):6.2f}")

print("\nensembles (mean probability)")
rows = []
for k in (2, 3, 4, 5):
    for combo in itertools.combinations(oof, k):
        p = np.mean([oof[c] for c in combo], axis=0)
        t = np.mean([test[c][0] for c in combo], axis=0)
        rows.append((acc(p, y_ref), acc(t, test[combo[0]][1]), combo))
for a, t, combo in sorted(rows, reverse=True)[:12]:
    print(f"  OOF {100 * a:6.2f}  test {100 * t:6.2f}  {' + '.join(c.replace('rp_', '') for c in combo)}")
