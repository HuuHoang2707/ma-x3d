"""Progress bars that also work in log files.

In a terminal this is a tqdm bar. When output goes to a file (nohup, `ma-x3d sweep`),
a plain line is printed every `every` fraction of the loop instead of carriage returns.
Set MA_X3D_PROGRESS=0 to silence both, or =1 to force the tqdm bar.
"""

from __future__ import annotations

import os
import sys
import time

from tqdm import tqdm


def _mode() -> str:
    env = os.environ.get("MA_X3D_PROGRESS")
    if env == "0":
        return "off"
    if env == "1" or sys.stderr.isatty():
        return "bar"
    return "lines"


class Progress:
    def __init__(self, iterable, desc: str, every: float = 0.25):
        self.iterable, self.desc, self.every = iterable, desc, every
        self.total = len(iterable) if hasattr(iterable, "__len__") else None
        self.mode = _mode()
        self.stats: dict[str, str] = {}
        self._bar = None

    def set(self, **stats) -> None:
        """Update the numbers shown next to the bar, e.g. set(loss=0.41, acc=0.83)."""
        self.stats = {k: (f"{v:.4f}" if isinstance(v, float) else str(v)) for k, v in stats.items()}
        if self._bar is not None:
            self._bar.set_postfix(self.stats, refresh=False)

    def __iter__(self):
        if self.mode == "bar":
            self._bar = tqdm(self.iterable, desc=self.desc, total=self.total, leave=False,
                             dynamic_ncols=True, file=sys.stderr)
            yield from self._bar
            self._bar.close()
            return
        t0, next_mark = time.time(), self.every
        for i, item in enumerate(self.iterable, 1):
            yield item
            if self.mode == "lines" and self.total and i / self.total >= next_mark:
                elapsed = time.time() - t0
                eta = elapsed / i * (self.total - i)
                extra = " ".join(f"{k} {v}" for k, v in self.stats.items())
                print(f"  {self.desc} {i}/{self.total} ({100 * i / self.total:.0f}%) "
                      f"{elapsed:.0f}s elapsed, eta {eta:.0f}s {extra}", flush=True)
                next_mark += self.every
