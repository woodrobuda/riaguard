"""Irregular ticks -> time / tick / volume / dollar bars; compare return kurtosis."""
import matplotlib.pyplot as plt, numpy as np
from scipy.stats import kurtosis
from _common import OUT
from riaguard import data, bars
tk = data.synthetic_ticks(40000, seed=2)
gaps = tk.index.to_series().diff().dt.total_seconds()
print(f"ticks={len(tk)}, gap mean={gaps.mean():.2f}s, max={gaps.max():.1f}s (irregular)")
B = {"time 1min": bars.time_bars(tk, "1min"), "tick 50": bars.tick_bars(tk, 50),
     "volume 12.5k": bars.volume_bars(tk, 12500), "dollar 1.25M": bars.dollar_bars(tk, 1.25e6)}
for k, b in B.items():
    r = np.log(b["close"]).diff().dropna(); print(f"{k:13s} bars={len(b):5d} excess kurtosis={kurtosis(r):+.2f}")
fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(tk.index, tk["price"], lw=.3, c="gray", label="ticks")
for k, b in B.items(): ax.plot(b.index, b["close"], ".", ms=2, label=k)
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/02_bars.png", dpi=90)
print("note: synthetic ticks have constant intensity, so kurtosis differences here are small; on real data activity clocks matter more")
