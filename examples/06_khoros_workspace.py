"""Run a Cantata-style dataflow workspace using khoros-cantata-python + SIGTRADE operators."""
import os, numpy as np, matplotlib.pyplot as plt
from _common import OUT
from signal_trading import data, khoros_adapter as ka
if not ka.AVAILABLE: raise SystemExit("install khoros-cantata-python first (pip install -e .[khoros])")
px = data.synthetic_prices(800, seed=6)["close"].to_numpy(); got = {}
r = ka.register_operators()
r.register("USER", "prices", lambda p, i: {"o": px})
r.register("USER", "capture", lambda p, i: (got.__setitem__(p.get("name", "x"), i["i"]), {})[1])
wk = os.path.join(os.path.dirname(__file__), "trend.wksp")
ws = ka.run_workspace(wk, r)
print("nodes executed:", len(ws.nodes), "| outputs captured:", sorted(got))
print("long fraction:", round(float(np.mean(got["signal"])), 3))
fig, ax = plt.subplots(2, 1, figsize=(10, 6)); ax[0].plot(px, "k", lw=.6); ax[0].plot(got["kalman"], label="kkalman"); ax[0].legend()
ax[1].plot(got["signal"]); ax[1].set_title("kcross(kema -span 20, ksma -n 100)  [ksma uses khoros conv1d_khoros]")
plt.tight_layout(); plt.savefig(f"{OUT}/06_khoros_workspace.png", dpi=90)
