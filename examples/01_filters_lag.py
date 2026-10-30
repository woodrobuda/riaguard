"""Smoothing vs lag: EMA, SMA, Butterworth (causal) vs filtfilt (non-causal), group delay."""
import warnings; warnings.filterwarnings('ignore', module='scipy')
import numpy as np, matplotlib.pyplot as plt
from _common import OUT
from signal_trading import data, streaming as st, spectral as sp
px = data.synthetic_prices(600, seed=1, cycle_amp=0.05)["close"].to_numpy()
b, a = sp.butter_lowpass(30)
ys = {"EMA(20)": st.run(st.EMA(span=20), px), "SMA(20)": st.run(st.FIR.sma(20), px),
      "Butter2(30) causal": st.run(st.IIR(b, a), px), "filtfilt (LOOK-AHEAD)": sp.zero_phase(px, b, a)}
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(px, "k", lw=.6, label="close")
for k, y in ys.items(): ax[0].plot(y, label=k)
ax[0].legend(); ax[0].set_title("Causal filters lag; zero-phase filtfilt does not (because it sees the future)")
for name, (bb, aa) in {"SMA(20)": (np.ones(20)/20, 1), "EMA(20)": ([2/21], [1, -(1-2/21)]), "Butter2(30)": (b, a)}.items():
    f, gd = sp.group_delay(bb, aa); ax[1].plot(f, gd, label=name)
ax[1].set_xlim(0, .1); ax[1].set_xlabel("cycles/bar"); ax[1].set_ylabel("group delay (bars)"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/01_filters_lag.png", dpi=90)
print("EMA(20) low-freq lag (bars):", round(st.EMA(span=20).lag, 2), "| SMA(20):", 9.5)
