"""Periodogram of returns, wavelet denoising/MRA, Hilbert period - and how a fake 'cycle' appears in pure noise."""
import numpy as np, matplotlib.pyplot as plt
from _common import OUT
from signal_trading import data, spectral as sp
cyc = data.synthetic_prices(1024, seed=3, cycle_amp=0.08, cycle_period=64)["close"]
rw = data.synthetic_prices(1024, seed=4)["close"]
fig, ax = plt.subplots(4, 1, figsize=(10, 13))
for name, s in [("with 64-bar cycle", cyc), ("pure random walk", rw)]:
    f, p = sp.periodogram(np.log(s)); ax[0].semilogy(f[1:], p[1:], label=f"log price: {name}")
    print(f"{name}: dominant period of log PRICE = {sp.dominant_period(np.log(s.to_numpy())):.1f} bars; "
          f"of RETURNS = {sp.dominant_period(np.diff(np.log(s.to_numpy()))):.1f}")
ax[0].legend(); ax[0].set_title("Random walks have 1/f^2 spectra: 'dominant cycle' on prices is mostly the longest window")
lx = np.log(cyc.to_numpy()); den = sp.wavelet_denoise(lx)
ax[1].plot(lx, lw=.6, label="log price"); ax[1].plot(den, label="wavelet denoised (db4, soft, universal thr.)"); ax[1].legend()
for n, c in sp.wavelet_mra(lx - lx.mean(), level=5): ax[2].plot(c, lw=.7, label=n)
ax[2].legend(ncol=6); ax[2].set_title("Wavelet multiresolution (components sum to the series)")
bp = [c for n, c in sp.wavelet_mra(lx, level=6) if n in ("D5", "D6")]; band = bp[0] + bp[1]
amp, ph, per = sp.hilbert_phase(band); ax[3].plot(per, label="Hilbert inst. period (D5+D6 band)"); ax[3].axhline(64, c="k", ls="--")
ax[3].set_ylim(0, 200); ax[3].legend()
print("median Hilbert period of D5+D6 band:", round(float(np.nanmedian(per[100:-100])), 1), "(true 64)")
plt.tight_layout(); plt.savefig(f"{OUT}/03_spectral_wavelet.png", dpi=90)
