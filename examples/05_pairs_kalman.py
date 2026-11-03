"""Kalman dynamic hedge ratio + innovation z-score (synthetic cointegrated pair)."""
import numpy as np, matplotlib.pyplot as plt
from _common import OUT
from riaguard import streaming as st
rng = np.random.default_rng(5); n = 1500
x = 50 + np.cumsum(rng.normal(0, 1, n)); beta = 1.2 + 0.4 * np.sin(np.arange(n) / 300)
spread = np.zeros(n)
for t in range(1, n): spread[t] = 0.95 * spread[t-1] + rng.normal(0, 1)
y = beta * x + 5 + spread
k = st.KalmanHedgeRatio(delta=1e-4, r=1.0); out = np.array([k.update(a, b) for a, b in zip(x, y)])
z = out[:, 2] / np.sqrt(out[:, 3])
pos = np.where(z < -1.5, 1, np.where(z > 1.5, -1, 0)).astype(float)  # long spread when y cheap
pos_held = np.r_[0, pos[:-1]]                                         # trade next bar
pnl = pos_held * (np.diff(np.r_[y[0], y]) - out[:, 0] * np.diff(np.r_[x[0], x]))
print("final beta est", round(out[-1, 0], 3), "true", round(beta[-1], 3), "| beta RMSE (after 200)", round(float(np.sqrt(np.mean((out[200:, 0]-beta[200:])**2))), 3))
print("toy spread P&L units (no costs, synthetic):", round(float(pnl.sum()), 1))
fig, ax = plt.subplots(3, 1, figsize=(10, 9)); ax[0].plot(beta, "k", label="true beta"); ax[0].plot(out[:, 0], label="Kalman beta"); ax[0].set_ylim(0, 3); ax[0].legend()
ax[1].plot(z, lw=.5); ax[1].axhline(1.5, c="r"); ax[1].axhline(-1.5, c="r"); ax[1].set_title("innovation z-score")
ax[2].plot(np.cumsum(pnl)); ax[2].set_title("toy cumulative P&L (synthetic, no costs)"); plt.tight_layout(); plt.savefig(f"{OUT}/05_pairs_kalman.png", dpi=90)
