"""Walk-forward comparison of filter signals vs buy-and-hold (5 bps/side). ILLUSTRATIVE ONLY."""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import OUT
from riaguard import data, backtest as bt, strategies as S
ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
df, src = data.fetch_yfinance(ticker, start="2010-01-01")
px = df["close"]; print("data source:", src, "| bars:", len(px), px.index[0].date(), "->", px.index[-1].date())
for f in (S.ema_cross, S.sma_trend, S.kalman_slope): bt.assert_causal(f, px)
res = [bt.run_signal(px, S.buy_and_hold(px), "buy&hold", 0),
       bt.run_signal(px, S.ema_cross(px, (20, 100)), "EMA 20/100", 5),
       bt.run_signal(px, S.sma_trend(px, 200), "SMA200 trend", 5),
       bt.run_signal(px, S.kalman_slope(px), "Kalman slope>0", 5)]
wf, chosen = bt.walk_forward(px, S.ema_cross, [(5, 20), (10, 50), (20, 100), (50, 200)], train=756, test=126, cost_bps=5, name="WF EMA (OOS)")
start = wf.returns.index[0]
rows = []
for r in res + [wf]:
    sub = bt.Result(r.name, (1 + r.returns.loc[start:]).cumprod(), r.returns.loc[start:], r.position.loc[start:] if len(r.position) else r.position)
    rows.append(sub.stats())
tab = pd.DataFrame(rows).set_index("strategy"); pd.set_option("display.float_format", "{:.3f}".format)
print(f"\nCommon OOS period {start.date()} -> {px.index[-1].date()} (fixed-param strategies shown on the same period):")
print(tab.to_string())
print("WF chosen params (window start, params):", [(str(d.date()), p) for d, p in chosen][:4], "...")
fig, ax = plt.subplots(figsize=(10, 5))
for r in res + [wf]: ax.plot((1 + r.returns.loc[start:]).cumprod(), label=r.name)
ax.set_yscale("log"); ax.legend(); ax.set_title(f"{src} - illustrative, not advice"); plt.tight_layout()
plt.savefig(f"{OUT}/04_backtest_{ticker if src.startswith('yfinance') else 'synthetic'}.png", dpi=90)
tab.to_csv(f"{OUT}/04_backtest_{ticker if src.startswith('yfinance') else 'synthetic'}.csv")
