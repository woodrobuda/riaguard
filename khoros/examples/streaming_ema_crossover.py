"""Stream a synthetic price series one tick at a time through an EMA-crossover workspace.

Run: python examples/streaming_ema_crossover.py
"""
from pathlib import Path
import numpy as np
from khoros_cantata.streaming import StreamingExecutor, batch_execute

WK = Path(__file__).with_name('streaming_ema_crossover.wk')


def synthetic_prices(n=500, seed=7):
    g = np.random.default_rng(seed)
    drift = 0.0005 * np.sin(2 * np.pi * np.arange(n) / 200)        # slow regime changes
    return 100 * np.exp(np.cumsum(drift + g.normal(0, 0.01, n)))


def main():
    px = synthetic_prices(); ex = StreamingExecutor(WK); signals = []
    for t, p in enumerate(px):                     # one sample per push, as a live feed would
        out = ex.push(price=p)
        s = out['signal'][0]
        if s:
            signals.append((t, 'BUY' if s > 0 else 'SELL', p))
            print(f"t={t:4d}  {'BUY ' if s > 0 else 'SELL'}  price={p:8.3f}  fast={out['fast'][0]:8.3f}  slow={out['slow'][0]:8.3f}")
    batch = batch_execute(WK, {'price': px})
    stream_sig = np.array([1.0 if (t, 'BUY', p) in signals else 0 for t, p in enumerate(px)])
    assert np.array_equal(np.flatnonzero(batch['signal'] > 0), np.flatnonzero(stream_sig))
    print(f'{len(signals)} signals; streaming output matches batch execution')
    return signals


if __name__ == '__main__':
    main()
