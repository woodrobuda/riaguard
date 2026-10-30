# Causal streaming mode

`khoros_cantata.streaming` runs a Cantata workspace **one sample (or chunk) at a time**,
for live feeds such as market data. Each glyph keeps its own state between pushes, so
output sample *t* depends only on input samples *≤ t*.

```python
from khoros_cantata.streaming import StreamingExecutor, batch_execute
ex = StreamingExecutor('examples/streaming_ema_crossover.wk')
for price in feed:
    out = ex.push(price=price)          # {'signal': array([...]), 'fast': ..., 'slow': ...}
ex.run({'price': prices}, chunk=50)     # whole record, streamed in chunks
batch_execute('...wk', {'price': prices})  # same graph on the batch WorkspaceExecutor
```

## Graph glyphs
| glyph | role |
|---|---|
| `STREAM:kinput -name X` | stream source; fed by `push(X=...)` |
| `STREAM:koutput -name Y` | collected output; `push` returns `{Y: chunk}` |
| `STREAM:krolling -op NAME -window N [opts]` | trailing-window adapter around any same-length batch operator |

## Operator contract
A streamable operator registers a factory with
`register_streamable(name, factory)`; `factory(params, batch_op, connected_ports)` returns a
`StreamOp` with `reset()` and `step(inputs: dict[port, chunk]) -> dict[port, chunk]`
(axis 0 is time; output chunks have the input length). The factory raises `NonCausal(reason)`
for option combinations that are not causal. Built-in implementations:

- `Pointwise` — stateless sample-wise ops (all arithmetic/logical/complex/compare/clip ops);
- `FIRStream` (`kfir -mode causal`, `klinearop -w0 0`), `IIRStream` (`kiir`, SOS state),
  `EMAStream` (`kema`), `RunStatsStream` (`krunstats`, cumulative or `-window`),
  `DelayStream` (`kdelay`), `DiffStream` (`kdiff`), `CrossStream` (`kcross`).

`NON_CAUSAL` lists operators that need the whole record, with the reason (FFT/spectra,
Hilbert, wavelets, Wiener, resample, histogram equalisation, whole-record normalisation,
centred image filters, ...). `classify(name, params, connected)` returns
`streamable` / `non-causal` / `undeclared`.

## Rejection and the rolling adapter
`StreamingExecutor` validates the graph when it is built: any glyph that is non-causal or
undeclared raises `NonCausalGraphError` (listing glyph id, operator and reason) unless it is
wrapped in `STREAM:krolling`. The adapter keeps the last `-window` samples of each input,
re-runs the batch operator on that window at every step and emits only the final output sample;
at start-up the window is shorter (if the operator cannot run on a short window, that sample is
NaN). This is causal by construction but is *not* the same as the batch operator on the full
record, and costs O(window) work per sample. Graph cycles are rejected.

## Look-ahead verification
`find_lookahead(fn, x)` / `assert_no_lookahead(fn, x)` perturb samples after a set of
checkpoints and report any output at or before the checkpoint that changed. The tests apply it
to every causal operator, the rolling adapter, and (as negative controls) filtfilt, Hilbert,
centred FIR and whole-record z-scoring, all of which it flags.
