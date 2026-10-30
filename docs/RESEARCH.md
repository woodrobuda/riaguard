# Signal processing on market data: research notes

Compiled 2026-10-30. Every source listed under "Sources opened" was actually retrieved while writing
these notes. Items under "Further reading" are standard references that were named in those sources
but **not** opened. These are educational notes, not trading advice.

## 1. Price series as signals, and why that analogy is weak
- **Prices behave close to random walks.** Their power spectrum falls off roughly as 1/f². Any FFT of a
  price window is therefore dominated by its lowest bins, so the "dominant cycle" it reports is mostly
  set by the window length. `examples/03_spectral_wavelet.py` shows this. On a pure random walk the
  "dominant period" of log price came out as 256 bars, which is nperseg, the Welch segment length.
  On the returns it came out as 7.8 bars, which is noise. On a series with a real 64-bar cycle injected,
  both methods recovered 64. Detrend or difference before any spectral analysis, and treat peaks as
  hypotheses, not facts.
- **The series are non-stationary.** Volatility, regime and microstructure all change over time.
  Fixed-coefficient filters therefore have a time-varying effective signal-to-noise ratio. The
  Hilbert–Huang transform was proposed specifically for non-stationary, nonlinear data, and its
  literature also documents the method's end effects and mode mixing. [HHT]

## 2. Moving averages are low-pass filters. Smoothing costs lag.
- **SMA(N)** is an FIR filter with N equal taps. It has linear phase, so its group delay is constant
  at (N−1)/2 bars.
- **EMA** y_t = αx_t + (1−α)y_{t−1} is a first-order IIR filter. Ehlers's MAMA paper writes the EMA
  this way and calls α the knob that trades responsiveness against smoothing. [MAMA] Its
  low-frequency group delay is (1−α)/α. With span 20 (α = 2/21) that is 9.5 bars, the same as SMA(20).
  The EMA's delay varies with frequency, though.
- **Group delay** is the negative derivative of phase with respect to frequency. It is the delay of
  the signal's envelope. [GD] A linear-phase (symmetric FIR) filter has constant delay. Any causal
  smoother must lag at low frequencies; "negative group delay" designs only work for band-limited,
  predictable signals and break down on surprises. [GD]
- **SciPy's `lfilter`** implements the direct-form-II-transposed difference equation and recommends
  second-order sections (`sosfilt`) for numerical robustness. `lfilter_zi` provides steady-state
  initial conditions, which avoid a start-up transient. The streaming `IIR` class in this repo copies
  that initialisation, and the tests check it against `lfilter`. [LFILTER]
- **Ehlers on lag:** "Two characteristics of moving averages are that they smooth the input data and
  they lag the input data. Their use and application is almost always a tradeoff between these two
  characteristics." [EHLERS-HIL]

## 3. Causal vs non-causal filtering, and look-ahead bias
- **`filtfilt`** runs the filter forward and then backward. That gives zero phase, but it uses future
  samples. [LFILTER] The same holds for centred moving averages, FFT-domain filtering, wavelet
  denoising and `scipy.signal.hilbert`, which is FFT-based: it zeroes the negative frequencies of the
  whole window. [HILBERT] Applied once to a full history and then backtested, all of these leak the
  future and give spectacular but fake results.
- **The test suite checks this.** `test_filtfilt_is_noncausal_and_rolling_fixes_it` shows that
  filtfilt outputs change when future data is removed. The fix is to re-evaluate the tool on a
  trailing window at every step (`spectral.rolling_causal`) and use only the last sample. That point
  sits at the window's edge, so it suffers the end effects documented for HHT and wavelets. [HHT]
- **`backtest.assert_causal`** applies an empirical guard to any signal function. It truncates the
  input and checks that past outputs do not change. A deliberately cheating signal (`close.shift(-1)`)
  is caught in the tests.
- **The backtest convention is position_{t+1} = signal_t.** Costs are charged on position changes.

## 4. Kalman filtering
- **What it is.** A Kalman filter is a recursive linear-MMSE estimator with two steps, predict and
  update. It needs only the previous state and covariance, so it is naturally a streaming algorithm.
  It stays the best *linear* estimator even without Gaussian noise if the covariances are known. In
  practice the noise covariances Q and R are hard to know and are often tuned as hyper-parameters.
  When the filter is well specified, its innovations should be white noise. [KF]
- **Local linear trend model.** The state is a level μ plus a slope ν: y_t = μ_t + ε_t,
  μ_{t+1} = μ_t + ν_t + ξ_t, ν_{t+1} = ν_t + ζ_t. statsmodels shows how to estimate the three variances
  by maximum likelihood. [SM-LLT] That model is `KalmanLocalLinearTrend` here, and the
  "Kalman slope>0" signal uses it.
- **Dynamic hedge ratio for pairs.** Treat (β_t, α_t) as random-walk states and the observation as
  y_t = β_t x_t + α_t + e_t. The innovation e_t and its variance S_t then give a z-score for spread
  trading. This is the formulation popularised by E. Chan's *Algorithmic Trading* and by QuantStart.
  The QuantStart article returned HTTP 503 when fetched, so it is listed under further reading.
  `KalmanHedgeRatio` returns the innovation computed from the *prior* state, so it is out-of-sample at
  each step. In `05_pairs_kalman.py` the filter tracked a time-varying β with RMSE 0.12. That was
  synthetic data; the result depends on δ and R.

## 5. FFT, spectral and cyclic analysis: pitfalls
- **Leakage and windowing.** Use Hann or another taper, detrend, and prefer Welch averaging to a single
  periodogram.
- **Resolution versus stationarity.** Resolving a 60-bar cycle needs several hundred bars, and over
  that span the "cycle" may not persist.
- **End effects and Gibbs phenomenon.** Frequency-domain processing of finite windows distorts the
  newest samples, which are the only ones a trader can act on. [HHT]
- **Ehlers's own caveat.** His Hilbert transformer has about a 4-bar group delay. For an 8-bar cycle
  that is half a cycle, so the signal would be "exactly the wrong trading signal". [EHLERS-HIL]
- **MESA as a measurement tool.** Ehlers's approach uses MESA/Hilbert measurements to *adapt* other
  indicators, for example by setting a lookback to half the measured dominant cycle. It does not use
  them as trade triggers. [BST]
- **MAMA/FAMA.** These adapt the EMA's α to the rate of change of Hilbert phase, with FastLimit 0.5 and
  SlowLimit 0.05. The paper's single reported test (100 stocks, 1998–2001, $6,403 gross over 1,317
  trades) is an author-run in-sample anecdote, not independent evidence. [MAMA]

## 6. Wavelets and multiresolution
- **Shrinkage denoising.** Donoho–Johnstone wavelet shrinkage transforms the data, applies a soft or
  hard threshold to the detail coefficients, and inverts. PyWavelets provides the `soft`, `hard`,
  `garrote` and `firm` threshold modes. [PYWT] `wavelet_denoise` uses the universal threshold
  σ√(2 ln N), with σ estimated from the median absolute deviation of the finest details. In the tests
  it halves the mean squared error on a noisy sinusoid.
- **Multiresolution.** `wavelet_mra` splits a series into an approximation A_J plus details D_1..D_J
  that sum back exactly. The tests verify this. A band such as D5+D6 can be fed to the Hilbert
  transform; on synthetic data with a 64-bar cycle this recovered a median period of about 66 bars.
- **Caveats.** The DWT is non-causal within its window and shift-variant. Use it inside
  `rolling_causal` for trading. EMD/HHT is an adaptive alternative, but its theory is empirical and it
  suffers end effects and mode mixing. [HHT]

## 7. Irregular ticks, resampling and bars
- **Activity bars.** Tick data arrive irregularly. López de Prado (*Advances in Financial Machine
  Learning*, ch. 2) argues for sampling on activity (tick, volume or dollar bars) instead of the clock.
  The claim is that time bars over-sample quiet periods and under-sample bursts, producing returns with
  fat tails and serial correlation. Dollar bars are robust to price level and to share-count changes.
  A suggested threshold is about 1/50 of average daily dollar volume, following Easley, López de Prado
  & O'Hara, "The Volume Clock". [MLFIN] [RD]
- **Independent replication.** Daru Finance reproduced the Gaussianisation effect in crypto, equities
  and FX. They add two caveats. The benefit depends on granularity, and equities need session-aware
  handling. On their equity regular-hours data, volume bars beat dollar bars on normality. [DARU]
- **Information-driven bars.** Imbalance and run bars sample when signed order flow exceeds an
  EWMA-based expectation. mlfinpy warns there is "very little to no academic papers" on them. [MLFIN]
- **In this repo.** `bars.py` builds time, tick, volume and dollar bars. Empty time intervals are
  dropped, not forward-filled; forward-filling invents zero-return observations. On *synthetic*
  constant-intensity ticks all bar types gave near-zero excess kurtosis. That is expected, because the
  benefit comes from time-varying activity, which the generator does not simulate. Do not read that
  output as evidence either way.

## 8. Streaming and online implementation
- **Design.** All filters here expose `update(x) -> y` with O(1) state (O(N) for FIR), so the same code
  runs in a backtest and on a live tick feed.
- **What runs online and what doesn't.** EMA, IIR filters and the Kalman variants are fully recursive.
  FFT, wavelets and Hilbert are windowed batch operations; online use needs a trailing window
  re-evaluated per bar, at a cost of O(W log W) per update.
- **Latency.** Algorithmic latency is the group delay. Compute latency matters only at high frequency.
  For bar-level strategies, the filter's group delay usually dominates.

## 9. Backtesting hygiene
- **The overfitting risk is large.** Bailey, Borwein, López de Prado & Zhu show that testing many
  strategy variants on one history makes overfitting likely. They argue that standard hold-out methods
  are unreliable for investment backtests, and propose estimating the probability of backtest
  overfitting. [BAILEY-PBO] [BAILEY-AMS]
- **The repo's own illustration.** On a *pure GBM* synthetic series with no exploitable structure,
  "Kalman slope>0" posted a higher Sharpe ratio (0.75) than buy-and-hold (0.63). That is luck, which is
  exactly the trap. Use multiple seeds, multiple assets, walk-forward selection, realistic costs and
  deflated Sharpe ratios.

## Sources opened
- [HHT] Wikipedia, "Hilbert–Huang transform": https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform
- [KF] Wikipedia, "Kalman filter": https://en.wikipedia.org/wiki/Kalman_filter
- [GD] Wikipedia, "Group delay and phase delay": https://en.wikipedia.org/wiki/Group_delay_and_phase_delay
- [LFILTER] SciPy, `scipy.signal.lfilter`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
- [HILBERT] SciPy, `scipy.signal.hilbert`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
- [PYWT] PyWavelets, thresholding functions: https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
- [SM-LLT] statsmodels, "State space modeling: Local Linear Trends": https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_local_linear_trend.html
- [MAMA] J. Ehlers, "MAMA – The Mother of Adaptive Moving Averages": https://www.mesasoftware.com/papers/MAMA.pdf
- [MLFIN] mlfinpy docs, "Financial Data Structures": https://mlfinpy.readthedocs.io/en/stable/FinancialDataStructure.html
- [BAILEY-PBO] Bailey, Borwein, López de Prado, Zhu, "The Probability of Backtest Overfitting": https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
- [BAILEY-AMS] Bailey et al., "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance", Notices of the AMS 61(5), 2014: https://www.ams.org/notices/201405/rnoti-p458.pdf
- Search-result excerpts only (pages not opened in full):
  - [EHLERS-HIL] Ehlers, "Hilbert Indicators Tell You When To Trade", S&C V.18:3, and the "Moving Average Lag" PDF: https://store.traders.com/v11627hilint.html and http://technicalanalysis.org.uk/moving-averages/Ehle.pdf. A direct fetch of the PDF returned HTTP 500.
  - [BST] Better System Trader episode 48 with John Ehlers: https://bettersystemtrader.com/048-john-ehlers/
  - [RD] Reasonable Deviations, notes on *Advances in Financial ML*: https://reasonabledeviations.com/notes/adv_fin_ml/
  - [DARU] Daru Finance, "Information-Driven Bars": https://daru.finance/research-review/lopez-de-prado/information-driven-bars

## Further reading (not opened)
- M. López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, ch. 2 (bars) and ch. 7, 11–12 (CV and backtesting).
- J. Ehlers, *Rocket Science for Traders* (2001) and *Cybernetic Analysis for Stocks and Futures* (2004).
- E. Chan, *Algorithmic Trading: Winning Strategies and Their Rationale*, Wiley 2013, ch. 3 (Kalman hedge ratio).
- QuantStart, "Dynamic Hedge Ratio Between ETF Pairs Using the Kalman Filter" (returned HTTP 503 when fetched).
- Durbin & Koopman, *Time Series Analysis by State Space Methods*, 2nd ed., OUP 2012.
- Donoho & Johnstone, "Ideal spatial adaptation by wavelet shrinkage", Biometrika 81(3), 1994.
- Easley, López de Prado & O'Hara, "The Volume Clock", JPM 39(1), 2012.
