# Operator status (operators branch)

Source list: `reports/final_134_operator_closure_matrix.csv` (134 operators). Status:
- **implemented** — new native implementation on this branch, registered in `NativeOperatorRegistry`, with a pytest test (`tests/test_registry_ops.py` plus module tests).
- **previously existed** — present before this branch (now also registered/extended where noted).
- **skipped** — not implemented, with reason.
- **streamable** — whether `StreamingExecutor` accepts the glyph directly (causal, stateful update). Non-causal and undeclared operators can still be streamed inside `STREAM:krolling -op NAME -window N`. See `docs/STREAMING.md`.

Module key: S = `khoros_cantata.ops_signal`, I = `ops_image`, D = `ops_datamanip`.

| # | toolbox | operator | status | streamable | implementation | notes |
|---|---|---|---|---|---|---|
| 1 | bootstrap | conductor | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 2 | bootstrap | ghostwriter | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 3 | bootstrap | kcp | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 4 | bootstrap | kecho | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 5 | bootstrap | kexec | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 6 | bootstrap | kgendepend | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 7 | bootstrap | kgenimake | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 8 | bootstrap | kgenmanual | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 9 | bootstrap | kgenobj | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 10 | bootstrap | kgentb | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 11 | bootstrap | kman | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 12 | bootstrap | krename | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 13 | bootstrap | krm | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 14 | bootstrap | kset | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 15 | bootstrap | phantomd | skipped | n/a | — | system/build tool (toolbox generation, file utilities, daemons); not a data operator |
| 16 | datamanip | kappend | implemented | no — record concatenation; use krolling | `D.kappend` |  |
| 17 | datamanip | karith1 | implemented | yes | `D.karith1` |  |
| 18 | datamanip | karith2 | implemented | yes | `D.karith2` |  |
| 19 | datamanip | kasc2loc | implemented | no (undeclared; use krolling if same-length) | `D.kasc2seg` | ASCII -> location segment |
| 20 | datamanip | kasc2map | implemented | no (undeclared; use krolling if same-length) | `D.kasc2seg` | ASCII -> map segment |
| 21 | datamanip | kasc2mask | implemented | no (undeclared; use krolling if same-length) | `D.kasc2seg` | ASCII -> mask segment |
| 22 | datamanip | kasc2time | implemented | no (undeclared; use krolling if same-length) | `D.kasc2seg` | ASCII -> time segment |
| 23 | datamanip | kasc2val | implemented | no (undeclared; use krolling if same-length) | `D.kasc2seg` | ASCII -> value segment |
| 24 | datamanip | kaxis | implemented | no (undeclared; use krolling if same-length) | `D.kaxis` |  |
| 25 | datamanip | kbessel | implemented | yes | `D.kbessel` |  |
| 26 | datamanip | kbitwise | implemented | yes | `D.kbitwise` |  |
| 27 | datamanip | kblend | implemented | yes | `D.kblend` |  |
| 28 | datamanip | kclip | previously existed | yes | default registry op `kclip` | fixed: lower bound 0 was treated as unset; ndarray support |
| 29 | datamanip | kcmplx | implemented | yes | `D.kcmplx` |  |
| 30 | datamanip | kcmplx2real | previously existed | yes | default registry op `kcmplx2real` | re-registered: -real/-imag/-mag/-phase/-logmag, ndarray support |
| 31 | datamanip | kcomment | implemented | no (undeclared; use krolling if same-length) | `D.kcomment` |  |
| 32 | datamanip | kcompare | implemented | yes | `D.kcompare` |  |
| 33 | datamanip | kconvert | implemented | yes (not with -norm 1) | `D.kconvert` |  |
| 34 | datamanip | kcpfromval | implemented | no (undeclared; use krolling if same-length) | `D.kcpfromval` |  |
| 35 | datamanip | kcpseg | implemented | no (undeclared; use krolling if same-length) | `D.kcpseg` |  |
| 36 | datamanip | kcptoval | implemented | no (undeclared; use krolling if same-length) | `D.kcptoval` |  |
| 37 | datamanip | kextract | implemented | no — region selection by absolute index; use krolling | `D.kextract` |  |
| 38 | datamanip | kfft | previously existed | no — FFT over the whole record; use krolling | `phase21_numeric.khoros_fft` | now registered as DATAMANIP:kfft (-dir -scale -center -w -h ...) |
| 39 | datamanip | kflip | implemented | no — time reversal; use krolling | `D.kflip` |  |
| 40 | datamanip | kformats | skipped | n/a | — | lists supported file formats (informational CLI) |
| 41 | datamanip | kgate | implemented | yes | `D.kgate` |  |
| 42 | datamanip | kgconst | implemented | no (undeclared; use krolling if same-length) | `S.kgconst` |  |
| 43 | datamanip | kgeneric | skipped | n/a | — | generic external-command wrapper; semantics depend on the GUI form |
| 44 | datamanip | kgenloc | implemented | no (undeclared; use krolling if same-length) | `D.kgenloc` |  |
| 45 | datamanip | kgnoise | implemented | no — block generator; use krolling | `S.kgnoise` | gaussian/uniform/rayleigh/exponential/poisson |
| 46 | datamanip | kgsin | previously existed | no — block generator; use krolling | `phase21_numeric.khoros_sine` | now registered as DATAMANIP:kgsin |
| 47 | datamanip | khisto | implemented | no — histogram of the whole record; use krolling | `S.khisto` |  |
| 48 | datamanip | khistops | implemented | no — histogram equalisation uses the whole-record CDF; use krolling | `S.khistops` | equalize / stretch / match |
| 49 | datamanip | kimportasc | implemented | no (undeclared; use krolling if same-length) | `D.kimportasc` |  |
| 50 | datamanip | kimportraw | implemented | no (undeclared; use krolling if same-length) | `D.kimportraw` |  |
| 51 | datamanip | kimpulse | implemented | no (undeclared; use krolling if same-length) | `S.kimpulse` |  |
| 52 | datamanip | kinsertseg | implemented | no (undeclared; use krolling if same-length) | `D.kinsertseg` |  |
| 53 | datamanip | kinset | implemented | no (undeclared; use krolling if same-length) | `D.kinset` |  |
| 54 | datamanip | kinverse | implemented | yes | `D.karith1(op="recip")` | semantics not confirmed from Khoros docs; implemented as multiplicative inverse 1/x |
| 55 | datamanip | kkmeans | implemented | no (undeclared; use krolling if same-length) | `I.kmeans` |  |
| 56 | datamanip | klinearop | previously existed | yes (-w0 0, kernel from -n) | `phase21_numeric.conv1d_khoros` | vectorized (np.convolve); registered as DATAMANIP:klinearop |
| 57 | datamanip | klocxform | implemented | no (undeclared; use krolling if same-length) | `D.klocxform` |  |
| 58 | datamanip | klogexp | implemented | yes | `D.klogexp` |  |
| 59 | datamanip | kmapdata | implemented | yes | `D.kmapdata` |  |
| 60 | datamanip | kmsquish | skipped | n/a | — | semantics not established from available sources |
| 61 | datamanip | knoise | implemented | no (undeclared; use krolling if same-length) | `S.knoise` | additive/multiplicative/signal-dependent Poisson |
| 62 | datamanip | knormal | implemented | no — normalises by whole-record min/max/mean; use krolling | `D.knormal` |  |
| 63 | datamanip | korient | implemented | no — axis reordering; use krolling | `D.korient` |  |
| 64 | datamanip | kpad | implemented | no — absolute-index padding; use krolling | `D.kpad` |  |
| 65 | datamanip | kprdata | implemented | no (undeclared; use krolling if same-length) | `D.kprdata` | returns text |
| 66 | datamanip | kprval | implemented | no (undeclared; use krolling if same-length) | `D.kprval` | returns text |
| 67 | datamanip | kreal2cmplx | implemented | yes | `D.kreal2cmplx` |  |
| 68 | datamanip | kreplace | implemented | yes | `D.kreplace` |  |
| 69 | datamanip | kresample | implemented | no — resampling interpolates between neighbours on both sides; use krolling | `S.kresample` | per-axis -wsize..-esize; interp zero/linear/fft |
| 70 | datamanip | krmseg | implemented | no (undeclared; use krolling if same-length) | `D.krmseg` |  |
| 71 | datamanip | ksampline | implemented | no (undeclared; use krolling if same-length) | `D.ksampline` |  |
| 72 | datamanip | ksegcmp | implemented | no (undeclared; use krolling if same-length) | `D.ksegcmp` |  |
| 73 | datamanip | ksetdattr | implemented | no (undeclared; use krolling if same-length) | `D.ksetdattr` |  |
| 74 | datamanip | kshot | implemented | no — noise uses whole-record min/max by default; use krolling | `S.kshot` | salt-and-pepper impulse noise |
| 75 | datamanip | kstats | previously existed | no — whole-record summary; use krolling | `phase21_numeric.stats` | now registered as DATAMANIP:kstats |
| 76 | datamanip | kunmask | implemented | yes | `D.kunmask` |  |
| 77 | datamanip | kwiener | implemented | no — frequency-domain / centred-window Wiener filter; use krolling | `S.kwiener` | frequency-domain Wiener restoration (PSF on i2, -nsr) or adaptive local Wiener (no PSF) |
| 78 | dataserv | kgdbm_dump | skipped | n/a | — | database dump utility (system tool); KDBM read support already lives in kdbm_compat |
| 79 | design | kcat | skipped | n/a | — | DESIGN toolbox entry; historical semantics not documented in available sources |
| 80 | envision | kautocolor | skipped | n/a | — | interactive display colour allocation (GUI/X11 visual) |
| 81 | envision | kcolorspace | implemented | no (undeclared; use krolling if same-length) | `D.kcolorspace` | rgb<->hsv, rgb<->yiq, rgb2grey |
| 82 | geometry | gapplytex | skipped | n/a | — | 3-D renderer texture mapping (render pipeline not reconstructed) |
| 83 | geometry | gcmaplegend | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 84 | geometry | gcolorize | implemented | no (undeclared; use krolling if same-length) | `D.gcolorize` |  |
| 85 | geometry | gcuberille | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 86 | geometry | gextents | implemented | no (undeclared; use krolling if same-length) | `D.gextents` |  |
| 87 | geometry | gflipnorms | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 88 | geometry | ggencmap | implemented | no (undeclared; use krolling if same-length) | `D.ggencmap` |  |
| 89 | geometry | ggridder2d | implemented | no (undeclared; use krolling if same-length) | `D.ggridder2d` |  |
| 90 | geometry | gimpfacet | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 91 | geometry | gisosurface | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 92 | geometry | glinecolor | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 93 | geometry | glinexyz | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 94 | geometry | goctmesh | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 95 | geometry | gprgeom | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 96 | geometry | gquadmesh | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 97 | geometry | gspheres | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 98 | geometry | gtubeness | skipped | n/a | — | GEOMETRY render/mesh object operator; the geometry object model and renderer are not reconstructed |
| 99 | geometry | kslicer | implemented | no (undeclared; use krolling if same-length) | `D.kslicer` |  |
| 100 | image | ifilt_design | implemented | no (undeclared; use krolling if same-length) | `I.ifilt_design` |  |
| 101 | image | igamut | implemented | no (undeclared; use krolling if same-length) | `I.igamut` | k-means colour quantization -> index image + colormap |
| 102 | image | igauss_func | implemented | no (undeclared; use krolling if same-length) | `I.igauss_func` |  |
| 103 | image | igeowarp | implemented | no (undeclared; use krolling if same-length) | `I.igeowarp` |  |
| 104 | image | igradient | implemented | no — centred differences; use krolling | `I.igradient` |  |
| 105 | image | imedian | implemented | no — centred median window; use krolling | `I.imedian` |  |
| 106 | image | ipostscr | skipped | n/a | — | PostScript printer output (obsolete device driver) |
| 107 | image | irotate | implemented | no — 2-D resampling; use krolling | `I.irotate` |  |
| 108 | image | itexture | implemented | no (undeclared; use krolling if same-length) | `I.itexture` |  |
| 109 | matrix | mcovar | implemented | no (undeclared; use krolling if same-length) | `D.mcovar` |  |
| 110 | matrix | meigen | implemented | no (undeclared; use krolling if same-length) | `D.meigen` |  |
| 111 | matrix | mexchg | implemented | no (undeclared; use krolling if same-length) | `D.mexchg` |  |
| 112 | matrix | mextract_diag | implemented | no (undeclared; use krolling if same-length) | `D.mextract_diag` |  |
| 113 | matrix | mgdiag | implemented | no (undeclared; use krolling if same-length) | `D.mgdiag` |  |
| 114 | matrix | minvert | implemented | no (undeclared; use krolling if same-length) | `D.minvert` |  |
| 115 | matrix | mlse | implemented | no (undeclared; use krolling if same-length) | `D.mlse` |  |
| 116 | matrix | mlud | implemented | no (undeclared; use krolling if same-length) | `D.mlud` |  |
| 117 | matrix | mmul | implemented | no (undeclared; use krolling if same-length) | `D.mmul` |  |
| 118 | matrix | mpow | implemented | no (undeclared; use krolling if same-length) | `D.mpow` |  |
| 119 | matrix | mrcsum | implemented | no (undeclared; use krolling if same-length) | `D.mrcsum` |  |
| 120 | matrix | mreplicate | implemented | no (undeclared; use krolling if same-length) | `D.mreplicate` |  |
| 121 | matrix | msvd | implemented | no (undeclared; use krolling if same-length) | `D.msvd` |  |
| 122 | migration | dbcnvrt | skipped | n/a | — | Khoros 1->2 migration utility for obsolete file/toolbox formats |
| 123 | migration | kdarwin | skipped | n/a | — | Khoros 1->2 migration utility for obsolete file/toolbox formats |
| 124 | migration | ktbupdb | skipped | n/a | — | Khoros 1->2 migration utility for obsolete file/toolbox formats |
| 125 | migration | mkproto | skipped | n/a | — | Khoros 1->2 migration utility for obsolete file/toolbox formats |
| 126 | migration | uisupdate | skipped | n/a | — | Khoros 1->2 migration utility for obsolete file/toolbox formats |
| 127 | support | kdbmcat | skipped | n/a | — | developer/support utility (man pages, toolbox index, object search) |
| 128 | support | kdiffobj | skipped | n/a | — | developer/support utility (man pages, toolbox index, object search) |
| 129 | support | kfindobj | skipped | n/a | — | developer/support utility (man pages, toolbox index, object search) |
| 130 | support | kgenman | skipped | n/a | — | developer/support utility (man pages, toolbox index, object search) |
| 131 | support | kgentbindex | skipped | n/a | — | developer/support utility (man pages, toolbox index, object search) |
| 132 | support | kgenwhatis | skipped | n/a | — | developer/support utility (man pages, toolbox index, object search) |
| 133 | support | kgrepobj | skipped | n/a | — | developer/support utility (man pages, toolbox index, object search) |
| 134 | support | ksync | skipped | n/a | — | developer/support utility (man pages, toolbox index, object search) |

**Totals:** implemented 80, previously existed 6, skipped 48 (of 134).

## Extended operators (not in the 134 list)

Registered operators requested for the library but absent from the historical matrix (windows, FIR/IIR, correlation, spectra, wavelets, morphology, thresholding, etc.) and single-op aliases:

`IMAGE:iclose`, `IMAGE:idilate`, `IMAGE:iedge`, `IMAGE:ierode`, `IMAGE:ilabel`, `IMAGE:ilaplacian`, `IMAGE:imorph`, `IMAGE:iopen`, `IMAGE:iscale`, `IMAGE:ithresh`, `IMAGE:itranslate`, `DATAMANIP:kabsdiff`, `DATAMANIP:kadd`, `DATAMANIP:katan`, `DATAMANIP:kceil`, `SIGNAL:kconvolve`, `SIGNAL:kcorr`, `DATAMANIP:kcos`, `SIGNAL:kcross`, `SIGNAL:kdelay`, `SIGNAL:kdiff`, `DATAMANIP:kdiv`, `DATAMANIP:kdivinto`, `SIGNAL:kdwt`, `SIGNAL:kema`, `DATAMANIP:kexp`, `SIGNAL:kfir`, `SIGNAL:kgwindow`, `SIGNAL:khilbert`, `SIGNAL:kidwt`, `SIGNAL:kiir`, `DATAMANIP:klog`, `DATAMANIP:klog10`, `DATAMANIP:kmag`, `DATAMANIP:kmax`, `DATAMANIP:kmin`, `DATAMANIP:kmul`, `DATAMANIP:kpow`, `SIGNAL:kpsd`, `SIGNAL:krunstats`, `DATAMANIP:ksign`, `DATAMANIP:ksin`, `SIGNAL:kspectrum`, `DATAMANIP:ksub`, `DATAMANIP:ktan`, `SIGNAL:kwindow`

## Conventions

- Ports: `i` (single input), `i1`/`i2` (dual), `o` (+ `o2`, `o3`, ... for secondary outputs); `kcmplx2real`/`kmag` output `om`, as in historical workspaces.
- Options come from the glyph command (`-name value`); dual-input arithmetic falls back to the `-real` constant when `i2` is unconnected (Khoros behaviour).
- Generators take sizes from `-wsize -hsize -dsize -tsize -esize`.
- `ViffObject` inputs: operator runs on the `value` segment and returns a deep copy with new value data (attributes and other segments preserved).
- Image axes: axis 0 = width (x), axis 1 = height (y), matching VIFF value layout; N-D inputs are processed per (w,h) plane.
- Matrices: ndarray in NumPy (row, col) order; ViffObject converted with rows = height.
- Every operator is also resolvable under toolbox `*`.

## Streamability of extended operators

| operator | streamable |
|---|---|
| IMAGE:iclose | no (undeclared; use krolling if same-length) |
| IMAGE:idilate | no (undeclared; use krolling if same-length) |
| IMAGE:iedge | no — centred differences; use krolling |
| IMAGE:ierode | no (undeclared; use krolling if same-length) |
| IMAGE:ilabel | no (undeclared; use krolling if same-length) |
| IMAGE:ilaplacian | no — centred differences; use krolling |
| IMAGE:imorph | no (undeclared; use krolling if same-length) |
| IMAGE:iopen | no (undeclared; use krolling if same-length) |
| IMAGE:iscale | no (undeclared; use krolling if same-length) |
| IMAGE:ithresh | no (undeclared; use krolling if same-length) |
| IMAGE:itranslate | no (undeclared; use krolling if same-length) |
| DATAMANIP:kabsdiff | yes |
| DATAMANIP:kadd | yes |
| DATAMANIP:katan | yes |
| DATAMANIP:kceil | yes |
| SIGNAL:kconvolve | no — full/centred convolution (use kfir or klinearop -w0 0); use krolling |
| SIGNAL:kcorr | no — two-sided correlation; use krolling |
| DATAMANIP:kcos | yes |
| SIGNAL:kcross | yes |
| SIGNAL:kdelay | yes |
| SIGNAL:kdiff | yes |
| DATAMANIP:kdiv | yes |
| DATAMANIP:kdivinto | yes |
| SIGNAL:kdwt | no — block wavelet transform (wavelet denoise needs the whole block); use krolling |
| SIGNAL:kema | yes |
| DATAMANIP:kexp | yes |
| SIGNAL:kfir | yes (-mode causal, taps from options) |
| SIGNAL:kgwindow | no (undeclared; use krolling if same-length) |
| SIGNAL:khilbert | no — Hilbert transform (FFT, two-sided); use krolling |
| SIGNAL:kidwt | no — block inverse wavelet transform; use krolling |
| SIGNAL:kiir | yes (not with -zerophase 1) |
| DATAMANIP:klog | yes |
| DATAMANIP:klog10 | yes |
| DATAMANIP:kmag | yes |
| DATAMANIP:kmax | yes |
| DATAMANIP:kmin | yes |
| DATAMANIP:kmul | yes |
| DATAMANIP:kpow | yes |
| SIGNAL:kpsd | no — spectral estimate over the whole record; use krolling |
| SIGNAL:krunstats | yes |
| DATAMANIP:ksign | yes |
| DATAMANIP:ksin | yes |
| SIGNAL:kspectrum | no — FFT over the whole record; use krolling |
| DATAMANIP:ksub | yes |
| DATAMANIP:ktan | yes |
| SIGNAL:kwindow | no (undeclared; use krolling if same-length) |

Streaming I/O glyphs: `STREAM:kinput -name X`, `STREAM:koutput -name Y`, `STREAM:krolling -op NAME -window N [inner options]`.

## FINANCE toolbox (authoring branch)

| operator | streamable |
|---|---|
| FINANCE:khold | yes |
| FINANCE:kkalman | yes |
| FINANCE:kkalmanpair | yes |
| FINANCE:klogret | yes |
| FINANCE:kthreshold3 | yes |
| FINANCE:kzscore | yes |

Full per-operator metadata (ports, parameters, lag, cost): `docs/CATALOG.md`.
