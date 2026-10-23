BC7 quality and correctness measurements
=======================================

Measured 2026-10-15 on Windows, Rust `1.98.1 (48a229cea 2026-09-01)`, release
profile. Workspace base commit `6e18cf57fad1a8bc8b0456555919e63b171cbd27`.
Evaluated uncommitted production source SHA256:

* `src/image/block/bc7.rs`:
  `dfd0eea932860bc9be602ece3593227e72e70f38f5f7c1ca70e93d2f150efcbe`
* `src/image/block/bc7/encode.rs`:
  `3b629d98ef164fe59d0aea4633f7b7c37f04d0b7c3039c93c33ef34b6931fe15`
* `src/image/block/bc7/tables.rs`:
  `302e540319fbe329cce8f250e96775f00cd175e1b221fdaddb42e89f5860c6e4`

Pinned sources, fixture generation, exact metric, configuration, reproduction
commands, and scope limits are in [bc7-reference.md](bc7-reference.md).

All 4,448 independently generated bcdec decoder fixtures agree exactly on every
normalized RGBA component. The four integration tests pass, covering those
fixtures, independent quality-reference decoding, all-bit FixedCode roundtrip,
mask independence, deterministic encoding, and sanitization.

The independently decoded encoder audit also passes: bcdec agrees byte-for-byte
with JKL decoding for 64 emitted quality blocks plus 32 partial-mask stress
blocks. All eight encoding modes occur in the 64 quality outputs; mode counts
0 through 7 are `[1, 6, 1, 13, 16, 5, 19, 3]`.

Uniform RGBA byte-space SSE, eight cases per category:

| Category | JKL SSE | bc7enc SSE | JKL/reference | JKL no worse | JKL zero SSE |
|---|---:|---:|---:|---:|---:|
| Flat | 0 | 80 | 0.000000 | 8/8 | 8/8 |
| Ramp | 206 | 248 | 0.830645 | 8/8 | 0/8 |
| Two groups | 1054 | 1207 | 0.873239 | 8/8 | 0/8 |
| Three groups | 1533 | 36161 | 0.042394 | 8/8 | 0/8 |
| Independent alpha | 33440 | 69487 | 0.481241 | 8/8 | 0/8 |
| Correlated | 3232 | 3370 | 0.959050 | 8/8 | 0/8 |
| Random | 898033 | 1098416 | 0.817571 | 8/8 | 0/8 |
| Representable | 477 | 129523 | 0.003683 | 6/8 | 4/8 |
| Total | 937975 | 1338492 | 0.700770 | 62/64 | 12/64 |

JKL's aggregate SSE is about 29.9% lower on this synthetic corpus. This is
corpus-scoped evidence against this pinned reference configuration, not a
universal quality ranking. The independent reference searches modes 1/5/6/7;
JKL searches all eight modes. Four of eight known-representable cases recover
the attainable zero SSE. The remaining four total 477 SSE; two are worse than
the reference on those individual blocks. The encoder does not promise global
optimality, including on representable inputs. The ignored regression audit
uses the recorded per-category JKL SSE values as nonregression ceilings.

One release encoder-only pass over preconstructed inputs took `168.1424 ms`,
or `2.627 ms/block`. This excludes fixture parsing, float input construction,
decoding, scoring, export, and the 32 additional stress encodes. This single
host run is not a statistical throughput estimate. The complete audit test
took `0.22 s`. No GPU sampling, full-image texture corpus, or JKLI file-path
integration is established by these standalone CPU-block checks.
