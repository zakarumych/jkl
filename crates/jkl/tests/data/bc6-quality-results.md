BC6H quality measurements
=========================

Measured 2026-10-13 on Windows, Rust `1.98.1 (48a229cea 2026-09-01)`, release
profile. Production `src/image/block/bc6.rs` SHA256:
`e65e1c728b6c2093cb14edaecfe1fe75dc9f8622fa1e04ed05a55a2446024650`.
Workspace base commit: `10b5904727965b067952eaf6b0d20e80fd00acc7`.
The source hash identifies the evaluated uncommitted encoder implementation.

Fixture provenance, pinned Microsoft reference sources, exact metric, and
generation instructions are in [bc6-quality.md](bc6-quality.md).

The 184-case independent quality comparison passed: every encoded block had
decoded half-code SSE no greater than the exhaustive-partition DirectXTex
reference block. All 56 sampled representable inputs recovered the known
global minimum exactly: zero SSE.

| Interpretation | Category | Cases | Production SSE | Reference SSE | SSE ratio | Worst block ratio |
|---|---|---:|---:|---:|---:|---:|
| UF16 | Half-code gradient | 8 | 1736593 | 142181105 | 0.012214 | 0.042009 |
| UF16 | Two groups | 8 | 543860 | 688934 | 0.789422 | 1.000000 |
| UF16 | Correlated noise | 8 | 4827642 | 90976983 | 0.053064 | 0.317502 |
| UF16 | Independent noise | 8 | 9624611666 | 13351139716 | 0.720883 | 0.819503 |
| UF16 | Three clusters | 8 | 2290931778 | 2830903096 | 0.809258 | 0.998979 |
| UF16 | Subnormal/sign | 8 | 10371281 | 11567241 | 0.896608 | 0.957029 |
| UF16 | Narrow bright | 8 | 161292 | 166587 | 0.968215 | 0.995415 |
| UF16 | Linear gradient | 8 | 55488413 | 177859399 | 0.311979 | 0.833455 |
| UF16 | Known optimum | 28 | 0 | 317387670 | 0.000000 | 1.000000 |
| SF16 | Half-code gradient | 8 | 3711870 | 970703390 | 0.003824 | 0.016537 |
| SF16 | Two groups | 8 | 736960 | 791136 | 0.931521 | 1.000000 |
| SF16 | Correlated noise | 8 | 13894281 | 527324634 | 0.026349 | 0.233936 |
| SF16 | Independent noise | 8 | 34082579292 | 52447694982 | 0.649839 | 0.761941 |
| SF16 | Three clusters | 8 | 13363556778 | 20013352563 | 0.667732 | 0.924973 |
| SF16 | Subnormal/sign | 8 | 14568442 | 16695390 | 0.872603 | 0.941509 |
| SF16 | Narrow bright | 8 | 165465 | 171469 | 0.964985 | 1.000000 |
| SF16 | Linear gradient | 8 | 76542326 | 183668143 | 0.416743 | 0.797066 |
| SF16 | Known optimum | 28 | 0 | 4594404688 | 0.000000 | 0.000000 |

The UF16 worst known-optimum ratio of 1 represents a block for which both
production and reference SSE are zero. Aggregate production/reference SSE:
UF16 `11988672525 / 16922870731 = 0.708430`; SF16
`47555755414 / 78754806395 = 0.603846`.

The full representable audit passed for all 952 signed interpretations of the
476 independent decoder fixtures. Each source mode contributed 34 inputs in
each interpretation: randomized endpoint/index data across partition patterns
0-31, plus zero and all-one endpoint payloads. Constant and duplicate inputs
are included in these counts. No nonfinite cases were skipped.

| Interpretation | Cases | Exact global optimum | SSE | Half-code RMSE | RMSE / full code range | RMSE / block standard deviation |
|---|---:|---:|---:|---:|---:|---:|
| UF16 | 476 | 476 | 0 | 0 | 0 | 0 |
| SF16 | 476 | 476 | 0 | 0 | 0 | 0 |

The ignored regression audits require zero SSE for all 56 sampled and all 952
full representable inputs, plus independent-reference aggregate quality floors
for each of the 18 signed/category cohorts. Ordinary test runs also cross-check
production decoding against the independent reference and known exact witnesses.

Encoder-only timing: one release pass over the preconstructed 184-case corpus,
`55.0337061 s`, or `299.096 ms/block`. Input conversion, fixture construction,
decoding, scoring, and reference encoding are outside the timed interval.
This is one host measurement, not a statistical throughput estimate. The
full representable quality audit took `87.73 s`; that duration includes checks
and scoring and is not an encoder-only benchmark.

Commands, run from the `crates` workspace directory:

```sh
cargo test -p jkl --test bc6_quality independent_reference_decoding_and_known_optimum
cargo test -p jkl --release --test bc6_quality independent_quality_report -- --ignored --nocapture
cargo test -p jkl --release --test bc6_quality full_reference_representable_audit -- --ignored --nocapture
cargo test -p jkl --release --test bc6_quality encoder_timing -- --ignored --nocapture
```

These results establish exact global optimality on the representable fixture
inputs and independent reference quality on the structured corpus. Endpoint
search remains bounded and heuristic on arbitrary inputs; these measurements
do not provide a universal global-optimum bound. The objective is decoded
half-code SSE, not absolute linear-light error or a perceptual image metric.

Larger-budget convergence audit, evaluated on the same production source hash:
production uses 64 finalists, 256 sample-pair starts, 24 fit iterations, eight
discrete sweeps, and six ordered-refinement passes. The audit uses 128 finalists,
512 sample-pair starts, 48 fit iterations, twelve sweeps, and twelve passes,
retaining the production result as an incumbent. Production matches the audit
on 183 of 184 inputs. The remaining input, SF16 narrow-bright case 115, scores
17,673 versus 17,566 SSE: a 0.609131% gap. Aggregate production and audit SSE are
59,544,427,939 and 59,544,427,832, respectively. All 56 sampled representable
inputs have zero error with both budgets. This checks search convergence; the
larger search is heuristic and does not certify a global bound.

```sh
cargo test -p jkl --release --lib image::block::bc6::tests::quality_search_report -- --ignored --nocapture
```
