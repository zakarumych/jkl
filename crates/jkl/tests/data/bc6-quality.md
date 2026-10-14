BC6H independent quality corpus
===============================

`bc6-quality.bin` has 184 records of 228 bytes, little-endian:

SHA256: `017b52fa7f3414fb01fa0b69ca40f367b53becb454d94a13e47d5fa7fe889d61`.

* 1 byte: signed interpretation (0 = UF16, 1 = SF16).
* 1 byte: category (listed below).
* 2 bytes: ordinal within category.
* 96 bytes: 16 input texels, RGB binary16 channels in row-major order.
* 16 bytes: block encoded by the independent reference encoder.
* 96 bytes: reference encoder block decoded by its independent decoder.
* 16 bytes: exact source block for category 8, zero bytes otherwise.

Categories 0-7 have eight cases in each signed interpretation (128 total):
half-code gradients, two spatial groups, correlated noise, independent noise,
three cross-channel clusters, subnormal/sign crossings, narrow bright ranges,
and linear-light gradients. Inputs use fixed xorshift32 seed `0xbc6a110f`.

Category 8 has two randomized, independently decoded blocks from each of the
14 BC6H modes in both interpretations (56 total). Source blocks and binary16
inputs come from `bc6-reference.bin`, patterns 0 and 17. These inputs have a
known global minimum of **zero** half-code squared error: the source block
reproduces every input exactly. They test endpoint-search limitations directly.

Reference encoder: Microsoft DirectXTex `BC6HBC7.cpp` and `BC.h`, MIT license,
commit `1acf4ebe6330e06b7eb52e8db63d4f824ad77427`:
https://github.com/microsoft/DirectXTex/tree/1acf4ebe6330e06b7eb52e8db63d4f824ad77427

Reference math: Microsoft DirectXMath, MIT license,
commit `5bbe2ceeec3236c27ad44c877715e1e0f3b9817a`:
https://github.com/microsoft/DirectXMath/tree/5bbe2ceeec3236c27ad44c877715e1e0f3b9817a

The reference explores all 14 modes. For this corpus its partition refinement
count is increased to all partitions by replacing this BC6H-only line:

```cpp
const size_t uItems = std::max<size_t>(1u, size_t(uShapes >> 2));
```

with `const size_t uItems = uShapes;`. This refines all 324 mode/partition
configurations rather than ranking 32 and refining only eight per two-subset
mode. Endpoint fitting remains heuristic, so reference comparisons alone
cannot establish a global-optimum gap.

Build the generator with a C++17 compiler. Put the pinned reference
`BC6HBC7.cpp` and `BC.h` in an isolated directory. Provide this minimal
`DirectXTexP.h` there instead of the library's platform header:

```cpp
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#define UNREFERENCED_PARAMETER(x) (void)(x)
#define TEX_COMPRESS_BC7_QUICK 0x100000
```

GCC/MinGW provides SAL annotations through its `sal.h`. On another platform,
use compatible SAL headers or the standard DirectXMath build instructions.
No encoder arithmetic, mode layouts, or endpoint algorithms are modified.

```sh
g++ -std=c++17 -O2 -I/path/to/DirectXMath/Inc -I/path/to/reference \
    bc6-quality-generate.cpp /path/to/reference/BC6HBC7.cpp -o bc6-quality-generate
./bc6-quality-generate bc6-reference.bin bc6-quality.bin
```

Quality metric: sum of squared differences of final decoded binary16 numeric
codes. Unsigned code is the half's magnitude bits; signed code is negative
magnitude for a negative half. Negative zero equals zero. Measure against
actual decoder output rather than unscaled endpoints. Use integer `u64` sums.
This metric weights relative HDR variation; it is not linear-light squared error.

Integration harness `bc6_quality.rs` checks both independent reference decoder
outputs and exact-source decoder outputs. Its ignored release quality report
measures each signed/category cohort and asserts aggregate SSE does not exceed
the independent encoder's SSE in any cohort. It also requires the production
encoder to reproduce all 56 known-optimum inputs exactly, with zero decoded SSE.
It prints individual known-optimum
cases, aggregate and worst-case ratios, absolute half-code RMSE, RMSE divided by
the full half-code range, and RMSE divided by each block's per-channel standard
deviation. Variance normalization exposes failures in narrow-range blocks.
Measured source hashes, cohort scores, exact recovery counts, and timing are
recorded in [bc6-quality-results.md](bc6-quality-results.md).
Neither reference superiority nor small range-normalized RMSE proves closeness
to a global optimum on arbitrary inputs.

```sh
cargo test -p jkl --release --test bc6_quality independent_quality_report -- --ignored --nocapture
cargo test -p jkl --release --test bc6_quality encoder_timing -- --ignored --nocapture
cargo test -p jkl --release --test bc6_quality full_reference_representable_audit -- --ignored --nocapture
```

The full representable audit uses all 476 source blocks from `bc6-reference.bin`
in both signed interpretations, covering all partitions rather than sampling
patterns 0 and 17. It excludes and counts nonfinite decoded inputs because the
public encoder clamps them, invalidating the source block's zero-error witness.
Finite inputs have an exact source witness checked through the production
decoder against the independent fixture before encoding. This audit reports
recovery counts and residuals and requires exact recovery of all 952 finite
fixture interpretations. Its sampling is not a bound for arbitrary blocks.
