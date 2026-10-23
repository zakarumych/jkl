BC7 independent decoder and quality fixtures
==========================================

The production codec has no third-party codec dependency. These deterministic
fixtures use independent implementations during offline generation only.
Sources, papers, license information, and pinned revisions are catalogued in
[bc7-research.md](bc7-research.md).

Decoder source: iOrange/bcdec, commit
`80859ed3b7afb1c527a2a99d70c61457bea72d0c`, `bcdec.h` version 0.985,
SHA256 `134520764d96f70a27db814173616c89ad85db95fbd3328417c7a8c77e7ca189`.

`generate-bc7-reference.c` uses xorshift32 seeded `0xb7c72026` and writes
4,448 records of 80 bytes: raw BC7 bytes `[16]`, then bcdec RGBA8 output `[64]`
in row-major texel/channel order. Every legal mode/partition/rotation/index
selection configuration (285 total) has eight payload patterns: zero, all-one,
alternating `aa`, alternating `55`, and four random patterns. All per-mode
pbit combinations have additional random records. Each mode also has 256
random payloads. Sixteen zero-prefix reserved headers verify transparent-black
behavior. Every two/three-subset partition and its nonzero anchors occurs.
These are differential fixtures, not an exhaustive 128-bit payload proof.

Fixture SHA256:
`b3c3c044ff46c788638d4ae47ad40997ce2325600054687408525c0cb2c046b6`.

Quality source: richgel999/bc7enc_rdo, commit
`b9438627eef73a1157e84201b6fa6eb2ffd6d9f0`, `bc7enc.cpp` SHA256
`b90444a091530a13c61fa6375583fcc9bc9ca47e474fd71f818876de68c1b581`,
`bc7enc.h` SHA256
`f40f4a956ed6a8845b7909e83831de09512424408b32ca29551bf49aee2fecf6`.

`generate-bc7-quality.cpp` uses xorshift32 seeded `0x0bc72026` and writes
64 records of 148 bytes: category `[1]`, ordinal `[1]`, zero reserved `[2]`,
source RGBA8 `[64]`, bc7enc block `[16]`, bcdec-decoded reference RGBA8 `[64]`.
Eight cases each cover flat colors, ramps, two spatial groups, three spatial
groups, independent alpha, correlated noise, independent noise, and decoded
representable blocks (one from each BC7 mode). Group cases have opaque alpha;
other categories include varying alpha. These synthetic 4x4 blocks are no
substitute for a natural-image or texture corpus.

Reference parameters: defaults initialized, then linear weights selected
(`m_perceptual=false`, RGBA weights all 1), `m_uber_level=4`,
`m_max_partitions=64`, `m_mode17_partition_estimation_filterbank=false`.
Least-squares enabled, all supported reference modes enabled, alpha detection
automatic, no rate-distortion optimization or gamma conversion. The source
supports modes 1/5/6/7; its nominal mode mask does not imply all eight modes.

Quality fixture SHA256:
`dad710bb1a72997012a86fa96993f1d78301e4459563c7e97cdcc986a1918227`.

Metric: sum over all sixteen texels and all RGBA channels of squared byte
differences. JKL receives exact byte/255 normalized floats; decoded channels
are converted back with nearest-byte rounding. Reference output is decoded
independently with bcdec. Lower SSE is better. Representable source blocks
establish zero as attainable, without requiring a bounded encoder to find it.
No global optimality or universal quality claim follows from this corpus.

Reproduction from repository root, with pinned reference sources placed in
`build/bc7-research/`:

```powershell
gcc -O2 -I build/bc7-research crates/jkl/tests/data/generate-bc7-reference.c -o build/bc7-research/generate-reference.exe
./build/bc7-research/generate-reference.exe crates/jkl/tests/data/bc7-reference.bin
g++ -O2 -I build/bc7-research crates/jkl/tests/data/generate-bc7-quality.cpp build/bc7-research/bc7enc.cpp -o build/bc7-research/generate-quality.exe
./build/bc7-research/generate-quality.exe crates/jkl/tests/data/bc7-quality.bin
gcc -O2 -I build/bc7-research crates/jkl/tests/data/check-bc7-encoded.c -o build/bc7-research/check-encoded.exe
```

From `crates/`:

```powershell
cargo test -p jkl --test bc7_reference
cargo test -p jkl --release --test bc7_quality -- --ignored --nocapture
../build/bc7-research/check-encoded.exe target/bc7-encoded.bin
```

The ignored audit exports 64 quality-corpus blocks plus 32 partially masked
blocks derived from independent fixtures. `BC7_EXPORT` can override the output
path. The C checker compares independently decoded bytes against JKL's output,
so a shared error between JKL's encoder and decoder cannot pass this check.
Encoder-only timing covers the 64 quality cases, excluding input construction,
decoding, scoring, export, and partially masked stress blocks. All evidence is
for standalone CPU blocks; no GPU/runtime or JKLI integration is implied.
