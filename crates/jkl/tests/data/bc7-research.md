# BC7 research and implementation references

Retrieved 2026-10-15. Downloaded papers and implementation sources are saved in
`build/bc7-research/` at the repository root. These are research inputs, not
runtime dependencies. JKL's encoder and decoder are native Rust.

## Papers

* Pavel Krajcevski, Adam Lake, Dinesh Manocha, **FasTC: Accelerated Fixed-Rate
  Texture Encoding**, I3D 2013.
  [Author PDF](https://gamma-web.iacs.umd.edu/FasTC/extra/paper.pdf).
  Uses bounding-box estimates to rank partitions, generalized cluster fitting
  in quantized space, and optional simulated annealing. Relevant to endpoint
  fitting and the quality/throughput tradeoff.
* Pavel Krajcevski, Dinesh Manocha, **SegTC: Fast Texture Compression using
  Image Segmentation**, HPG 2014.
  [Author PDF](https://gamma-web.iacs.umd.edu/FasTC/extra/SegTC.pdf).
  Uses whole-image superpixel segmentation and matching against predefined
  partition patterns. Useful for future image-level acceleration; a standalone
  block API cannot use whole-image context.

## Normative format references

* [Khronos DataFormat BPTC chapter](https://github.com/KhronosGroup/DataFormat/blob/main/bptc.txt):
  endpoint expansion, integer interpolation, partition patterns, and fixup anchors.
* [Microsoft BC7 mode reference](https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format-mode-reference):
  eight modes and their bit allocations.
* [Microsoft BC7 format](https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format):
  decoding conventions and component rotation.

## Production open-source implementations

| Project | Useful implementation detail | License / scope |
| --- | --- | --- |
| [Microsoft DirectXTex](https://github.com/microsoft/DirectXTex/blob/main/DirectXTex/BC6HBC7.cpp) | Mode/shape enumeration, endpoint optimization, and scalar decoding | MIT; BC6H and BC7 |
| [AMD Compressonator](https://github.com/GPUOpen-Tools/compressonator/blob/master/cmp_compressonatorlib/bc7/bc7_encode.cpp) | Single-index versus separate-alpha fitting, p-bit setup, fixup normalization | MIT; dedicated BC7 encoder |
| [Intel ISPCTextureCompressor](https://github.com/GameTechDev/ISPCTextureCompressor/blob/master/ispc_texcomp/kernel.ispc) | SIMD-oriented mode groups, partition ranking, bounded refinement | MIT; Intel has discontinued maintenance |
| [Richard Geldreich's bc7enc_rdo](https://github.com/richgel999/bc7enc_rdo) | Compact quantization, least-squares refinement, quality controls; independent quality baseline | MIT or public domain for the downloaded bc7enc source; encoder uses modes 1/6 for RGB and 5/6/7 for RGBA |
| [iOrange bcdec](https://github.com/iOrange/bcdec) | Small independent decoder for differential fixtures | MIT or Unlicense; validation only |

The bc7enc header contains older comments describing only modes 1 and 6;
the implementation's mode paths determine the actual comparison scope.
Its quality scores do not represent an exhaustive all-mode BC7 optimum.

## Design and evidence boundaries

The codec uses stored-channel RGBA byte squared error, without gamma conversion
or perceptual channel weights. An all-mode search includes all legal partitions,
all mode-4 rotation/index-selection combinations, and all mode-5 rotations.
Quantization and index assignment must score the exact integer interpolation
palette. Separate RGB and alpha index streams need independent fixup inversions.

Format coverage, endpoint-search quality, and throughput are separate claims.
Reference-vector parity establishes decoding correctness on the covered inputs;
encoder comparisons establish corpus-scoped error and timing. Neither proves a
global optimum. See `bc7-reference.md` and `bc7-quality-results.md` for measured
validation and reproduction.

## Download fingerprints (SHA256)

Upstream revisions at retrieval:

```text
KhronosGroup/DataFormat 8020b6258d969562e82d3005cba24bd8e6633940
microsoft/DirectXTex 1acf4ebe6330e06b7eb52e8db63d4f824ad77427
GPUOpen-Tools/compressonator f4b53d79ec5abbb50924f58aebb7bf2793200b94
GameTechDev/ISPCTextureCompressor 79ddbc90334fc31edd438e68ccb0fe99b4e15aab
richgel999/bc7enc_rdo b9438627eef73a1157e84201b6fa6eb2ffd6d9f0
iOrange/bcdec 80859ed3b7afb1c527a2a99d70c61457bea72d0c
```

```text
FasTC-2013.pdf 9e93c135862d9818f81fb2eaaedeb19bf56d5fe8a654a7bc321234bf33d93602
SegTC-2014.pdf 4a265d330ecdb1060d806384ac386bb671b6259e8a99007d9f1f2d3a14cd9e88
bptc.txt c4d68c53a07464bc546cc4a584a6fd1a605d10b79e675bc9bc4f78e28b29482f
DirectXTex-BC6HBC7.cpp 8962b6f2e28fce5c2003747f1e62219da8352fb79ba8245f60d53c2f20722e11
compressonator-bc7.cpp 85c520865fcaa39d5649c7a74dd9fda402d3a4788112a3f94c203670f2a64178
ispc-kernel.ispc 9562c59daf23621917ce91f6fb2f100343f3fc64ea4036202f3c5258ece4f479
bc7enc.cpp b90444a091530a13c61fa6375583fcc9bc9ca47e474fd71f818876de68c1b581
bc7enc.h f40f4a956ed6a8845b7909e83831de09512424408b32ca29551bf49aee2fecf6
bcdec.h 134520764d96f70a27db814173616c89ad85db95fbd3328417c7a8c77e7ca189
```
