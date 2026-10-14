BC6H reference vectors
=====================

`bc6-reference.bin` contains 476 records of 208 bytes each:

* 16 bytes: raw BC6H block.
* 96 bytes: 48 little-endian binary16 channels decoded as UF16.
* 96 bytes: 48 little-endian binary16 channels decoded as SF16.

The independent reference is iOrange/bcdec, `bcdec_bc6h_half`, retrieved from
https://raw.githubusercontent.com/iOrange/bcdec/master/bcdec.h on 2026-10-13.
Header SHA256: `134520764d96f70a27db814173616c89ad85db95fbd3328417c7a8c77e7ca189`.
bcdec offers MIT or Unlicense terms; its source is not bundled here.

The generator covers every valid mode, each of the 32 partitions, randomized
nonzero endpoints/deltas/indices, and zero/all-one endpoint payloads. Both
signed interpretations are recorded. The seed and random algorithm are fixed
in `generate-bc6-reference.c`.

To reproduce, place the reference header on the include path and run:

```sh
cc -I /path/to/bcdec generate-bc6-reference.c -o generate-bc6-reference
./generate-bc6-reference > bc6-reference.bin
```

Production endpoint layouts and arithmetic follow Khronos DataFormat's BC6H
specification: https://github.com/KhronosGroup/DataFormat/blob/main/bptc.txt.
Signed finalization follows Microsoft's BC6H pseudocode, which determines the
sign after scaling and returns positive zero for negative underflow:
https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc6h-format.
