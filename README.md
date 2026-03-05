<div align="center"><img src="https://raw.githubusercontent.com/zakarumych/jkl/main/logo.svg" alt="JKL logo" width="1440" alt="JKL"></div>

---

JKL is a batteries-included Rust crate for compressing data
for reduced footprint and blazing fast decompression on
CPU *or* directly on the GPU. If you ship textures, sprite atlases, tile
maps, or any bulk binary payload that must be ready the instant it hits
VRAM, JKL was built for you.

## Feature highlights

* **GPU-oriented block texture codecs** - encode and decode BC1 through BC5 blocks.
  Cluster-fit algorithm finds endpoints for best perceptual quality.
* **Layered entropy coding** - Not all data is equally compressible,
  stack loseless encoders such as rANS, LZ77 or simple RLE.
  Find the optimal combination balancing compression ratio and decompression speed.
* **Bit I/O** - the [`bits`] module provides bit-level IO wrapper for `std::io::Read` and `std::io::Write`.
  Bitstreams are used extensively in JKL for variable-length coding of symbols.
* **Jackal Image format** - the [`jackal::image`] module provides API for working with Jackal Image (JKLI) file format.
  Write entire images and read them back as tiles, supporting parallelized decompression.
* **Atlas** - [`max_rects`] packs rectangles for
  atlas generation

## Quick orientation - "how do I …?"

| I want to …                              | Start here                                                                           |
|-------------------------------------------|--------------------------------------------------------------------------------------|
| Compress a texture to BC1–BC5             | [`bc1::Block::encode`], [`bc2::Block::encode`], …[`bc5::Block::encode`]              |
| Decompress a BC block                     | [`bc1::Block::decode`], [`bc2::Block::decode`], …[`bc5::Block::decode`]              |
| Write / read a complete Jackal Image file | [`jackal::image::write_image`], [`jackal::image::JackalReader`]                      |
| Entropy-code a symbol stream              | Build an [`ans::Context`], then use [`ans::Encoder`] / [`ans::Decoder`]              |
| Run-length encode an iterator             | [`rle::rle`], [`rle::rle_power_of_two`], or [`rle::rle_with_cfg`]                   |
| LZ77-compress a value stream              | [`lz77::Encoder`] → [`lz77::Token`] stream → [`lz77::Decoder`]                      |
| LZ78-compress a value stream              | [`lz78::Encoder`] → [`lz78::Token`] stream → [`lz78::Decoder`]                      |
| Encode small unsigned ints compactly      | [`vle::encode`] / [`vle::decode`] (Elias delta), or wrap with [`vle::Vle`]           |
| Map signed ints for better compression    | [`zigzaq::ZigZag::zigzag`] before VLE encoding                                      |
| Read / write individual bits              | [`bits::WriteBits`], [`bits::ReadBits`], or the [`bits::write_bits_scope`] helper    |
| Bin-pack rectangles into an atlas         | [`max_rects::MaximalRectangles::new`] then [`insert`][mr_insert] in a loop           |
| Work with 2D pixel buffers                | [`image::Image2DRef`] / [`image::Image2DMut`]                                        |
| Use vector math or pixel types            | [`math::Vec3`], [`math::Rgb565`], [`math::Rgba32F`], etc.                            |

## Module overview

### Compression

| Module        | Purpose                                                                |
|---------------|------------------------------------------------------------------------|
| [`ans`]       | Asymmetric Numeral Systems (ANS) entropy coder                         |
| [`rle`]       | Run-length encoding with configurable max length and power-of-two mode |
| [`vle`]       | Variable-length Elias delta coding for unsigned integers               |
| [`lz77`]      | LZ77 sliding-window compressor / decompressor                          |
| [`lz78`]      | LZ78 dictionary compressor (GPU-friendly variant)                      |
| [`zigzaq`]    | Zigzag signed ↔ unsigned mapping for better VLE compression            |

### Block-texture codecs

| Module          | Format                     | Block size | Channels        |
|-----------------|----------------------------|------------|-----------------|
| [`bc1`]         | BC1 / DXT1                 | 8 bytes    | RGB + 1-bit A   |
| [`bc2`]         | BC2 / DXT3                 | 16 bytes   | RGBA (4-bit A)  |
| [`bc3`]         | BC3 / DXT5                 | 16 bytes   | RGBA (interp A) |
| [`bc4`]         | BC4 / RGTC1                | 8 bytes    | R               |
| [`bc5`]         | BC5 / RGTC2                | 16 bytes   | RG              |
| [`cluster_fit`] | Shared quantizer for above | –          | –               |

### Image and math

| Module    | Purpose                                                                 |
|-----------|-------------------------------------------------------------------------|
| [`image`] | Non-owning 2D / 3D image views ([`Image2DRef`][i2dr], [`Image2DMut`][i2dm], …) |
| [`math`]  | Vectors ([`Vec2`][v2]–[`Vec4`][v4]), pixel types, [`Rect`][rect], PCA, bit-interleaving |

### I/O and serialization

| Module     | Purpose                                                                |
|------------|------------------------------------------------------------------------|
| [`bits`]   | Bit-level buffered reader / writer over byte streams                   |
| [`encode`] | [`FixedCode`][fc] and [`VarCode`][vc] serialization traits             |

### Packing and spatial

| Module        | Purpose                                                               |
|---------------|-----------------------------------------------------------------------|
| [`max_rects`] | 2D bin packing (maximal-rectangles algorithm) for atlas generation     |
| [`z_curve`]   | Z-order / Morton curve coordinate iteration                           |

### File formats

| Module            | Purpose                                                      |
|-------------------|--------------------------------------------------------------|
| [`jackal::image`] | Tiled, GPU-decompressible image container format             |

<!-- docs.rs link definitions -->
[`bits`]: https://docs.rs/jkl/0.2.1/jkl/bits/index.html
[`jackal::image`]: https://docs.rs/jkl/0.2.1/jkl/jackal/image/index.html
[`max_rects`]: https://docs.rs/jkl/0.2.1/jkl/max_rects/index.html
[`z_curve`]: https://docs.rs/jkl/0.2.1/jkl/z_curve/index.html
[`ans`]: https://docs.rs/jkl/0.2.1/jkl/ans/index.html
[`rle`]: https://docs.rs/jkl/0.2.1/jkl/rle/index.html
[`vle`]: https://docs.rs/jkl/0.2.1/jkl/vle/index.html
[`lz77`]: https://docs.rs/jkl/0.2.1/jkl/lz77/index.html
[`lz78`]: https://docs.rs/jkl/0.2.1/jkl/lz78/index.html
[`zigzaq`]: https://docs.rs/jkl/0.2.1/jkl/zigzaq/index.html
[`bc1`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc1/index.html
[`bc2`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc2/index.html
[`bc3`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc3/index.html
[`bc4`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc4/index.html
[`bc5`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc5/index.html
[`cluster_fit`]: https://docs.rs/jkl/0.2.1/jkl/cluster_fit/index.html
[`image`]: https://docs.rs/jkl/0.2.1/jkl/image/index.html
[`math`]: https://docs.rs/jkl/0.2.1/jkl/math/index.html
[`encode`]: https://docs.rs/jkl/0.2.1/jkl/encode/index.html
[`bc1::Block::encode`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc1/struct.Block.html#method.encode
[`bc2::Block::encode`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc2/struct.Block.html#method.encode
[`bc5::Block::encode`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc5/struct.Block.html#method.encode
[`bc1::Block::decode`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc1/struct.Block.html#method.decode
[`bc2::Block::decode`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc2/struct.Block.html#method.decode
[`bc5::Block::decode`]: https://docs.rs/jkl/0.2.1/jkl/image/block/bc5/struct.Block.html#method.decode
[`jackal::image::write_image`]: https://docs.rs/jkl/0.2.1/jkl/jackal/image/fn.write_image.html
[`jackal::image::JackalReader`]: https://docs.rs/jkl/0.2.1/jkl/jackal/image/struct.JackalReader.html
[`ans::Context`]: https://docs.rs/jkl/0.2.1/jkl/ans/struct.Context.html
[`ans::Encoder`]: https://docs.rs/jkl/0.2.1/jkl/ans/struct.Encoder.html
[`ans::Decoder`]: https://docs.rs/jkl/0.2.1/jkl/ans/struct.Decoder.html
[`rle::rle`]: https://docs.rs/jkl/0.2.1/jkl/rle/fn.rle.html
[`rle::rle_power_of_two`]: https://docs.rs/jkl/0.2.1/jkl/rle/fn.rle_power_of_two.html
[`rle::rle_with_cfg`]: https://docs.rs/jkl/0.2.1/jkl/rle/fn.rle_with_cfg.html
[`lz77::Encoder`]: https://docs.rs/jkl/0.2.1/jkl/lz77/struct.Encoder.html
[`lz77::Token`]: https://docs.rs/jkl/0.2.1/jkl/lz77/enum.Token.html
[`lz77::Decoder`]: https://docs.rs/jkl/0.2.1/jkl/lz77/struct.Decoder.html
[`lz78::Encoder`]: https://docs.rs/jkl/0.2.1/jkl/lz78/struct.Encoder.html
[`lz78::Token`]: https://docs.rs/jkl/0.2.1/jkl/lz78/struct.Token.html
[`lz78::Decoder`]: https://docs.rs/jkl/0.2.1/jkl/lz78/struct.Decoder.html
[`vle::encode`]: https://docs.rs/jkl/0.2.1/jkl/vle/fn.encode.html
[`vle::decode`]: https://docs.rs/jkl/0.2.1/jkl/vle/fn.decode.html
[`vle::Vle`]: https://docs.rs/jkl/0.2.1/jkl/vle/struct.Vle.html
[`zigzaq::ZigZag::zigzag`]: https://docs.rs/jkl/0.2.1/jkl/zigzaq/trait.ZigZag.html#tymethod.zigzag
[`bits::WriteBits`]: https://docs.rs/jkl/0.2.1/jkl/bits/struct.WriteBits.html
[`bits::ReadBits`]: https://docs.rs/jkl/0.2.1/jkl/bits/struct.ReadBits.html
[`bits::write_bits_scope`]: https://docs.rs/jkl/0.2.1/jkl/bits/fn.write_bits_scope.html
[`max_rects::MaximalRectangles::new`]: https://docs.rs/jkl/0.2.1/jkl/max_rects/struct.MaximalRectangles.html#method.new
[mr_insert]: https://docs.rs/jkl/0.2.1/jkl/max_rects/struct.MaximalRectangles.html#method.insert
[`image::Image2DRef`]: https://docs.rs/jkl/0.2.1/jkl/image/struct.Image2DRef.html
[`image::Image2DMut`]: https://docs.rs/jkl/0.2.1/jkl/image/struct.Image2DMut.html
[`math::Vec3`]: https://docs.rs/jkl/0.2.1/jkl/math/struct.Vec3.html
[`math::Rgb565`]: https://docs.rs/jkl/0.2.1/jkl/math/struct.Rgb565.html
[`math::Rgba32F`]: https://docs.rs/jkl/0.2.1/jkl/math/struct.Rgba32F.html
[i2dr]: https://docs.rs/jkl/0.2.1/jkl/image/struct.Image2DRef.html
[i2dm]: https://docs.rs/jkl/0.2.1/jkl/image/struct.Image2DMut.html
[v2]: https://docs.rs/jkl/0.2.1/jkl/math/struct.Vec2.html
[v4]: https://docs.rs/jkl/0.2.1/jkl/math/struct.Vec4.html
[rect]: https://docs.rs/jkl/0.2.1/jkl/math/struct.Rect.html
[fc]: https://docs.rs/jkl/0.2.1/jkl/encode/trait.FixedCode.html
[vc]: https://docs.rs/jkl/0.2.1/jkl/encode/trait.VarCode.html

## License

See [Cargo.toml](Cargo.toml) for license information.
