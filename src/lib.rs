//! # JKL - blazingly fast compression and image processing toolkit
//!
//! JKL is a batteries-included Rust crate for compressing, encoding and
//! packing data that needs to travel fast and decompress even faster - on the
//! CPU *or* directly on the GPU. If you ship textures, sprite atlases, tile
//! maps, or any bulk binary payload that must be ready the instant it hits
//! VRAM, JKL was built for you.
//!
//! # Feature highlights
//!
//! * **GPU-oriented block texture codecs** - encode and decode BC1 through BC5
//!   blocks with cluster-fit quality ([`bc1`], [`bc2`], [`bc3`], [`bc4`], [`bc5`], powered by [`cluster_fit`]).
//! * **Layered entropy coding** - stack [`lz77`] or [`rle`] with [`ans`] for
//!   near-optimal compression ratios while keeping decompression blazingly fast
//!   - simple enough for a GPU compute shader.
//! * **Bit-perfect I/O** - the [`bits`] module reads and writes individual bits
//!   so every encoder can squeeze out every last fraction of a byte.
//! * **Jackal Image format** - the [`jackal::image`] module ties everything together
//!   into a tiled, mip-mapped, GPU-ready image container.
//! * **Packing and spatial indexing** - [`max_rects`] bin-packs rectangles for
//!   atlas generation; [`z_curve`] provides Morton-order iteration for
//!   cache-friendly traversal.
//!
//! # Quick orientation - "how do I …?"
//!
//! | I want to …                              | Start here                                                                              |
//! |-------------------------------------------|-----------------------------------------------------------------------------------------|
//! | Compress a texture to BC1–BC5             | [`bc1::Block::encode`], [`bc2::Block::encode`], [`bc3::Block::encode`], [`bc4::Block::encode`], [`bc5::Block::encode`]                             |
//! | Decompress a BC block                     | [`bc1::Block::decode`], [`bc2::Block::decode`], [`bc3::Block::decode`], [`bc4::Block::decode`], [`bc5::Block::decode`]                                      |
//! | Write / read a complete Jackal Image file | [`jackal::image::write_image`], [`jackal::image::JackalReader`]                         |
//! | Entropy-code a symbol stream              | Build an [`ans::Context`], then use [`ans::Encoder`] / [`ans::Decoder`]                 |
//! | Run-length encode an iterator             | [`rle::rle`], [`rle::rle_power_of_two`], or [`rle::rle_with_cfg`]                       |
//! | LZ77-compress a value stream              | [`lz77::Encoder`] → [`lz77::Token`] stream → [`lz77::Decoder`]                         |
//! | LZ78-compress a value stream              | [`lz78::Encoder`] → [`lz78::Token`] stream → [`lz78::Decoder`]                         |
//! | Encode small unsigned ints compactly      | [`vle::encode`] / [`vle::decode`] (Elias delta), or wrap with [`vle::Vle`]              |
//! | Map signed ints for better compression    | [`zigzaq::ZigZag::zigzag`] before VLE encoding                                         |
//! | Read / write individual bits              | [`bits::WriteBits`], [`bits::ReadBits`], or the [`bits::write_bits_scope`] helper       |
//! | Bin-pack rectangles into an atlas         | [`max_rects::MaximalRectangles::new`] then [`insert`](max_rects::MaximalRectangles::insert) in a loop |
//! | Work with 2D pixel buffers                | [`image::Image2DRef`] / [`image::Image2DMut`]                                           |
//! | Use vector math or pixel types            | [`math::Vec3`], [`math::Rgb565`], [`math::Rgba32F`], etc.                               |
//!
//! # Module overview
//!
//! ## Compression
//!
//! | Module          | Purpose                                                                |
//! |-----------------|------------------------------------------------------------------------|
//! | [`ans`]         | Asymmetric Numeral Systems (ANS) entropy coder                         |
//! | [`rle`]         | Run-length encoding with configurable max length and power-of-two mode |
//! | [`vle`]         | Variable-length Elias delta coding for unsigned integers               |
//! | [`lz77`]        | LZ77 sliding-window compressor / decompressor                          |
//! | [`lz78`]        | LZ78 dictionary compressor (GPU-friendly variant)                      |
//! | [`zigzaq`]      | Zigzag signed ↔ unsigned mapping for better VLE compression            |
//!
//! ## Block-texture codecs
//!
//! | Module          | Format                     | Block size | Channels        |
//! |-----------------|----------------------------|------------|-----------------|
//! | [`bc1`]         | BC1 / DXT1                 | 8 bytes    | RGB + 1-bit A   |
//! | [`bc2`]         | BC2 / DXT3                 | 16 bytes   | RGBA (4-bit A)  |
//! | [`bc3`]         | BC3 / DXT5                 | 16 bytes   | RGBA (interp A) |
//! | [`bc4`]         | BC4 / RGTC1                | 8 bytes    | R               |
//! | [`bc5`]         | BC5 / RGTC2                | 16 bytes   | RG              |
//! | [`cluster_fit`] | Shared quantizer for above | -          | -               |
//!
//! ## Image and math
//!
//! | Module      | Purpose                                                                 |
//! |-------------|-------------------------------------------------------------------------|
//! | [`image`]   | Non-owning 2D / 3D image views (`Image2DRef`, `Image2DMut`, …)          |
//! | [`math`]    | Vectors (`Vec2`–`Vec4`), pixel types, `Rect`, PCA, bit-interleaving     |
//!
//! ## I/O and serialization
//!
//! | Module      | Purpose                                                                 |
//! |-------------|-------------------------------------------------------------------------|
//! | [`bits`]    | Bit-level buffered reader / writer over byte streams                    |
//! | [`encode`]  | `FixedCode` and `VarCode` serialization traits                          |
//!
//! ## Packing and spatial
//!
//! | Module        | Purpose                                                               |
//! |---------------|-----------------------------------------------------------------------|
//! | [`max_rects`] | 2D bin packing (maximal-rectangles algorithm) for atlas generation    |
//! | [`z_curve`]   | Z-order / Morton curve coordinate iteration                           |
//!
//! ## File formats
//!
//! | Module             | Purpose                                                     |
//! |--------------------|-------------------------------------------------------------|
//! | [`jackal::image`]  | Tiled, GPU-decompressible image container format            |

macro_rules! impl_fixedcode_struct {
    ($name:ident { $($field_name:ident: $field_ty:ty),* $(,)? } | $error:ty) => {
        impl $crate::encode::FixedCode for $name {
            const SIZE: usize = 0 $(+ <$field_ty as $crate::encode::FixedCode>::SIZE)*;
            type Array = [u8; Self::SIZE];
            type Error = $error;

            fn fix_encode(&self) -> [u8; Self::SIZE] {
                #![allow(unused_assignments)]

                let mut output = [0u8; Self::SIZE];
                let mut offset = 0;
                $(
                    output[offset..offset + <$field_ty as $crate::encode::FixedCode>::SIZE].copy_from_slice(&self.$field_name.fix_encode());
                    offset += <$field_ty as $crate::encode::FixedCode>::SIZE;
                )*

                output
            }

            fn fix_decode(input: &[u8; Self::SIZE]) -> Result<Self, $error> {
                #![allow(unused_assignments)]

                let mut offset = 0;
                Ok($name {
                    $(
                        $field_name: {
                            let value = <$field_ty as $crate::encode::FixedCode>::fix_decode(input[offset..offset + <$field_ty as $crate::encode::FixedCode>::SIZE].as_array().unwrap())?;
                            offset += <$field_ty as $crate::encode::FixedCode>::SIZE;
                            value
                        },
                    )*
                })
            }
        }
    };

    ($name:ident ( $($field_name:ident: $field_ty:ty),* $(,)? ) | $error:ty) => {
        impl $crate::encode::FixedCode for $name {
            const SIZE: usize = 0 $(+ <$field_ty as $crate::encode::FixedCode>::SIZE)*;
            type Array = [u8; Self::SIZE];
            type Error = $error;

            fn fix_encode(&self) -> [u8; Self::SIZE] {
                #![allow(unused_assignments)]

                let $name($($field_name,)*) = self;

                let mut output = [0u8; Self::SIZE];
                let mut offset = 0;
                $(
                    output[offset..offset + <$field_ty as $crate::encode::FixedCode>::SIZE].copy_from_slice(&$field_name.fix_encode());
                    offset += <$field_ty as $crate::encode::FixedCode>::SIZE;
                )*
                output
            }

            fn fix_decode(input: &[u8; Self::SIZE]) -> Result<Self, $error> {
                #![allow(unused_assignments)]

                let mut offset = 0;
                Ok($name(
                    $(
                        {
                            let $field_name = <$field_ty as $crate::encode::FixedCode>::fix_decode(input[offset..offset + <$field_ty as $crate::encode::FixedCode>::SIZE].as_array().unwrap())?;
                            offset += <$field_ty as $crate::encode::FixedCode>::SIZE;
                            $field_name
                        },
                    )*
                ))
            }
        }
    }
}

macro_rules! impl_fixedcode_array {
    ($name:ident([$e:ty; $n:literal]) | $error:ty) => {
        impl $crate::encode::FixedCode for $name {
            const SIZE: usize = <$e as $crate::encode::FixedCode>::SIZE * $n;
            type Array = [u8; Self::SIZE];
            type Error = $error;

            fn fix_encode(&self) -> Self::Array {
                let es = <$e as $crate::encode::FixedCode>::SIZE;
                let mut output = [0u8; Self::SIZE];
                for (i, item) in self.0.iter().enumerate() {
                    output[i * es..][..es].copy_from_slice(&item.fix_encode());
                }
                output
            }

            fn fix_decode(input: &Self::Array) -> Result<Self, Self::Error> {
                let es = <$e as $crate::encode::FixedCode>::SIZE;
                let mut result = [const { None }; $n];

                for (i, slot) in result.iter_mut().enumerate() {
                    match <$e as $crate::encode::FixedCode>::fix_decode(input[i * es..][..es].as_array().unwrap()) {
                        Ok(value) => {
                            *slot = Some(value)
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }

                Ok($name(result.map(|slot| slot.unwrap())))
            }
        }
    };
}

macro_rules! for_tuple {
    ($macro:ident) => {
        for_tuple!($macro for A B C D E F G H I J K L M N O P);
    };
    ($macro:ident for ) => {
        $macro!();
    };
    ($macro:ident for $head:ident $($tail:ident)*) => {
        for_tuple!($macro for $($tail)*);
        $macro!($head $($tail)*);
    };
}

macro_rules! impl_fixedcode_zero {
    ($name:ty) => {
        impl $crate::encode::FixedCode for $name {
            const SIZE: usize = 0;
            type Array = [u8; 0];
            type Error = std::convert::Infallible;

            fn fix_encode(&self) -> Self::Array {
                []
            }

            fn fix_decode(_: &Self::Array) -> Result<Self, Self::Error> {
                Ok(Self)
            }
        }
    };
}

pub mod ans;
pub mod bc1;
pub mod bc2;
pub mod bc3;
pub mod bc4;
pub mod bc5;
pub mod bits;
pub mod cluster_fit;
pub mod encode;
pub mod image;
pub mod jackal;
pub mod lz77;
pub mod lz78;
pub mod math;
pub mod max_rects;
// pub mod reference_map;
pub mod rle;
pub mod vle;
pub mod z_curve;
pub mod zigzaq;
