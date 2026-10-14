//! BC1 (DXT1) block texture compression.
//!
//! BC1 compresses RGB indices into 8-byte blocks. Each block stores two [`Rgb565`]
//! endpoint colors and a 4×4 grid of 2-bit indices into a 4-entry interpolated palette.

use std::convert::Infallible;

use crate::{
    image::{ImageMut, ImageRef},
    math::{Rgb32F, Rgb565, Rgba32F},
};

/// A block of 4x4 indices compressed with BC1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Block {
    pub color0: Rgb565,
    pub color1: Rgb565,
    pub indices: [u8; 4],
}

impl_fixedcode_struct!(
    Block {
        color0: Rgb565,
        color1: Rgb565,
        indices: [u8; 4],
    } | Infallible
);

impl Block {
    pub const BLACK: Block = Block {
        color0: Rgb565::BLACK,
        color1: Rgb565::BLACK,
        indices: [0x00; 4],
    };

    pub const WHITE: Block = Block {
        color0: Rgb565::WHITE,
        color1: Rgb565::WHITE,
        indices: [0x00; 4],
    };

    pub const TRANSPARENT: Block = Block {
        color0: Rgb565::BLACK,
        color1: Rgb565::BLACK,
        indices: [0xFF; 4],
    };

    /// Returns the raw 8-byte representation of this block.
    pub fn bytes(&self) -> [u8; 8] {
        let color0 = self.color0.bytes();
        let color1 = self.color1.bytes();
        let indices = self.indices;

        [
            color0[0], color0[1], color1[0], color1[1], indices[0], indices[1], indices[2],
            indices[3],
        ]
    }

    /// Constructs a `Block` from its raw 8-byte representation.
    pub fn from_bytes(bytes: [u8; 8]) -> Block {
        let color0 = Rgb565::from_bytes([bytes[0], bytes[1]]);
        let color1 = Rgb565::from_bytes([bytes[2], bytes[3]]);
        let indices = [bytes[4], bytes[5], bytes[6], bytes[7]];

        Block {
            color0,
            color1,
            indices,
        }
    }

    /// Decodes single BC1 block.
    pub fn decode(self) -> [[Rgb32F; 4]; 4] {
        // Decode endpoints.
        let color0 = self.color0.into_f32();
        let color1 = self.color1.into_f32();

        // Prepare local variables.
        let mut colors = [[Rgb32F::BLACK; 4]; 4];
        let indices = self.indices;

        // Check mode and build palette.
        let palette = if self.color0.bits() > self.color1.bits() {
            // Interpolate two intermediate colors.
            [
                color0,
                color1,
                Rgb32F::lerp(color0, color1, 1.0 / 3.0),
                Rgb32F::lerp(color0, color1, 2.0 / 3.0),
            ]
        } else {
            // Interpolate one intermediate color.
            [
                color0,
                color1,
                Rgb32F::lerp(color0, color1, 1.0 / 2.0),
                Rgb32F::BLACK,
            ]
        };

        // Decode indices.
        for y in 0..4 {
            for x in 0..4 {
                let index = (indices[y] >> (2 * x)) & 0b11;

                colors[y][x] = palette[index as usize];
            }
        }

        colors
    }

    /// Decodes single BC1 block.
    pub fn decode_with_alpha(self) -> [[Rgba32F; 4]; 4] {
        // Decode endpoints.
        let color0 = self.color0.into_f32();
        let color1 = self.color1.into_f32();

        // Prepare local variables.
        let mut colors = [[Rgba32F::TRANSPARENT; 4]; 4];
        let indices = self.indices;

        // Check mode and build palette.
        let palette = if self.color0.bits() > self.color1.bits() {
            // Interpolate two intermediate colors.
            [
                color0.with_alpha(1.0),
                color1.with_alpha(1.0),
                Rgb32F::lerp(color0, color1, 1.0 / 3.0).with_alpha(1.0),
                Rgb32F::lerp(color0, color1, 2.0 / 3.0).with_alpha(1.0),
            ]
        } else {
            // Interpolate one intermediate color.
            [
                color0.with_alpha(1.0),
                color1.with_alpha(1.0),
                Rgb32F::lerp(color0, color1, 1.0 / 2.0).with_alpha(1.0),
                Rgba32F::TRANSPARENT,
            ]
        };

        // Decode indices.
        for y in 0..4 {
            for x in 0..4 {
                let index = (indices[y] >> (2 * x)) & 0b11;

                colors[y][x] = palette[index as usize];
            }
        }

        colors
    }

    /// Encodes a 4×4 grid of RGB colors into a BC1 block.
    pub fn encode(colors: [[(Rgb32F, bool); 4]; 4]) -> Self {
        super::fast_fit::rgb(colors, false)
    }

    /// Encode block into BC1 setting indices to TRANSPARENT if alpha < threshold.
    pub fn encode_with_alpha(colors: [[(Rgba32F, bool); 4]; 4], threshold: f32) -> Self {
        let mut opaque = [[(Rgb32F::BLACK, false); 4]; 4];
        let mut transparent = false;
        let mut count = 0;
        for y in 0..4 {
            for x in 0..4 {
                let (c, valid) = colors[y][x];
                if valid && c.a() >= threshold {
                    opaque[y][x] = (c.rgb(), true);
                    count += 1;
                } else if valid {
                    transparent = true;
                }
            }
        }
        if count == 0 {
            return if transparent {
                Self::TRANSPARENT
            } else {
                Self::BLACK
            };
        }
        let mut block = super::fast_fit::rgb(opaque, transparent);
        if transparent {
            for y in 0..4 {
                for x in 0..4 {
                    if !opaque[y][x].1 {
                        block.indices[y] |= 3 << (x * 2);
                    }
                }
            }
        }
        block
    }
}

pub fn encode_image<T>(
    input: ImageRef<'_, T>,
    map: impl Fn(T) -> Rgb32F,
    mut output: ImageMut<'_, Block>,
) where
    T: Copy,
{
    assert_eq!(output.width(), input.width().div_ceil(4));
    assert_eq!(output.height(), input.height().div_ceil(4));
    assert_eq!(output.depth(), input.depth());
    assert_eq!(output.layers(), input.layers());

    let input = input.reinterpret_as_3d();
    let mut output = output.as_mut_3d();

    for z in 0..output.depth() {
        for y in 0..output.height() {
            for x in 0..output.width() {
                let mut block_colors = [[(Rgb32F::BLACK, false); 4]; 4];

                for by in 0..4 {
                    for bx in 0..4 {
                        if bx >= input.width() - x * 4 || by >= input.height() - y * 4 {
                            continue;
                        }
                        let c = map(*input.get_pixel(x * 4 + bx, y * 4 + by, z));
                        block_colors[by][bx] = (c, true);
                    }
                }

                let block = Block::encode(block_colors);
                output.set(x, y, z, block);
            }
        }
    }
}

pub fn encode_image_with_alpha<T>(
    input: ImageRef<'_, T>,
    map: impl Fn(T) -> Rgba32F,
    threshold: f32,
    mut output: ImageMut<'_, Block>,
) where
    T: Copy,
{
    assert_eq!(output.width(), input.width().div_ceil(4));
    assert_eq!(output.height(), input.height().div_ceil(4));
    assert_eq!(output.depth(), input.depth());
    assert_eq!(output.layers(), input.layers());

    let input = input.reinterpret_as_3d();
    let mut output = output.as_mut_3d();

    for z in 0..output.depth() {
        for y in 0..output.height() {
            for x in 0..output.width() {
                let mut block_colors = [[(Rgba32F::BLACK, false); 4]; 4];

                for by in 0..4 {
                    for bx in 0..4 {
                        if bx >= input.width() - x * 4 || by >= input.height() - y * 4 {
                            continue;
                        }
                        let c = map(*input.get_pixel(x * 4 + bx, y * 4 + by, z));
                        block_colors[by][bx] = (c, true);
                    }
                }

                let block = Block::encode_with_alpha(block_colors, threshold);
                output.set(x, y, z, block);
            }
        }
    }
}

pub fn decode_image<T>(
    input: ImageRef<'_, Block>,
    map: impl Fn(Rgba32F) -> T,
    mut output: ImageMut<'_, T>,
) where
    T: Copy,
{
    assert_eq!(output.width().div_ceil(4), input.width());
    assert_eq!(output.height().div_ceil(4), input.height());
    assert_eq!(output.depth(), input.depth());
    assert_eq!(output.layers(), input.layers());

    let input = input.reinterpret_as_3d();
    let mut output = output.as_mut_3d();

    for z in 0..input.depth() {
        for y in 0..input.height() {
            for x in 0..input.width() {
                let block = input.get_pixel(x, y, z);
                let block_colors = Block::decode_with_alpha(*block);

                for by in 0..4 {
                    for bx in 0..4 {
                        if bx >= output.width() - x * 4 || by >= output.height() - y * 4 {
                            continue;
                        }
                        let c = map(block_colors[by][bx]);
                        output.set(x * 4 + bx, y * 4 + by, z, c);
                    }
                }
            }
        }
    }
}
