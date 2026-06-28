//! BC2 (DXT3) block texture compression.
//!
//! BC2 compresses RGBA indices into 16-byte blocks. Alpha is stored as explicit
//! 4-bit values per texel, while the RGB portion uses the same encoding as [`bc1`](crate::bc1).

use std::{convert::Infallible, mem::swap};

use crate::{
    algos::cluster_fit::cluster_fit,
    image::{ImageMut, ImageRef},
    math::{Rgb32F, Rgb565, Rgba32F, Vec3, Yiq32F},
};

/// A block of 4x4 indices compressed with BC2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Block {
    pub alpha: [u8; 8],
    pub color0: Rgb565,
    pub color1: Rgb565,
    pub indices: [u8; 4],
}

impl_fixedcode_struct!(
    Block {
        alpha: [u8; 8],
        color0: Rgb565,
        color1: Rgb565,
        indices: [u8; 4],
    } | Infallible
);

#[allow(clippy::needless_range_loop)]
impl Block {
    pub const BLACK: Block = Block {
        alpha: [0xFF; 8],
        color0: Rgb565::WHITE,
        color1: Rgb565::BLACK,
        indices: [0xFF; 4],
    };

    pub const WHITE: Block = Block {
        alpha: [0xFF; 8],
        color0: Rgb565::WHITE,
        color1: Rgb565::BLACK,
        indices: [0x00; 4],
    };

    pub const TRANSPARENT: Block = Block {
        alpha: [0x00; 8],
        color0: Rgb565::BLACK,
        color1: Rgb565::BLACK,
        indices: [0xFF; 4],
    };

    pub fn bytes(&self) -> [u8; 16] {
        let alpha = self.alpha;
        let color0 = self.color0.bytes();
        let color1 = self.color1.bytes();
        let indices = self.indices;

        [
            alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5], alpha[6], alpha[7],
            color0[0], color0[1], color1[0], color1[1], indices[0], indices[1], indices[2],
            indices[3],
        ]
    }

    pub fn from_bytes(bytes: [u8; 16]) -> Block {
        let alpha = [
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ];

        let color0 = Rgb565::from_bytes([bytes[8], bytes[9]]);
        let color1 = Rgb565::from_bytes([bytes[10], bytes[11]]);
        let indices = [bytes[12], bytes[13], bytes[14], bytes[15]];

        Block {
            alpha,
            color0,
            color1,
            indices,
        }
    }

    /// Decodes single BC2 block.
    pub fn decode(self) -> [[Rgb32F; 4]; 4] {
        // Decode endpoints.
        let color0 = self.color0.into_f32();
        let color1 = self.color1.into_f32();

        let indices = self.indices;

        // Check mode and build palette.
        let palette = [
            color0,
            color1,
            Rgb32F::lerp(color0, color1, 1.0 / 3.0),
            Rgb32F::lerp(color0, color1, 2.0 / 3.0),
        ];

        let mut colors = [[Rgb32F::BLACK; 4]; 4];

        // Decode indices.
        for i in 0..4 {
            for j in 0..4 {
                let index = (indices[i] >> (2 * j)) & 0b11;
                colors[i][j] = palette[index as usize];
            }
        }

        colors
    }

    /// Decodes single BC2 block.
    pub fn decode_with_alpha(self) -> [[Rgba32F; 4]; 4] {
        // Decode endpoints.
        let color0 = self.color0.into_f32();
        let color1 = self.color1.into_f32();

        // Prepare local variables.
        let mut colors = [[Rgba32F::TRANSPARENT; 4]; 4];
        let indices = self.indices;

        // Check mode and build palette.
        let palette = [
            color0,
            color1,
            Rgb32F::lerp(color0, color1, 1.0 / 3.0),
            Rgb32F::lerp(color0, color1, 2.0 / 3.0),
        ];

        // Decode indices.
        for y in 0..4 {
            for x in 0..4 {
                let index = (indices[y] >> (2 * x)) & 0b11;
                let alpha = (self.alpha[y * 2 + x / 2] >> (4 * (x % 2))) & 0b1111;
                let alpha = alpha as f32 / 15.0;

                colors[y][x] = palette[index as usize].with_alpha(alpha);
            }
        }

        colors
    }

    pub fn encode(colors: [[(Rgb32F, bool); 4]; 4]) -> Self {
        let mut samples = [Vec3::ZERO; 16];

        let mut count = 0;

        for row in &colors {
            for &(c, v) in row {
                if v {
                    samples[count] = c.into();
                    count += 1;
                }
            }
        }

        assert_ne!(count, 0, "At least one sample must valid");

        let mut cf = cluster_fit::<Vec3, 4, 16>(
            &samples[..count],
            |a: Vec3| Rgb565::from_f32(a.into()).into_f32().into(),
            |a: Vec3, b: Vec3| Yiq32F::perceptual_distance(Yiq32F(a.0), Yiq32F(b.0)),
        );

        let (color0, color1) = cf.endpoints;

        let mut color0 = Rgb565::from_f32(Rgb32F::from(color0));
        let mut color1 = Rgb565::from_f32(Rgb32F::from(color1));

        // This is not really required for BC2,
        // but it's just more consistent and may reduce entropy in the output.
        if color0 == color1 {
            return Block {
                alpha: [0xFF; 8],
                color0,
                color1: Rgb565::BLACK,
                indices: [0x00; 4],
            };
        } else if color0.bits() < color1.bits() {
            swap(&mut color0, &mut color1);
            for index in &mut cf.indices {
                *index = 3 - *index;
            }
        }

        let mut indices = [0; 4];
        let mut index_index = 0;
        for y in 0..4 {
            for x in 0..4 {
                if colors[y][x].1 {
                    continue;
                }
                let idx = match cf.indices[index_index] {
                    0 => 0,
                    1 => 2,
                    2 => 3,
                    3 => 1,
                    _ => unreachable!(),
                };
                indices[y] |= idx << (x * 2);
                index_index += 1;
            }
        }

        Block {
            alpha: [0xFF; 8],
            color0,
            color1,
            indices,
        }
    }

    /// Encode block into BC2 with alpha.
    pub fn encode_with_alpha(colors: [[(Rgba32F, bool); 4]; 4]) -> Self {
        let mut samples = [Vec3::ZERO; 16];

        let mut count = 0;

        for row in &colors {
            for &(c, v) in row {
                if v {
                    samples[count] = c.rgb().into();
                    count += 1;
                }
            }
        }

        assert_ne!(count, 0, "At least one sample must valid");

        let mut cf = cluster_fit::<Vec3, 4, 16>(
            &samples[..count],
            |a: Vec3| Rgb565::from_f32(a.into()).into_f32().into(),
            |a: Vec3, b: Vec3| Yiq32F::perceptual_distance(Yiq32F(a.0), Yiq32F(b.0)),
        );

        let mut alpha = [0; 8];
        for y in 0..4 {
            for x in 0..4 {
                let a = (colors[y][x].0.a() * 15.0).round() as u8;
                alpha[y * 2 + x / 2] |= (a & 0b1111) << (4 * (x % 2));
            }
        }

        let (color0, color1) = cf.endpoints;

        let mut color0 = Rgb565::from_f32(Rgb32F::from(color0));
        let mut color1 = Rgb565::from_f32(Rgb32F::from(color1));

        // This is not really required for BC2,
        // but it's just more consistent and may reduce entropy in the output.
        if color0 == color1 {
            return Block {
                alpha,
                color0,
                color1: Rgb565::BLACK,
                indices: [0x00; 4],
            };
        } else if color0.bits() < color1.bits() {
            swap(&mut color0, &mut color1);
            for index in &mut cf.indices {
                *index = 3 - *index;
            }
        }

        let mut indices = [0; 4];
        let mut index_index = 0;
        for y in 0..4 {
            for x in 0..4 {
                if colors[y][x].1 {
                    continue;
                }
                let idx = match cf.indices[index_index] {
                    0 => 0,
                    1 => 2,
                    2 => 3,
                    3 => 1,
                    _ => unreachable!(),
                };
                indices[y] |= idx << (x * 2);
                index_index += 1;
            }
        }

        Block {
            alpha,
            color0,
            color1,
            indices,
        }
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

                let block = Block::encode_with_alpha(block_colors);
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
