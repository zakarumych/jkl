//! BC2 (DXT3) block texture compression.
//!
//! BC2 compresses RGBA indices into 16-byte blocks. Alpha is stored as explicit
//! 4-bit values per texel, while the RGB portion uses the same encoding as [`bc1`](crate::bc1).

use std::{convert::Infallible, mem::swap};

use crate::{
    cluster_fit::cluster_fit,
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

    pub fn encode(colors: [[Rgb32F; 4]; 4]) -> Self {
        let mut samples = [Vec3::ZERO; 16];

        for y in 0..4 {
            for x in 0..4 {
                samples[y * 4 + x] = colors[y][x].into();
            }
        }

        let mut cf = cluster_fit::<Vec3, 4, 16>(
            &samples,
            |a: Vec3, b: Vec3| {
                let a = Rgb565::from_f32(a.into());
                let b = Rgb565::from_f32(b.into());

                (a.into_f32().into(), b.into_f32().into())
            },
            |a: Vec3, b: Vec3| {
                let a = Rgb32F::from(a);
                let b = Rgb32F::from(b);

                let a = Yiq32F::from_rgb(a);
                let b = Yiq32F::from_rgb(b);

                Yiq32F::perceptual_distance(a, b)
            },
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
        for y in 0..4 {
            for x in 0..4 {
                let idx = match cf.indices[y * 4 + x] {
                    0 => 0,
                    1 => 2,
                    2 => 3,
                    3 => 1,
                    _ => unreachable!(),
                };
                indices[y] |= idx << (x * 2);
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
    pub fn encode_with_alpha(colors: [[Rgba32F; 4]; 4]) -> Self {
        let mut samples = [Vec3::ZERO; 16];

        for y in 0..4 {
            for x in 0..4 {
                samples[y * 4 + x] = colors[y][x].rgb().into();
            }
        }

        let mut cf = cluster_fit::<Vec3, 4, 16>(
            &samples,
            |a: Vec3, b: Vec3| {
                let a = Rgb565::from_f32(a.into());
                let b = Rgb565::from_f32(b.into());

                (a.into_f32().into(), b.into_f32().into())
            },
            |a: Vec3, b: Vec3| {
                let a = Rgb32F::from(a);
                let b = Rgb32F::from(b);

                let a = Yiq32F::from_rgb(a);
                let b = Yiq32F::from_rgb(b);

                Yiq32F::perceptual_distance(a, b)
            },
        );

        let mut alpha = [0; 8];
        for y in 0..4 {
            for x in 0..4 {
                let a = (colors[y][x].a() * 15.0).round() as u8;
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
        for y in 0..4 {
            for x in 0..4 {
                let idx = match cf.indices[y * 4 + x] {
                    0 => 0,
                    1 => 2,
                    2 => 3,
                    3 => 1,
                    _ => unreachable!(),
                };
                indices[y] |= idx << (x * 2);
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
