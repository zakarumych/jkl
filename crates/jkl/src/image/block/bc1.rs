//! BC1 (DXT1) block texture compression.
//!
//! BC1 compresses RGB indices into 8-byte blocks. Each block stores two [`Rgb565`]
//! endpoint colors and a 4×4 grid of 2-bit indices into a 4-entry interpolated palette.

use std::{convert::Infallible, mem::swap};

use crate::{
    cluster_fit::cluster_fit,
    math::{Rgb32F, Rgb565, Rgba32F, Vec3, Yiq32F},
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

        if color0 == color1 {
            return Block {
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
            color0,
            color1,
            indices,
        }
    }

    /// Encode block into BC1 setting indices to TRANSPARENT if alpha <= threshold.
    pub fn encode_with_alpha(colors: [[Rgba32F; 4]; 4], threshold: f32) -> Self {
        #![allow(clippy::needless_range_loop)]
        let mut samples = [Vec3::ZERO; 16];

        let mut num_samples = 0;

        for row in &colors {
            for &c in row {
                if c.a() <= threshold {
                    continue;
                }

                samples[num_samples] = c.rgb().into();
                num_samples += 1;
            }
        }

        match num_samples {
            0 => Self::TRANSPARENT,
            1..16 => {
                // Some samples are transparent.
                let mut cf = cluster_fit::<Vec3, 3, 16>(
                    &samples[..num_samples],
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

                if color0.bits() > color1.bits() {
                    swap(&mut color0, &mut color1);
                    for index in &mut cf.indices {
                        *index = 2 - *index;
                    }
                }

                let mut indices = [0; 4];
                for y in 0..4 {
                    for x in 0..4 {
                        let c = colors[y][x];
                        if c.a() < threshold {
                            indices[y] |= 0b11 << (x * 2);
                        } else {
                            let idx = match cf.indices[y * 4 + x] {
                                0 => 0,
                                1 => 2,
                                2 => 1,
                                _ => unreachable!(),
                            };
                            indices[y] |= idx << (x * 2);
                        }
                    }
                }

                Block {
                    color0,
                    color1,
                    indices,
                }
            }
            16 => {
                // Solid case.
                let mut cf = cluster_fit::<Vec3, 4, 16>(
                    &samples[..num_samples],
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

                if color0 == color1 {
                    return Block {
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
                    color0,
                    color1,
                    indices,
                }
            }
            _ => unreachable!(),
        }
    }
}
