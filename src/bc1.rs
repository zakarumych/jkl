//! BC1 implementation.
//!

use crate::{
    cluster_fit::cluster_fit,
    math::{Rgb32F, Rgb565, Rgba32F, Vec3, Yiq32F},
};

/// A block of 4x4 texels compressed with BC1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Block {
    pub color0: Rgb565,
    pub color1: Rgb565,
    pub texels: [u8; 4],
}

impl Block {
    pub const BLACK: Block = Block {
        color0: Rgb565::BLACK,
        color1: Rgb565::BLACK,
        texels: [0; 4],
    };

    pub const WHITE: Block = Block {
        color0: Rgb565::WHITE,
        color1: Rgb565::BLACK,
        texels: [0xFF; 4],
    };

    pub const TRANSPARENT: Block = Block {
        color0: Rgb565::BLACK,
        color1: Rgb565::BLACK,
        texels: [0xFF; 4],
    };

    pub fn bytes(&self) -> [u8; 8] {
        let color0 = self.color0.bytes();
        let color1 = self.color1.bytes();
        let texels = self.texels;

        [
            color0[0], color0[1], color1[0], color1[1], texels[0], texels[1], texels[2], texels[3],
        ]
    }

    pub fn from_bytes(bytes: [u8; 8]) -> Block {
        let color0 = Rgb565::from_bytes([bytes[0], bytes[1]]);
        let color1 = Rgb565::from_bytes([bytes[2], bytes[3]]);
        let texels = [bytes[4], bytes[5], bytes[6], bytes[7]];

        Block {
            color0,
            color1,
            texels,
        }
    }

    /// Decodes single BC1 block.
    pub fn decode(self) -> [[Rgb32F; 4]; 4] {
        // Decode endpoints.
        let color0 = self.color0.into_f32();
        let color1 = self.color1.into_f32();

        // Prepare local variables.
        let mut colors = [[Rgb32F::BLACK; 4]; 4];
        let texels = self.texels;

        // Check mode and build palette.
        let palette = if self.color0.bits() > self.color1.bits() {
            // Interpolate two intermediate colors.
            [
                color0,
                Rgb32F::lerp(color0, color1, 1.0 / 3.0),
                Rgb32F::lerp(color0, color1, 2.0 / 3.0),
                color1,
            ]
        } else {
            // Interpolate one intermediate color.
            [
                color0,
                Rgb32F::lerp(color0, color1, 1.0 / 2.0),
                color1,
                Rgb32F::BLACK,
            ]
        };

        // Decode texels.
        for i in 0..4 {
            for j in 0..4 {
                let index = (texels[i] >> 2 * j) & 0b11;

                colors[i][j] = palette[index as usize];
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
        let texels = self.texels;

        // Check mode and build palette.
        let palette = if self.color0.bits() > self.color1.bits() {
            // Interpolate two intermediate colors.
            [
                color0.with_alpha(1.0),
                Rgb32F::lerp(color0, color1, 1.0 / 3.0).with_alpha(1.0),
                Rgb32F::lerp(color0, color1, 2.0 / 3.0).with_alpha(1.0),
                color1.with_alpha(1.0),
            ]
        } else {
            // Interpolate one intermediate color.
            [
                color0.with_alpha(1.0),
                Rgb32F::lerp(color0, color1, 1.0 / 2.0).with_alpha(1.0),
                color1.with_alpha(1.0),
                Rgba32F::TRANSPARENT,
            ]
        };

        // Decode texels.
        for i in 0..4 {
            for j in 0..4 {
                let index = (texels[i] >> 2 * j) & 0b11;

                colors[i][j] = palette[index as usize];
            }
        }

        colors
    }

    pub fn encode(colors: [[Rgb32F; 4]; 4]) -> Self {
        let mut samples = [Vec3::ZERO; 16];

        for i in 0..4 {
            for j in 0..4 {
                samples[i * 4 + j] = colors[i][j].into();
            }
        }

        let cf = cluster_fit::<Vec3, 4, 16>(
            &samples,
            |a: Vec3, b: Vec3| {
                let mut a = Rgb565::from_f32(a.into());
                let mut b = Rgb565::from_f32(b.into());

                if a == b {
                    b = Rgb565::from_bits(!a.bits());
                }
                if a.bits() < b.bits() {
                    core::mem::swap(&mut a, &mut b);
                }

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
        let mut texels = [0; 4];
        for i in 0..4 {
            for j in 0..4 {
                let idx = (cf.indices[i * 4 + j] as u8) & 0b11;
                texels[i] |= idx << (j * 2);
            }
        }

        Block {
            color0: Rgb32F::from(color0).into(),
            color1: Rgb32F::from(color1).into(),
            texels,
        }
    }

    /// Encode block into BC1 setting texels to TRANSPARENT if alpha <= threshold.
    pub fn encode_with_alpha(colors: [[Rgba32F; 4]; 4], threshold: f32) -> Self {
        let mut samples = [Vec3::ZERO; 16];

        let mut num_samples = 0;

        for i in 0..4 {
            for j in 0..4 {
                let c = colors[i][j];

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
                let cf = cluster_fit::<Vec3, 3, 16>(
                    &samples[..num_samples],
                    |a: Vec3, b: Vec3| {
                        let mut a = Rgb565::from_f32(a.into());
                        let mut b = Rgb565::from_f32(b.into());

                        if a == b {
                            b = Rgb565::from_bits(!a.bits());
                        }
                        if a.bits() > b.bits() {
                            core::mem::swap(&mut a, &mut b);
                        }

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
                let mut texels = [0; 4];
                for i in 0..4 {
                    for j in 0..4 {
                        let c = colors[i][j];
                        if c.a() < threshold {
                            texels[i] |= 0b11 << (j * 2);
                        } else {
                            let idx = (cf.indices[i * 4 + j] as u8) & 0b11;
                            texels[i] |= idx << (j * 2);
                        }
                    }
                }

                Block {
                    color0: Rgb32F::from(color0).into(),
                    color1: Rgb32F::from(color1).into(),
                    texels,
                }
            }
            16 => {
                let cf = cluster_fit::<Vec3, 4, 16>(
                    &samples[..num_samples],
                    |a: Vec3, b: Vec3| {
                        let mut a = Rgb565::from_f32(a.into());
                        let mut b = Rgb565::from_f32(b.into());

                        if a == b {
                            b = Rgb565::from_bits(!a.bits());
                        }
                        if a.bits() < b.bits() {
                            core::mem::swap(&mut a, &mut b);
                        }

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
                let mut texels = [0; 4];
                for i in 0..4 {
                    for j in 0..4 {
                        let idx = (cf.indices[i * 4 + j] as u8) & 0b11;
                        texels[i] |= idx << (j * 2);
                    }
                }

                Block {
                    color0: Rgb32F::from(color0).into(),
                    color1: Rgb32F::from(color1).into(),
                    texels,
                }
            }
            _ => unreachable!(),
        }
    }
}
