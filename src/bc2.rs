//! BC2 implementation.
//!

use crate::{
    cluster_fit::cluster_fit,
    math::{Rgb32F, Rgb565, Rgba32F, Vec3, Yiq32F},
};

/// A block of 4x4 texels compressed with BC2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Block {
    pub alpha: [u8; 8],
    pub color0: Rgb565,
    pub color1: Rgb565,
    pub texels: [u8; 4],
}

impl Block {
    pub const BLACK: Block = Block {
        alpha: [0xFF; 8],
        color0: Rgb565::WHITE,
        color1: Rgb565::BLACK,
        texels: [0xFF; 4],
    };

    pub const WHITE: Block = Block {
        alpha: [0xFF; 8],
        color0: Rgb565::WHITE,
        color1: Rgb565::BLACK,
        texels: [0x00; 4],
    };

    pub const TRANSPARENT: Block = Block {
        alpha: [0x00; 8],
        color0: Rgb565::BLACK,
        color1: Rgb565::BLACK,
        texels: [0xFF; 4],
    };

    pub fn bytes(&self) -> [u8; 16] {
        let alpha = self.alpha;
        let color0 = self.color0.bytes();
        let color1 = self.color1.bytes();
        let texels = self.texels;

        [
            alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5], alpha[6], alpha[7],
            color0[0], color0[1], color1[0], color1[1], texels[0], texels[1], texels[2], texels[3],
        ]
    }

    pub fn from_bytes(bytes: [u8; 16]) -> Block {
        let alpha = [
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ];

        let color0 = Rgb565::from_bytes([bytes[8], bytes[9]]);
        let color1 = Rgb565::from_bytes([bytes[10], bytes[11]]);
        let texels = [bytes[12], bytes[13], bytes[14], bytes[15]];

        Block {
            alpha,
            color0,
            color1,
            texels,
        }
    }

    /// Decodes single BC2 block.
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

    /// Decodes single BC2 block.
    pub fn decode_with_alpha(self) -> [[Rgba32F; 4]; 4] {
        // Decode endpoints.
        let color0 = self.color0.into_f32();
        let color1 = self.color1.into_f32();

        // Prepare local variables.
        let mut colors = [[Rgba32F::TRANSPARENT; 4]; 4];
        let texels = self.texels;

        // Check mode and build palette.
        let palette = [
            color0,
            Rgb32F::lerp(color0, color1, 1.0 / 3.0),
            Rgb32F::lerp(color0, color1, 2.0 / 3.0),
            color1,
        ];

        // Decode texels.
        for i in 0..4 {
            for j in 0..4 {
                let index = (texels[i] >> 2 * j) & 0b11;
                let alpha = (self.alpha[i * 2 + j / 2] >> 4 * (j % 2)) & 0b1111;
                let alpha = alpha as f32 / 15.0;

                colors[i][j] = palette[index as usize].with_alpha(alpha);
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
            alpha: [0xFF; 8],
            color0: Rgb565::from_f32(Rgb32F::from(color0)),
            color1: Rgb565::from_f32(Rgb32F::from(color1)),
            texels,
        }
    }

    /// Encode block into BC2 with alpha.
    pub fn encode_with_alpha(colors: [[Rgba32F; 4]; 4]) -> Self {
        let mut samples = [Vec3::ZERO; 16];

        for i in 0..4 {
            for j in 0..4 {
                samples[i * 4 + j] = colors[i][j].rgb().into();
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

        let mut alpha = [0; 8];
        for i in 0..4 {
            for j in 0..4 {
                let a = (colors[i][j].a() * 15.0).round() as u8;
                alpha[i * 2 + j / 2] |= (a & 0b1111) << (4 * (j % 2));
            }
        }

        Block {
            alpha,
            color0: Rgb565::from_f32(Rgb32F::from(color0)),
            color1: Rgb565::from_f32(Rgb32F::from(color1)),
            texels,
        }
    }
}
