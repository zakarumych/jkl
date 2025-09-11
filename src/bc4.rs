//! BC4 implementation.
//!

use crate::{
    cluster_fit::cluster_fit,
    math::{R32F, R8U},
};

/// A block of 4x4 texels compressed with BC4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Block {
    pub color0: R8U,
    pub color1: R8U,
    pub texels: [u8; 6],
}

impl Block {
    pub const BLACK: Block = Block {
        color0: R8U::WHITE,
        color1: R8U::BLACK,
        texels: [0xFF; 6],
    };

    pub const WHITE: Block = Block {
        color0: R8U::WHITE,
        color1: R8U::BLACK,
        texels: [0x00; 6],
    };

    pub fn bytes(&self) -> [u8; 8] {
        let color0 = self.color0.bits();
        let color1 = self.color1.bits();
        let texels = self.texels;

        [
            color0, color1, texels[0], texels[1], texels[2], texels[3], texels[4], texels[5],
        ]
    }

    pub fn from_bytes(bytes: [u8; 8]) -> Block {
        let color0 = R8U::new(bytes[0]);
        let color1 = R8U::new(bytes[1]);
        let texels = [bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]];

        Block {
            color0,
            color1,
            texels,
        }
    }

    /// Decodes single BC4 block.
    pub fn decode(self) -> [[R32F; 4]; 4] {
        // Decode endpoints.
        let color0 = self.color0.into_f32();
        let color1 = self.color1.into_f32();

        // Prepare local variables.
        let mut colors = [[R32F::BLACK; 4]; 4];
        let texels = self.texels;

        // Check mode and build palette.
        let palette = if self.color0.bits() > self.color1.bits() {
            [
                color0,
                R32F::lerp(color0, color1, 1.0 / 7.0),
                R32F::lerp(color0, color1, 2.0 / 7.0),
                R32F::lerp(color0, color1, 3.0 / 7.0),
                R32F::lerp(color0, color1, 4.0 / 7.0),
                R32F::lerp(color0, color1, 5.0 / 7.0),
                R32F::lerp(color0, color1, 6.0 / 7.0),
                color1,
            ]
        } else {
            [
                color0,
                R32F::lerp(color0, color1, 1.0 / 5.0),
                R32F::lerp(color0, color1, 2.0 / 5.0),
                R32F::lerp(color0, color1, 3.0 / 5.0),
                R32F::lerp(color0, color1, 4.0 / 5.0),
                color1,
                R32F::BLACK,
                R32F::WHITE,
            ]
        };

        // Decode texels.
        for i in 0..4 {
            for j in 0..4 {
                let start_bit = (i * 4 + j) * 3;
                let start_byte = start_bit / 8;

                let mut index = (texels[start_byte] >> (start_bit & 7)) & 0b111;
                if start_bit & 7 > 5 {
                    index |= (texels[start_byte + 1] << (8 - (start_bit & 7))) & 0b111;
                }

                colors[i][j] = palette[index as usize];
            }
        }

        colors
    }

    pub fn encode(colors: [[R32F; 4]; 4]) -> Self {
        let mut samples = [0.0; 16];

        for i in 0..4 {
            for j in 0..4 {
                samples[i * 4 + j] = colors[i][j].r();
            }
        }

        let cf = cluster_fit::<f32, 8, 16>(
            &samples,
            |a: f32, b: f32| {
                let mut a = R8U::from_f32(R32F::new(a));
                let mut b = R8U::from_f32(R32F::new(b));

                if a == b {
                    b = R8U::from_bits(!a.bits());
                }
                if a.bits() < b.bits() {
                    core::mem::swap(&mut a, &mut b);
                }

                (a.into_f32().r(), b.into_f32().r())
            },
            |a: f32, b: f32| {
                let a = R32F::new(a);
                let b = R32F::new(b);

                R32F::distance(a, b)
            },
        );

        let (color0, color1) = cf.endpoints;
        let mut texels = [0; 6];
        for i in 0..4 {
            for j in 0..4 {
                let idx = (cf.indices[i * 4 + j] as u8) & 0b111;

                let start_bit = (i * 4 + j) * 3;
                let start_byte = start_bit / 8;

                texels[start_byte] |= idx << (start_bit & 7);
                if start_bit & 7 > 5 {
                    texels[start_byte + 1] |= idx >> (8 - (start_bit & 7));
                }
            }
        }

        Block {
            color0: R8U::from_f32(R32F::new(color0)),
            color1: R8U::from_f32(R32F::new(color1)),
            texels,
        }
    }
}
