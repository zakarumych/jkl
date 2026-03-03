//! BC3 implementation.
//!
//! BC3 compresses RGBA texels into 16-byte blocks. Each block stores alpha via a
//! [`bc4::Block`] and RGB via a [`bc1::Block`].

use std::convert::Infallible;

use crate::{
    bc1, bc4,
    math::{Rgb32F, Rgba32F, R32F},
};

/// A block of 4×4 texels compressed with BC3.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Block {
    pub alpha: bc4::Block,
    pub rgb: bc1::Block,
}

impl_fixedcode_struct!(
    Block {
        alpha: bc4::Block,
        rgb: bc1::Block,
    } | Infallible
);

impl Block {
    pub const BLACK: Block = Block {
        alpha: bc4::Block::WHITE,
        rgb: bc1::Block::BLACK,
    };

    pub const WHITE: Block = Block {
        alpha: bc4::Block::WHITE,
        rgb: bc1::Block::WHITE,
    };

    pub const TRANSPARENT: Block = Block {
        alpha: bc4::Block::BLACK,
        rgb: bc1::Block::BLACK,
    };

    /// Returns the raw 16-byte representation of this block.
    pub fn bytes(&self) -> [u8; 16] {
        let alpha = self.alpha.bytes();
        let rgb = self.rgb.bytes();

        [
            alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5], alpha[6], alpha[7], rgb[0],
            rgb[1], rgb[2], rgb[3], rgb[4], rgb[5], rgb[6], rgb[7],
        ]
    }

    /// Constructs a `Block` from its raw 16-byte representation.
    pub fn from_bytes(bytes: [u8; 16]) -> Block {
        let alpha = [
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ];
        let rgb = [
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ];

        Block {
            alpha: bc4::Block::from_bytes(alpha),
            rgb: bc1::Block::from_bytes(rgb),
        }
    }

    /// Decodes the RGB channels of this BC3 block, discarding alpha.
    pub fn decode(self) -> [[Rgb32F; 4]; 4] {
        self.rgb.decode()
    }

    /// Decodes the full RGBA block, combining BC4 alpha and BC1 RGB.
    pub fn decode_with_alpha(self) -> [[Rgba32F; 4]; 4] {
        let alpha = self.alpha.decode();
        let rgb = self.rgb.decode();

        let mut colors = [[Rgba32F::BLACK; 4]; 4];

        for x in 0..4 {
            for y in 0..4 {
                colors[y][x] = rgb[y][x].with_alpha(alpha[y][x].r());
            }
        }

        colors
    }

    /// Encodes a 4×4 grid of RGB colors into a BC3 block with full alpha.
    pub fn encode(colors: [[Rgb32F; 4]; 4]) -> Self {
        let rgb = bc1::Block::encode(colors);

        Block {
            alpha: bc4::Block::WHITE,
            rgb,
        }
    }

    /// Encodes a 4×4 grid of RGBA colors into a BC3 block with per-texel alpha.
    pub fn encode_with_alpha(colors: [[Rgba32F; 4]; 4]) -> Self {
        let alpha_texels = colors.map(|row| row.map(|c| R32F::new(c.a())));
        let alpha = bc4::Block::encode(alpha_texels);

        let rgb_texels = colors.map(|row| row.map(|c| c.rgb()));
        let rgb = bc1::Block::encode(rgb_texels);

        Block { alpha, rgb }
    }
}
