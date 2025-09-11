//! BC4 implementation.
//!

use crate::{
    bc4,
    math::{Rg32F, R32F},
};

/// A block of 4x4 texels compressed with BC4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Block {
    pub red: bc4::Block,
    pub green: bc4::Block,
}

impl Block {
    pub const BLACK: Block = Block {
        red: bc4::Block::BLACK,
        green: bc4::Block::BLACK,
    };

    pub const WHITE: Block = Block {
        red: bc4::Block::WHITE,
        green: bc4::Block::WHITE,
    };

    pub fn bytes(&self) -> [u8; 16] {
        let red = self.red.bytes();
        let green = self.green.bytes();

        [
            red[0], red[1], red[2], red[3], red[4], red[5], red[6], red[7], green[0], green[1],
            green[2], green[3], green[4], green[5], green[6], green[7],
        ]
    }

    pub fn from_bytes(bytes: [u8; 16]) -> Block {
        let red = [
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ];
        let green = [
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ];

        Block {
            red: bc4::Block::from_bytes(red),
            green: bc4::Block::from_bytes(green),
        }
    }

    /// Decodes single BC4 block.
    pub fn decode(self) -> [[Rg32F; 4]; 4] {
        let red = self.red.decode();
        let green = self.green.decode();

        let mut colors = [[Rg32F::BLACK; 4]; 4];

        for i in 0..4 {
            for j in 0..4 {
                colors[i][j] = Rg32F::new(red[i][j].r(), green[i][j].r());
            }
        }

        colors
    }

    pub fn encode(colors: [[Rg32F; 4]; 4]) -> Self {
        let mut red = [[R32F::BLACK; 4]; 4];
        let mut green = [[R32F::BLACK; 4]; 4];

        for i in 0..4 {
            for j in 0..4 {
                red[i][j] = R32F::new(colors[i][j].r());
                green[i][j] = R32F::new(colors[i][j].g());
            }
        }

        let red = bc4::Block::encode(red);
        let green = bc4::Block::encode(green);

        Block { red, green }
    }
}
