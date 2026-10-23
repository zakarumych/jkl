//! BC7 / BPTC UNORM compression for RGBA colors.
//!
//! A block stores 4×4 texels in 16 bytes. Decoding supports all eight modes,
//! partitions, rotations, and index selections. Encoding searches every legal
//! configuration with multiple endpoint seeds, alternating least-squares fits,
//! and bounded discrete refinement. The search minimizes RGBA byte-space
//! squared error; no gamma conversion or global-optimum guarantee is implied.

use std::convert::Infallible;

use crate::math::Rgba32F;

mod encode;
include!("bc7/tables.rs");

/// A BC7 block containing normalized RGBA colors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Block {
    pub data: [u8; 16],
}

impl_fixedcode_struct!(Block { data: [u8; 16] } | Infallible);

impl Block {
    /// Opaque black.
    pub const BLACK: Self = Self {
        data: constant(0, 255),
    };
    /// Opaque white.
    pub const WHITE: Self = Self {
        data: constant(255, 255),
    };
    /// Transparent black.
    pub const TRANSPARENT: Self = Self {
        data: constant(0, 0),
    };

    /// Returns the raw BC7 representation.
    pub const fn bytes(&self) -> [u8; 16] {
        self.data
    }

    /// Constructs a block without interpreting its mode or payload.
    pub const fn from_bytes(data: [u8; 16]) -> Self {
        Self { data }
    }

    /// Decodes normalized RGBA. Reserved headers produce transparent black.
    pub fn decode(self) -> [[Rgba32F; 4]; 4] {
        let pixels = self.decode_bytes();
        std::array::from_fn(|y| {
            std::array::from_fn(|x| Rgba32F(pixels[y * 4 + x].map(|v| v as f32 / 255.0)))
        })
    }

    /// Encodes RGBA, ignoring texels whose masks are `false`.
    ///
    /// Channels are clamped to `[0, 1]` and rounded to the nearest byte.
    /// NaNs become zero. Empty blocks produce [`Self::TRANSPARENT`]. Both
    /// linear and sRGB textures use their stored channel values directly.
    pub fn encode(colors: [[(Rgba32F, bool); 4]; 4]) -> Self {
        encode::encode(colors)
    }

    fn decode_bytes(self) -> [[u8; 4]; 16] {
        let mut input = Bits::new(self.data);
        let mode = self.data[0].trailing_zeros() as usize;
        if mode >= 8 {
            return [[0; 4]; 16];
        }
        input.position = mode + 1;
        let m = MODES[mode];
        let partition = input.read(m.partition_bits) as usize;
        let rotation = if m.dual() { input.read(2) } else { 0 };
        let selection = if mode == 4 { input.read(1) } else { 0 };
        let mut endpoints = [[0; 4]; 6];
        for channel in 0..4 {
            let bits = m.bits(channel);
            if bits != 0 {
                for endpoint in &mut endpoints[..m.subsets * 2] {
                    endpoint[channel] = input.read(bits);
                }
            }
        }
        let mut pbits = [0; 6];
        if mode == 1 {
            for subset in 0..2 {
                let p = input.read(1);
                pbits[2 * subset] = p;
                pbits[2 * subset + 1] = p;
            }
        } else if m.pbits {
            for p in &mut pbits[..m.subsets * 2] {
                *p = input.read(1);
            }
        }
        for e in 0..m.subsets * 2 {
            endpoints[e] = expand_endpoint(endpoints[e], pbits[e], m);
        }
        let (subsets, anchors) = shape(m.subsets, partition);
        let mut indices = [0; 16];
        for (i, index) in indices.iter_mut().enumerate() {
            *index = input.read(m.index_bits - ((anchors >> i) & 1) as usize);
        }
        let mut output = [[0; 4]; 16];
        for i in 0..16 {
            let secondary = if m.dual() {
                input.read(m.secondary_bits - usize::from(i == 0))
            } else {
                indices[i]
            };
            let subset = ((subsets >> (i * 2)) & 3) as usize;
            for c in 0..4 {
                let use_secondary = m.dual() && ((c == 3) == (selection == 0));
                let (index, bits) = if use_secondary {
                    (secondary, m.secondary_bits)
                } else {
                    (indices[i], m.index_bits)
                };
                output[i][c] = interpolate(
                    endpoints[subset * 2][c],
                    endpoints[subset * 2 + 1][c],
                    weights(bits)[index as usize],
                );
            }
            if rotation != 0 {
                output[i].swap(3, rotation as usize - 1);
            }
        }
        debug_assert_eq!(input.position, 128);
        output
    }
}

// Mode 5 can represent independent RGB and alpha constants exactly at 0/255.
const fn constant(rgb: u8, alpha: u8) -> [u8; 16] {
    let mut raw = 1u128 << 5;
    let mut position = 8;
    let mut c = 0;
    while c < 3 {
        let q = (rgb >> 1) as u128;
        raw |= q << position;
        raw |= q << (position + 7);
        position += 14;
        c += 1;
    }
    raw |= (alpha as u128) << position;
    raw |= (alpha as u128) << (position + 8);
    raw.to_le_bytes()
}

#[derive(Clone, Copy)]
struct Mode {
    subsets: usize,
    partition_bits: usize,
    rgb_bits: usize,
    alpha_bits: usize,
    pbits: bool,
    index_bits: usize,
    secondary_bits: usize,
}

impl Mode {
    fn bits(self, channel: usize) -> usize {
        if channel == 3 {
            self.alpha_bits
        } else {
            self.rgb_bits
        }
    }

    fn dual(self) -> bool {
        self.secondary_bits != 0
    }
}

const MODES: [Mode; 8] = [
    Mode {
        subsets: 3,
        partition_bits: 4,
        rgb_bits: 4,
        alpha_bits: 0,
        pbits: true,
        index_bits: 3,
        secondary_bits: 0,
    },
    Mode {
        subsets: 2,
        partition_bits: 6,
        rgb_bits: 6,
        alpha_bits: 0,
        pbits: true,
        index_bits: 3,
        secondary_bits: 0,
    },
    Mode {
        subsets: 3,
        partition_bits: 6,
        rgb_bits: 5,
        alpha_bits: 0,
        pbits: false,
        index_bits: 2,
        secondary_bits: 0,
    },
    Mode {
        subsets: 2,
        partition_bits: 6,
        rgb_bits: 7,
        alpha_bits: 0,
        pbits: true,
        index_bits: 2,
        secondary_bits: 0,
    },
    Mode {
        subsets: 1,
        partition_bits: 0,
        rgb_bits: 5,
        alpha_bits: 6,
        pbits: false,
        index_bits: 2,
        secondary_bits: 3,
    },
    Mode {
        subsets: 1,
        partition_bits: 0,
        rgb_bits: 7,
        alpha_bits: 8,
        pbits: false,
        index_bits: 2,
        secondary_bits: 2,
    },
    Mode {
        subsets: 1,
        partition_bits: 0,
        rgb_bits: 7,
        alpha_bits: 7,
        pbits: true,
        index_bits: 4,
        secondary_bits: 0,
    },
    Mode {
        subsets: 2,
        partition_bits: 6,
        rgb_bits: 5,
        alpha_bits: 5,
        pbits: true,
        index_bits: 2,
        secondary_bits: 0,
    },
];

fn shape(subsets: usize, partition: usize) -> (u32, u16) {
    if subsets == 1 {
        (0, 1)
    } else {
        (
            PARTITIONS[subsets - 2][partition],
            ANCHORS[subsets - 2][partition],
        )
    }
}

fn weights(bits: usize) -> &'static [u8] {
    match bits {
        2 => &[0, 21, 43, 64],
        3 => &[0, 9, 18, 27, 37, 46, 55, 64],
        4 => &[0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64],
        _ => unreachable!(),
    }
}

fn expand(value: u8, bits: usize) -> u8 {
    let v = (value as u16) << (8 - bits);
    (v | (v >> bits)) as u8
}

fn expand_endpoint(q: [u8; 4], p: u8, mode: Mode) -> [u8; 4] {
    std::array::from_fn(|c| {
        let bits = mode.bits(c);
        if bits == 0 {
            255
        } else if mode.pbits {
            expand((q[c] << 1) | p, bits + 1)
        } else {
            expand(q[c], bits)
        }
    })
}

fn interpolate(a: u8, b: u8, weight: u8) -> u8 {
    (((64 - weight as u32) * a as u32 + weight as u32 * b as u32 + 32) >> 6) as u8
}

struct Bits {
    raw: u128,
    position: usize,
}

impl Bits {
    fn new(data: [u8; 16]) -> Self {
        Self {
            raw: u128::from_le_bytes(data),
            position: 0,
        }
    }

    fn read(&mut self, bits: usize) -> u8 {
        if bits == 0 {
            return 0;
        }
        let value = ((self.raw >> self.position) & ((1 << bits) - 1)) as u8;
        self.position += bits;
        value
    }

    fn write(&mut self, value: u8, bits: usize) {
        if bits == 0 {
            return;
        }
        debug_assert!((value as u16) < (1 << bits));
        self.raw |= (value as u128) << self.position;
        self.position += bits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_decode_exactly() {
        for (block, color) in [
            (Block::BLACK, [0, 0, 0, 255]),
            (Block::WHITE, [255; 4]),
            (Block::TRANSPARENT, [0; 4]),
        ] {
            assert_eq!(block.decode_bytes(), [color; 16]);
            assert_eq!(
                block.decode(),
                [[Rgba32F(color.map(|v| v as f32 / 255.0)); 4]; 4]
            );
        }
    }
}
