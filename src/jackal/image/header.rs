use std::convert::Infallible;

use crate::{
    encode::FixedCode,
    image::{Dimensions, Extent},
    jackal::image::{format::Format, tiles::TileSize},
};

use super::DecodeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct MipLevels(pub u16);

impl FixedCode for MipLevels {
    const SIZE: usize = 2;
    type Array = [u8; 2];
    type Error = DecodeError;

    #[inline]
    fn fix_encode(&self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    #[inline]
    fn fix_decode(bytes: &[u8; 2]) -> Result<Self, DecodeError> {
        let levels = u16::from_le_bytes(*bytes);
        if levels == 0 {
            return Err(DecodeError::MipZero);
        }
        Ok(MipLevels(levels))
    }
}

const MAGIC_NUMBER: u32 = 0x494C4B4Au32; // "JKLI"

#[derive(Clone, Copy)]
pub struct Magic;

impl FixedCode for Magic {
    const SIZE: usize = 4;
    type Array = [u8; 4];
    type Error = DecodeError;

    #[inline]
    fn fix_encode(&self) -> [u8; 4] {
        MAGIC_NUMBER.to_le_bytes()
    }

    #[inline]
    fn fix_decode(bytes: &[u8; 4]) -> Result<Self, DecodeError> {
        let value = u32::from_le_bytes(*bytes);
        if value != MAGIC_NUMBER {
            return Err(DecodeError::InvalidMagic);
        }
        Ok(Magic)
    }
}

/// Compression method used for the texel data.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Compression {
    None = 0,
    Lz77 = 1,
    Ans = 2,
    Lz77Ans = 3,
    RleAns = 4,
}

impl FixedCode for Compression {
    const SIZE: usize = 1;
    type Array = [u8; 1];
    type Error = DecodeError;

    #[inline]
    fn fix_encode(&self) -> [u8; 1] {
        [*self as u8]
    }

    #[inline]
    fn fix_decode(input: &[u8; 1]) -> Result<Self, DecodeError> {
        let value = input[0];
        let compression = match value {
            0 => Compression::None,
            1 => Compression::Lz77,
            2 => Compression::Ans,
            3 => Compression::Lz77Ans,
            4 => Compression::RleAns,
            _ => return Err(DecodeError::InvalidCompression),
        };
        Ok(compression)
    }
}

impl FixedCode for Extent {
    const SIZE: usize = 25;
    type Array = [u8; 25];
    type Error = DecodeError;

    fn fix_encode(&self) -> [u8; 25] {
        let mut output = [0u8; 25];
        output[0] = match self.dimensions() {
            Dimensions::D1 => 0u8,
            Dimensions::D2 => 1u8,
            Dimensions::D3 => 2u8,
            Dimensions::D1Array => 3u8,
            Dimensions::D2Array => 4u8,
        };

        let raw_size = self.raw_size();

        output[1..9].copy_from_slice(
            &u64::try_from(raw_size[0])
                .expect("Dimension exceeds u64::MAX")
                .to_le_bytes(),
        );
        output[9..17].copy_from_slice(
            &u64::try_from(raw_size[1])
                .expect("Dimension exceeds u64::MAX")
                .to_le_bytes(),
        );
        output[17..25].copy_from_slice(
            &u64::try_from(raw_size[2])
                .expect("Dimension exceeds u64::MAX")
                .to_le_bytes(),
        );
        output
    }

    fn fix_decode(input: &[u8; 25]) -> Result<Self, DecodeError> {
        let dimensions = match input[0] {
            0u8 => Dimensions::D1,
            1u8 => Dimensions::D2,
            2u8 => Dimensions::D3,
            3u8 => Dimensions::D1Array,
            4u8 => Dimensions::D2Array,
            _ => return Err(DecodeError::InvalidDimensions),
        };

        let width = u64::from_le_bytes(*input[1..9].as_array().unwrap())
            .try_into()
            .map_err(|_| DecodeError::TooLarge)?;
        let height = u64::from_le_bytes(*input[9..17].as_array().unwrap())
            .try_into()
            .map_err(|_| DecodeError::TooLarge)?;
        let depth = u64::from_le_bytes(*input[17..25].as_array().unwrap())
            .try_into()
            .map_err(|_| DecodeError::TooLarge)?;

        let raw_size = [width, height, depth];

        Extent::from_raw_size(raw_size, dimensions).ok_or(DecodeError::InvalidExtent)
    }
}

#[derive(Clone, Copy)]
pub struct JackalHeader {
    pub magic: Magic,

    /// Compression method used for the texel data.
    pub compression: Compression,

    // Format of the blocks.
    pub format: Format,

    /// Extent of the image at mip-0.
    pub extent: Extent,

    // Number of texture mip levels.
    pub levels: MipLevels,

    // Size of compression tiles.
    pub tile_size: TileSize,
}

impl_fixedcode_struct! {
    JackalHeader {
        magic: Magic,
        compression: Compression,
        format: Format,
        extent: Extent,
        levels: MipLevels,
        tile_size: TileSize,
    } | DecodeError
}

impl JackalHeader {
    pub const fn new() -> JackalHeader {
        JackalHeader {
            magic: Magic,
            compression: Compression::None,
            format: Format::R8,
            extent: Extent::D1 { width: 1 },
            levels: MipLevels(1),
            tile_size: TileSize {
                width: 1,
                height: 1,
            },
        }
    }

    #[inline]
    pub fn tiles_count(&self) -> usize {
        self.tile_size.tiles_count(self.extent)
    }

    #[inline]
    pub fn tiles(&self) -> [usize; 3] {
        self.tile_size.tiles(self.extent)
    }
}

#[derive(Clone, Copy)]
pub struct JackalBlock {
    pub offset: u64,
}

impl_fixedcode_struct!(JackalBlock { offset: u64 } | Infallible);
