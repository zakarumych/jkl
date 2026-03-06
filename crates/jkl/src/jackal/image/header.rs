use crate::{
    encode::FixedCode,
    image::{format::Format, tiles::TileSize, Dimensions, Extent},
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

#[derive(Clone, Copy)]
struct JackalFormat(Format);

impl FixedCode for JackalFormat {
    const SIZE: usize = 2;
    type Array = [u8; 2];
    type Error = DecodeError;

    #[inline]
    fn fix_encode(&self) -> [u8; 2] {
        (self.0 as u16).to_le_bytes()
    }

    #[inline]
    fn fix_decode(bytes: &[u8; 2]) -> Result<Self, DecodeError> {
        let value = u16::from_le_bytes(*bytes);

        let format = match value {
            0 => Format::R8,
            1 => Format::RG8,
            2 => Format::RGB8,
            3 => Format::RGBA8,
            256 => Format::BC1,
            257 => Format::BC2,
            258 => Format::BC3,
            259 => Format::BC4,
            260 => Format::BC5,
            261 => Format::BC6,
            262 => Format::BC7,
            _ => return Err(DecodeError::InvalidFormat),
        };

        Ok(JackalFormat(format))
    }
}

#[derive(Clone, Copy)]
struct JackalExtent(Extent);

impl FixedCode for JackalExtent {
    const SIZE: usize = 25;
    type Array = [u8; 25];
    type Error = DecodeError;

    fn fix_encode(&self) -> [u8; 25] {
        let mut output = [0u8; 25];
        output[0] = match self.0.dimensions() {
            Dimensions::D1 => 0u8,
            Dimensions::D2 => 1u8,
            Dimensions::D3 => 2u8,
            Dimensions::D1Array => 3u8,
            Dimensions::D2Array => 4u8,
        };

        let raw_size = self.0.raw_size();

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

        let extent =
            Extent::from_raw_size(raw_size, dimensions).ok_or(DecodeError::InvalidDimensions)?;
        Ok(JackalExtent(extent))
    }
}

#[derive(Clone, Copy)]
pub(super) struct JackalHeader {
    magic: Magic,

    /// Compression method used for the texel data.
    compression: Compression,

    // Format of the blocks.
    format: JackalFormat,

    /// Extent of the image at mip-0.
    extent: JackalExtent,

    // Number of texture mip levels.
    levels: MipLevels,

    // Size of compression tiles.
    tile_size: TileSize,
}

impl JackalHeader {
    pub fn new(
        compression: Compression,
        format: Format,
        extent: Extent,
        levels: u16,
        tile_size: TileSize,
    ) -> JackalHeader {
        JackalHeader {
            magic: Magic,
            compression,
            format: JackalFormat(format),
            extent: JackalExtent(extent),
            levels: MipLevels(levels),
            tile_size,
        }
    }

    pub fn compression(&self) -> Compression {
        self.compression
    }

    pub fn format(&self) -> Format {
        self.format.0
    }

    pub fn extent(&self) -> Extent {
        self.extent.0
    }

    // pub fn levels(&self) -> u16 {
    //     self.levels.0
    // }

    pub fn tile_size(&self) -> TileSize {
        self.tile_size
    }
}

impl_fixedcode_struct! {
    JackalHeader {
        magic: Magic,
        compression: Compression,
        format: JackalFormat,
        extent: JackalExtent,
        levels: MipLevels,
        tile_size: TileSize,
    } | DecodeError
}

impl JackalHeader {
    #[inline]
    pub fn tiles_count(&self) -> usize {
        self.tile_size.tiles_count(self.extent.0)
    }
}
