use crate::{
    encode::FixedCode,
    image::{Extent, format::Format, tiles::TileSize},
};

use super::DecodeError;

define_magic!(Magic => b"JKLI");

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

#[derive(Clone, Copy, Debug)]
pub(crate) struct JackalImageHeader {
    magic: Magic,

    /// Compression method used for the texel data.
    pub compression: Compression,

    /// Format of the blocks.
    pub format: Format,

    /// Extent of the image at mip-0.
    pub extent: Extent,

    /// Number of texture mip levels.
    pub levels: u16,

    /// Size of compression tiles.
    pub tile_size: TileSize,
}

impl JackalImageHeader {
    pub fn new(
        compression: Compression,
        format: Format,
        extent: Extent,
        levels: u16,
        tile_size: TileSize,
    ) -> JackalImageHeader {
        JackalImageHeader {
            magic: Magic,
            compression,
            format,
            extent,
            levels,
            tile_size,
        }
    }
}

impl_fixedcode_struct! {
    JackalImageHeader {
        magic: Magic,
        compression: Compression,
        format: Format,
        extent: Extent,
        levels: u16,
        tile_size: TileSize,
    } | DecodeError
}
