use std::fmt;

use crate::encode::FixedCode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Format {
    /// 8-bit single channel.
    R8,

    /// 8-bit two channels.
    RG8,

    /// 8-bit three channels.
    RGB8,

    /// 8-bit four channels.
    RGBA8,

    /// BC1 (DXT1) block compression format.
    BC1 = 256,

    /// BC2 (DXT3) block compression format.
    BC2,

    /// BC3 (DXT5) block compression format.
    BC3,

    /// BC4 block compression format.
    BC4,

    /// BC5 block compression format.
    BC5,

    /// BC6 block compression format.
    BC6,

    /// BC7 block compression format.
    BC7,
}

impl Format {
    pub const fn tile_width_granularity(&self) -> u16 {
        match self {
            Format::R8 => 4,
            Format::RG8 => 2,
            Format::RGB8 | Format::RGBA8 => 1,
            Format::BC1
            | Format::BC2
            | Format::BC3
            | Format::BC4
            | Format::BC5
            | Format::BC6
            | Format::BC7 => 4,
        }
    }

    pub const fn tile_height_granularity(&self) -> u16 {
        match self {
            Format::R8 | Format::RG8 | Format::RGB8 | Format::RGBA8 => 1,
            Format::BC1
            | Format::BC2
            | Format::BC3
            | Format::BC4
            | Format::BC5
            | Format::BC6
            | Format::BC7 => 4,
        }
    }

    /// Returns width of a block in pixels for the format.
    pub const fn block_width(&self) -> u16 {
        match self {
            Format::R8 | Format::RG8 | Format::RGB8 | Format::RGBA8 => 1,
            Format::BC1
            | Format::BC2
            | Format::BC3
            | Format::BC4
            | Format::BC5
            | Format::BC6
            | Format::BC7 => 4,
        }
    }

    /// Returns height of a block in pixels for the format.
    pub const fn block_height(&self) -> u16 {
        match self {
            Format::R8 | Format::RG8 | Format::RGB8 | Format::RGBA8 => 1,
            Format::BC1
            | Format::BC2
            | Format::BC3
            | Format::BC4
            | Format::BC5
            | Format::BC6
            | Format::BC7 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InvalidFormat;

impl fmt::Display for InvalidFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid format")
    }
}

impl std::error::Error for InvalidFormat {}

impl FixedCode for Format {
    const SIZE: usize = 2;
    type Array = [u8; 2];
    type Error = InvalidFormat;

    #[inline]
    fn fix_encode(&self) -> [u8; 2] {
        (*self as u16).to_le_bytes()
    }

    #[inline]
    fn fix_decode(bytes: &[u8; 2]) -> Result<Self, InvalidFormat> {
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
            _ => return Err(InvalidFormat),
        };

        Ok(format)
    }
}
