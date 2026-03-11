#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
