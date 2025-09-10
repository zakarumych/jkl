use std::io::{Read, Write};

use crate::jackal::{DecodeError, DecompressError};

/// Size of the super-block in number of blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuperBlockSize {
    pub width: u16,
    pub height: u16,
}

fn super_block_from_extent(extent: u32) -> u16 {
    match extent {
        0..64 => 16,
        64..128 => 64,
        128..256 => 128,
        256..512 => 256,
        512..1024 => 512,
        _ => 512,
    }
}

impl SuperBlockSize {
    pub fn encode(&self) -> [u8; 2] {
        debug_assert!(self.width.is_power_of_two());
        debug_assert!(self.height.is_power_of_two());

        let w = self.width.trailing_zeros();
        let h = self.height.trailing_zeros();

        debug_assert!(w < 16);
        debug_assert!(h < 16);

        [w as u8, h as u8]
    }

    pub fn decode(bytes: [u8; 2]) -> Result<Self, DecodeError> {
        let w = bytes[0];
        let h = bytes[1];

        if w >= 16 || h >= 16 {
            return Err(DecodeError::InvalidHeader);
        }

        let width = 1 << w;
        let height = 1 << h;

        Ok(SuperBlockSize { width, height })
    }

    pub fn from_size(width: u32, height: u32) -> Self {
        SuperBlockSize {
            width: super_block_from_extent(width),
            height: super_block_from_extent(height),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum Format {
    BC1,
    BC3,
    BC4,
    BC5,
    BC6,
    BC7,
}

impl Format {
    pub fn encode(&self) -> [u8; 2] {
        match self {
            Format::BC1 => 0u16.to_le_bytes(),
            Format::BC3 => 1u16.to_le_bytes(),
            Format::BC4 => 2u16.to_le_bytes(),
            Format::BC5 => 3u16.to_le_bytes(),
            Format::BC6 => 4u16.to_le_bytes(),
            Format::BC7 => 5u16.to_le_bytes(),
        }
    }

    pub fn decode(bytes: [u8; 2]) -> Result<Self, DecodeError> {
        match u16::from_le_bytes(bytes) {
            0 => Ok(Format::BC1),
            1 => Ok(Format::BC3),
            2 => Ok(Format::BC4),
            3 => Ok(Format::BC5),
            4 => Ok(Format::BC6),
            5 => Ok(Format::BC7),
            _ => Err(DecodeError::InvalidHeader),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum Dimensions {
    D1,
    D2,
    D3,
    D1Array,
    D2Array,
}

impl Dimensions {
    pub fn encode(&self) -> [u8; 2] {
        match self {
            Dimensions::D1 => 0u16.to_le_bytes(),
            Dimensions::D2 => 1u16.to_le_bytes(),
            Dimensions::D3 => 2u16.to_le_bytes(),
            Dimensions::D1Array => 3u16.to_le_bytes(),
            Dimensions::D2Array => 4u16.to_le_bytes(),
        }
    }

    pub fn decode(bytes: [u8; 2]) -> Result<Self, DecodeError> {
        match u16::from_le_bytes(bytes) {
            0 => Ok(Dimensions::D1),
            1 => Ok(Dimensions::D2),
            2 => Ok(Dimensions::D3),
            3 => Ok(Dimensions::D1Array),
            4 => Ok(Dimensions::D2Array),
            _ => Err(DecodeError::InvalidHeader),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct MipLevels(pub u16);

impl MipLevels {
    pub fn encode(&self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    pub fn decode(bytes: [u8; 2]) -> Result<Self, DecodeError> {
        let levels = u16::from_le_bytes(bytes);
        if levels == 0 {
            return Err(DecodeError::InvalidHeader);
        }
        Ok(MipLevels(levels))
    }
}

const MAGIC_NUMBER: u32 = 0x494C4B4Au32; // "JKLI"

#[derive(Clone, Copy)]
pub struct JackalHeader {
    // Number of texture mip levels.
    pub levels: MipLevels,

    // Format of the blocks.
    pub format: Format,

    // SuperBlockSize of super-blocks.
    pub super_block_size: SuperBlockSize,

    /// Extent of the image. Decoded based on dimensions.
    pub extent: Extent,
}

impl JackalHeader {
    pub const BYTES_SIZE: usize = 26;

    pub fn write_to(&self, mut write: impl Write) -> std::io::Result<()> {
        let mut bytes = [0; Self::BYTES_SIZE];

        bytes[0..4].copy_from_slice(&MAGIC_NUMBER.to_le_bytes());

        bytes[4..6].copy_from_slice(&self.levels.encode());
        bytes[6..8].copy_from_slice(&self.format.encode());
        bytes[8..10].copy_from_slice(&self.super_block_size.encode());
        bytes[10..12].copy_from_slice(&self.extent.dimensions().encode());

        let raw_size = self.extent.raw_size();
        bytes[12..16].copy_from_slice(&raw_size[0].to_le_bytes());
        bytes[16..20].copy_from_slice(&raw_size[1].to_le_bytes());
        bytes[20..24].copy_from_slice(&raw_size[2].to_le_bytes());

        write.write_all(&bytes)?;
        Ok(())
    }

    pub fn read_from(mut read: impl Read) -> Result<Self, DecompressError> {
        let mut bytes = [0; 24];
        read.read_exact(&mut bytes)?;

        let mut magic_bytes = [0; 4];
        magic_bytes.copy_from_slice(&bytes[0..4]);
        let magic = u32::from_le_bytes(magic_bytes);
        if magic != MAGIC_NUMBER {
            return Err(DecodeError::InvalidMagic.into());
        }

        let mut levels_bytes = [0; 2];
        levels_bytes.copy_from_slice(&bytes[4..6]);
        let levels = MipLevels::decode(levels_bytes)?;

        let mut format_bytes = [0; 2];
        format_bytes.copy_from_slice(&bytes[6..8]);
        let format = Format::decode(format_bytes)?;

        let mut super_block_size_bytes = [0; 2];
        super_block_size_bytes.copy_from_slice(&bytes[8..10]);
        let super_block_size = SuperBlockSize::decode(super_block_size_bytes)?;

        let mut dimensions_bytes = [0; 2];
        dimensions_bytes.copy_from_slice(&bytes[10..12]);
        let dimensions = Dimensions::decode(dimensions_bytes)?;

        let mut extent_bytes = [0; 4];
        extent_bytes.copy_from_slice(&bytes[12..16]);
        let width = u32::from_le_bytes(extent_bytes);
        extent_bytes.copy_from_slice(&bytes[16..20]);
        let height = u32::from_le_bytes(extent_bytes);
        extent_bytes.copy_from_slice(&bytes[20..24]);
        let depth = u32::from_le_bytes(extent_bytes);

        let raw_size = [width, height, depth];
        let extent = Extent::from_raw_size(raw_size, dimensions)?;

        Ok(JackalHeader {
            levels,
            format,
            super_block_size,
            extent,
        })
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn extent(&self) -> Extent {
        self.extent
    }

    pub fn jackal_blocks_count(&self) -> usize {
        let [width, height, depth] = self.jackal_blocks_extent();
        (width * height * depth) as usize
    }

    pub fn jackal_blocks_extent(&self) -> [u32; 3] {
        let raw_size = self.extent.raw_size();
        let jackal_blocks_width = (raw_size[0] + self.super_block_size.width as u32 - 1)
            / self.super_block_size.width as u32;
        let jackal_blocks_height = (raw_size[1] + self.super_block_size.height as u32 - 1)
            / self.super_block_size.height as u32;
        let jackal_blocks_depth = raw_size[2];

        [
            jackal_blocks_width,
            jackal_blocks_height,
            jackal_blocks_depth,
        ]
    }

    pub fn blocks_count(&self) -> usize {
        let raw_size = self.extent.raw_size();
        raw_size[0] as usize * raw_size[1] as usize * raw_size[2] as usize
    }
}

#[derive(Clone, Copy)]
pub struct JackalBlock {
    pub offset: u64,
}

impl JackalBlock {
    pub const BYTES_SIZE: usize = size_of::<u64>();

    pub fn write_to(&self, mut write: impl Write) -> std::io::Result<()> {
        write.write_all(&self.offset.to_le_bytes())
    }

    pub fn read_from(mut read: impl Read) -> Result<Self, DecompressError> {
        let mut bytes = [0; Self::BYTES_SIZE];
        read.read_exact(&mut bytes)?;
        let offset = u64::from_le_bytes(bytes);
        Ok(JackalBlock { offset })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extent {
    D1 {
        width: u32,
    },
    D2 {
        width: u32,
        height: u32,
    },
    D3 {
        width: u32,
        height: u32,
        depth: u32,
    },
    D1Array {
        width: u32,
        layers: u32,
    },
    D2Array {
        width: u32,
        height: u32,
        layers: u32,
    },
}

impl Extent {
    pub fn width(&self) -> u32 {
        match *self {
            Extent::D1 { width } => width,
            Extent::D2 { width, .. } => width,
            Extent::D3 { width, .. } => width,
            Extent::D1Array { width, .. } => width,
            Extent::D2Array { width, .. } => width,
        }
    }

    pub fn height(&self) -> u32 {
        match *self {
            Extent::D1 { .. } => 1,
            Extent::D2 { height, .. } => height,
            Extent::D3 { height, .. } => height,
            Extent::D1Array { .. } => 1,
            Extent::D2Array { height, .. } => height,
        }
    }

    pub fn depth(&self) -> u32 {
        match *self {
            Extent::D1 { .. } => 1,
            Extent::D2 { .. } => 1,
            Extent::D3 { depth, .. } => depth,
            Extent::D1Array { .. } => 1,
            Extent::D2Array { .. } => 1,
        }
    }

    pub fn layers(&self) -> u32 {
        match *self {
            Extent::D1 { .. } => 1,
            Extent::D2 { .. } => 1,
            Extent::D3 { .. } => 1,
            Extent::D1Array { layers, .. } => layers,
            Extent::D2Array { layers, .. } => layers,
        }
    }

    fn dimensions(self) -> Dimensions {
        match self {
            Extent::D1 { .. } => Dimensions::D1,
            Extent::D2 { .. } => Dimensions::D2,
            Extent::D3 { .. } => Dimensions::D3,
            Extent::D1Array { .. } => Dimensions::D1Array,
            Extent::D2Array { .. } => Dimensions::D2Array,
        }
    }

    pub fn raw_size(self) -> [u32; 3] {
        match self {
            Extent::D1 { width } => [width, 1, 1],
            Extent::D2 { width, height } => [width, height, 1],
            Extent::D3 {
                width,
                height,
                depth,
            } => [width, height, depth],
            Extent::D1Array { width, layers } => [width, layers, 1],
            Extent::D2Array {
                width,
                height,
                layers,
            } => [width, height, layers],
        }
    }

    fn from_raw_size(value: [u32; 3], dimensions: Dimensions) -> Result<Self, DecodeError> {
        match dimensions {
            Dimensions::D1 => {
                if value[1] != 1 || value[2] != 1 {
                    return Err(DecodeError::InvalidHeader);
                }
                Ok(Extent::D1 { width: value[0] })
            }
            Dimensions::D2 => {
                if value[2] != 1 {
                    return Err(DecodeError::InvalidHeader);
                }
                Ok(Extent::D2 {
                    width: value[0],
                    height: value[1],
                })
            }
            Dimensions::D3 => Ok(Extent::D3 {
                width: value[0],
                height: value[1],
                depth: value[2],
            }),
            Dimensions::D1Array => {
                if value[2] != 1 {
                    return Err(DecodeError::InvalidHeader);
                }
                Ok(Extent::D1Array {
                    width: value[0],
                    layers: value[1],
                })
            }
            Dimensions::D2Array => Ok(Extent::D2Array {
                width: value[0],
                height: value[1],
                layers: value[2],
            }),
        }
    }
}
