use core::f32;
use std::convert::Infallible;

use crate::{encode::FixedCode, jackal::DecodeError};

/// Size of the super-block in number of blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuperBlockSize {
    pub width: u16,
    pub height: u16,
}

impl FixedCode for SuperBlockSize {
    const SIZE: usize = 1;
    type Array = [u8; 1];
    type Error = DecodeError;

    fn encode(&self) -> [u8; 1] {
        debug_assert!(self.width.is_power_of_two());
        debug_assert!(self.height.is_power_of_two());

        let w = self.width.trailing_zeros();
        let h = self.height.trailing_zeros();

        debug_assert!(w < 16);
        debug_assert!(h < 16);

        [((w << 4) | h) as u8]
    }

    fn decode(input: &[u8; 1]) -> Result<Self, DecodeError> {
        let byte = input[0];
        let w = byte >> 4;
        let h = byte & 0x0F;

        let width = 1 << w;
        let height = 1 << h;

        Ok(SuperBlockSize { width, height })
    }
}

impl SuperBlockSize {
    /// Finds the optimal super-block size for the given extent and block size.
    ///
    /// The optimal super-block size is the one that minimizes the cost function:
    /// cost = (flat_cost + size_cost * superblock_size) * ceil(number_of_superblocks / 64) * 64
    /// i.e. super block cost is linear to size plus a float amount and it is multiplied by the number of super blocks rounded up to the nearest multiple of 64.
    /// This cost function models the GPU decompression cost, which runs thread per super-block, each thread has a fixed overhead of dispatching and a cost proportional to the super-block size,
    /// and the GPU executes threads in warps of 32 or 64 threads, so the number of super-blocks is rounded up to the nearest multiple of 64.
    pub fn find_optimal(extent: Extent, block_size: u16, flat_cost: f32, size_cost: f32) -> Self {
        assert!(block_size.is_power_of_two());

        // The max super-block size to consider is either enough to cover the entire extent,
        // or 32768 blocks, which is the maximum super-block size.
        let max_superblock_width =
            u16::try_from(extent.width().next_power_of_two()).unwrap_or(1 << 15);
        let max_superblock_height =
            u16::try_from(extent.height().next_power_of_two()).unwrap_or(1 << 15);

        let mut min_cost = f32::INFINITY;
        let mut best_candidate = SuperBlockSize {
            width: block_size,
            height: block_size,
        };

        let mut superblock_width = u32::from(block_size);
        let mut superblock_height = u32::from(block_size);

        loop {
            let row_size = (extent.width() + superblock_width - 1) / superblock_width;
            let column_size = (extent.height() + superblock_height - 1) / superblock_height;
            let planes = extent.depth();

            let superblocks_count = row_size * column_size * planes;
            let warps_count = (superblocks_count + 63) / 64;

            let cost = (flat_cost
                + size_cost * (superblock_width as f32 * superblock_height as f32))
                * warps_count as f32;

            if cost < min_cost {
                min_cost = cost;
                best_candidate = SuperBlockSize {
                    width: superblock_width as u16,
                    height: superblock_height as u16,
                };
            }

            if superblock_width == u32::from(max_superblock_width) {
                if superblock_height == u32::from(max_superblock_height) {
                    break;
                } else {
                    superblock_height *= 2;
                    superblock_width = u32::from(block_size);
                }
            }
        }

        best_candidate
    }
}

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

impl FixedCode for Format {
    const SIZE: usize = 2;
    type Array = [u8; 2];
    type Error = DecodeError;

    #[inline]
    fn encode(&self) -> [u8; 2] {
        (*self as u16).to_le_bytes()
    }

    #[inline]
    fn decode(bytes: &[u8; 2]) -> Result<Self, DecodeError> {
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

        Ok(format)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct MipLevels(pub u16);

impl FixedCode for MipLevels {
    const SIZE: usize = 2;
    type Array = [u8; 2];
    type Error = DecodeError;

    #[inline]
    fn encode(&self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    #[inline]
    fn decode(bytes: &[u8; 2]) -> Result<Self, DecodeError> {
        let levels = u16::from_le_bytes(*bytes);
        if levels == 0 {
            return Err(DecodeError::MipZero);
        }
        Ok(MipLevels(levels))
    }
}

const MAGIC_NUMBER: u32 = 0x494C4B4Au32; // "JKLI"

#[derive(Clone, Copy)]
struct Magic;

impl FixedCode for Magic {
    const SIZE: usize = 4;
    type Array = [u8; 4];
    type Error = DecodeError;

    #[inline]
    fn encode(&self) -> [u8; 4] {
        MAGIC_NUMBER.to_le_bytes()
    }

    #[inline]
    fn decode(bytes: &[u8; 4]) -> Result<Self, DecodeError> {
        let value = u32::from_le_bytes(*bytes);
        if value != MAGIC_NUMBER {
            return Err(DecodeError::InvalidMagic);
        }
        Ok(Magic)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Dimensions {
    D1,
    D2,
    D3,
    D1Array,
    D2Array,
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
                    return Err(DecodeError::InvalidExtent);
                }
                Ok(Extent::D1 { width: value[0] })
            }
            Dimensions::D2 => {
                if value[2] != 1 {
                    return Err(DecodeError::InvalidExtent);
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
                    return Err(DecodeError::InvalidExtent);
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

impl FixedCode for Extent {
    const SIZE: usize = 13;
    type Array = [u8; 13];
    type Error = DecodeError;

    fn encode(&self) -> [u8; 13] {
        let mut output = [0u8; 13];
        output[0] = match self.dimensions() {
            Dimensions::D1 => 0u8,
            Dimensions::D2 => 1u8,
            Dimensions::D3 => 2u8,
            Dimensions::D1Array => 3u8,
            Dimensions::D2Array => 4u8,
        };

        let raw_size = self.raw_size();
        output[1..5].copy_from_slice(&raw_size[0].to_le_bytes());
        output[5..9].copy_from_slice(&raw_size[1].to_le_bytes());
        output[9..13].copy_from_slice(&raw_size[2].to_le_bytes());
        output
    }

    fn decode(input: &[u8; 13]) -> Result<Self, DecodeError> {
        let dimensions = match input[0] {
            0 => Dimensions::D1,
            1 => Dimensions::D2,
            2 => Dimensions::D3,
            3 => Dimensions::D1Array,
            4 => Dimensions::D2Array,
            _ => return Err(DecodeError::InvalidDimensions),
        };

        let width = u32::from_le_bytes(*input[1..5].as_array().unwrap());
        let height = u32::from_le_bytes(*input[5..9].as_array().unwrap());
        let depth = u32::from_le_bytes(*input[9..13].as_array().unwrap());

        let raw_size = [width, height, depth];
        let extent = Extent::from_raw_size(raw_size, dimensions)?;

        Ok(extent)
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
    fn encode(&self) -> [u8; 1] {
        [*self as u8]
    }

    #[inline]
    fn decode(input: &[u8; 1]) -> Result<Self, DecodeError> {
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
pub struct JackalHeader {
    magic: Magic,

    /// Compression method used for the texel data.
    pub compression: Compression,

    // Format of the blocks.
    pub format: Format,

    /// Extent of the image at mip-0.
    pub extent: Extent,

    // Number of texture mip levels.
    pub levels: MipLevels,

    // SuperBlockSize of super-blocks.
    pub superblock_size: SuperBlockSize,
}

impl_fixedcode_struct! {
    JackalHeader {
        magic: Magic,
        compression: Compression,
        format: Format,
        extent: Extent,
        levels: MipLevels,
        superblock_size: SuperBlockSize,
    } | DecodeError
}

impl JackalHeader {
    /// Constructs a new `JackalHeader` with the given fields.
    pub fn new(
        compression: Compression,
        format: Format,
        extent: Extent,
        levels: MipLevels,
        superblock_size: SuperBlockSize,
    ) -> Self {
        JackalHeader {
            magic: Magic,
            compression,
            format,
            extent,
            levels,
            superblock_size,
        }
    }

    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    #[inline]
    pub fn extent(&self) -> Extent {
        self.extent
    }

    #[inline]
    pub fn superblocks_count(&self) -> usize {
        let [width, height, depth] = self.superblocks_extent();
        (width * height * depth) as usize
    }

    #[inline]
    pub fn superblocks_extent(&self) -> [u32; 3] {
        let raw_size = self.extent.raw_size();
        let superblocks_width = (raw_size[0] + self.superblock_size.width as u32 - 1)
            / self.superblock_size.width as u32;
        let superblocks_height = (raw_size[1] + self.superblock_size.height as u32 - 1)
            / self.superblock_size.height as u32;
        let superblocks_depth = raw_size[2];

        [superblocks_width, superblocks_height, superblocks_depth]
    }

    #[inline]
    pub fn blocks_count(&self) -> usize {
        let raw_size = self.extent.raw_size();
        raw_size[0] as usize * raw_size[1] as usize * raw_size[2] as usize
    }
}

#[derive(Clone, Copy)]
pub struct JackalBlock {
    pub offset: u64,
}

impl_fixedcode_struct!(JackalBlock { offset: u64 } | Infallible);
