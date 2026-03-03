use core::f32;
use std::convert::Infallible;

use crate::{
    encode::FixedCode,
    image::{Dimensions, Image2DRef, Image3DRef, ImageRef},
};

use super::DecodeError;

/// Size of the super-block in number of blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSize {
    pub width: u16,
    pub height: u16,
}

impl FixedCode for TileSize {
    const SIZE: usize = 1;
    type Array = [u8; 1];
    type Error = Infallible;

    fn fix_encode(&self) -> [u8; 1] {
        debug_assert!(self.width.is_power_of_two());
        debug_assert!(self.height.is_power_of_two());

        let w = self.width.trailing_zeros();
        let h = self.height.trailing_zeros();

        debug_assert!(w < 16);
        debug_assert!(h < 16);

        [((w << 4) | h) as u8]
    }

    fn fix_decode(input: &[u8; 1]) -> Result<Self, Infallible> {
        let byte = input[0];
        let w = byte >> 4;
        let h = byte & 0x0F;

        let width = 1 << w;
        let height = 1 << h;

        Ok(TileSize { width, height })
    }
}

impl TileSize {
    /// Finds the optimal super-block size for the given extent and block size.
    ///
    /// The optimal super-block size is the one that minimizes the cost function:
    /// cost = (flat_cost + size_cost * tile_size) * ceil(number_of_tiles / 64) * 64
    /// i.e. super block cost is linear to size plus a float amount and it is multiplied by the number of super blocks rounded up to the nearest multiple of 64.
    /// This cost function models the GPU decompression cost, which runs thread per super-block, each thread has a fixed overhead of dispatching and a cost proportional to the super-block size,
    /// and the GPU executes threads in warps of 32 or 64 threads, so the number of super-blocks is rounded up to the nearest multiple of 64.
    pub fn find_optimal(extent: Extent, block_size: u16, flat_cost: f32, size_cost: f32) -> Self {
        assert!(block_size.is_power_of_two());

        // The max super-block size to consider is either enough to cover the entire extent,
        // or 32768 blocks, which is the maximum super-block size.
        let max_tile_width = u16::try_from(extent.width().next_power_of_two()).unwrap_or(1 << 15);
        let max_tile_height = u16::try_from(extent.height().next_power_of_two()).unwrap_or(1 << 15);

        let mut min_cost = f32::INFINITY;
        let mut best_candidate = TileSize {
            width: block_size,
            height: block_size,
        };

        let mut tile_width = u64::from(block_size);
        let mut tile_height = u64::from(block_size);

        loop {
            let row_size = (extent.width() + tile_width - 1) / tile_width;
            let column_size = (extent.height() + tile_height - 1) / tile_height;
            let planes = extent.depth();

            let tiles_count = row_size * column_size * planes;
            let warps_count = (tiles_count + 63) / 64;

            let cost = (flat_cost + size_cost * (tile_width as f32 * tile_height as f32))
                * warps_count as f32;

            if cost < min_cost {
                min_cost = cost;
                best_candidate = TileSize {
                    width: tile_width as u16,
                    height: tile_height as u16,
                };
            }

            if tile_width == u64::from(max_tile_width) {
                if tile_height == u64::from(max_tile_height) {
                    break;
                } else {
                    tile_height *= 2;
                    tile_width = u64::from(block_size);
                }
            }
        }

        best_candidate
    }

    pub fn iter_tiles<'a, T>(&self, image: ImageRef<'a, T>) -> TilesIter<'a, T> {
        TilesIter {
            image: image.as_3d(),
            tile_size: *self,
            x: 0,
            y: 0,
            z: 0,
        }
    }
}

struct TilesIter<'a, T> {
    image: Image3DRef<'a, T>,
    tile_size: TileSize,
    x: usize,
    y: usize,
    z: usize,
}

impl<'a, T> Clone for TilesIter<'a, T> {
    fn clone(&self) -> Self {
        TilesIter {
            image: self.image,
            tile_size: self.tile_size,
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl<'a, T> Iterator for TilesIter<'a, T> {
    type Item = Image2DRef<'a, T>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn next(&mut self) -> Option<Image2DRef<'a, T>> {
        let w = usize::min(
            usize::from(self.tile_size.width),
            self.image.width() - self.x,
        );
        let h = usize::min(
            usize::from(self.tile_size.height),
            self.image.height() - self.y,
        );

        if self.x >= self.image.width() {
            self.x = 0;
            self.y += h;
        }

        if self.y >= self.image.height() {
            self.y = 0;
            self.z += 1;
        }

        if self.z >= self.image.depth() {
            return None;
        }

        let plane = self.image.get_plane_xy(self.z);

        let tile = plane.get_range(self.x, self.y, w, h);

        self.x += w;

        Some(tile)
    }
}

impl<'a, T> ExactSizeIterator for TilesIter<'a, T> {
    fn len(&self) -> usize {
        let rows = (self.image.width() + usize::from(self.tile_size.width) - 1)
            / usize::from(self.tile_size.width);
        let columns = (self.image.height() + usize::from(self.tile_size.height) - 1)
            / usize::from(self.tile_size.height);
        let planes = self.image.depth();

        planes * columns * rows - self.z * columns * rows - self.y * rows - self.x
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
    fn fix_encode(&self) -> [u8; 2] {
        (*self as u16).to_le_bytes()
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
struct Magic;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extent {
    D1 { width: u64 },
    D2 { width: u64, height: u64 },
    D3 { width: u64, height: u64, depth: u64 },
}

impl Extent {
    pub fn width(&self) -> u64 {
        match *self {
            Extent::D1 { width } => width,
            Extent::D2 { width, .. } => width,
            Extent::D3 { width, .. } => width,
        }
    }

    pub fn height(&self) -> u64 {
        match *self {
            Extent::D1 { .. } => 1,
            Extent::D2 { height, .. } => height,
            Extent::D3 { height, .. } => height,
        }
    }

    pub fn depth(&self) -> u64 {
        match *self {
            Extent::D1 { .. } => 1,
            Extent::D2 { .. } => 1,
            Extent::D3 { depth, .. } => depth,
        }
    }

    pub fn dimensions(self) -> Dimensions {
        match self {
            Extent::D1 { .. } => Dimensions::D1,
            Extent::D2 { .. } => Dimensions::D2,
            Extent::D3 { .. } => Dimensions::D3,
        }
    }

    pub fn raw_size(self) -> [u64; 3] {
        match self {
            Extent::D1 { width } => [width, 1, 1],
            Extent::D2 { width, height } => [width, height, 1],
            Extent::D3 {
                width,
                height,
                depth,
            } => [width, height, depth],
        }
    }

    pub fn from_raw_size(value: [u64; 3], dimensions: Dimensions) -> Option<Self> {
        match dimensions {
            Dimensions::D1 => {
                if value[1] != 1 || value[2] != 1 {
                    return None;
                }
                Some(Extent::D1 { width: value[0] })
            }
            Dimensions::D2 => {
                if value[2] != 1 {
                    return None;
                }
                Some(Extent::D2 {
                    width: value[0],
                    height: value[1],
                })
            }
            Dimensions::D3 => Some(Extent::D3 {
                width: value[0],
                height: value[1],
                depth: value[2],
            }),
        }
    }

    #[inline]
    pub fn tiles_count(&self, tile: TileSize) -> usize {
        let [width, height, depth] = self.tiles(tile);
        (width * height * depth) as usize
    }

    #[inline]
    pub fn tiles(&self, tile: TileSize) -> [u64; 3] {
        let raw_size = self.raw_size();
        let tiles_width = (raw_size[0] + tile.width as u64 - 1) / tile.width as u64;
        let tiles_height = (raw_size[1] + tile.height as u64 - 1) / tile.height as u64;
        let tiles_depth = raw_size[2];

        [tiles_width, tiles_height, tiles_depth]
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
        };

        let raw_size = self.raw_size();
        output[1..9].copy_from_slice(&raw_size[0].to_le_bytes());
        output[9..17].copy_from_slice(&raw_size[1].to_le_bytes());
        output[17..25].copy_from_slice(&raw_size[2].to_le_bytes());
        output
    }

    fn fix_decode(input: &[u8; 25]) -> Result<Self, DecodeError> {
        let dimensions = match input[0] {
            0u8 => Dimensions::D1,
            1u8 => Dimensions::D2,
            2u8 => Dimensions::D3,
            _ => return Err(DecodeError::InvalidDimensions),
        };

        let width = u64::from_le_bytes(*input[1..9].as_array().unwrap());
        let height = u64::from_le_bytes(*input[9..17].as_array().unwrap());
        let depth = u64::from_le_bytes(*input[17..25].as_array().unwrap());

        let raw_size = [width, height, depth];

        Extent::from_raw_size(raw_size, dimensions).ok_or(DecodeError::InvalidExtent)
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

    // TileSize of super-blocks.
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
    pub fn format(&self) -> Format {
        self.format
    }

    #[inline]
    pub fn extent(&self) -> Extent {
        self.extent
    }

    #[inline]
    pub fn tiles_count(&self) -> usize {
        self.extent.tiles_count(self.tile_size)
    }

    #[inline]
    pub fn tiles(&self) -> [u64; 3] {
        self.extent.tiles(self.tile_size)
    }
}

#[derive(Clone, Copy)]
pub struct JackalBlock {
    pub offset: u64,
}

impl_fixedcode_struct!(JackalBlock { offset: u64 } | Infallible);
