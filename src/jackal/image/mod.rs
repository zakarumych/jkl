//! # Jackal Image File Format (JKLI)
//!
//! This module provides comprehensive support for the Jackal Image (JKLI) file format,
//! a flexible and efficient image storage format designed for high-performance image encoding
//! and decoding with support for multiple compression methods and tile-based processing.
//!
//! ## Overview
//!
//! The JKLI format is optimized for storing images with varying levels of compression,
//! allowing users to balance between file size and processing speed. The format supports
//! tile-based encoding, which enables efficient streaming and partial image loading.
//!
//! ## Features
//!
//! - **Multiple Compression Methods**: Support for uncompressed, LZ77, ANS, LZ77+ANS, and RLE+ANS compression
//! - **Tile-Based Architecture**: Images are divided into optimally-sized tiles for efficient processing
//! - **Flexible Pixel Formats**: Support for various pixel formats including R8, RG8, RGB8, RGBA8, BC1, BC2, BC3, BC4, BC5 and more to come.
//! - **Random Access**: Tile offsets enable seeking and reading specific image regions without decompressing the entire file
//!
//! ## Writing Images
//!
//! Use [`write_image`] to encode an image to a stream. Configure compression and tiling behavior
//! through the [`Options`] struct:
//!
//! ```ignore
//! let options = Options::new().with_compression(Compression::Ans);
//! write_image(image, options, file)?;
//! ```
//!
//! ## Reading Images
//!
//! Use [`JackalReader`] to open and read JKLI files. The reader provides:
//! - Header and metadata inspection without full decompression
//! - Tile offset information for random access
//! - A [`JackalTileReader`] for efficient tile-by-tile decoding, requires user to provide generic type parameter for pixel format, which must match the image format.
//!
//! ```ignore
//! let mut reader = JackalReader::open(file)?;
//! if reader.format() == Format::RGB8 {
//!     let mut tile_reader = reader.tile_reader::<Rgb8U>()?;
//!     tile_reader.read_tile(0, tile_buffer)?;
//! }
//! ```
//!
//! ## File Structure
//!
//! A JKLI file consists of:
//! 1. **Header**: File metadata including format, dimensions, compression method, and tile size
//! 2. **Tile Offsets**: Lookup table for random access to individual tiles
//! 3. **Context Data**: Compression context information common for all tiles.
//! 4. **Tile Data**: Compressed or uncompressed pixel data for each tile.
//!
//! ## Error Handling
//!
//! Decoding errors are represented by the [`DecodeError`] enum, which covers validation failures
//! such as invalid magic numbers, unsupported formats, and platform-specific limitations.
//!

use std::{convert::Infallible, fmt, io};

use crate::{
    bits::read_bits_scope,
    encode::{FixedCode, VarCode},
    image::{Extent, Image2DMut, ImageRef},
    jackal::image::{
        compress::{AnsCompressor, LZ77Compressor, RleCompressor},
        format::{Offsets, WriteOffsets},
        header::JackalHeader,
    },
};

pub use self::{
    format::{Format, Pixel},
    header::{Compression, MipLevels},
    tiles::TileSize,
};

mod compress;
mod filter;
mod format;
mod header;
mod tiles;

/// Tile options for image compression.
pub enum TileOptions {
    /// Use fixed tile size for image compression.
    Size(TileSize),

    /// Use optimal tile size for image compression.
    Optimal {
        /// Flat cost factor for tile size cost function.
        flat_cost: f32,

        /// Size cost factor for tile size cost function.
        size_cost: f32,
    },
}

pub struct Options {
    compression: Compression,
    tile_options: TileOptions,
}

impl Default for Options {
    fn default() -> Self {
        Self::new()
    }
}

impl Options {
    /// Creates a new `Options` with default parameters:
    ///
    /// - No compression
    /// - Optimal tile calculation using pre-defined cost factors.
    pub const fn new() -> Self {
        Options {
            tile_options: TileOptions::Optimal {
                flat_cost: 64.0,
                size_cost: 1.0,
            },
            compression: Compression::None,
        }
    }

    pub const fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Configures fixed tile size for the image compression.
    ///
    /// Compression will panic if tile size is not a multiple of block size of pixel format.
    pub fn with_tile_size(mut self, tile_size: TileSize) -> Self {
        self.tile_options = TileOptions::Size(tile_size);
        self
    }

    /// Configures optimal tile size for the image compression.
    ///
    /// The optimal tile size is calculated based on the image extent, block size of pixel format,
    /// and the provided cost factors.
    pub fn with_optimal_tile_size(mut self, flat_cost: f32, size_cost: f32) -> Self {
        self.tile_options = TileOptions::Optimal {
            flat_cost,
            size_cost,
        };
        self
    }
}

/// Encode entire image to the IO stream.
///
/// Uses options to determine the compression method and tile size.
pub fn write_image<T>(
    input: ImageRef<T>,
    options: Options,
    mut write: impl io::Write + io::Seek,
) -> io::Result<()>
where
    T: Pixel,
{
    let extent = input.extent();

    let tile_size = match options.tile_options {
        TileOptions::Optimal {
            flat_cost,
            size_cost,
        } => TileSize::find_optimal(
            extent,
            T::FORMAT.block_width(),
            T::FORMAT.block_height(),
            flat_cost,
            size_cost,
        ),
        TileOptions::Size(size) => size,
    };

    assert!(tile_size.width.is_multiple_of(T::FORMAT.block_width()));
    assert!(tile_size.height.is_multiple_of(T::FORMAT.block_height()));

    let tiles_iter = tile_size.iter_tiles(input);

    let header = JackalHeader {
        compression: options.compression,
        format: T::FORMAT,
        extent,
        tile_size,
        ..JackalHeader::new()
    };

    header.fix_write(&mut write)?;

    match options.compression {
        Compression::None => {
            // Simply write all pixels using fixed code.

            let mut offsets = WriteOffsets::new(tiles_iter.len(), &mut write)?;

            for tile in tiles_iter {
                offsets.push_next(&mut write)?;

                tile.iter_pixels()
                    .try_for_each(|p| p.fix_write(&mut write))?
            }

            offsets.write(&mut write)?;
            Ok(())
        }
        Compression::Lz77 => T::compress_images(tiles_iter, LZ77Compressor::new(), write),
        Compression::Ans => T::compress_images(tiles_iter, AnsCompressor, write),
        Compression::Lz77Ans => {
            T::compress_images(tiles_iter, (LZ77Compressor::new(), AnsCompressor), write)
        }
        Compression::RleAns => {
            T::compress_images(tiles_iter, (RleCompressor, AnsCompressor), write)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DecodeError {
    /// Magic number invalid.
    InvalidMagic,

    /// Compression method is invalid.
    InvalidCompression,

    /// Format is invalid.
    InvalidFormat,

    /// Mip levels count is zero.
    MipZero,

    /// Dimensions are invalid.
    InvalidDimensions,

    /// Extent is invalid.
    InvalidExtent,

    // Data is invalid.
    // Such as position is out of bounds.
    InvalidData,

    /// Numeric values exceed the maximum allowed on current platform.
    /// For example dimensions exceed usize::MAX.
    ///
    /// In theory this may happen if image with dimension 2^32 or larger is created on 64-bit platform,
    /// and then attempted to be read on 32-bit platform, where `usize` can't represent dimensions as large as 2^32.
    TooLarge,
}

impl From<Infallible> for DecodeError {
    fn from(void: Infallible) -> Self {
        match void {}
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::InvalidMagic => write!(f, "Invalid magic number"),
            DecodeError::InvalidCompression => write!(f, "Invalid compression method"),
            DecodeError::InvalidFormat => write!(f, "Invalid format"),
            DecodeError::MipZero => write!(f, "Mip levels count is zero"),
            DecodeError::InvalidDimensions => write!(f, "Invalid dimensions"),
            DecodeError::InvalidExtent => write!(f, "Invalid extent"),
            DecodeError::InvalidData => write!(f, "Invalid data"),
            DecodeError::TooLarge => write!(f, "Numeric value is too large for current platform"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Convenience reader object for reading Jackal Images from a stream.
pub struct JackalReader<R> {
    compression: Compression,
    format: Format,
    extent: Extent,
    tile_size: TileSize,
    offsets: Offsets,
    read: R,
}

enum AnyContext<T: Pixel> {
    None,
    Lz77(T::Context<LZ77Compressor>),
    Ans(T::Context<AnsCompressor>),
    Lz77Ans(T::Context<(LZ77Compressor, AnsCompressor)>),
    RleAns(T::Context<(RleCompressor, AnsCompressor)>),
}

impl<T> AnyContext<T>
where
    T: Pixel,
{
    fn read_for_complression(
        compression: Compression,
        read: &mut impl io::Read,
    ) -> io::Result<Self> {
        read_bits_scope(read, |read| match compression {
            Compression::None => Ok(AnyContext::None),
            Compression::Lz77 => Ok(AnyContext::Lz77(T::Context::var_read(read)?)),
            Compression::Ans => Ok(AnyContext::Ans(T::Context::var_read(read)?)),
            Compression::Lz77Ans => Ok(AnyContext::Lz77Ans(T::Context::var_read(read)?)),
            Compression::RleAns => Ok(AnyContext::RleAns(T::Context::var_read(read)?)),
        })
    }
}

impl<R> JackalReader<R> {
    /// Opens a JackalReader, reads the header and tile offsets from the stream.
    pub fn open(mut read: R) -> io::Result<Self>
    where
        R: io::Read,
    {
        let header = JackalHeader::fix_read(&mut read)?;

        // Read tile offsets.
        let tiles_count = header.tiles_count();
        let offsets = Offsets::read(tiles_count, &mut read)?;

        Ok(JackalReader {
            compression: header.compression,
            format: header.format,
            extent: header.extent,
            tile_size: header.tile_size,
            offsets,
            read,
        })
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn extent(&self) -> Extent {
        self.extent
    }

    pub fn tiles(&self) -> usize {
        self.offsets.slice().len()
    }

    pub fn tile_offsets(&self) -> &[u64] {
        self.offsets.slice()
    }

    pub fn tile_size(&self) -> TileSize {
        self.tile_size
    }

    pub fn tile_pos(&self, tile_index: usize) -> [usize; 3] {
        self.tile_size.tile_pos(self.extent, tile_index)
    }

    pub fn pixel_reader<T>(&mut self) -> io::Result<JackalTileReader<'_, R, T>>
    where
        T: Pixel,
        R: io::Read + io::Seek,
    {
        assert_eq!(
            self.format,
            T::FORMAT,
            "Pixel type format does not match image format"
        );

        let context_pos = JackalHeader::SIZE + self.offsets.bytes_size();
        let context_pos = u64::try_from(context_pos)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, DecodeError::TooLarge))?;

        self.read.seek(io::SeekFrom::Start(context_pos))?;
        let context = AnyContext::read_for_complression(self.compression, &mut self.read)?;

        Ok(JackalTileReader {
            context,
            extent: self.extent,
            tile_size: self.tile_size,
            offsets: self.offsets.slice(),
            read: &mut self.read,
        })
    }
}

pub struct JackalTileReader<'a, R, T: Pixel> {
    context: AnyContext<T>,
    extent: Extent,
    tile_size: TileSize,
    offsets: &'a [u64],
    read: &'a mut R,
}

impl<'a, R, T> JackalTileReader<'a, R, T>
where
    T: Pixel,
    R: io::Read + io::Seek,
{
    pub fn extent(&self) -> Extent {
        self.extent
    }

    pub fn tiles(&self) -> usize {
        self.offsets.len()
    }

    pub fn tile_offsets(&self) -> &[u64] {
        self.offsets
    }

    pub fn tile_size(&self) -> TileSize {
        self.tile_size
    }

    pub fn tile_pos(&self, tile_index: usize) -> [usize; 3] {
        self.tile_size.tile_pos(self.extent, tile_index)
    }

    pub fn read_tile(&mut self, tile_index: usize, mut image: Image2DMut<'_, T>) -> io::Result<()> {
        assert!(tile_index < self.offsets.len(), "Tile index out of bounds");
        assert_eq!(
            usize::from(self.tile_size.width),
            image.width(),
            "Tile width mismatch"
        );

        assert_eq!(
            usize::from(self.tile_size.height),
            image.height(),
            "Tile height mismatch"
        );

        self.read
            .seek(io::SeekFrom::Start(self.offsets[tile_index]))?;

        match &self.context {
            AnyContext::None => {
                for pixel in image.iter_pixels_mut() {
                    *pixel = T::fix_read(&mut *self.read)?;
                }
                Ok(())
            }
            AnyContext::Lz77(context) => {
                T::decompress_image(LZ77Compressor::new(), context, &mut *self.read, image)
            }
            AnyContext::Ans(context) => {
                T::decompress_image(AnsCompressor, context, &mut *self.read, image)
            }
            AnyContext::Lz77Ans(context) => T::decompress_image(
                (LZ77Compressor::new(), AnsCompressor),
                context,
                &mut *self.read,
                image,
            ),
            AnyContext::RleAns(context) => T::decompress_image(
                (RleCompressor, AnsCompressor),
                context,
                &mut *self.read,
                image,
            ),
        }
    }
}

#[test]
fn jkli_smoke_test_rgb() {
    use crate::math::Rgb8U;

    let extent = Extent::D2 {
        width: 4,
        height: 4,
    };

    let pixels = [
        Rgb8U::RED,
        Rgb8U::GREEN,
        Rgb8U::BLUE,
        Rgb8U::WHITE,
        Rgb8U::BLACK,
        Rgb8U::RED,
        Rgb8U::BLACK,
        Rgb8U::WHITE,
        Rgb8U::GREEN,
        Rgb8U::BLUE,
        Rgb8U::WHITE,
        Rgb8U::BLACK,
        Rgb8U::BLUE,
        Rgb8U::WHITE,
        Rgb8U::BLACK,
        Rgb8U::RED,
    ];

    // let pixels = [Rgb8U::WHITE; 16];

    let image = ImageRef::new(crate::image::Dimensions::D2, [4, 4, 1], &pixels);

    let mut buffer = Vec::new();

    write_image(
        image,
        Options::new().with_compression(Compression::Ans),
        std::io::Cursor::new(&mut buffer),
    )
    .unwrap();

    let mut reader = JackalReader::open(std::io::Cursor::new(&buffer[..])).unwrap();

    assert_eq!(reader.format(), Format::RGB8);

    let mut reader = reader.pixel_reader::<Rgb8U>().unwrap();

    assert_eq!(reader.extent(), extent);

    let mut decoded_pixels = [Rgb8U::BLACK; 16];
    let mut decoded_image = Image2DMut::new(4, 4, &mut decoded_pixels);

    for tile_index in 0..reader.tiles() {
        let [x, y, z] = reader.tile_pos(tile_index);
        assert_eq!(z, 0);

        let decoded_tile = decoded_image.get_range_mut(
            x,
            y,
            usize::from(reader.tile_size().width),
            usize::from(reader.tile_size().height),
        );

        reader.read_tile(tile_index, decoded_tile).unwrap();
    }

    assert_eq!(decoded_pixels, pixels);
}

#[test]
fn jkli_smoke_test_bc1() {
    use crate::bc1::Block;

    let extent = Extent::D2 {
        width: 4,
        height: 4,
    };

    let blocks = [
        Block::BLACK,
        Block::TRANSPARENT,
        Block::WHITE,
        Block::WHITE,
        Block::BLACK,
        Block::BLACK,
        Block::BLACK,
        Block::WHITE,
        Block::TRANSPARENT,
        Block::WHITE,
        Block::WHITE,
        Block::BLACK,
        Block::TRANSPARENT,
        Block::WHITE,
        Block::BLACK,
        Block::BLACK,
    ];

    let image = ImageRef::new(crate::image::Dimensions::D2, [4, 4, 1], &blocks);

    let mut buffer = Vec::new();

    write_image(
        image,
        Options::new().with_compression(Compression::Ans),
        std::io::Cursor::new(&mut buffer),
    )
    .unwrap();

    let mut reader = JackalReader::open(std::io::Cursor::new(&buffer[..])).unwrap();

    assert_eq!(reader.format(), Format::BC1);

    let mut reader = reader.pixel_reader::<Block>().unwrap();

    assert_eq!(reader.extent(), extent);

    let mut decoded_blocks = [Block::BLACK; 16];
    let mut decoded_image = Image2DMut::new(4, 4, &mut decoded_blocks);

    for tile_index in 0..reader.tiles() {
        let [x, y, z] = reader.tile_pos(tile_index);
        assert_eq!(z, 0);

        let decoded_tile = decoded_image.get_range_mut(
            x,
            y,
            usize::from(reader.tile_size().width),
            usize::from(reader.tile_size().height),
        );

        reader.read_tile(tile_index, decoded_tile).unwrap();
    }

    assert_eq!(decoded_blocks, blocks);
}
