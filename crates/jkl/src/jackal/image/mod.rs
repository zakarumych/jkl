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
//! Use [`JackalImageReader`] to open and read JKLI files. The reader provides:
//! - Header and metadata inspection without full decompression
//! - Tile offset information for random access
//! - A [`JackalTileReader`] for efficient tile-by-tile decoding, requires user to provide generic type parameter for pixel format, which must match the image format.
//!
//! ```ignore
//! let mut reader = JackalImageReader::open(file)?;
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

use smallvec::SmallVec;

use crate::{
    bits::read_bits_scope,
    encode::{FixedCode, VarCode},
    image::{
        Extent, Image2DMut, ImageRef, InvalidExtent,
        compress::{AnsCompressor, LZ77Compressor, RleCompressor},
        format::{Format, InvalidFormat},
        tiles::{Tile, TileSize},
    },
    jackal::InvalidMagic,
};

use self::header::JackalImageHeader;

pub use self::{format::Pixel, header::Compression};

mod format;
mod header;

/// Array of tile offsets in the Jackal file,
/// provides population, reading and writing of tile offsets,
/// to ensure that implementation is consistent across the codebase.
pub(crate) struct Offsets {
    array: SmallVec<[u64; 16]>,
}

impl Offsets {
    pub fn read(len: usize, mut read: impl io::Read) -> io::Result<Self> {
        let mut array = SmallVec::with_capacity(len);
        for _ in 0..len {
            let offset = u64::fix_read(&mut read)?;
            array.push(offset);
        }
        Ok(Offsets { array })
    }

    pub fn slice(&self) -> &[u64] {
        &self.array
    }

    pub fn bytes_size(&self) -> usize {
        self.array.len() * <u64 as FixedCode>::SIZE
    }
}

pub(crate) struct WriteOffsets {
    offsets_start: u64,
    offsets: Offsets,
}

impl WriteOffsets {
    /// Creates a new Offsets object, reserves space for `len` offsets,
    /// and seeks the output stream to the end of the reserved space.
    pub fn new<W>(len: usize, write: &mut W) -> io::Result<Self>
    where
        W: io::Write + io::Seek,
    {
        let offsets_start = write.stream_position()?;
        let offsets_len = u64::try_from(<u64 as FixedCode>::SIZE * len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Too many tiles"))?; // 8 is the size of u64 in bytes
        let offsets_end = offsets_start + offsets_len;

        write.seek(io::SeekFrom::Start(offsets_end))?;

        Ok(WriteOffsets {
            offsets_start,
            offsets: Offsets {
                array: SmallVec::with_capacity(len),
            },
        })
    }

    pub fn push_next<W>(&mut self, write: &mut W) -> io::Result<()>
    where
        W: io::Seek,
    {
        let offset = write.stream_position()?;
        self.offsets.array.push(offset);
        Ok(())
    }

    /// Writes the offsets to the output stream at the reserved space.
    pub fn write<W>(&self, write: &mut W) -> io::Result<()>
    where
        W: io::Write + io::Seek,
    {
        write.seek(io::SeekFrom::Start(self.offsets_start))?;

        for offset in &self.offsets.array {
            offset.fix_write(write)?;
        }
        Ok(())
    }
}

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
    pub compression: Compression,
    pub tile_options: TileOptions,
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

pub(crate) fn tile_size(tile_options: TileOptions, extent: Extent, format: Format) -> TileSize {
    match tile_options {
        TileOptions::Optimal {
            flat_cost,
            size_cost,
        } => TileSize::find_optimal(
            extent,
            format.tile_width_granularity(),
            format.tile_height_granularity(),
            flat_cost,
            size_cost,
        ),
        TileOptions::Size(size) => size,
    }
}

pub(crate) fn write_tiles<T>(
    input: ImageRef<T>,
    compression: Compression,
    tile_size: TileSize,
    mut write: impl io::Write + io::Seek,
) -> io::Result<()>
where
    T: Pixel,
{
    let tiles_iter = tile_size.iter_tiles(input.extent()).map(|tile| {
        input
            .get_plane(tile.plane)
            .get_range(tile.rect.x, tile.rect.y, tile.rect.w, tile.rect.h)
    });

    match compression {
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

    let tile_size = tile_size(options.tile_options, extent, T::FORMAT);

    let header = JackalImageHeader::new(options.compression, T::FORMAT, extent, 1, tile_size);

    header.fix_write(&mut write)?;

    write_tiles(input, options.compression, tile_size, write)
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

impl From<InvalidMagic> for DecodeError {
    fn from(_: InvalidMagic) -> Self {
        DecodeError::InvalidMagic
    }
}

impl From<InvalidFormat> for DecodeError {
    fn from(_: InvalidFormat) -> Self {
        DecodeError::InvalidFormat
    }
}

impl From<InvalidExtent> for DecodeError {
    fn from(error: InvalidExtent) -> Self {
        match error {
            InvalidExtent::InvalidDimensions => DecodeError::InvalidDimensions,
            InvalidExtent::TooLarge => DecodeError::TooLarge,
        }
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
pub struct JackalImageReader<R> {
    compression: Compression,
    format: Format,
    extent: Extent,
    tile_size: TileSize,
    offsets: Offsets,
    read: R,
    end: u64,
}

/// Concatenated compressed tile payloads with per-tile byte offsets.
///
/// `offsets` always has `tiles + 1` items.
/// Tile `i` occupies byte range `offsets[i]..offsets[i + 1]` in `payload`.
pub struct TilePayloadBlob {
    pub payload: Vec<u8>,
    pub offsets: Vec<u64>,
}

/// Compact ANS symbol tables for RGB8 GPU decompression.
pub struct Rgb8AnsGpuContext {
    pub symbol_cumul: Vec<u32>,
    pub symbol_freq: Vec<u32>,
    /// Packed as 0x00RRGGBB.
    pub symbol_rgb8: Vec<u32>,
}

impl TilePayloadBlob {
    pub fn tiles(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn tile_range(&self, tile_index: usize) -> std::ops::Range<usize> {
        assert!(tile_index < self.tiles(), "Tile index out of bounds");

        let start = usize::try_from(self.offsets[tile_index])
            .expect("tile offset exceeds usize::MAX on this platform");
        let end = usize::try_from(self.offsets[tile_index + 1])
            .expect("tile offset exceeds usize::MAX on this platform");
        start..end
    }
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
    fn read_for_compression(
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

impl<R> JackalImageReader<R> {
    /// Opens a JackalImageReader, reads the header and tile offsets from the stream.
    pub fn open(mut read: R) -> io::Result<Self>
    where
        R: io::Read + io::Seek,
    {
        let end = read.seek(io::SeekFrom::End(0))?;
        read.seek(io::SeekFrom::Start(0))?;

        let header = JackalImageHeader::fix_read(&mut read)?;

        // Read tile offsets.
        let tiles_count = header.tile_size.tiles_count(header.extent);
        let offsets = Offsets::read(tiles_count, &mut read)?;

        Ok(JackalImageReader {
            compression: header.compression,
            format: header.format,
            extent: header.extent,
            tile_size: header.tile_size,
            offsets,
            read,
            end,
        })
    }

    pub(crate) fn new(
        compression: Compression,
        format: Format,
        extent: Extent,
        tile_size: TileSize,
        offsets: Offsets,
        read: R,
        end: u64,
    ) -> Self {
        JackalImageReader {
            compression,
            format,
            extent,
            tile_size,
            offsets,
            read,
            end,
        }
    }

    #[inline]
    pub(crate) fn reader(&mut self) -> &mut R {
        &mut self.read
    }

    /// Returns image format.
    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns compression method.
    #[inline]
    pub fn compression(&self) -> Compression {
        self.compression
    }

    /// Returns image extent.
    #[inline]
    pub fn extent(&self) -> Extent {
        self.extent
    }

    /// Returns number of tiles in the image.
    #[inline]
    pub fn tiles(&self) -> usize {
        self.offsets.slice().len()
    }

    /// Returns slice of tile offsets.
    #[inline]
    pub fn tile_offsets(&self) -> &[u64] {
        self.offsets.slice()
    }

    /// Returns tile size.
    #[inline]
    pub fn tile_size(&self) -> TileSize {
        self.tile_size
    }

    /// Returns tile information for the given tile index.
    #[inline]
    pub fn tile(&self, tile_index: usize) -> Tile {
        self.tile_size.tile(self.extent, tile_index)
    }

    /// Returns an object for reading individual tiles from the image.
    pub fn tile_reader<T>(&mut self) -> io::Result<JackalTileReader<'_, R, T>>
    where
        T: Pixel,
        R: io::Read + io::Seek,
    {
        assert_eq!(
            self.format,
            T::FORMAT,
            "Pixel type format does not match image format"
        );

        let context_pos = JackalImageHeader::SIZE + self.offsets.bytes_size();
        let context_pos = u64::try_from(context_pos)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, DecodeError::TooLarge))?;

        self.read.seek(io::SeekFrom::Start(context_pos))?;
        let context = AnyContext::read_for_compression(self.compression, &mut self.read)?;

        Ok(JackalTileReader {
            context,
            extent: self.extent,
            tile_size: self.tile_size,
            offsets: self.offsets.slice(),
            read: &mut self.read,
        })
    }

    /// Returns payload length of a tile.
    pub fn tile_payload_len(&self, tile_index: usize) -> u64 {
        assert!(
            tile_index < self.offsets.slice().len(),
            "Tile index out of bounds"
        );

        let start = self.offsets.slice()[tile_index];
        let end = if tile_index + 1 < self.offsets.slice().len() {
            self.offsets.slice()[tile_index + 1]
        } else {
            self.end
        };

        end - start
    }

    /// Returns payload length of all tiles combined
    #[inline]
    pub fn tiles_payload_len(&self) -> u64 {
        match self.offsets.slice().first() {
            None => 0,
            Some(&start) => self.end - start,
        }
    }

    /// Copies compressed payload of the tile into destination buffer.
    pub fn copy_tile_payload_into(&mut self, tile_index: usize, dst: &mut [u8]) -> io::Result<()>
    where
        R: io::Read + io::Seek,
    {
        assert!(
            tile_index < self.offsets.slice().len(),
            "Tile index out of bounds"
        );

        let start = self.offsets.slice()[tile_index];
        let end = if tile_index + 1 < self.offsets.slice().len() {
            self.offsets.slice()[tile_index + 1]
        } else {
            self.end
        };

        let len = end - start;

        match usize::try_from(len) {
            Ok(len) if len == dst.len() => {
                self.read.seek(io::SeekFrom::Start(start))?;
                self.read.read_exact(dst)
            }
            _ => {
                panic!(
                    "Tile payload length mismatch: expected {}, got {}",
                    dst.len(),
                    len
                );
            }
        }
    }

    /// Returns a reader for the context portion of the image.
    pub fn context_reader(&mut self) -> io::Result<impl io::Read + '_>
    where
        R: io::Read + io::Seek,
    {
        let start = JackalImageHeader::SIZE + self.offsets.bytes_size();
        let start = u64::try_from(start)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, DecodeError::TooLarge))?;

        let end = self.offsets.slice().first().copied().unwrap_or(self.end);

        self.read.seek(io::SeekFrom::Start(start))?;

        let len = end - start;
        let read_context = <&mut R as io::Read>::take(self.read.by_ref(), len);
        Ok(read_context)
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

    pub fn tile(&self, tile_index: usize) -> Tile {
        self.tile_size.tile(self.extent, tile_index)
    }

    pub fn read_tile(&mut self, tile_index: usize, mut image: Image2DMut<'_, T>) -> io::Result<()> {
        assert!(tile_index < self.offsets.len(), "Tile index out of bounds");
        assert!(
            image.width() <= usize::from(self.tile_size.width),
            "Tile width exceeds configured tile size"
        );

        assert!(
            image.height() <= usize::from(self.tile_size.height),
            "Tile height exceeds configured tile size"
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

    let mut reader = JackalImageReader::open(std::io::Cursor::new(&buffer[..])).unwrap();

    assert_eq!(reader.format(), Format::RGB8);

    let mut reader = reader.tile_reader::<Rgb8U>().unwrap();

    assert_eq!(reader.extent(), extent);

    let mut decoded_pixels = [Rgb8U::BLACK; 16];
    let mut decoded_image = Image2DMut::new(4, 4, &mut decoded_pixels);

    for tile_index in 0..reader.tiles() {
        let tile = reader.tile(tile_index);
        assert_eq!(tile.plane, 0);

        let decoded_tile = decoded_image.get_rect_mut(tile.rect);

        reader.read_tile(tile_index, decoded_tile).unwrap();
    }

    assert_eq!(decoded_pixels, pixels);
}

#[test]
fn jkli_smoke_test_bc1() {
    use crate::image::block::bc1::Block;

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

    let mut reader = JackalImageReader::open(std::io::Cursor::new(&buffer[..])).unwrap();

    assert_eq!(reader.format(), Format::BC1);

    let mut reader = reader.tile_reader::<Block>().unwrap();

    assert_eq!(reader.extent(), extent);

    let mut decoded_blocks = [Block::BLACK; 16];
    let mut decoded_image = Image2DMut::new(4, 4, &mut decoded_blocks);

    for tile_index in 0..reader.tiles() {
        let tile = reader.tile(tile_index);
        assert_eq!(tile.plane, 0);

        let decoded_tile = decoded_image.get_rect_mut(tile.rect);

        reader.read_tile(tile_index, decoded_tile).unwrap();
    }

    assert_eq!(decoded_blocks, blocks);
}
