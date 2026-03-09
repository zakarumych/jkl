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
    image::{
        Extent, Image2DMut, ImageRef,
        compress::{AnsCompressor, LZ77Compressor, RleCompressor},
        format::Format,
        tiles::{Tile, TileSize},
    },
    math::Rgb8U,
};

use self::{
    format::{Offsets, Pixel, WriteOffsets},
    header::JackalHeader,
};

pub use self::header::Compression;

mod format;
mod header;

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

    let tiles_iter = tile_size.iter_tiles(extent).map(|tile| {
        input
            .plane_ref(tile.plane)
            .get_range(tile.rect.x, tile.rect.y, tile.rect.w, tile.rect.h)
    });

    let header = JackalHeader::new(options.compression, T::FORMAT, extent, 1, tile_size);

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
    end: u64,
}

#[derive(Clone)]
struct TilePayloadLayout {
    payload_start: u64,
    payload_len: usize,
    offsets: Vec<u64>,
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
    fn tile_payload_layout(&self) -> io::Result<TilePayloadLayout> {
        let file_offsets = self.offsets.slice();

        if file_offsets.is_empty() {
            return Ok(TilePayloadLayout {
                payload_start: 0,
                payload_len: 0,
                offsets: vec![0],
            });
        }

        let start = file_offsets[0];
        if start > self.end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                DecodeError::InvalidData,
            ));
        }

        let mut rebased_offsets = Vec::with_capacity(file_offsets.len() + 1);
        let mut prev = start;

        for &offset in file_offsets {
            // Tile offsets must be monotonic and lie inside file bounds.
            if offset < prev || offset > self.end {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    DecodeError::InvalidData,
                ));
            }

            rebased_offsets.push(offset - start);
            prev = offset;
        }

        let payload_len_u64 = self.end - start;
        let payload_len = usize::try_from(payload_len_u64)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, DecodeError::TooLarge))?;

        rebased_offsets.push(payload_len_u64);

        Ok(TilePayloadLayout {
            payload_start: start,
            payload_len,
            offsets: rebased_offsets,
        })
    }

    /// Returns total payload size in bytes for all tiles.
    ///
    /// The value is cached inside the reader after first query.
    pub fn tile_payload_len_bytes(&mut self) -> io::Result<usize>
    where
        R: io::Read + io::Seek,
    {
        if self.offsets.slice().is_empty() {
            return Ok(0);
        }
        let start = self.offsets.slice()[0];

        let payload_len_u64 = self.end - start;
        let payload_len = usize::try_from(payload_len_u64)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, DecodeError::TooLarge))?;

        Ok(payload_len)
    }

    /// Opens a JackalReader, reads the header and tile offsets from the stream.
    pub fn open(mut read: R) -> io::Result<Self>
    where
        R: io::Read + io::Seek,
    {
        let end = read.seek(io::SeekFrom::End(0))?;
        read.seek(io::SeekFrom::Start(0))?;

        let header = JackalHeader::fix_read(&mut read)?;

        // Read tile offsets.
        let tiles_count = header.tiles_count();
        let offsets = Offsets::read(tiles_count, &mut read)?;

        Ok(JackalReader {
            compression: header.compression(),
            format: header.format(),
            extent: header.extent(),
            tile_size: header.tile_size(),
            offsets,
            read,
            end,
        })
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn compression(&self) -> Compression {
        self.compression
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

    pub fn tile(&self, tile_index: usize) -> Tile {
        self.tile_size.tile(self.extent, tile_index)
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

    /// Reads compressed payload bytes for all tiles at once.
    ///
    /// Returned offsets are rebased to the returned payload slice and include
    /// one extra sentinel offset at the end.
    pub fn read_all_tile_payloads(&mut self) -> io::Result<TilePayloadBlob>
    where
        R: io::Read + io::Seek,
    {
        let restore_pos = self.read.stream_position()?;

        let result = (|| {
            let layout = self.tile_payload_layout()?;

            let mut payload = vec![0; layout.payload_len];
            if layout.payload_len > 0 {
                self.read.seek(io::SeekFrom::Start(layout.payload_start))?;
                self.read.read_exact(&mut payload)?;
            }

            Ok(TilePayloadBlob {
                payload,
                offsets: layout.offsets,
            })
        })();

        self.read.seek(io::SeekFrom::Start(restore_pos))?;
        result
    }

    /// Reads all tile payload bytes into caller-provided destination memory.
    ///
    /// This is intended for zero-copy staging into pre-mapped GPU buffers.
    /// Returned offsets are rebased to `dst` and include an end sentinel.
    pub fn read_all_tile_payloads_into(&mut self, dst: &mut [u8]) -> io::Result<Vec<u64>>
    where
        R: io::Read + io::Seek,
    {
        let restore_pos = self.read.stream_position()?;

        let result = (|| {
            let layout = self.tile_payload_layout()?;

            if dst.len() < layout.payload_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Destination buffer is too small for tile payload",
                ));
            }

            if layout.payload_len > 0 {
                self.read.seek(io::SeekFrom::Start(layout.payload_start))?;
                self.read.read_exact(&mut dst[..layout.payload_len])?;
            }

            Ok(layout.offsets)
        })();

        self.read.seek(io::SeekFrom::Start(restore_pos))?;
        result
    }

    /// Reads RGB8 + ANS context and converts it into compact GPU symbol tables.
    pub fn read_rgb8_ans_gpu_context(&mut self) -> io::Result<Rgb8AnsGpuContext>
    where
        R: io::Read + io::Seek,
    {
        if self.format != Format::RGB8 || self.compression != Compression::Ans {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Expected RGB8 image with ANS compression",
            ));
        }

        let restore_pos = self.read.stream_position()?;

        let result = (|| {
            let context_pos = JackalHeader::SIZE + self.offsets.bytes_size();
            let context_pos = u64::try_from(context_pos)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, DecodeError::TooLarge))?;

            self.read.seek(io::SeekFrom::Start(context_pos))?;

            let context =
                match AnyContext::<Rgb8U>::read_for_complression(self.compression, &mut self.read)?
                {
                    AnyContext::Ans(context) => context,
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "Expected ANS context",
                        ));
                    }
                };

            let mut freqs = context.freqs().collect::<Vec<_>>();
            freqs.sort_unstable_by_key(|(symbol, _)| *symbol);

            let mut symbol_cumul = Vec::with_capacity(freqs.len());
            let mut symbol_freq = Vec::with_capacity(freqs.len());
            let mut symbol_rgb8 = Vec::with_capacity(freqs.len());

            let mut cumul = 0u32;

            for (symbol, freq) in freqs {
                let cumul_u32 = u32::try_from(cumul).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, DecodeError::TooLarge)
                })?;
                let freq_u32 = u32::try_from(freq).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, DecodeError::TooLarge)
                })?;

                symbol_cumul.push(cumul_u32);
                symbol_freq.push(freq_u32);

                let rgb = Rgb8U::from_bits_interleaved(symbol.0);
                let packed =
                    (u32::from(rgb.r()) << 16) | (u32::from(rgb.g()) << 8) | u32::from(rgb.b());
                symbol_rgb8.push(packed);

                cumul = cumul.checked_add(freq.get()).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, DecodeError::TooLarge)
                })?;
            }

            Ok(Rgb8AnsGpuContext {
                symbol_cumul,
                symbol_freq,
                symbol_rgb8,
            })
        })();

        self.read.seek(io::SeekFrom::Start(restore_pos))?;
        result
    }
}

const RGB8U_RANS_WGSL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/rgb8u_rans.wgsl"
));

/// Returns a WGSL decompression kernel for selected image format and compression.
///
/// Currently available combination:
/// - `Format::RGB8` + `Compression::Ans` (rANS)
pub fn decompress_wgsl_kernel(format: Format, compression: Compression) -> &'static str {
    match (format, compression) {
        (Format::RGB8, Compression::Ans) => RGB8U_RANS_WGSL,
        _ => {
            unimplemented!(
                "WGSL kernel is not implemented for format {format:?} and compression {compression:?}"
            )
        }
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

    let mut reader = JackalReader::open(std::io::Cursor::new(&buffer[..])).unwrap();

    assert_eq!(reader.format(), Format::RGB8);

    let mut reader = reader.pixel_reader::<Rgb8U>().unwrap();

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

    let mut reader = JackalReader::open(std::io::Cursor::new(&buffer[..])).unwrap();

    assert_eq!(reader.format(), Format::BC1);

    let mut reader = reader.pixel_reader::<Block>().unwrap();

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

#[test]
fn wgsl_kernel_exists_for_rgb8_ans() {
    let kernel = decompress_wgsl_kernel(Format::RGB8, Compression::Ans);
    assert!(!kernel.is_empty());
}
