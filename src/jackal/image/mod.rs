//! Jackal image file format.

use std::{convert::Infallible, fmt, io};

use crate::{
    bits::read_bits_scope,
    encode::{FixedCode, VarCode},
    image::{Extent, Image2DMut, ImageRef},
    jackal::image::{
        compress::{AnsCompressor, LZ77Compressor, RleCompressor},
        format::Offsets,
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

pub struct Options {
    flat_cost: f32,
    size_cost: f32,
    compression: Compression,
}

impl Options {
    pub const fn new() -> Self {
        Options {
            flat_cost: 64.0,
            size_cost: 1.0,
            compression: Compression::None,
        }
    }

    pub const fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
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

    let tile_size = TileSize::find_optimal(
        extent,
        T::FORMAT.block_width(),
        T::FORMAT.block_height(),
        options.flat_cost,
        options.size_cost,
    );
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
            input
                .iter_pixels()
                .try_for_each(|p| p.fix_write(&mut write))
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

/// Convenience reader object for reading Jackal images from a stream.
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
            Compression::Lz77 => Ok(AnyContext::Lz77(T::Context::<LZ77Compressor>::var_read(
                read,
            )?)),
            Compression::Ans => Ok(AnyContext::Ans(T::Context::<AnsCompressor>::var_read(
                read,
            )?)),
            _ => unimplemented!(),
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

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn pixel_reader<T>(&mut self) -> io::Result<JackalPixelReader<'_, R, T>>
    where
        T: Pixel,
        R: io::Read,
    {
        assert_eq!(
            self.format,
            T::FORMAT,
            "Pixel type format does not match image format"
        );

        let context = AnyContext::read_for_complression(self.compression, &mut self.read)?;

        Ok(JackalPixelReader {
            context,
            extent: self.extent,
            tile_size: self.tile_size,
            offsets: self.offsets.slice(),
            read: &mut self.read,
        })
    }
}

pub struct JackalPixelReader<'a, R, T: Pixel> {
    context: AnyContext<T>,
    extent: Extent,
    tile_size: TileSize,
    offsets: &'a [u64],
    read: &'a mut R,
}

impl<'a, R, T> JackalPixelReader<'a, R, T>
where
    T: Pixel,
    R: io::Read + io::Seek,
{
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

    pub fn read_tile(&mut self, tile_index: usize, mut image: Image2DMut<'a, T>) -> io::Result<()> {
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
        }
    }
}
