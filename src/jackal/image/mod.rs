//! Jackal image file format.

use std::{convert::Infallible, fmt, io};

use crate::{
    encode::{FixedCode, VarCode},
    image::{Image, ImageRef},
    jackal::image::{compress::Symbol, format::AnyFormat, header::Extent},
    math::{Rgb8U, R8U},
};

pub use self::{
    compress::LZ77Compressor,
    header::{Compression, Format, JackalBlock, JackalHeader, MipLevels, TileSize},
};

mod compress;
mod filter;
mod format;
mod header;

pub(super) trait Pixel: Symbol + FixedCode + AnyFormat {}

fn image_extent(input: ImageRef<impl Pixel>) -> Extent {
    match input {
        ImageRef::D1(input) => Extent::D1 {
            width: u64::try_from(input.len()).expect("Image length exceeds u64::MAX"),
        },
        ImageRef::D2(input) => Extent::D2 {
            width: u64::try_from(input.width()).expect("Image width exceeds u64::MAX"),
            height: u64::try_from(input.height()).expect("Image height exceeds u64::MAX"),
        },
        ImageRef::D3(input) => Extent::D3 {
            width: u64::try_from(input.width()).expect("Image width exceeds u64::MAX"),
            height: u64::try_from(input.height()).expect("Image height exceeds u64::MAX"),
            depth: u64::try_from(input.depth()).expect("Image depth exceeds u64::MAX"),
        },
    }
}

pub struct Config {
    pub flat_cost: f32,
    pub size_cost: f32,
}

pub(super) fn encode_image<T>(
    input: ImageRef<T>,
    compression: Compression,
    config: &Config,
    mut write: impl io::Write + io::Seek,
) -> io::Result<()>
where
    T: Pixel,
{
    let extent = image_extent(input);
    let header = JackalHeader {
        compression,
        format: T::FORMAT,
        extent,
        ..JackalHeader::new()
    };

    match compression {
        Compression::None => match input {
            ImageRef::D1(input) => {
                for pixel in input {
                    pixel.fix_write(&mut write)?;
                }
                Ok(())
            }
            ImageRef::D2(input) => {
                for pixel in input.iter_pixels() {
                    pixel.fix_write(&mut write)?;
                }
                Ok(())
            }
            ImageRef::D3(input) => {
                for pixel in input.iter_pixels() {
                    pixel.fix_write(&mut write)?;
                }
                Ok(())
            }
        },
        Compression::Lz77 => {
            let tile_size = TileSize::find_optimal(extent, 1, config.flat_cost, config.size_cost);
            let tiles_iter = tile_size.iter_tiles(input);

            JackalHeader {
                tile_size,
                ..header
            }
            .fix_write(&mut write)?;

            let compressor = LZ77Compressor { window_size: 1024 };
            T::compress_images(tiles_iter, compressor, write)
        }
        _ => unimplemented!(),
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
        }
    }
}

impl std::error::Error for DecodeError {}

/// Read Jackal header from the stream.
pub fn read_header(mut read: impl io::Read) -> io::Result<JackalHeader> {
    let mut bytes = [0; JackalHeader::SIZE];
    read.read_exact(&mut bytes)?;
    JackalHeader::fix_decode(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Read super-blocks from the stream.
pub fn read_jackal_blocks(
    mut read: impl io::Read,
    jackal_blocks: &mut [JackalBlock],
) -> io::Result<()> {
    let mut buffer = [0; JackalBlock::SIZE];
    for block in jackal_blocks.iter_mut() {
        read.read_exact(&mut buffer)?;
        *block = JackalBlock::fix_decode(&buffer)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    }
    Ok(())
}
