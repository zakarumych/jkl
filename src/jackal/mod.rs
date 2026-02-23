// Jackal compression format.
//
// It is hybrid compression algorithm designed to work on blocks that have
// color data and indices.
// Color data is compressed using combination of run-length, hash and diff encoding.
// Indices are compressed by LZW algorithm with parameters predefined for each block format.
//
// Jackal format compresses super-blocks (blocks of blocks) independently.
// This allows parallel processing of super-blocks on multi-core CPU and GPU.
// Although small textures may have just one super-block.

use std::{fmt, io};

use crate::encode::FixedCode;

pub use self::header::{Extent, Format, JackalBlock, JackalHeader, MipLevels, SuperBlockSize};

mod block;
mod header;

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

#[derive(Debug)]
pub enum DecompressError {
    Io(std::io::Error),
    Decode(DecodeError),
}

impl From<std::io::Error> for DecompressError {
    #[inline(always)]
    fn from(err: std::io::Error) -> Self {
        DecompressError::Io(err)
    }
}

impl From<DecodeError> for DecompressError {
    #[inline(always)]
    fn from(err: DecodeError) -> Self {
        DecompressError::Decode(err)
    }
}

/// Read Jackal header from the stream.
pub fn read_header(mut read: impl io::Read) -> Result<JackalHeader, DecompressError> {
    let mut bytes = [0; JackalHeader::SIZE];
    read.read_exact(&mut bytes)?;
    let header = JackalHeader::decode(&bytes)?;
    Ok(header)
}

/// Read super-blocks from the stream.
pub fn read_jackal_blocks(
    jackal_blocks: &mut [JackalBlock],
    mut read: impl io::Read,
) -> Result<(), DecompressError> {
    let mut buffer = [0; JackalBlock::SIZE];
    for block in jackal_blocks.iter_mut() {
        read.read_exact(&mut buffer)?;
        *block = JackalBlock::decode(&buffer).unwrap();
    }
    Ok(())
}

/// Visits blocks of one superblock.
///
/// `header` is Jackal header.
/// `super_pos` is coordinate of the superblock.
/// `visit` is a visitor function that will be called for each block of the superblock.
fn visit_superblock<B, E>(
    header: &JackalHeader,
    super_pos: [u32; 3],
    mut visit: impl FnMut(usize) -> Result<(), E>,
) -> Result<(), E> {
    let raw_size = header.extent.raw_size();

    let x_start = super_pos[0] * header.superblock_size.width as u32;
    let x_end = if raw_size[0] - x_start < header.superblock_size.width as u32 {
        raw_size[0]
    } else {
        x_start + header.superblock_size.width as u32
    };

    let y_start = super_pos[1] * header.superblock_size.height as u32;
    let y_end = if raw_size[1] - y_start < header.superblock_size.height as u32 {
        raw_size[1]
    } else {
        y_start + header.superblock_size.height as u32
    };

    let z = super_pos[2];

    let width = x_end - x_start;
    let height = y_end - y_start;

    debug_assert!(width <= u16::MAX as u32);
    debug_assert!(height <= u16::MAX as u32);

    // let bound_curve = BoundZCurve::new(width as u16, height as u16);
    let bound_curve = (0..height * width).map(|index| {
        let x = index % width;
        let y = index / width;
        (x, y)
    });

    for (x0, y0) in bound_curve {
        let x = x_start + x0 as u32;
        let y = y_start + y0 as u32;
        let width = raw_size[0] as usize;
        let height = raw_size[1] as usize;
        let index = x as usize + y as usize * width + z as usize * width * height;
        visit(index)?;
    }

    Ok(())
}
