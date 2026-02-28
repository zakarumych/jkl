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

pub use self::compress::{AnsCompressor, LZ77Compressor, RleCompressor};
pub use self::header::{Compression, Extent, Format, JackalBlock, JackalHeader, MipLevels, SuperBlockSize};

mod compress;
mod format;
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

/// Configuration for encoding a texture with Jackal compression.
///
/// `Config` controls how the encoder partitions the input image into superblocks
/// and which compression algorithm is applied. The superblock size is chosen
/// automatically to minimise the estimated GPU decompression cost:
///
/// ```text
/// cost = (flat_cost + size_cost * superblock_area) * ceil(superblock_count / 64) * 64
/// ```
///
/// where `flat_cost` models the fixed per-superblock GPU dispatch overhead and
/// `size_cost` models the per-texel work. Adjust these two values to match the
/// characteristics of your target GPU.
#[derive(Clone, Debug)]
pub struct Config {
    /// Fixed cost per superblock, modelling GPU dispatch overhead.
    ///
    /// A higher value encourages larger superblocks (fewer dispatches).
    pub flat_cost: f32,

    /// Cost per texel within a superblock, modelling per-texel GPU work.
    ///
    /// A higher value encourages smaller superblocks (less work per dispatch).
    pub size_cost: f32,

    /// Target pixel / block format stored in the output file.
    pub format: Format,

    /// Compression algorithm applied to the texel data.
    pub compression: Compression,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            flat_cost: 64.0,
            size_cost: 1.0,
            format: Format::RGB8,
            compression: Compression::RleAns,
        }
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
