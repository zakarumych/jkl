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

use crate::{
    bits::{ReadBits, WriteBits},
    encode::{Encode, FixedCode},
    image::{ImageMut, ImageRef},
};

pub use self::compress::{AnsCompressor, Compressor, LZ77Compressor, RleCompressor};
pub use self::header::{Compression, Extent, Format, JackalBlock, JackalHeader, MipLevels, SuperBlockSize};

use self::format::AnyFormat;

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
/// cost = (superblock_flat_cost + superblock_size_cost * superblock_area) * ceil(superblock_count / 64) * 64
/// ```
///
/// where `superblock_flat_cost` models the fixed per-superblock GPU dispatch
/// overhead and `superblock_size_cost` models the per-texel decompression work.
/// Adjust these two values to match the characteristics of your target GPU.
#[derive(Clone, Debug)]
pub struct Config {
    /// Fixed cost per superblock for GPU superblock decompression dispatch.
    ///
    /// Models the overhead of launching each superblock on the GPU.
    /// A higher value encourages larger superblocks (fewer dispatches).
    pub superblock_flat_cost: f32,

    /// Per-texel cost within a superblock for GPU superblock decompression.
    ///
    /// Models the work per texel during GPU decompression.
    /// A higher value encourages smaller superblocks (less work per dispatch).
    pub superblock_size_cost: f32,

    /// Target pixel / block format stored in the output file.
    pub format: Format,

    /// Compression algorithm applied to the texel data.
    pub compression: Compression,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            superblock_flat_cost: 64.0,
            superblock_size_cost: 1.0,
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

/// Encode a 2-D image as a Jackal file.
///
/// The image is partitioned into superblocks whose size is chosen to minimise
/// the GPU decompression cost model configured in `config`. All superblocks are
/// compressed into memory first so that block offsets are known before any bytes
/// are written; the output is therefore a single forward pass to `write`:
/// header → block-offset table → compressed context → compressed token data.
#[allow(private_bounds)]
pub fn encode<P: AnyFormat>(
    image: ImageRef<'_, P>,
    config: &Config,
    write: &mut impl io::Write,
) -> io::Result<()> {
    let extent = Extent::D2 {
        width: image.width() as u32,
        height: image.height() as u32,
    };
    let superblock_size = SuperBlockSize::find_optimal(
        extent,
        1,
        config.superblock_flat_cost,
        config.superblock_size_cost,
    );
    let header = JackalHeader::new(
        config.compression,
        config.format,
        extent,
        MipLevels(1),
        superblock_size,
    );
    let sw = superblock_size.width as usize;
    let sh = superblock_size.height as usize;
    let superblocks = split_superblocks(image, sw, sh);

    match config.compression {
        Compression::None => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Compression::None is not implemented",
        )),
        Compression::Lz77 => encode_impl(
            superblocks.iter().copied(),
            LZ77Compressor { window_size: 256 },
            header,
            write,
        ),
        Compression::Ans => {
            encode_impl(superblocks.iter().copied(), AnsCompressor, header, write)
        }
        Compression::Lz77Ans => encode_impl(
            superblocks.iter().copied(),
            (LZ77Compressor { window_size: 256 }, AnsCompressor),
            header,
            write,
        ),
        Compression::RleAns => encode_impl(
            superblocks.iter().copied(),
            (RleCompressor, AnsCompressor),
            header,
            write,
        ),
    }
}

/// Splits `image` into a row-major grid of superblocks of size `sw × sh`.
/// Edge blocks are smaller when the image dimensions are not exact multiples.
fn split_superblocks<'a, P>(
    image: ImageRef<'a, P>,
    sw: usize,
    sh: usize,
) -> Vec<ImageRef<'a, P>> {
    let width = image.width();
    let height = image.height();
    let n_x = (width + sw - 1) / sw;
    let n_y = (height + sh - 1) / sh;
    let mut blocks = Vec::with_capacity(n_x * n_y);
    for by in 0..n_y {
        for bx in 0..n_x {
            let x = bx * sw;
            let y = by * sh;
            let w = sw.min(width - x);
            let h = sh.min(height - y);
            blocks.push(image.get_range(x, y, w, h));
        }
    }
    blocks
}

/// Compress `superblocks` with `compressor`, then write the complete Jackal
/// payload (header → block-offset table → context → token data) to `write`.
fn encode_impl<'a, P, C>(
    superblocks: impl Iterator<Item = ImageRef<'a, P>> + Clone,
    compressor: C,
    header: JackalHeader,
    write: &mut impl io::Write,
) -> io::Result<()>
where
    P: AnyFormat,
    C: Compressor,
{
    let n_blocks = header.superblocks_count();

    // Compress all superblocks into an in-memory buffer and capture relative offsets.
    let mut token_buf = io::Cursor::new(Vec::<u8>::new());
    let mut rel_offsets = vec![0u64; n_blocks];
    let cx = P::compress_images(superblocks, compressor, &mut token_buf, &mut rel_offsets)?;

    // Serialise the context to a byte buffer.
    let mut ctx_bytes: Vec<u8> = Vec::new();
    {
        let mut wbits = WriteBits::new(&mut ctx_bytes);
        cx.write(&mut wbits)?;
        wbits.finish()?;
    }

    // Absolute offset of block i = (header + table + context) + relative offset.
    let base_offset =
        (JackalHeader::SIZE + n_blocks * JackalBlock::SIZE + ctx_bytes.len()) as u64;

    // Write header.
    write.write_all(&header.encode())?;

    // Write block table with absolute offsets.
    for &rel in &rel_offsets {
        write.write_all(&JackalBlock { offset: base_offset + rel }.encode())?;
    }

    // Write context bytes.
    write.write_all(&ctx_bytes)?;

    // Write token data.
    write.write_all(token_buf.get_ref())?;

    Ok(())
}

/// Reads Jackal metadata from `read` and returns a decoder that can
/// decompress individual superblocks on demand.
pub struct JackalDecoder<R> {
    read: R,
    header: JackalHeader,
    blocks: Vec<JackalBlock>,
    /// Raw serialised context bytes, cached at construction time.
    /// Re-parsed on each [`decompress_superblock`] call because the context
    /// type depends on the caller-supplied pixel type `P` and compressor `C`.
    context_bytes: Vec<u8>,
}

impl<R: io::Read + io::Seek> JackalDecoder<R> {
    /// Read the file header, block-offset table and compressed context from
    /// `read`, then return a decoder ready to decompress individual superblocks.
    pub fn new(mut read: R) -> Result<Self, DecompressError> {
        let header = read_header(&mut read)?;
        let n_blocks = header.superblocks_count();
        let mut blocks = vec![JackalBlock { offset: 0 }; n_blocks];
        read_jackal_blocks(&mut blocks, &mut read)?;

        // Read context bytes: everything from here up to the first block's data.
        let context_end = blocks.first().map_or_else(
            || read.seek(io::SeekFrom::End(0)),
            |b| Ok(b.offset),
        )?;
        let context_start = read.stream_position()?;
        let context_len = context_end.saturating_sub(context_start) as usize;
        let mut context_bytes = vec![0u8; context_len];
        read.read_exact(&mut context_bytes)?;

        Ok(JackalDecoder { read, header, blocks, context_bytes })
    }

    /// Returns the decoded file header.
    pub fn header(&self) -> &JackalHeader {
        &self.header
    }

    /// Returns the block-offset table.
    pub fn blocks(&self) -> &[JackalBlock] {
        &self.blocks
    }

    /// Decompresses superblock `index` into `image`.
    ///
    /// `P` must be the pixel type matching `self.header().format` and
    /// `C` must be the compressor matching `self.header().compression`.
    #[allow(private_bounds)]
    pub fn decompress_superblock<P, C>(
        &mut self,
        index: usize,
        compressor: C,
        image: ImageMut<'_, P>,
    ) -> io::Result<()>
    where
        P: AnyFormat,
        C: Compressor,
    {
        if index >= self.blocks.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "superblock index {index} out of range ({})",
                    self.blocks.len()
                ),
            ));
        }

        // Deserialise the context from the cached bytes.
        let cx = {
            let mut cursor = io::Cursor::new(&self.context_bytes);
            let mut rbits = ReadBits::new(&mut cursor);
            <P::Context<C> as Encode>::read(&mut rbits)?
        };

        // Seek to this superblock's token data and decompress.
        self.read.seek(io::SeekFrom::Start(self.blocks[index].offset))?;
        P::decompress_image(compressor, &cx, &mut self.read, image)
    }
}
