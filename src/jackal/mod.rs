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

use std::{
    io::{Read, Seek, SeekFrom, Write},
    u32,
};

use crate::bc1;

pub use self::{
    block::AnyBlock,
    header::{Extent, Format, JackalBlock, JackalHeader, MipLevels, SuperBlockSize},
};

mod block;
mod header;

#[derive(Clone, Copy, Debug)]
pub enum DecodeError {
    /// Magic number invalid.
    InvalidMagic,

    /// Header is invalid.
    /// Data is corrupted or comes from other version with extended capabilities.
    InvalidHeader,

    // Data is invalid.
    // Such as position is out of bounds.
    InvalidData,
}

pub fn compress_bc1_texture(
    extent: Extent,
    blocks: &[bc1::Block],
    write: impl Write + Seek,
) -> std::io::Result<()> {
    compress_texture(extent, blocks, write)
}

fn compress_texture<B>(
    extent: Extent,
    blocks: &[B],
    mut write: impl Write + Seek,
) -> std::io::Result<()>
where
    B: AnyBlock,
{
    let raw_size = extent.raw_size();

    assert_eq!(blocks.len() as u32, raw_size[0] * raw_size[1] * raw_size[2]);

    let super_block_size = SuperBlockSize::from_size(raw_size[0], raw_size[1]);

    let header = JackalHeader {
        levels: MipLevels(1),
        format: Format::BC1,
        super_block_size,
        extent,
    };

    let start = write.seek(SeekFrom::Current(0))?;
    header.write_to(&mut write)?;

    let jackal_blocks_width =
        (raw_size[0] + super_block_size.width as u32 - 1) / super_block_size.width as u32;
    let jackal_blocks_height =
        (raw_size[1] + super_block_size.height as u32 - 1) / super_block_size.height as u32;
    let jackal_blocks_depth = raw_size[2];

    let jackal_blocks_count = jackal_blocks_width * jackal_blocks_height * jackal_blocks_depth;

    let jackal_blocks_start = start + JackalHeader::BYTES_SIZE as u64;
    let jackal_blocks_end =
        jackal_blocks_start + JackalBlock::BYTES_SIZE as u64 * jackal_blocks_count as u64;

    let mut next_jackal_block_pos = jackal_blocks_start;
    let mut next_data_pos = jackal_blocks_end;

    for z in 0..raw_size[2] {
        for y_start in (0..raw_size[1]).step_by(super_block_size.height as usize) {
            let y_end = if raw_size[1] - y_start < header.super_block_size.height as u32 {
                raw_size[1]
            } else {
                y_start + header.super_block_size.height as u32
            };

            for x_start in (0..raw_size[0]).step_by(super_block_size.width as usize) {
                let x_end = if raw_size[0] - x_start < header.super_block_size.width as u32 {
                    raw_size[0]
                } else {
                    x_start + header.super_block_size.width as u32
                };

                write.seek(SeekFrom::Start(next_jackal_block_pos))?;

                // Write a jackal_block.
                let sb = JackalBlock {
                    offset: next_data_pos,
                };
                sb.write_to(&mut write)?;
                next_jackal_block_pos += JackalBlock::BYTES_SIZE as u64;

                write.seek(SeekFrom::Start(next_data_pos))?;
                compress_any_block::<B>(
                    x_start, x_end, y_start, y_end, z, raw_size, blocks, &mut write,
                )?;
                next_data_pos = write.seek(SeekFrom::Current(0))?;
            }
        }
    }

    Ok(())
}

pub fn compress_bc1_blocks(
    header: &JackalHeader,
    super_pos: [u32; 3],
    jackal_block: JackalBlock,
    blocks: &[bc1::Block],
    mut write: impl Write + Seek,
) -> std::io::Result<()> {
    let raw_size = header.extent.raw_size();

    let x_start = super_pos[0] * header.super_block_size.width as u32;
    let x_end = if raw_size[0] - x_start < header.super_block_size.width as u32 {
        raw_size[0]
    } else {
        x_start + header.super_block_size.width as u32
    };

    let y_start = super_pos[1] * header.super_block_size.height as u32;
    let y_end = if raw_size[1] - y_start < header.super_block_size.height as u32 {
        raw_size[1]
    } else {
        y_start + header.super_block_size.height as u32
    };

    let z = super_pos[2];

    write.seek(SeekFrom::Start(jackal_block.offset))?;
    compress_any_block(x_start, x_end, y_start, y_end, z, raw_size, blocks, write)
}

fn compress_any_block<B>(
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
    z: u32,
    raw_size: [u32; 3],
    blocks: &[B],
    write: impl Write,
) -> std::io::Result<()>
where
    B: AnyBlock,
{
    // let mut encoder = lzw::Encoder::<B::EncoderElement>::new();
    // let mut write = WriteBits::new(write);
    let mut encoder = brotli::CompressorWriter::new(write, 4096, 11, 22);

    compress_any_block_aspect::<B, 0>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut encoder,
        // &mut write,
    )?;

    compress_any_block_aspect::<B, 1>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut encoder,
        // &mut write,
    )?;

    compress_any_block_aspect::<B, 2>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut encoder,
        // &mut write,
    )?;

    compress_any_block_aspect::<B, 3>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut encoder,
        // &mut write,
    )?;

    compress_any_block_aspect::<B, 4>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut encoder,
        // &mut write,
    )?;

    compress_any_block_aspect::<B, 5>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut encoder,
        // &mut write,
    )?;

    compress_any_block_aspect::<B, 6>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut encoder,
        // &mut write,
    )?;

    compress_any_block_aspect::<B, 7>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut encoder,
        // &mut write,
    )?;

    // encoder.finish(&mut write)?;
    // write.finish()?;

    encoder.flush()?;

    Ok(())
}

fn compress_any_block_aspect<B, const ASPECT: usize>(
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
    z: u32,
    blocks: &[B],
    raw_size: [u32; 3],
    // encoder: &mut lzw::Encoder<B::EncoderElement>,
    // write: &mut WriteBits<impl Write>,
    encoder: &mut brotli::CompressorWriter<impl Write>,
) -> std::io::Result<()>
where
    B: AnyBlock,
{
    if B::ASPECTS <= ASPECT {
        return Ok(());
    }

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
        let block = &blocks[index as usize];

        block.compress::<ASPECT>(&mut *encoder)?;
    }

    Ok(())
}

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
pub fn read_header(read: impl Read) -> Result<JackalHeader, DecompressError> {
    JackalHeader::read_from(read)
}

/// Read super-blocks from the stream.
pub fn read_jackal_blocks(
    jackal_blocks: &mut [JackalBlock],
    mut read: impl Read,
) -> Result<(), DecompressError> {
    for sb in jackal_blocks.iter_mut() {
        *sb = JackalBlock::read_from(&mut read)?;
    }
    Ok(())
}

/// Read blocks of one jackal_block.
///
/// `header` is Jackal header.
/// `super_x` is x coordinate of the jackal_block.
/// `super_y` is y coordinate of the jackal_block.
/// `jackal_block` is jackal_block data.
/// `blocks` array of pre-allocated blocks of all jackal_blocks.
pub fn decompress_bc1_blocks(
    header: &JackalHeader,
    super_pos: [u32; 3],
    jackal_block: JackalBlock,
    blocks: &mut [bc1::Block],
    read: impl Read + Seek,
) -> Result<(), DecompressError> {
    decompress_any_block(header, super_pos, jackal_block, blocks, read)
}

fn decompress_any_block<B>(
    header: &JackalHeader,
    super_pos: [u32; 3],
    jackal_block: JackalBlock,
    blocks: &mut [B],
    mut read: impl Read + Seek,
) -> Result<(), DecompressError>
where
    B: AnyBlock,
{
    let raw_size = header.extent.raw_size();

    let x_start = super_pos[0] * header.super_block_size.width as u32;
    let x_end = if raw_size[0] - x_start < header.super_block_size.width as u32 {
        raw_size[0]
    } else {
        x_start + header.super_block_size.width as u32
    };

    let y_start = super_pos[1] * header.super_block_size.height as u32;
    let y_end = if raw_size[1] - y_start < header.super_block_size.height as u32 {
        raw_size[1]
    } else {
        y_start + header.super_block_size.height as u32
    };

    let z = super_pos[2];

    read.seek(SeekFrom::Start(jackal_block.offset))?;

    // let mut decoder = lzw::Decoder::<B::EncoderElement>::new();
    // let mut read = ReadBits::new(read);
    let mut decoder = brotli::reader::Decompressor::new(read, 4096);

    decompress_any_block_aspect::<B, 0>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut decoder,
        // &mut read,
    )?;

    decompress_any_block_aspect::<B, 1>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut decoder,
        // &mut read,
    )?;

    decompress_any_block_aspect::<B, 2>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut decoder,
        // &mut read,
    )?;

    decompress_any_block_aspect::<B, 3>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut decoder,
        // &mut read,
    )?;

    decompress_any_block_aspect::<B, 4>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut decoder,
        // &mut read,
    )?;

    decompress_any_block_aspect::<B, 5>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut decoder,
        // &mut read,
    )?;

    decompress_any_block_aspect::<B, 6>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut decoder,
        // &mut read,
    )?;

    decompress_any_block_aspect::<B, 7>(
        x_start,
        x_end,
        y_start,
        y_end,
        z,
        blocks,
        raw_size,
        &mut decoder,
        // &mut read,
    )?;

    // decoder.finish();

    Ok(())
}

fn decompress_any_block_aspect<B, const ASPECT: usize>(
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
    z: u32,
    blocks: &mut [B],
    raw_size: [u32; 3],
    // decoder: &mut lzw::Decoder<B::EncoderElement>,
    // read: &mut ReadBits<impl Read>,
    decoder: &mut brotli::reader::Decompressor<impl Read>,
) -> Result<(), DecompressError>
where
    B: AnyBlock,
{
    if B::ASPECTS <= ASPECT {
        return Ok(());
    }

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
        let mut block = blocks[index];

        block.decompress::<ASPECT>(&mut *decoder)?;

        blocks[index as usize] = block;
    }

    Ok(())
}

pub fn decompress_bc1_texture(
    mut read: impl Read + Seek,
) -> Result<(Extent, Vec<bc1::Block>), DecompressError> {
    let header = read_header(&mut read)?;
    let mut jackal_blocks = vec![JackalBlock { offset: 0 }; header.jackal_blocks_count()];
    read_jackal_blocks(&mut jackal_blocks, &mut read)?;

    let mut blocks = vec![bc1::Block::BLACK; header.blocks_count()];

    let jackal_blocks_extent = header.jackal_blocks_extent();

    for z in 0..jackal_blocks_extent[2] {
        for y in 0..jackal_blocks_extent[1] {
            for x in 0..jackal_blocks_extent[0] {
                decompress_bc1_blocks(
                    &header,
                    [x, y, z],
                    jackal_blocks[(x
                        + y * jackal_blocks_extent[0]
                        + z * jackal_blocks_extent[0] * jackal_blocks_extent[1])
                        as usize],
                    &mut blocks,
                    &mut read,
                )?;
            }
        }
    }

    Ok((header.extent(), blocks))
}

#[test]
fn roundtrip() {
    use crate::math::Rgb32F;

    let pixels = [
        [Rgb32F::BLACK, Rgb32F::WHITE, Rgb32F::BLACK, Rgb32F::WHITE],
        [Rgb32F::WHITE, Rgb32F::BLACK, Rgb32F::WHITE, Rgb32F::BLACK],
        [Rgb32F::BLACK, Rgb32F::WHITE, Rgb32F::BLACK, Rgb32F::WHITE],
        [Rgb32F::WHITE, Rgb32F::BLACK, Rgb32F::WHITE, Rgb32F::BLACK],
    ];

    let block = bc1::Block::encode(pixels);

    assert_eq!(block.decode(), pixels);

    let blocks = vec![block; 2];

    // eprintln!("\n\nCompress");

    let mut output = Vec::new();
    compress_bc1_texture(
        Extent::D2 {
            width: 2,
            height: 1,
        },
        &blocks,
        std::io::Cursor::new(&mut output),
    )
    .unwrap();

    // eprintln!("\n\nDecompress");

    let (extent, decompressed) = decompress_bc1_texture(std::io::Cursor::new(&output)).unwrap();

    assert_eq!(
        extent,
        Extent::D2 {
            width: 2,
            height: 1,
        }
    );

    assert_eq!(decompressed[..], blocks[..]);
}
