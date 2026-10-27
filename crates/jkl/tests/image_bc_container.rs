use std::{fmt::Debug, io::Cursor};

use jkl::{
    encode::FixedCode,
    image::{
        ImageRef, OwnedImage,
        block::{bc1, bc2, bc3, bc4, bc5, bc6, bc7},
        tiles::TileSize,
    },
    jackal::image::{Compression, JackalImageReader, Options, Pixel, write_image},
};

const COMPRESSIONS: [Compression; 5] = [
    Compression::None,
    Compression::Lz77,
    Compression::Ans,
    Compression::Lz77Ans,
    Compression::RleAns,
];

fn encoded<T: Pixel>(blocks: &[T], compression: Compression) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    write_image(
        ImageRef::new_2d(9, 5, blocks),
        Options::new()
            .with_compression(compression)
            .with_tile_size(TileSize {
                width: 4,
                height: 4,
            }),
        &mut out,
    )
    .unwrap();
    out.into_inner()
}

fn decoded<T: Pixel>(bytes: &[u8], zero: T) -> std::io::Result<Vec<T>> {
    let mut reader = JackalImageReader::open(Cursor::new(bytes))?;
    assert_eq!(reader.format(), T::FORMAT);
    assert_eq!(reader.tiles(), 6);
    let extent = reader.extent();
    let mut result = OwnedImage::new_2d(9, 5, vec![zero; 45].into_boxed_slice());
    let mut tiles = reader.tile_reader::<T>()?;
    let mut output = result.as_mut();
    let mut plane = output.get_plane_mut(0);
    for (index, tile) in tiles.tile_size().iter_tiles(extent).enumerate() {
        tiles.read_tile(index, plane.get_rect_mut(tile.rect))?;
    }
    Ok(result.data().to_vec())
}

fn exercise<T, const N: usize>(from_bytes: impl Fn([u8; N]) -> T)
where
    T: Pixel + FixedCode<Array = [u8; N]> + Debug,
{
    // Distinct blocks and bytes make both block ordering and byte ordering observable.
    let canonical: Vec<[u8; N]> = (0..45)
        .map(|block| std::array::from_fn(|byte| (block * 17 + byte * 11) as u8))
        .collect();
    let blocks: Vec<_> = canonical.iter().copied().map(from_bytes).collect();
    for compression in COMPRESSIONS {
        let bytes = encoded(&blocks, compression);
        assert_eq!(
            decoded(&bytes, blocks[0]).unwrap(),
            blocks,
            "{compression:?}"
        );

        // Removing the final tile payload must not produce a successful image.
        let reader = JackalImageReader::open(Cursor::new(&bytes)).unwrap();
        let last_offset = *reader.tile_offsets().last().unwrap() as usize;
        assert!(
            decoded(&bytes[..last_offset], blocks[0]).is_err(),
            "{compression:?}"
        );
    }

    // Verify raw tile layout independently of Pixel::decompress_image.
    let bytes = encoded(&blocks, Compression::None);
    let reader = JackalImageReader::open(Cursor::new(&bytes)).unwrap();
    for (index, tile) in reader.tile_size().iter_tiles(reader.extent()).enumerate() {
        let mut expected = Vec::new();
        for y in tile.rect.y..tile.rect.y + tile.rect.h {
            for x in tile.rect.x..tile.rect.x + tile.rect.w {
                expected.extend_from_slice(&canonical[y * 9 + x]);
            }
        }
        let offset = reader.tile_offsets()[index] as usize;
        assert_eq!(&bytes[offset..offset + expected.len()], expected);
    }
}

#[test]
fn bc3_all_compressions() {
    exercise(bc3::Block::from_bytes);
}
#[test]
fn bc4_all_compressions() {
    exercise(bc4::Block::from_bytes);
}
#[test]
fn bc5_all_compressions() {
    exercise(bc5::Block::from_bytes);
}
#[test]
fn bc6_all_compressions() {
    exercise(bc6::Block::from_bytes);
}
#[test]
fn bc7_all_compressions() {
    exercise(bc7::Block::from_bytes);
}

#[test]
fn bc1_existing_layout_roundtrip() {
    exercise(bc1::Block::from_bytes);
}
#[test]
fn bc2_existing_layout_roundtrip() {
    exercise(bc2::Block::from_bytes);
}
