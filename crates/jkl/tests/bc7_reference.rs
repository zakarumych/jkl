//! Independent BC7 decoding and encoder invariants. Provenance: data/bc7-reference.md.
use jkl::{encode::FixedCode, image::block::bc7::Block, math::Rgba32F};

const REFERENCE: &[u8] = include_bytes!("data/bc7-reference.bin");

fn assert_pixels(block: Block, expected: &[u8], case: usize) {
    for (i, pixel) in block.decode().iter().flatten().enumerate() {
        for c in 0..4 {
            assert_eq!(
                pixel.0[c],
                f32::from(expected[i * 4 + c]) / 255.0,
                "case {case}, texel {i}, channel {c}, block {:02x?}",
                block.bytes()
            );
        }
    }
}

#[test]
fn independently_decodes_all_modes_partitions_rotations_and_pbits() {
    assert_eq!(REFERENCE.len(), 4448 * 80);
    let mut modes = [0usize; 9];
    for (case, record) in REFERENCE.chunks_exact(80).enumerate() {
        let block = Block::from_bytes(record[..16].try_into().unwrap());
        modes[block.bytes()[0].trailing_zeros().min(8) as usize] += 1;
        assert_pixels(block, &record[16..], case);
    }
    assert!(modes.iter().all(|&n| n > 0), "{modes:?}");
}

#[test]
fn fixed_code_preserves_every_bit() {
    assert_eq!(Block::SIZE, 16);
    for record in REFERENCE.chunks_exact(80) {
        let raw: [u8; 16] = record[..16].try_into().unwrap();
        let block = Block::from_bytes(raw);
        assert_eq!(block.fix_encode(), raw);
        assert_eq!(Block::fix_decode(&raw).unwrap(), block);
        let mut out = Vec::new();
        block.fix_write(&mut out).unwrap();
        assert_eq!(Block::fix_read(&mut out.as_slice()).unwrap(), block);
    }
}

#[test]
fn mask_sanitization_and_determinism() {
    let bad = Rgba32F([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0]);
    let mut input = [[(bad, false); 4]; 4];
    assert_eq!(Block::encode(input), Block::TRANSPARENT);
    input[1][2] = (Rgba32F([0.1, 0.3, 0.7, 0.9]), true);
    let encoded = Block::encode(input);
    for row in &mut input {
        for (pixel, valid) in row {
            if !*valid {
                *pixel = Rgba32F([1.0, 0.2, 0.5, 0.8]);
            }
        }
    }
    assert_eq!(encoded, Block::encode(input));
    assert_eq!(encoded, Block::encode(input));
    assert_eq!(
        Block::encode([[(bad, true); 4]; 4]),
        Block::encode([[(Rgba32F([0.0, 1.0, 0.0, 0.0]), true); 4]; 4])
    );
    assert_eq!(
        Block::encode([[(Rgba32F([-0.1, 1.1, -f32::MAX, f32::MAX]), true); 4]; 4]),
        Block::encode([[(Rgba32F([0.0, 1.0, 0.0, 1.0]), true); 4]; 4])
    );
}

#[test]
fn independent_quality_reference_decodes() {
    let data = include_bytes!("data/bc7-quality.bin");
    assert_eq!(data.len(), 64 * 148);
    for (case, record) in data.chunks_exact(148).enumerate() {
        assert_pixels(
            Block::from_bytes(record[68..84].try_into().unwrap()),
            &record[84..],
            case,
        );
    }
}
