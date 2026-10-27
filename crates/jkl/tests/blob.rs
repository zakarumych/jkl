use jkl::jackal::blob::{Compression as C, JackalBlobReader, Options, write_blob};
use std::io::{Cursor, ErrorKind};
const CODECS: [C; 7] = [
    C::None,
    C::Lz77,
    C::Ans,
    C::Lz77Ans,
    C::RleAns,
    C::Rle,
    C::Lz78,
];
fn encode(data: &[u8], chunk_size: usize, allowed_compressions: Vec<C>) -> Vec<u8> {
    let mut encoded = Vec::new();
    write_blob(
        data,
        &Options {
            chunk_size,
            allowed_compressions,
        },
        &mut encoded,
    )
    .unwrap();
    encoded
}
#[test]
fn every_codec_roundtrips_empty_tiny_and_chunks() {
    for codec in CODECS {
        for data in [
            vec![],
            vec![0],
            vec![255],
            vec![7; 513],
            (0..1203).map(|i| ((i * 13 + i / 7) % 256) as u8).collect(),
        ] {
            let encoded = encode(&data, 128, vec![codec]);
            let mut reader = JackalBlobReader::open(Cursor::new(encoded)).unwrap();
            assert_eq!(reader.len(), data.len() as u64);
            assert_eq!(reader.is_empty(), data.is_empty());
            assert_eq!(reader.read_blob().unwrap(), data, "{codec:?}");
            for (i, part) in data.chunks(128).enumerate().rev() {
                assert_eq!(reader.read_chunk(i).unwrap(), part);
            }
        }
    }
}
#[test]
fn selection_uses_context_and_honors_allowlist() {
    let data = [vec![42; 128], (0..128).collect(), vec![11; 3]].concat();
    let mut reader = JackalBlobReader::open(Cursor::new(encode(
        &data,
        128,
        vec![C::None, C::Ans, C::Rle],
    )))
    .unwrap();
    assert_eq!(
        reader
            .chunks()
            .iter()
            .map(|c| c.compression)
            .collect::<Vec<_>>(),
        [C::Rle, C::None, C::None]
    );
    assert_eq!(reader.read_blob().unwrap(), data);
    let reader = JackalBlobReader::open(Cursor::new(encode(&[1], 128, vec![C::Ans]))).unwrap();
    assert_eq!(reader.chunks()[0].compression, C::Ans);
    // A single RLE token and a single LZ78 token both occupy eight bytes.
    let reader =
        JackalBlobReader::open(Cursor::new(encode(&[1], 128, vec![C::Rle, C::Lz78]))).unwrap();
    assert_eq!(reader.chunks()[0].compression, C::Rle);
}
#[test]
fn invalid_options_and_gpu_capabilities() {
    for options in [
        Options {
            chunk_size: 0,
            ..Options::default()
        },
        Options {
            allowed_compressions: vec![],
            ..Options::default()
        },
        Options {
            chunk_size: usize::MAX,
            ..Options::default()
        },
    ] {
        assert_eq!(
            write_blob(&[], &options, Vec::new()).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
    for codec in CODECS {
        let mut reader =
            JackalBlobReader::open(Cursor::new(encode(&[4; 129], 64, vec![codec]))).unwrap();
        let prepared = reader.prepare_gpu();
        if codec.supports_gpu() {
            let prepared = prepared.unwrap();
            assert_eq!(prepared.len, 129);
            assert_eq!(prepared.chunks.len(), 3);
            assert_eq!(prepared.chunks[2].output_offset, 128);
        } else {
            assert_eq!(prepared.unwrap_err().kind(), ErrorKind::Unsupported);
        }
    }
}
#[test]
fn malformed_and_truncated_input_never_panics() {
    for codec in CODECS {
        let encoded = encode(&[5; 32], 32, vec![codec]);
        for end in 0..encoded.len() {
            let truncated = &encoded[..end];
            assert!(
                JackalBlobReader::open(Cursor::new(truncated))
                    .and_then(|mut r| r.read_blob())
                    .is_err()
            );
        }
        for i in 0..encoded.len() {
            let mut changed = encoded.clone();
            changed[i] ^= 255;
            let result = std::panic::catch_unwind(|| {
                JackalBlobReader::open(Cursor::new(changed)).and_then(|mut r| r.read_blob())
            });
            assert!(result.is_ok(), "codec {codec:?}, byte {i}");
        }
    }
}
#[test]
fn hostile_ans_tables_are_rejected() {
    let encoded = encode(&[0, 1, 0, 1], 4, vec![C::Ans]);
    for (offset, word) in [
        (52, 0u32),
        (52, u32::MAX),
        (64, 0),
        (64, u32::MAX),
        (68, 0),
        (76, u32::MAX),
    ] {
        let mut changed = encoded.clone();
        changed[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
        let mut reader = JackalBlobReader::open(Cursor::new(changed)).unwrap();
        assert!(reader.read_blob().is_err());
        assert!(reader.prepare_gpu().is_err());
    }
}
#[test]
fn embedded_blob_offsets_are_relative() {
    let mut encoded = vec![99; 17];
    write_blob(b"hello", &Options::default(), &mut encoded).unwrap();
    let mut cursor = Cursor::new(encoded);
    cursor.set_position(17);
    assert_eq!(
        JackalBlobReader::open(cursor).unwrap().read_blob().unwrap(),
        b"hello"
    );
}
#[test]
fn lz77_long_references() {
    let data = vec![0; 131_079];
    for codec in [C::Lz77, C::Lz77Ans] {
        let mut reader =
            JackalBlobReader::open(Cursor::new(encode(&data, data.len(), vec![codec]))).unwrap();
        assert_eq!(reader.read_blob().unwrap(), data);
    }
}
