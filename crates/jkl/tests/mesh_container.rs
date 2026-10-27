use jkl::{
    jackal::mesh::{JackalMeshReader, write_compressed, write_mesh, write_prepared},
    mesh::{
        codec, compress,
        data::{EncodingOptions, MeshData},
    },
};
use std::io::Cursor;

fn fixture() -> MeshData {
    let positions = vec![
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.1],
        [2., 0., 0.],
        [2., 1., 0.2],
        [7., 2., 1.],
    ];
    let n = positions.len();
    MeshData {
        positions,
        triangles: vec![[0, 1, 2], [2, 1, 3], [1, 4, 3], [3, 4, 5]],
        normals: Some(vec![[0., 0., 1.]; n]),
        tangents: Some(vec![[1., 0., 0., -1.]; n]),
        tex_coords: vec![
            vec![
                [0., 0.],
                [1., 0.],
                [0., 1.],
                [1., 1.],
                [2., 0.],
                [2., 1.],
                [7., 2.],
            ],
            vec![[0.25, 0.5]; n],
        ],
        colors: Some((0..n).map(|i| [i as u8, 80, 90, 255]).collect()),
        primitive_ids: vec![0, 0, 1, 1],
        materials: vec![4, 7],
    }
}
fn options() -> EncodingOptions {
    EncodingOptions {
        max_vertices: 3,
        weld_epsilon: None,
        ..Default::default()
    }
}

#[test]
fn complete_attributes_random_patch_reads_and_gpu_preparation() {
    for bits in [2, 10, 20] {
        let mut opts = options();
        opts.position_bits = bits;
        opts.uv_bits = bits;
        opts.normal_bits = bits;
        opts.tangent_bits = bits;
        let prepared = codec::prepare(&fixture(), &opts).unwrap();
        let reference = prepared
            .patches
            .iter()
            .map(|p| codec::decode_patch(&prepared.schema, &p.info, &p.streams).unwrap())
            .collect::<Vec<_>>();
        let reference =
            codec::assemble(&prepared.schema, &reference, prepared.materials.clone()).unwrap();
        let mut bytes = Vec::new();
        write_prepared(&prepared, &mut bytes).unwrap();
        let mut reader = JackalMeshReader::open(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.vertex_count(), 7);
        assert_eq!(reader.triangle_count(), 4);
        assert!(
            reader
                .patches()
                .iter()
                .any(|p| p.info.triangle_count > p.info.internal_triangle_count)
        );
        let patches = (0..reader.patches().len())
            .rev()
            .map(|i| reader.read_patch(i).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            codec::assemble(reader.schema(), &patches, reader.materials().to_vec()).unwrap(),
            reference
        );
        assert_eq!(reader.read_mesh().unwrap(), reference);
        let gpu = reader.prepare_gpu().unwrap();
        let compressed = compress::compress(&prepared).unwrap();
        assert_eq!(
            gpu.streams.len(),
            prepared.patches.len() * prepared.schema.stream_kinds().len()
        );
        for (stream, original) in gpu
            .streams
            .iter()
            .zip(compressed.patches.iter().flat_map(|p| &p.streams))
        {
            assert_eq!(
                &gpu.payload[stream.payload_offset as usize
                    ..(stream.payload_offset + stream.payload_len) as usize],
                original.words.iter().rev().copied().collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn empty_and_loose_only_meshes() {
    for mesh in [
        MeshData::default(),
        MeshData {
            positions: vec![[0., 0., 0.], [0.5, 1., 2.]],
            ..Default::default()
        },
    ] {
        let mut bytes = Vec::new();
        write_mesh(&mesh, &options(), &mut bytes).unwrap();
        let mut reader = JackalMeshReader::open(Cursor::new(bytes)).unwrap();
        let decoded = reader.read_mesh().unwrap();
        assert_eq!(decoded.positions.len(), mesh.positions.len());
        assert!(decoded.triangles.is_empty());
        reader.prepare_gpu().unwrap();
    }
}

#[test]
fn rejects_truncated_sections_and_bad_header_counts() {
    let mut bytes = Vec::new();
    write_mesh(&fixture(), &options(), &mut bytes).unwrap();
    for n in 0..bytes.len() {
        assert!(
            JackalMeshReader::open(Cursor::new(&bytes[..n])).is_err(),
            "accepted length {n}"
        );
    }
    for offset in [4, 8, 12, 24, 28, 32, 36, 40, 44, 52, 60] {
        let mut bad = bytes.clone();
        bad[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(
            JackalMeshReader::open(Cursor::new(bad)).is_err(),
            "accepted field {offset}"
        );
    }
    let mut beyond = Cursor::new(bytes);
    beyond.set_position(u64::MAX);
    assert!(JackalMeshReader::open(beyond).is_err());
}

#[test]
fn table_and_payload_corruption_are_checked() {
    let prepared = codec::prepare(&fixture(), &options()).unwrap();
    let mut compressed = compress::compress(&prepared).unwrap();
    let mut bytes = Vec::new();
    write_compressed(&compressed, &mut bytes).unwrap();
    let table_start = 68 + compressed.materials.len() * 4;
    let mut bad = bytes.clone();
    bad[table_start..table_start + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(JackalMeshReader::open(Cursor::new(bad)).is_err());
    let mut bad = bytes;
    let n = bad.len();
    bad[n - 4..].fill(0);
    let mut reader = JackalMeshReader::open(Cursor::new(bad)).unwrap();
    assert!(reader.read_mesh().is_err());
    compressed.tables[2][0].count += 1;
    assert!(write_compressed(&compressed, &mut Vec::new()).is_err());
}

#[test]
fn entropy_roundtrip_handles_extreme_symbols_and_rejects_invalid_states() {
    use jkl::algos::ans;
    let values = vec![0, u32::MAX, 0, 17, 0, 0, u32::MAX];
    let ctx = ans::Context::from_input(values.iter().copied());
    let mut encoder = ans::Encoder::new(&ctx);
    let mut words = Vec::new();
    for &s in values.iter().rev() {
        if let Some(w) = encoder.encode(s) {
            words.push(w);
        }
    }
    words.extend(encoder.finish());
    let mut stream = compress::EncodedStream {
        symbol_count: values.len() as u32,
        words,
    };
    assert_eq!(compress::decode_stream(&ctx, &stream).unwrap(), values);
    stream.words.fill(u32::MAX);
    assert!(compress::decode_stream(&ctx, &stream).is_err());
    assert!(
        compress::decode_stream(
            &ctx,
            &compress::EncodedStream {
                symbol_count: 0,
                words: vec![0]
            }
        )
        .is_err()
    );
    assert!(
        compress::context(&[compress::Frequency {
            symbol: 1,
            count: 0
        }])
        .is_err()
    );
}
