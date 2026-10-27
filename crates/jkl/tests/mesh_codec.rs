use jkl::mesh::{
    codec,
    data::{EncodingOptions, MeshData},
    normal, spatial, tangent,
};

fn fixture() -> MeshData {
    let positions = vec![
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [2., 0., 0.],
        [2., 1., 0.],
        [5., 5., 5.],
    ];
    let n = positions.len();
    MeshData {
        tex_coords: vec![
            positions
                .iter()
                .map(|p| [p[0] * 2. - 3., p[1] * 3. + 7.])
                .collect(),
            vec![[9., -2.]; n],
        ],
        positions,
        triangles: vec![[0, 1, 2], [2, 1, 3], [1, 4, 3], [3, 4, 5], [0, 0, 1]],
        normals: Some(vec![[0., 0., 1.]; n]),
        tangents: Some(vec![[1., 0., 0., -1.]; n]),
        colors: Some((0..n).map(|i| [i as u8, 17, 200, 255]).collect()),
        primitive_ids: vec![0, 1, 0, 1, 0],
        materials: vec![7, 42],
    }
}
fn roundtrip(mesh: &MeshData, options: &EncodingOptions) -> (codec::PreparedMesh, MeshData) {
    let prepared = codec::prepare(mesh, options).unwrap();
    let mut decoded: Vec<_> = prepared
        .patches
        .iter()
        .map(|p| codec::decode_patch(&prepared.schema, &p.info, &p.streams).unwrap())
        .collect();
    decoded.reverse();
    let out = codec::assemble(&prepared.schema, &decoded, prepared.materials.clone()).unwrap();
    (prepared, out)
}
#[test]
fn all_attributes_independent_patches_and_exact_color() {
    let mesh = fixture();
    let options = EncodingOptions {
        max_vertices: 3,
        weld_epsilon: None,
        ..Default::default()
    };
    let (prepared, out) = roundtrip(&mesh, &options);
    assert!(prepared.patches.len() > 1);
    assert!(
        prepared
            .patches
            .iter()
            .any(|p| p.info.triangle_count > p.info.internal_triangle_count)
    );
    assert_eq!(out.positions.len(), mesh.positions.len());
    assert_eq!(out.triangles.len(), mesh.triangles.len());
    assert_eq!(out.materials, mesh.materials);
    let colors = out.colors.as_ref().unwrap();
    let mut map = vec![0u32; colors.len()];
    for (i, color) in colors.iter().enumerate() {
        let source = color[0] as usize;
        map[i] = source as u32;
        assert_eq!(*color, mesh.colors.as_ref().unwrap()[source]);
        for k in 0..3 {
            assert!((out.positions[i][k] - mesh.positions[source][k]).abs() < 0.03);
        }
        for set in 0..2 {
            for k in 0..2 {
                assert!((out.tex_coords[set][i][k] - mesh.tex_coords[set][source][k]).abs() < 0.03);
            }
        }
        assert!(out.normals.as_ref().unwrap()[i][2] > 0.999);
        let t = out.tangents.as_ref().unwrap()[i];
        assert!(t[0] > 0.999);
        assert_eq!(t[3], -1.);
    }
    fn canonical(mut t: [u32; 3]) -> [u32; 3] {
        let a = t;
        let b = [t[1], t[2], t[0]];
        let c = [t[2], t[0], t[1]];
        t = a.min(b).min(c);
        t
    }
    let mut actual: Vec<_> = out
        .triangles
        .iter()
        .zip(&out.primitive_ids)
        .map(|(t, p)| (canonical(t.map(|v| map[v as usize])), *p))
        .collect();
    let mut expected: Vec<_> = mesh
        .triangles
        .iter()
        .zip(&mesh.primitive_ids)
        .map(|(&t, &p)| (canonical(t), p))
        .collect();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
}
#[test]
fn high_precision_and_empty() {
    let mesh = fixture();
    let options = EncodingOptions {
        position_bits: 20,
        uv_bits: 20,
        normal_bits: 20,
        tangent_bits: 20,
        weld_epsilon: None,
        ..Default::default()
    };
    let (_, out) = roundtrip(&mesh, &options);
    for (i, c) in out.colors.as_ref().unwrap().iter().enumerate() {
        for k in 0..3 {
            assert!((out.positions[i][k] - mesh.positions[c[0] as usize][k]).abs() < 0.0001);
        }
    }
    let (p, d) = roundtrip(&MeshData::default(), &EncodingOptions::default());
    assert!(p.patches.is_empty());
    assert!(d.positions.is_empty());
}
#[test]
fn wrapped_predictions_skip_known_and_degenerate_triangles() {
    let head = [[0, 1, 2], [2, 1, 3], [0, 1, 2], [3, 3, 4], [4, 4, 4]];
    let values = [[0, 63], [63, 0], [0, 0], [63, 63], [31, 32], [2, 61]];
    let s = spatial::encode(&head, &values, 6).unwrap();
    assert!(s.iter().all(|&v| v < 64));
    let (out, order) = spatial::decode::<2>(&head, values.len(), 6, &s).unwrap();
    assert_eq!(out, values);
    assert_eq!(order, vec![0, 1, 2, 3, 4, 5]);
    assert!(spatial::decode::<2>(&head, values.len(), 6, &s[..s.len() - 1]).is_err());
    let mut bad = s;
    bad[0] = 64;
    assert!(spatial::decode::<2>(&head, values.len(), 6, &bad).is_err());
}
#[test]
fn normals_cover_directions_and_zero_fallback() {
    for bits in [6, 10, 20] {
        for predicted in [[0., 0., 0.], [1., 2., 3.], [-1., 0., 0.]] {
            for reference in [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [-1., 0., 0.],
                [0., -1., 0.],
                [0., 0., -1.],
            ] {
                let s = normal::encode(reference, predicted, bits).unwrap();
                let d = normal::decode(s, predicted, bits).unwrap();
                let error = reference
                    .into_iter()
                    .zip(d)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    error < 8. / (1u32 << bits) as f32,
                    "{bits} {reference:?} {predicted:?} {error}"
                );
            }
        }
    }
    assert!(normal::decode([u32::MAX, 0], [0., 0., 1.], 10).is_err());
}
#[test]
fn degenerate_uv_tangent_and_loose_vertices() {
    let positions = vec![[0., 0., 0.]; 3];
    let normals = vec![[0., 0., 1.]; 3];
    let zero = tangent::predictors(&positions, &[[0., 0.]; 3], &normals, &[[0, 1, 2]]).unwrap();
    for i in 0..3 {
        let (a, s) = tangent::encode([1., 0., 0., 1.], normals[i], zero[i], 10).unwrap();
        let t = tangent::decode(a, s, normals[i], zero[i], 10).unwrap();
        assert!(t[0] > 0.9999);
    }
    let mut mesh = fixture();
    mesh.triangles.clear();
    mesh.primitive_ids.clear();
    let (p, out) = roundtrip(
        &mesh,
        &EncodingOptions {
            max_vertices: 2,
            ..Default::default()
        },
    );
    assert!(p.patches.iter().all(|p| p.info.internal_vertex_count == 0));
    assert!(out.triangles.is_empty());
}
#[test]
fn rejects_corrupt_metadata_streams_and_overlapping_assembly() {
    let prepared = codec::prepare(&fixture(), &EncodingOptions::default()).unwrap();
    let p = &prepared.patches[0];
    let mut bad = p.clone();
    bad.info.internal_vertex_count = 0;
    assert!(codec::decode_patch(&prepared.schema, &bad.info, &bad.streams).is_err());
    let mut bad = p.clone();
    bad.streams[0][0] = u32::MAX;
    assert!(codec::decode_patch(&prepared.schema, &bad.info, &bad.streams).is_err());
    let decoded = codec::decode_patch(&prepared.schema, &p.info, &p.streams).unwrap();
    assert!(
        codec::assemble(
            &prepared.schema,
            &[decoded.clone(), decoded],
            prepared.materials
        )
        .is_err()
    );
}
#[test]
fn topology_transition_and_extreme_deltas() {
    let head = [[0, 1, 2], [2, 1, 3]];
    let tail = [[103, 200, 101]];
    let s = codec::encode_indices(&head, &tail, 100).unwrap();
    assert_eq!(
        codec::decode_indices(&s, 2, 100).unwrap(),
        (head.to_vec(), tail.to_vec())
    );
    assert!(codec::encode_indices(&[], &[[u32::MAX, 0, 0]], 0).is_err());
}

#[test]
fn standalone_spatial_helpers_reject_invalid_grids() {
    let inverse = [[0.; 4]; 4];
    for bits in [0, 1, 21, 32, 255] {
        assert!(spatial::positions(&[[0; 3]], &inverse, bits).is_err());
        assert!(spatial::quantize_uv(&[[0.; 2]], bits).is_err());
        assert!(spatial::tex_coords(&[[0; 2]], &[0.; 4], bits).is_err());
    }
    assert!(spatial::positions(&[[1024, 0, 0]], &inverse, 10).is_err());
    assert!(spatial::quantize_uv(&[[f32::NAN, 0.]], 10).is_err());
    assert!(spatial::tex_coords(&[[0; 2]], &[0., 0., -1., 1.], 10).is_err());
    let (q, bounds) = spatial::quantize_uv(&[[7., -2.]; 3], 10).unwrap();
    assert_eq!(q, [[0; 2]; 3]);
    assert_eq!(
        spatial::tex_coords(&q, &bounds, 10).unwrap(),
        [[7., -2.]; 3]
    );
}

#[test]
fn quantized_tangent_determinants_and_cancellation_are_exact() {
    let positions = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]];
    let normals = [[0., 0., 1.]; 3];
    let collinear = [[0, 0], [1_048_574, 524_287], [524_288, 262_144]];
    let fallback =
        tangent::predictors_quantized(&positions, &collinear, &normals, &[[0, 1, 2]]).unwrap();
    assert_eq!(fallback, [[0., 1., 0.]; 3]);
    let m = 1_048_575;
    // Products are approximately 2^40, while their exact difference is minus one.
    let minus_one = [[0, 0], [m, m - 1], [m - 1, m - 2]];
    let direction =
        tangent::predictors_quantized(&positions, &minus_one, &normals, &[[0, 1, 2]]).unwrap();
    assert!(direction[0][0] < -0.7 && direction[0][1] > 0.7);
    let plus_one = [[0, 0], [m - 1, m - 2], [m, m - 1]];
    let direction =
        tangent::predictors_quantized(&positions, &plus_one, &normals, &[[0, 1, 2]]).unwrap();
    assert!(direction[0][0] > 0.7 && direction[0][1] < -0.7);
    let positions = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
    ];
    let uv = [[100, 100], [200, 100], [100, 200], [0, 100], [100, 200]];
    let direction =
        tangent::predictors_quantized(&positions, &uv, &[[0., 0., 1.]; 5], &[[0, 1, 2], [0, 3, 4]])
            .unwrap();
    assert_eq!(direction[0], [0., 1., 0.]);
    let near_axis = [[1., 1. + f32::EPSILON, 1.]; 3];
    let direction =
        tangent::predictors_quantized(&positions[..3], &collinear, &near_axis, &[[0, 1, 2]])
            .unwrap();
    assert_eq!(direction[0], normal::frame(near_axis[0]).1);
}
