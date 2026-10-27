use jkl::mesh::{
    data::{EncodingOptions, MeshData, weld},
    layout::build_layout,
};

#[test]
fn weld_preserves_attribute_seams_and_primitive_metadata() {
    let mut mesh = MeshData {
        positions: vec![[0.; 3]; 7],
        normals: Some(vec![[0., 0., 1.]; 7]),
        tex_coords: vec![vec![[0., 0.]; 7]],
        colors: Some(vec![[255; 4]; 7]),
        ..Default::default()
    };
    mesh.positions[1][0] = 0.00001;
    mesh.normals.as_mut().unwrap()[2] = [1., 0., 0.];
    mesh.tex_coords[0][3] = [1., 0.];
    mesh.colors.as_mut().unwrap()[4] = [0; 4];
    mesh.triangles = vec![[0, 1, 2], [3, 4, 5], [6, 6, 6]];
    mesh.primitive_ids = vec![0, 0, 1];
    mesh.materials = vec![2, 3];
    let welded = weld(&mesh, 0.001).unwrap();
    assert_eq!(welded.old_to_new, vec![0, 0, 1, 2, 3, 0, 0]);
    assert_eq!(welded.source_vertices, vec![0, 2, 3, 4]);
    assert_eq!(welded.mesh.primitive_ids, mesh.primitive_ids);
    assert_eq!(welded.mesh.materials, mesh.materials);
    assert_eq!(weld(&mesh, 0.001).unwrap(), welded);
}

#[test]
fn weld_uses_euclidean_distance_and_first_representative() {
    let mesh = MeshData {
        positions: vec![[0., 0., 0.], [0.8, 0.8, 0.], [0.5, 0., 0.], [1., 0., 0.]],
        ..Default::default()
    };
    let w = weld(&mesh, 1.).unwrap();
    assert_eq!(w.old_to_new, vec![0, 1, 0, 0]);
    let extreme = MeshData {
        positions: vec![[f32::MAX, 0., 0.], [f32::MAX, 0., 0.], [-f32::MAX, 0., 0.]],
        ..Default::default()
    };
    assert_eq!(
        weld(&extreme, f32::from_bits(1)).unwrap().old_to_new,
        vec![0, 0, 1]
    );
}

fn cycle(mut t: [u32; 3]) -> [u32; 3] {
    let mut best = t;
    for _ in 0..2 {
        t = [t[1], t[2], t[0]];
        best = best.min(t);
    }
    best
}
fn check_layout(mesh: &MeshData, cap: usize) {
    let l = build_layout(mesh, cap).unwrap();
    assert_eq!(l, build_layout(mesh, cap).unwrap());
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut vb = 0;
    let mut tb = 0;
    for p in &l.patches {
        assert_eq!(p.vertex_base, vb);
        assert_eq!(p.triangle_base, tb);
        assert!(p.vertices.len() <= cap);
        vertices.extend(p.vertices.iter().copied());
        let mut encountered = std::collections::BTreeSet::new();
        for t in &p.head {
            for &v in t {
                assert!((v as usize) < p.vertices.len());
                if !encountered.contains(&v) {
                    assert_eq!(
                        v,
                        encountered.len() as u32,
                        "local IDs follow stored first appearance"
                    );
                    encountered.insert(v);
                }
            }
        }
        assert_eq!(encountered.len(), p.internal_vertex_count as usize);
        assert_eq!(
            encountered.into_iter().collect::<Vec<_>>(),
            (0..p.internal_vertex_count).collect::<Vec<_>>()
        );
        for (ordinal, (stored, &f)) in p
            .head
            .iter()
            .chain(&p.tail)
            .zip(p.head_face_ids.iter().chain(&p.tail_face_ids))
            .enumerate()
        {
            let mut decoded = *stored;
            if ordinal < p.head.len() {
                decoded = decoded.map(|v| v + p.vertex_base);
            }
            if ordinal & 1 != 0 {
                decoded.swap(1, 2);
            }
            assert_eq!(
                cycle(decoded),
                cycle(mesh.triangles[f as usize].map(|v| l.old_to_new[v as usize]))
            );
            faces.push(f);
        }
        vb += p.vertices.len() as u32;
        tb += (p.head.len() + p.tail.len()) as u32;
    }
    vertices.sort_unstable();
    faces.sort_unstable();
    assert_eq!(
        vertices,
        (0..mesh.positions.len() as u32).collect::<Vec<_>>()
    );
    assert_eq!(faces, (0..mesh.triangles.len() as u32).collect::<Vec<_>>());
}

#[test]
fn layout_empty_loose_internal_and_cross_patch_faces() {
    check_layout(&MeshData::default(), 256);
    let mut mesh = MeshData {
        positions: (0..17).map(|v| [v as f32, (v % 3) as f32, 0.]).collect(),
        ..Default::default()
    };
    check_layout(&mesh, 4);
    mesh.triangles = vec![
        [0, 1, 2],
        [2, 1, 3],
        [3, 4, 5],
        [1, 3, 5],
        [6, 7, 8],
        [8, 9, 10],
        [1, 12, 16],
        [0, 0, 1],
        [11, 11, 11],
    ];
    for cap in [1, 2, 4, 8, 256] {
        check_layout(&mesh, cap);
    }
    let l = build_layout(&mesh, 1).unwrap();
    assert!(
        l.patches
            .iter()
            .any(|p| p.head.is_empty() && !p.tail.is_empty())
    );
}

#[test]
fn validates_options_attributes_and_metadata() {
    let mut mesh = MeshData {
        positions: vec![[0.; 3]],
        ..Default::default()
    };
    let mut o = EncodingOptions::default();
    o.validate(&mesh).unwrap();
    o.position_bits = 20;
    o.validate(&mesh).unwrap();
    o.position_bits = 21;
    assert!(o.validate(&mesh).is_err());
    mesh.normals = Some(vec![]);
    assert!(mesh.validate().is_err());
    mesh.normals = Some(vec![[0.; 3]]);
    mesh.tangents = Some(vec![[1., 0., 0., 1.]]);
    assert!(EncodingOptions::default().validate(&mesh).is_err());
    mesh.tex_coords.push(vec![[0.; 2]]);
    EncodingOptions::default().validate(&mesh).unwrap();
    mesh.triangles.push([0, 0, 0]);
    mesh.primitive_ids.push(2);
    mesh.materials.push(1);
    assert!(mesh.validate().is_err());
    assert!(weld(&MeshData::default(), 0.).is_err());
    assert!(build_layout(&MeshData::default(), 257).is_err());
}
