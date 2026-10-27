use jkl::mesh::{SplitPositionsError, split_positions};

fn assert_partition(positions: &[[f64; 3]], cap: usize) -> Vec<Vec<usize>> {
    let patches = split_positions(positions, cap).unwrap();
    let count = positions.len().div_ceil(cap);
    assert_eq!(patches.len(), count);
    let mut indices: Vec<_> = patches.iter().flatten().copied().collect();
    indices.sort_unstable();
    assert_eq!(indices, (0..positions.len()).collect::<Vec<_>>());
    if count != 0 {
        let base = positions.len() / count;
        assert!(patches.iter().all(|p| p.len() <= cap && p.len() >= base));
        assert!(patches.iter().all(|p| p.len() <= base + 1));
    }
    patches
}

#[test]
fn exhaustive_coverage_cap_and_global_balance() {
    let positions: Vec<_> = (0..257)
        .map(|i| [(i * 97 % 257) as f64, (i % 13) as f64, -(i as f64)])
        .collect();
    for n in 0..=positions.len() {
        for cap in (1..=65).chain([usize::MAX]) {
            assert_partition(&positions[..n], cap);
        }
    }
}

#[test]
fn duplicates_and_coordinate_ties_are_deterministic() {
    for positions in [
        vec![[7.0; 3]; 73],
        (0..73).map(|i| [(i % 3) as f64, 0.0, 0.0]).collect(),
        (0..73)
            .map(|i| [if i % 2 == 0 { -0.0 } else { 0.0 }, 0.0, 0.0])
            .collect(),
    ] {
        let expected = assert_partition(&positions, 8);
        for _ in 0..5 {
            assert_eq!(split_positions(&positions, 8).unwrap(), expected);
        }
    }
    let patches = split_positions(&[[1.0; 3]; 12], 3).unwrap();
    for (patch, expected) in patches.iter().zip((0..12).collect::<Vec<_>>().chunks(3)) {
        let mut sorted = patch.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, expected);
    }
}

#[test]
fn separated_clusters_stay_separate_despite_interleaved_input() {
    let positions: Vec<_> = (0..16)
        .flat_map(|i| {
            [
                [-1000.0 + i as f64, (i % 4) as f64, 0.0],
                [1000.0 + i as f64, (i % 4) as f64, 0.0],
            ]
        })
        .collect();
    let patches = assert_partition(&positions, 16);
    assert!(patches[0].iter().all(|&i| positions[i][0] < 0.0));
    assert!(patches[1].iter().all(|&i| positions[i][0] > 0.0));
}

#[test]
fn planar_linear_and_identical_positions_are_valid() {
    for positions in [
        (0..100)
            .map(|i| [(i % 10) as f64, (i / 10) as f64, 0.0])
            .collect(),
        (0..100).map(|i| [0.0, 0.0, i as f64]).collect(),
        vec![[0.0; 3]; 100],
    ] {
        assert_partition(&positions, 13);
    }
    assert!(split_positions(&[], 1).unwrap().is_empty());
    assert_eq!(
        split_positions(&[[1.0; 3]], usize::MAX).unwrap(),
        vec![vec![0]]
    );
}

#[test]
fn extreme_finite_spans_choose_longest_axis_without_overflow() {
    // Both X and Y spans overflow, but Y is longer and has different rank order.
    let m = f64::MAX;
    let positions = [
        [-0.75 * m, -m, 0.0],
        [0.75 * m, -m, 0.0],
        [-0.75 * m, m, 0.0],
        [0.75 * m, m, 0.0],
    ];
    let patches = assert_partition(&positions, 2);
    assert!(patches[0].iter().all(|&i| i < 2));
    assert!(patches[1].iter().all(|&i| i >= 2));

    // Tiny spans must not disappear through blanket rescaling.
    let tiny = f64::from_bits(1);
    let positions = [[0.0, tiny, 0.0], [0.0, 0.0, 0.0]];
    assert_eq!(
        split_positions(&positions, 1).unwrap(),
        vec![vec![1], vec![0]]
    );
}

#[test]
fn invalid_inputs_report_vertex_and_component() {
    assert_eq!(
        split_positions(&[], 0),
        Err(SplitPositionsError::ZeroMaxVertices)
    );
    assert_eq!(
        split_positions(&[[f64::NAN; 3]], 0),
        Err(SplitPositionsError::ZeroMaxVertices)
    );
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for component in 0..3 {
            let mut positions = [[0.0; 3]; 4];
            positions[2][component] = value;
            let error = split_positions(&positions, 2).unwrap_err();
            assert_eq!(
                error,
                SplitPositionsError::NonFiniteCoordinate {
                    vertex: 2,
                    component
                }
            );
            assert!(error.to_string().contains("vertex 2"));
            let _: &dyn std::error::Error = &error;
        }
    }
    assert_eq!(
        split_positions(&[[0.0, f64::NAN, f64::INFINITY]], 1),
        Err(SplitPositionsError::NonFiniteCoordinate {
            vertex: 0,
            component: 1
        })
    );
}

#[test]
fn permutation_preserves_triangles_and_vertex_attributes() {
    let positions = [
        [2.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-2.0, 0.0, 0.0],
    ];
    let attributes = [10, 20, 30, 40];
    let triangles = [[0, 1, 2], [1, 3, 2]];
    let patches = split_positions(&positions, 2).unwrap();
    let permutation: Vec<_> = patches.into_iter().flatten().collect();
    assert_ne!(permutation, vec![0, 1, 2, 3]);
    let reordered_positions: Vec<_> = permutation.iter().map(|&i| positions[i]).collect();
    let reordered_attributes: Vec<_> = permutation.iter().map(|&i| attributes[i]).collect();
    let mut inverse = vec![0; positions.len()];
    for (new, &old) in permutation.iter().enumerate() {
        inverse[old] = new;
    }
    let remapped = triangles.map(|triangle| triangle.map(|old| inverse[old]));
    for (original, reordered) in triangles.iter().zip(remapped) {
        for (&old, new) in original.iter().zip(reordered) {
            assert_eq!(positions[old], reordered_positions[new]);
            assert_eq!(attributes[old], reordered_attributes[new]);
        }
    }
    // Shared triangle corners retain one shared vertex after remapping.
    assert_eq!(remapped[0][1], remapped[1][0]);
    assert_eq!(remapped[0][2], remapped[1][2]);
}
