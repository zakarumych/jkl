use jkl::mesh::order::{OrderError, order_quantized};

fn score(points: &[[u32; 3]], order: &[usize]) -> f64 {
    let mut sum = 0.0;
    for edge in order.windows(2) {
        for axis in 0..3 {
            let delta = i64::from(points[edge[0]][axis]) - i64::from(points[edge[1]][axis]);
            sum += (1.0 + delta.abs() as f64).log2();
        }
    }
    sum
}

#[test]
fn full_grids_follow_continuous_hilbert_curve() {
    for bits in 1..=4 {
        let side = 1 << bits;
        let points: Vec<_> = (0..side)
            .flat_map(|x| (0..side).flat_map(move |y| (0..side).map(move |z| [x, y, z])))
            .collect();
        let order = order_quantized(&points, bits).unwrap();
        let mut sorted = order.hilbert.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..points.len()).collect::<Vec<_>>());
        for edge in order.hilbert.windows(2) {
            let a = points[edge[0]];
            let b = points[edge[1]];
            assert_eq!((0..3).map(|i| a[i].abs_diff(b[i])).sum::<u32>(), 1);
        }
    }
}

#[test]
fn one_bit_curve_has_expected_orientation() {
    let points = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ];
    assert_eq!(
        order_quantized(&points, 1).unwrap().hilbert,
        [0, 1, 3, 2, 6, 7, 5, 4]
    );
}

#[test]
fn validation_and_degenerate_inputs() {
    for bits in [0, 21, 255] {
        assert_eq!(order_quantized(&[], bits), Err(OrderError::InvalidBits));
    }
    assert_eq!(
        order_quantized(&[[0, 7, 0], [8, 0, 0]], 3),
        Err(OrderError::CoordinateOutOfRange {
            vertex: 1,
            component: 0
        })
    );
    assert_eq!(
        order_quantized(&[[0, u32::MAX, 0]], 16),
        Err(OrderError::CoordinateOutOfRange {
            vertex: 0,
            component: 1
        })
    );
    for points in [vec![], vec![[65535; 3]], vec![[65535; 3]; 99]] {
        let result = order_quantized(&points, 16).unwrap();
        assert_eq!(result.hilbert, (0..points.len()).collect::<Vec<_>>());
        assert_eq!(result.refined, result.hilbert);
        assert_eq!(result.hilbert_cost, 0.0);
        assert_eq!(result.refined_cost, 0.0);
        assert_eq!(result.accepted_moves, 0);
    }
}

#[test]
fn deterministic_permutations_and_independently_measured_improvement() {
    let mut seed = 17u32;
    let mut improved = false;
    for bits in [1, 6, 8, 10, 16, 17, 20] {
        for count in [2, 3, 17, 32, 33, 256] {
            let maximum = (1 << bits) - 1;
            let points: Vec<[u32; 3]> = (0..count)
                .map(|_| {
                    std::array::from_fn(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        seed & maximum
                    })
                })
                .collect();
            let saved = points.clone();
            let result = order_quantized(&points, bits).unwrap();
            assert_eq!(result, order_quantized(&points, bits).unwrap());
            assert_eq!(points, saved);
            for order in [&result.hilbert, &result.refined] {
                let mut sorted = order.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..count).collect::<Vec<_>>());
            }
            let before = score(&points, &result.hilbert);
            let after = score(&points, &result.refined);
            assert!((before - result.hilbert_cost).abs() < 1e-9);
            assert!((after - result.refined_cost).abs() < 1e-9);
            assert!(after <= before + 1e-9);
            assert!(result.passes <= 4);
            improved |= after < before - 1e-9;
        }
    }
    assert!(improved);
}
