use jkl::mesh::fit::{CONTAINMENT_TOLERANCE, FitError, PatchFit, fit_patch, transform_point};

fn rotated(p: [f64; 3]) -> [f64; 3] {
    let (s, c) = 0.61f64.sin_cos();
    let (t, d) = 0.37f64.sin_cos();
    let x = c * p[0] - s * p[1];
    let y = s * p[0] + c * p[1];
    [x + 1.25, d * y - t * p[2] - 0.7, t * y + d * p[2] + 0.2]
}

fn check(points: &[[f64; 3]], bits: u8) -> PatchFit {
    let patch = fit_patch(points, bits).unwrap();
    assert_eq!(patch.bits, bits);
    assert!((2..=3202).contains(&patch.evaluations));
    assert!(
        patch.refined.errors.mean_squared_distance <= patch.axis_seed.errors.mean_squared_distance
    );
    assert!(
        patch.refined.errors.mean_squared_distance
            <= patch.diagonal_seed.errors.mean_squared_distance
    );
    let q = (1u32 << bits) - 1;
    for fit in [&patch.axis_seed, &patch.diagonal_seed, &patch.refined] {
        assert_eq!(fit.forward[3], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(fit.inverse[3], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(fit.quantized.len(), points.len());
        for row in 0..4 {
            for column in 0..4 {
                let product: f64 = (0..4)
                    .map(|k| fit.inverse[row][k] * fit.forward[k][column])
                    .sum();
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!(
                    (product - expected).abs() < 2e-7,
                    "inverse product [{row},{column}] = {product}"
                );
            }
        }
        let mut mean = 0.0;
        let mut mse = 0.0;
        let mut max = 0.0f64;
        for (&point, &encoded) in points.iter().zip(&fit.quantized) {
            let u = fit.normalized_point(point);
            assert!(
                u.iter()
                    .all(|v| (-CONTAINMENT_TOLERANCE..=1.0 + CONTAINMENT_TOLERANCE).contains(v)),
                "{u:?}"
            );
            assert!(encoded.iter().all(|&v| v <= q));
            assert_eq!(fit.encode_point(point, bits).unwrap(), encoded);
            let roundtrip = transform_point(&fit.inverse, u);
            for i in 0..3 {
                assert!(
                    (roundtrip[i] - point[i]).abs() < 2e-7 * (1.0 + point[i].abs()),
                    "{roundtrip:?} vs {point:?}"
                );
            }
            let decoded = fit.decode_point(encoded, bits);
            let d = (decoded[0] - point[0])
                .hypot(decoded[1] - point[1])
                .hypot(decoded[2] - point[2]);
            mean += d / points.len() as f64;
            mse += d * d / points.len() as f64;
            max = max.max(d);
        }
        assert_eq!(fit.errors.mean_distance, mean);
        assert_eq!(fit.errors.mean_squared_distance, mse);
        assert_eq!(fit.errors.max_distance, max);
    }
    patch
}

#[test]
fn full_pca_frame_aligns_rotated_anisotropic_box() {
    let corners: Vec<_> = (0..8)
        .map(|i| {
            rotated([
                if i & 1 == 0 { -10.0 } else { 10.0 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { -0.1 } else { 0.1 },
            ])
        })
        .collect();
    let patch = check(&corners, 8);
    for point in corners {
        for coordinate in patch.axis_seed.normalized_point(point) {
            assert!(coordinate.abs().min((1.0 - coordinate).abs()) < 1e-10);
        }
    }
    assert!(patch.axis_seed.errors.max_distance < 1e-10);
}

#[test]
fn rotated_anisotropic_cloud_preserves_contract_and_is_deterministic() {
    let points: Vec<_> = (0..80)
        .map(|i| {
            rotated([
                (i * 37 % 83) as f64 / 9.0,
                (i * 17 % 79) as f64 / 100.0,
                (i * 11 % 71) as f64 / 1000.0,
            ])
        })
        .collect();
    for bits in [1, 8, 12, 16, 17, 20] {
        check(&points, bits);
    }
    assert_eq!(
        fit_patch(&points, 8).unwrap(),
        fit_patch(&points, 8).unwrap()
    );
}

#[test]
fn degenerate_patches_remain_invertible_without_midpoint_error() {
    let constant = [[2.0, -3.0, 7.0]; 5];
    let patch = check(&constant, 8);
    assert_eq!(patch.axis_seed.quantized, vec![[0; 3]; 5]);
    assert_eq!(patch.refined.errors.max_distance, 0.0);
    let plane: Vec<_> = (0..36)
        .map(|i| rotated([(i % 6) as f64, (i / 6) as f64, 0.0]))
        .collect();
    check(&plane, 8);
    let line: Vec<_> = (0..32)
        .map(|i| rotated([i as f64 / 10.0, 0.0, 0.0]))
        .collect();
    check(&line, 8);
    check(&[[0.0; 3]], 1);
}

#[test]
fn refined_fit_improves_nontrivial_quantization() {
    let points: Vec<_> = (0..45)
        .map(|i| {
            rotated([
                (i as f64 * 0.73).sin(),
                (i as f64 * 0.31).cos(),
                (i as f64 * 0.43).sin() * 0.3,
            ])
        })
        .collect();
    let patch = check(&points, 6);
    assert!(
        patch.refined.errors.mean_squared_distance
            < patch
                .axis_seed
                .errors
                .mean_squared_distance
                .min(patch.diagonal_seed.errors.mean_squared_distance)
    );
}

#[test]
fn affine_matrix_layout_is_row_major() {
    let m = [
        [2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0, 13.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    assert_eq!(transform_point(&m, [1.0, 2.0, 3.0]), [25.0, 53.0, 81.0]);
}

#[test]
fn invalid_and_unrepresentable_inputs_are_explicit_errors() {
    for bits in [0, 21, 255] {
        assert_eq!(fit_patch(&[[0.0; 3]], bits), Err(FitError::InvalidBits));
    }
    assert_eq!(fit_patch(&[], 8), Err(FitError::EmptyPatch));
    for component in 0..3 {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut points = [[0.0; 3]; 2];
            points[1][component] = value;
            assert_eq!(
                fit_patch(&points, 8),
                Err(FitError::NonFiniteCoordinate {
                    vertex: 1,
                    component
                })
            );
        }
    }
    assert_eq!(
        fit_patch(&[[-f64::MAX; 3], [f64::MAX; 3]], 8),
        Err(FitError::NumericalRange)
    );
    assert_eq!(
        fit_patch(&[[0.0; 3], [f64::from_bits(1); 3]], 8),
        Err(FitError::NumericalRange)
    );
    let patch = fit_patch(&[[0.0; 3], [1.0; 3]], 8).unwrap();
    assert_eq!(
        patch.refined.encode_point([1000.0; 3], 8),
        Err(FitError::OutsideCube)
    );
    assert_eq!(
        patch.refined.encode_point([0.0; 3], 0),
        Err(FitError::InvalidBits)
    );
}
