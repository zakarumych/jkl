//! UV/position-predicted tangent angles with exact handedness.
use super::{
    data::MeshError,
    normal::{self, add, cross, dot, mul, sub, unit},
};

/// Build tangent zero-angle directions from decoded attributes.
pub fn predictors(
    positions: &[[f32; 3]],
    uv: &[[f32; 2]],
    normals: &[[f32; 3]],
    head: &[[u32; 3]],
) -> Result<Vec<[f32; 3]>, MeshError> {
    if positions.len() != uv.len() || positions.len() != normals.len() {
        return Err(MeshError::InvalidData("tangent attribute lengths"));
    }
    let mut sums = vec![[0.; 3]; positions.len()];
    for (i, &t) in head.iter().enumerate() {
        let mut t = t.map(|x| x as usize);
        if t.iter().any(|&x| x >= positions.len()) {
            return Err(MeshError::InvalidData("tangent triangle index"));
        }
        if i & 1 != 0 {
            t.swap(1, 2);
        }
        let [a, b, c] = t;
        let e1 = sub(positions[b], positions[a]);
        let e2 = sub(positions[c], positions[a]);
        let d1 = [uv[b][0] - uv[a][0], uv[b][1] - uv[a][1]];
        let d2 = [uv[c][0] - uv[a][0], uv[c][1] - uv[a][1]];
        let det = d1[0] * d2[1] - d1[1] * d2[0];
        if det == 0. || !det.is_finite() {
            continue;
        }
        let tangent = mul(sub(mul(e1, d2[1]), mul(e2, d1[1])), 1. / det);
        if tangent.iter().any(|x| !x.is_finite()) {
            continue;
        }
        for id in t {
            sums[id] = add(sums[id], tangent);
        }
    }
    Ok(project(sums, normals))
}

/// Build tangent predictions directly from the decoded UV integer lattice.
///
/// Determinants use exact signed 64-bit arithmetic for grids through 20 bits.
/// The common positive factor `(2^bits - 1) / patch_u_extent` cancels during
/// normalization. Degenerate UV triangles therefore have an exact shared
/// decision on CPU and GPU, independent of UV offset reconstruction roundoff.
pub fn predictors_quantized(
    positions: &[[f32; 3]],
    uv: &[[u32; 2]],
    normals: &[[f32; 3]],
    head: &[[u32; 3]],
) -> Result<Vec<[f32; 3]>, MeshError> {
    if positions.len() != uv.len() || positions.len() != normals.len() {
        return Err(MeshError::InvalidData("tangent attribute lengths"));
    }
    if uv.iter().flatten().any(|&v| v >= 1 << 20) {
        return Err(MeshError::InvalidData("tangent UV grid exceeds 20 bits"));
    }
    let mut sums = vec![[0.; 3]; positions.len()];
    for (i, &triangle) in head.iter().enumerate() {
        let mut t = triangle.map(|v| v as usize);
        if t.iter().any(|&v| v >= positions.len()) {
            return Err(MeshError::InvalidData("tangent triangle index"));
        }
        if i & 1 != 0 {
            t.swap(1, 2);
        }
        let [a, b, c] = t;
        let d1 = [
            uv[b][0] as i64 - uv[a][0] as i64,
            uv[b][1] as i64 - uv[a][1] as i64,
        ];
        let d2 = [
            uv[c][0] as i64 - uv[a][0] as i64,
            uv[c][1] as i64 - uv[a][1] as i64,
        ];
        let determinant = d1[0] * d2[1] - d1[1] * d2[0];
        if determinant == 0 {
            continue;
        }
        let numerator = sub(
            mul(sub(positions[b], positions[a]), d2[1] as f32),
            mul(sub(positions[c], positions[a]), d1[1] as f32),
        );
        let tangent = mul(numerator, 1. / determinant as f32);
        if tangent.iter().any(|x| !x.is_finite()) {
            continue;
        }
        for id in t {
            sums[id] = add(sums[id], tangent);
        }
    }
    Ok(project(sums, normals))
}

fn project(sums: Vec<[f32; 3]>, normals: &[[f32; 3]]) -> Vec<[f32; 3]> {
    sums.into_iter()
        .zip(normals)
        .map(|(sum, &n)| {
            let projected = sub(sum, mul(n, dot(sum, n)));
            if dot(projected, projected) > 0. && projected.iter().all(|x| x.is_finite()) {
                unit(projected)
            } else {
                normal::frame(n).1
            }
        })
        .collect()
}

/// Encode a tangent as one periodic linear angle.
pub fn encode(
    tangent: [f32; 4],
    normal: [f32; 3],
    zero: [f32; 3],
    bits: u8,
) -> Result<(u32, u32), MeshError> {
    if !(2..=20).contains(&bits) || !(tangent[3] == 1. || tangent[3] == -1.) {
        return Err(MeshError::InvalidInput("tangent precision or handedness"));
    }
    let t = [tangent[0], tangent[1], tangent[2]];
    let angle = dot(t, cross(zero, normal)).atan2(dot(t, zero));
    Ok((
        normal::angle_encode(angle, bits),
        u32::from(tangent[3] < 0.),
    ))
}
/// Decode a tangent angle and its exact sign.
pub fn decode(
    angle: u32,
    sign: u32,
    normal: [f32; 3],
    zero: [f32; 3],
    bits: u8,
) -> Result<[f32; 4], MeshError> {
    if !(2..=20).contains(&bits) || sign > 1 {
        return Err(MeshError::InvalidData("tangent precision or handedness"));
    }
    let a = normal::angle_decode(angle, bits)?;
    let t = unit(add(mul(zero, a.cos()), mul(cross(zero, normal), a.sin())));
    Ok([t[0], t[1], t[2], if sign == 0 { 1. } else { -1. }])
}
