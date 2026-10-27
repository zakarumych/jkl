//! Two signed rotations around a deterministic predicted-normal frame.
use super::data::MeshError;

pub(crate) fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|k| a[k] + b[k])
}
pub(crate) fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|k| a[k] - b[k])
}
pub(crate) fn mul(a: [f32; 3], s: f32) -> [f32; 3] {
    a.map(|x| x * s)
}
pub(crate) fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}
pub(crate) fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub(crate) fn unit(a: [f32; 3]) -> [f32; 3] {
    let length = dot(a, a).sqrt();
    if length > 0. && length.is_finite() {
        mul(a, 1. / length)
    } else {
        [0., 0., 1.]
    }
}

/// Deterministic predicted-normal frame with `v = u × n`; ties prefer X, Y, then Z.
pub fn frame(predicted: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let n = unit(predicted);
    let mut axis = 0;
    for k in 1..3 {
        if n[k].abs() < n[axis].abs() {
            axis = k;
        }
    }
    let mut a = [0.; 3];
    a[axis] = 1.;
    let u = unit(cross(n, a));
    (n, u, cross(u, n))
}
pub(crate) fn zigzag(v: i64) -> Result<u32, MeshError> {
    u32::try_from(if v >= 0 { v * 2 } else { -v * 2 - 1 })
        .map_err(|_| MeshError::InvalidData("signed residual overflow"))
}
pub(crate) fn unzigzag(v: u32) -> i64 {
    if v & 1 == 0 {
        (v / 2) as i64
    } else {
        -((v / 2) as i64) - 1
    }
}
pub(crate) fn angle_encode(angle: f32, bits: u8) -> u32 {
    let n = 1i64 << bits;
    let q = (angle * ((n as f32) / std::f32::consts::TAU)).round_ties_even() as i64;
    zigzag((q + n / 2).rem_euclid(n) - n / 2).unwrap()
}
pub(crate) fn angle_decode(symbol: u32, bits: u8) -> Result<f32, MeshError> {
    let q = unzigzag(symbol);
    let half = 1i64 << (bits - 1);
    if q < -half || q >= half {
        return Err(MeshError::InvalidData("angular residual outside grid"));
    }
    Ok(q as f32 * (std::f32::consts::TAU / (1u32 << bits) as f32))
}

/// Encode a direction as signed latitude and longitude around its predictor.
pub fn encode(reference: [f32; 3], predicted: [f32; 3], bits: u8) -> Result<[u32; 2], MeshError> {
    if !(2..=20).contains(&bits) {
        return Err(MeshError::InvalidInput("normal precision"));
    }
    let (n, u, v) = frame(predicted);
    let r = unit(reference);
    let qu = dot(r, u);
    let qv = dot(r, v);
    let qn = dot(r, n);
    let radius = qu.hypot(qn);
    let limit = (1i64 << (bits - 1)) - 1;
    let a =
        (qv.atan2(radius) * (2. * limit as f32 / std::f32::consts::PI)).round_ties_even() as i64;
    let b = if radius <= 8. * f32::EPSILON || a.abs() == limit {
        0
    } else {
        angle_encode((-qu).atan2(qn), bits)
    };
    Ok([zigzag(a)?, b])
}

/// Decode a pair of signed angular residuals.
pub fn decode(symbols: [u32; 2], predicted: [f32; 3], bits: u8) -> Result<[f32; 3], MeshError> {
    if !(2..=20).contains(&bits) {
        return Err(MeshError::InvalidData("normal precision"));
    }
    let limit = (1i64 << (bits - 1)) - 1;
    let aq = unzigzag(symbols[0]);
    if aq.abs() > limit {
        return Err(MeshError::InvalidData("normal latitude outside grid"));
    }
    let a = aq as f32 * (std::f32::consts::PI / (2. * limit as f32));
    let b = angle_decode(symbols[1], bits)?;
    let (n, u, v) = frame(predicted);
    Ok(unit(add(
        sub(mul(n, a.cos() * b.cos()), mul(u, a.cos() * b.sin())),
        mul(v, a.sin()),
    )))
}

/// Accumulate area-weighted geometric normals in original triangle winding.
pub fn geometric(positions: &[[f32; 3]], head: &[[u32; 3]]) -> Result<Vec<[f32; 3]>, MeshError> {
    let mut out = vec![[0.; 3]; positions.len()];
    for (i, &t) in head.iter().enumerate() {
        let mut t = t.map(|x| x as usize);
        if t.iter().any(|&x| x >= positions.len()) {
            return Err(MeshError::InvalidData("normal triangle index"));
        }
        if i & 1 != 0 {
            t.swap(1, 2);
        }
        let g = cross(
            sub(positions[t[1]], positions[t[0]]),
            sub(positions[t[2]], positions[t[0]]),
        );
        for id in t {
            out[id] = add(out[id], g);
        }
    }
    Ok(out.into_iter().map(unit).collect())
}

pub(crate) fn stream(
    reference: Option<&[[f32; 3]]>,
    predictors: &[[f32; 3]],
    order: &[usize],
    internal: usize,
    bits: u8,
    input: &[u32],
) -> Result<(Vec<u32>, Vec<[f32; 3]>), MeshError> {
    if reference.is_none() && input.len() != order.len() * 2 {
        return Err(MeshError::InvalidData("normal stream length"));
    }
    let mut decoded = vec![[0.; 3]; order.len()];
    let mut symbols = Vec::with_capacity(order.len() * 2);
    for (j, &id) in order.iter().enumerate() {
        let predicted = if j < internal {
            predictors[id]
        } else if j > 0 {
            decoded[order[j - 1]]
        } else {
            [0., 0., 1.]
        };
        let s = if let Some(r) = reference {
            encode(r[id], predicted, bits)?
        } else {
            [input[j * 2], input[j * 2 + 1]]
        };
        decoded[id] = decode(s, predicted, bits)?;
        symbols.extend(s);
    }
    Ok((symbols, decoded))
}
