//! Integer spatial prediction shared by positions and texture coordinates.

use super::data::MeshError;

/// Unknown integer slots use this exact quiet-NaN bit pattern.
pub const SENTINEL: u32 = 0x7fc0_0001;

/// Encode absolute grid coordinates as wrapped triangle-predicted residuals.
pub fn encode<const D: usize>(
    head: &[[u32; 3]],
    values: &[[u32; D]],
    bits: u8,
) -> Result<Vec<u32>, MeshError> {
    process(head, values.len(), bits, Some(values), &[]).map(|(_, symbols, _)| symbols)
}

/// Decode residuals and return the first-appearance vertex order.
pub fn decode<const D: usize>(
    head: &[[u32; 3]],
    count: usize,
    bits: u8,
    symbols: &[u32],
) -> Result<(Vec<[u32; D]>, Vec<usize>), MeshError> {
    process(head, count, bits, None, symbols).map(|(values, _, order)| (values, order))
}

fn process<const D: usize>(
    head: &[[u32; 3]],
    count: usize,
    bits: u8,
    values: Option<&[[u32; D]]>,
    input: &[u32],
) -> Result<(Vec<[u32; D]>, Vec<u32>, Vec<usize>), MeshError> {
    if !(2..=20).contains(&bits) || D == 0 {
        return Err(MeshError::InvalidData("invalid spatial precision"));
    }
    let mask = (1u32 << bits) - 1;
    if values.is_none()
        && input.len()
            != count
                .checked_mul(D)
                .ok_or(MeshError::InvalidData("spatial size overflow"))?
    {
        return Err(MeshError::InvalidData("spatial stream length"));
    }
    let mut slots = vec![[SENTINEL; D]; count];
    let mut order = Vec::with_capacity(count);
    let mut output = Vec::new();
    let mut cursor = 0;
    let mut previous: Option<[u32; 3]> = None;
    for &triangle in head {
        if triangle.iter().any(|&v| v as usize >= count) {
            return Err(MeshError::InvalidData("local triangle index"));
        }
        let mut shared = Vec::with_capacity(3);
        if let Some(prev) = previous {
            for v in triangle {
                if prev.contains(&v) && !shared.contains(&v) {
                    shared.push(v);
                }
            }
        }
        let predictor = if shared.len() >= 2 {
            let opposite = previous.unwrap().into_iter().find(|v| !shared.contains(v));
            std::array::from_fn(|k| {
                slots[shared[0] as usize][k] as i64 + slots[shared[1] as usize][k] as i64
                    - opposite.map_or(0, |v| slots[v as usize][k] as i64)
            })
        } else if shared.len() == 1 {
            slots[shared[0] as usize].map(i64::from)
        } else {
            [0; D]
        };
        for (corner, id) in triangle.into_iter().enumerate() {
            let pred = if shared.is_empty() && corner != 0 {
                slots[triangle[0] as usize].map(i64::from)
            } else {
                predictor
            };
            emit(
                id as usize,
                pred,
                mask,
                values,
                input,
                &mut slots,
                &mut order,
                &mut output,
                &mut cursor,
            )?;
        }
        previous = Some(triangle);
    }
    for id in 0..count {
        let pred = order
            .last()
            .map_or([0; D], |&last| slots[last].map(i64::from));
        emit(
            id,
            pred,
            mask,
            values,
            input,
            &mut slots,
            &mut order,
            &mut output,
            &mut cursor,
        )?;
    }
    Ok((slots, output, order))
}

#[allow(clippy::too_many_arguments)]
fn emit<const D: usize>(
    id: usize,
    pred: [i64; D],
    mask: u32,
    values: Option<&[[u32; D]]>,
    input: &[u32],
    slots: &mut [[u32; D]],
    order: &mut Vec<usize>,
    output: &mut Vec<u32>,
    cursor: &mut usize,
) -> Result<(), MeshError> {
    if slots[id] != [SENTINEL; D] {
        return Ok(());
    }
    for k in 0..D {
        let symbol = if let Some(values) = values {
            if values[id][k] > mask {
                return Err(MeshError::InvalidData("spatial coordinate outside grid"));
            }
            let symbol = ((values[id][k] as i64 - pred[k]) & mask as i64) as u32;
            output.push(symbol);
            symbol
        } else {
            let symbol = input[*cursor];
            *cursor += 1;
            symbol
        };
        if symbol > mask {
            return Err(MeshError::InvalidData("spatial residual outside grid"));
        }
        slots[id][k] = ((pred[k] + symbol as i64) & mask as i64) as u32;
    }
    order.push(id);
    Ok(())
}

/// Reconstruct world positions using the serialized single-precision matrix.
pub fn positions(
    q: &[[u32; 3]],
    inverse: &[[f32; 4]; 4],
    bits: u8,
) -> Result<Vec<[f32; 3]>, MeshError> {
    let mask = levels(bits)?;
    if inverse.iter().flatten().any(|x| !x.is_finite()) || q.iter().flatten().any(|&x| x > mask) {
        return Err(MeshError::InvalidData("position transform or grid"));
    }
    let scale = mask as f32;
    let values: Vec<_> = q
        .iter()
        .map(|v| {
            let c = v.map(|x| x as f32 / scale);
            std::array::from_fn(|k| {
                ((c[0] * inverse[k][0] + c[1] * inverse[k][1]) + c[2] * inverse[k][2])
                    + inverse[k][3]
            })
        })
        .collect();
    if values.iter().flatten().any(|x| !x.is_finite()) {
        return Err(MeshError::InvalidData("nonfinite position reconstruction"));
    }
    Ok(values)
}

/// Fit UV coordinates with serialized offset and nonuniform scale.
pub fn quantize_uv(values: &[[f32; 2]], bits: u8) -> Result<(Vec<[u32; 2]>, [f32; 4]), MeshError> {
    let scale = levels(bits)? as f64;
    if values.iter().flatten().any(|x| !x.is_finite()) {
        return Err(MeshError::InvalidInput("nonfinite UV coordinate"));
    }
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for v in values {
        for k in 0..2 {
            min[k] = min[k].min(v[k]);
            max[k] = max[k].max(v[k]);
        }
    }
    if values.is_empty() {
        min = [0.; 2];
        max = [0.; 2];
    }
    let bounds = [min[0], min[1], max[0] - min[0], max[1] - min[1]];
    if bounds.iter().any(|x| !x.is_finite()) {
        return Err(MeshError::InvalidInput("UV transform exceeds f32 range"));
    }
    let q = values
        .iter()
        .map(|v| {
            std::array::from_fn(|k| {
                if bounds[k + 2] == 0. {
                    0
                } else {
                    (((v[k] as f64 - bounds[k] as f64) / bounds[k + 2] as f64).clamp(0., 1.)
                        * scale)
                        .round_ties_even() as u32
                }
            })
        })
        .collect();
    Ok((q, bounds))
}

/// Restore UV coordinates from their patch bounds.
pub fn tex_coords(q: &[[u32; 2]], bounds: &[f32; 4], bits: u8) -> Result<Vec<[f32; 2]>, MeshError> {
    let mask = levels(bits)?;
    if q.iter().flatten().any(|&x| x > mask)
        || bounds.iter().any(|x| !x.is_finite())
        || bounds[2] < 0.
        || bounds[3] < 0.
    {
        return Err(MeshError::InvalidData("UV transform or grid"));
    }
    let scale = mask as f32;
    let values: Vec<_> = q
        .iter()
        .map(|v| std::array::from_fn(|k| bounds[k] + v[k] as f32 / scale * bounds[k + 2]))
        .collect();
    if values.iter().flatten().any(|x| !x.is_finite()) {
        return Err(MeshError::InvalidData("nonfinite UV reconstruction"));
    }
    Ok(values)
}

fn levels(bits: u8) -> Result<u32, MeshError> {
    if !(2..=20).contains(&bits) {
        return Err(MeshError::InvalidData("spatial precision"));
    }
    Ok((1u32 << bits) - 1)
}
