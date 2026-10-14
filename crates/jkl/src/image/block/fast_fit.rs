//! Principal-axis initialization, least-squares fitting, and discrete endpoint refinement.
#![allow(clippy::needless_range_loop)]
use crate::math::{Rgb32F, Rgb565, Vec3, pca_axis3};

#[derive(Clone, Copy)]
struct Candidate<const N: usize> {
    endpoints: [u16; 2],
    indices: [u8; 16],
    error: f32,
}

fn snap(v: f32, maximum: u16) -> u16 {
    (v.clamp(0.0, 1.0) * maximum as f32).round() as u16
}
fn rgb_snap(v: [f32; 3]) -> u16 {
    (snap(v[0], 31) << 11) | (snap(v[1], 63) << 5) | snap(v[2], 31)
}
fn rgb_value(q: u16) -> [f32; 3] {
    [
        (q >> 11) as f32 / 31.0,
        ((q >> 5) & 63) as f32 / 63.0,
        (q & 31) as f32 / 31.0,
    ]
}
fn weight(k: usize, count: usize) -> Option<f32> {
    match k {
        0 => Some(0.0),
        1 => Some(1.0),
        _ if count == 6 && k >= 6 => None,
        _ => Some(
            (k - 1) as f32
                / match count {
                    3 => 2.0,
                    4 => 3.0,
                    6 => 5.0,
                    _ => 7.0,
                },
        ),
    }
}
fn evaluate<const N: usize>(
    samples: &[[f32; N]; 16],
    valid: &[bool; 16],
    endpoints: [u16; 2],
    rgb: bool,
    three: bool,
) -> Candidate<N> {
    let count = if rgb {
        if endpoints[0] == endpoints[1] {
            1
        } else if three {
            3
        } else {
            4
        }
    } else if endpoints[0] > endpoints[1] {
        8
    } else {
        6
    };
    let mut a = [0.0; N];
    let mut b = [0.0; N];
    if rgb {
        let av = rgb_value(endpoints[0]);
        let bv = rgb_value(endpoints[1]);
        for j in 0..N {
            a[j] = av[j];
            b[j] = bv[j];
        }
    } else {
        a[0] = endpoints[0] as f32 / 255.0;
        b[0] = endpoints[1] as f32 / 255.0;
    }
    let mut palette = [[0.0; N]; 8];
    for k in 0..if rgb { count } else { 8 } {
        for j in 0..N {
            palette[k][j] = match weight(k, count) {
                Some(t) => a[j] * (1.0 - t) + b[j] * t,
                None => {
                    if k == 7 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
        }
    }
    let mut c = Candidate {
        endpoints,
        indices: [0; 16],
        error: 0.0,
    };
    for i in 0..16 {
        if !valid[i] {
            continue;
        }
        let mut best = f32::INFINITY;
        for k in 0..if rgb { count } else { 8 } {
            let mut e = 0.0;
            for j in 0..N {
                let d = samples[i][j] - palette[k][j];
                e += d * d;
            }
            if e < best {
                best = e;
                c.indices[i] = k as u8;
            }
        }
        c.error += if rgb {
            best.sqrt()
        } else {
            (samples[i][0] - palette[c.indices[i] as usize][0]).abs()
        };
    }
    c
}
fn ordered(mut endpoints: [u16; 2], ascending: bool) -> [u16; 2] {
    if (endpoints[0] > endpoints[1]) == ascending {
        endpoints.swap(0, 1);
    }
    endpoints
}
fn refine<const N: usize>(
    samples: &[[f32; N]; 16],
    valid: &[bool; 16],
    mut c: Candidate<N>,
    rgb: bool,
    three: bool,
) -> Candidate<N> {
    for _ in 0..4 {
        let count = if rgb {
            if three { 3 } else { 4 }
        } else if c.endpoints[0] > c.endpoints[1] {
            8
        } else {
            6
        };
        let (mut a, mut b, mut d) = (0.0, 0.0, 0.0);
        let mut x = [0.0; N];
        let mut y = [0.0; N];
        for i in 0..16 {
            if !valid[i] {
                continue;
            }
            let Some(t) = weight(c.indices[i] as usize, count) else {
                continue;
            };
            let u = 1.0 - t;
            a += u * u;
            b += u * t;
            d += t * t;
            for j in 0..N {
                x[j] += samples[i][j] * u;
                y[j] += samples[i][j] * t;
            }
        }
        let det = a * d - b * b;
        if det.abs() < 1e-8 {
            break;
        }
        let mut e0 = [0.0; N];
        let mut e1 = [0.0; N];
        for j in 0..N {
            e0[j] = (x[j] * d - y[j] * b) / det;
            e1[j] = (y[j] * a - x[j] * b) / det;
        }
        let endpoints = if rgb {
            [
                rgb_snap([e0[0], e0[1], e0[2]]),
                rgb_snap([e1[0], e1[1], e1[2]]),
            ]
        } else {
            [snap(e0[0], 255), snap(e1[0], 255)]
        };
        let next = evaluate(
            samples,
            valid,
            ordered(endpoints, if rgb { three } else { count == 6 }),
            rgb,
            three,
        );
        if next.error >= c.error {
            break;
        }
        c = next;
    }
    c
}

pub(super) fn rgb(colors: [[(Rgb32F, bool); 4]; 4], three: bool) -> super::bc1::Block {
    let mut samples = [[0.0; 3]; 16];
    let mut valid = [false; 16];
    let mut sorted = [Vec3::ZERO; 16];
    let mut count = 0;
    for i in 0..16 {
        let (v, ok) = colors[i / 4][i % 4];
        samples[i] = [v.r(), v.g(), v.b()];
        valid[i] = ok;
        if ok {
            sorted[count] = Vec3(samples[i]);
            count += 1;
        }
    }
    if count == 0 {
        return super::bc1::Block::BLACK;
    }
    sorted[..count].sort_by(|a, b| {
        a.0[0]
            .total_cmp(&b.0[0])
            .then(a.0[1].total_cmp(&b.0[1]))
            .then(a.0[2].total_cmp(&b.0[2]))
    });
    let mut axis = pca_axis3(&sorted[..count]);
    if axis.length() < 1e-6 {
        axis = Vec3::new(1.0, 0.0, 0.0);
    }
    let mut major = 0;
    for j in 1..3 {
        if axis.0[j].abs() > axis.0[major].abs() {
            major = j;
        }
    }
    if axis.0[major] < 0.0 {
        for j in 0..3 {
            axis.0[j] = -axis.0[j];
        }
    }
    let mut lo = sorted[0];
    let mut hi = lo;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in &sorted[..count] {
        let dot = v.0[0] * axis.0[0] + v.0[1] * axis.0[1] + v.0[2] * axis.0[2];
        if dot < min {
            min = dot;
            lo = v;
        }
        if dot >= max {
            max = dot;
            hi = v;
        }
    }
    let mut c = refine(
        &samples,
        &valid,
        evaluate(
            &samples,
            &valid,
            ordered([rgb_snap(lo.0), rgb_snap(hi.0)], three),
            true,
            three,
        ),
        true,
        three,
    );
    let seed = c;
    for endpoint in 0..2 {
        let q = seed.endpoints[endpoint];
        let code = [(q >> 11) as i32, ((q >> 5) & 63) as i32, (q & 31) as i32];
        for r in -1..=1 {
            for g in -1..=1 {
                for b in -1..=1 {
                    if r == 0 && g == 0 && b == 0 {
                        continue;
                    }
                    let v = [code[0] + r, code[1] + g, code[2] + b];
                    if v[0] < 0 || v[0] > 31 || v[1] < 0 || v[1] > 63 || v[2] < 0 || v[2] > 31 {
                        continue;
                    }
                    let mut e = seed.endpoints;
                    e[endpoint] = ((v[0] << 11) | (v[1] << 5) | v[2]) as u16;
                    let next = evaluate(&samples, &valid, ordered(e, three), true, three);
                    if next.error < c.error {
                        c = next;
                    }
                }
            }
        }
    }
    let mut indices = [0; 4];
    for i in 0..16 {
        indices[i / 4] |= c.indices[i] << ((i % 4) * 2);
    }
    super::bc1::Block {
        color0: Rgb565::from_bytes(c.endpoints[0].to_le_bytes()),
        color1: Rgb565::from_bytes(c.endpoints[1].to_le_bytes()),
        indices,
    }
}

pub(super) fn scalar(colors: [[(crate::math::R32F, bool); 4]; 4]) -> super::bc4::Block {
    let mut samples = [[0.0; 1]; 16];
    let mut valid = [false; 16];
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for i in 0..16 {
        let (v, ok) = colors[i / 4][i % 4];
        samples[i] = [v.0];
        valid[i] = ok;
        if ok {
            lo = lo.min(v.0);
            hi = hi.max(v.0);
        }
    }
    if !valid.iter().any(|&v| v) {
        return super::bc4::Block {
            color0: crate::math::R8U::BLACK,
            color1: crate::math::R8U::BLACK,
            indices: [0; 6],
        };
    }
    let endpoints = [snap(hi, 255), snap(lo, 255)];
    let mut c = refine(
        &samples,
        &valid,
        evaluate(&samples, &valid, endpoints, false, false),
        false,
        false,
    );
    let other = refine(
        &samples,
        &valid,
        evaluate(&samples, &valid, [endpoints[1], endpoints[0]], false, false),
        false,
        false,
    );
    if other.error < c.error {
        c = other;
    }
    let seed = c;
    for endpoint in 0..2 {
        for step in [-1i32, 1] {
            let value = seed.endpoints[endpoint] as i32 + step;
            if !(0..=255).contains(&value) {
                continue;
            }
            let mut e = seed.endpoints;
            e[endpoint] = value as u16;
            let next = evaluate(&samples, &valid, e, false, false);
            if next.error < c.error {
                c = next;
            }
        }
    }
    let mut packed = 0u64;
    for i in 0..16 {
        packed |= (c.indices[i] as u64) << (i * 3);
    }
    let bytes = packed.to_le_bytes();
    super::bc4::Block {
        color0: crate::math::R8U::new(c.endpoints[0] as u8),
        color1: crate::math::R8U::new(c.endpoints[1] as u8),
        indices: bytes[..6].try_into().unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        image::block::{bc1, bc2, bc3, bc4, bc5},
        math::{R32F, Rg32F, Rgba32F},
    };
    fn gradient() -> [[(Rgba32F, bool); 4]; 4] {
        std::array::from_fn(|y| {
            std::array::from_fn(|x| {
                let t = (y * 4 + x) as f32 / 15.0;
                (Rgba32F::new(t, 1.0 - t, t * 0.5, t), true)
            })
        })
    }
    #[test]
    fn alpha_gradients_and_indices() {
        let input = gradient();
        let b2 = bc2::Block::encode_with_alpha(input);
        let b3 = bc3::Block::encode_with_alpha(input);
        let d2 = b2.decode_with_alpha();
        let d3 = b3.decode_with_alpha();
        for i in 0..16 {
            let y = i / 4;
            let x = i % 4;
            assert!((d2[y][x].a() - input[y][x].0.a()).abs() < 1e-6);
            assert!((d3[y][x].a() - input[y][x].0.a()).abs() < 0.1);
            assert!((d2[y][x].r() - input[y][x].0.r()).abs() < 0.2);
        }
        assert_ne!(b2.indices, [0; 4]);
        assert_ne!(b3.alpha.indices, [0; 6]);
    }
    #[test]
    fn punch_through_threshold_and_mask() {
        let mut input = gradient();
        input[0][0].1 = false;
        input[2][0].0 = Rgba32F::new(0.4, 0.5, 0.6, 0.5);
        let b = bc1::Block::encode_with_alpha(input, 0.5);
        assert!(b.color0.bits() <= b.color1.bits());
        let d = b.decode_with_alpha();
        for i in 1..16 {
            let y = i / 4;
            let x = i % 4;
            assert_eq!(d[y][x].a(), if input[y][x].0.a() < 0.5 { 0.0 } else { 1.0 });
        }
        let mut sparse = [[(Rgb32F::WHITE, false); 4]; 4];
        sparse[2][3] = (Rgb32F::new(0.2, 0.4, 0.6), true);
        let a = bc1::Block::encode(sparse);
        sparse[0][0].0 = Rgb32F::BLACK;
        assert_eq!(a, bc1::Block::encode(sparse));
    }
    #[test]
    fn empty_and_flat_blocks() {
        assert_eq!(
            bc1::Block::encode([[(Rgb32F::WHITE, false); 4]; 4]),
            bc1::Block::BLACK
        );
        assert_eq!(
            bc1::Block::encode_with_alpha([[(Rgba32F::TRANSPARENT, true); 4]; 4], 0.5),
            bc1::Block::TRANSPARENT
        );
        let d = bc4::Block::encode([[(R32F::new(0.4), true); 4]; 4]).decode();
        for row in d {
            for v in row {
                assert!((v.0 - 0.4).abs() < 1.0 / 255.0);
            }
        }
        let b = bc4::Block::encode([[(R32F::WHITE, false); 4]; 4]);
        assert_eq!(b.color0.bits(), 0);
        for row in
            bc1::Block::encode([[(Rgb32F::new(0.4, 0.5, 0.6), true); 4]; 4]).decode_with_alpha()
        {
            for v in row {
                assert_eq!(v.a(), 1.0);
            }
        }
    }
    #[test]
    fn scalar_constants_and_independent_channels() {
        let input = std::array::from_fn(|y| {
            std::array::from_fn(|x| {
                let v = match (y * 4 + x) % 4 {
                    0 => 0.0,
                    1 => 0.4,
                    2 => 0.6,
                    _ => 1.0,
                };
                (R32F::new(v), true)
            })
        });
        let b = bc4::Block::encode(input);
        assert!(b.color0.bits() <= b.color1.bits());
        let d = b.decode();
        for i in 0..16 {
            assert!((d[i / 4][i % 4].0 - input[i / 4][i % 4].0.0).abs() < 0.005);
        }
        let rg = gradient().map(|r| r.map(|(v, k)| (Rg32F::new(v.r(), v.g()), k)));
        let b5 = bc5::Block::encode(rg);
        assert_ne!(b5.red.indices, [0; 6]);
        assert_ne!(b5.green.indices, [0; 6]);
        let d = b5.decode();
        for i in 0..16 {
            assert!((d[i / 4][i % 4].r() - rg[i / 4][i % 4].0.r()).abs() < 0.1);
            assert!((d[i / 4][i % 4].g() - rg[i / 4][i % 4].0.g()).abs() < 0.1);
        }
    }
    #[test]
    fn bc3_always_four_colors() {
        let b = bc3::Block {
            alpha: bc4::Block::WHITE,
            rgb: bc1::Block {
                color0: Rgb565::BLACK,
                color1: Rgb565::WHITE,
                indices: [255; 4],
            },
        };
        assert!((b.decode()[0][0].r() - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(b.decode_with_alpha()[0][0].a(), 1.0);
    }
    #[test]
    #[ignore = "exhaustive RGB565 opaque flat-block validation; run in release"]
    fn all_rgb565_flat_blocks_stay_opaque() {
        for code in 0..=u16::MAX {
            let color = Rgb565::from_bytes(code.to_le_bytes()).into_f32();
            let block = bc1::Block::encode([[(color, true); 4]; 4]);
            for row in block.decode_with_alpha() {
                for texel in row {
                    assert_eq!(texel.a(), 1.0, "RGB565 code {code:#06x}");
                }
            }
        }
    }
}
