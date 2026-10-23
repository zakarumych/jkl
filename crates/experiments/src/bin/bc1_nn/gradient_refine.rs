//! CPU reference for the continuous RGB-norm gradient refinement experiment.
use super::{Texels, error, pack, rgb};
use jkl::image::block::bc1::Block;

const WEIGHTS: [f32; 4] = [0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0];
pub fn relaxed(t: &Texels, a: [f32; 3], b: [f32; 3]) -> (f32, [f32; 3], [f32; 3]) {
    relaxed_indices(t, a, b, None)
}
pub fn relaxed_fixed(
    t: &Texels,
    a: [f32; 3],
    b: [f32; 3],
    indices: [u8; 4],
) -> (f32, [f32; 3], [f32; 3]) {
    relaxed_indices(t, a, b, Some(indices))
}
fn relaxed_indices(
    t: &Texels,
    a: [f32; 3],
    b: [f32; 3],
    indices: Option<[u8; 4]>,
) -> (f32, [f32; 3], [f32; 3]) {
    let mut value = 0.0;
    let mut ga = [0.0; 3];
    let mut gb = [0.0; 3];
    for (i, &texel) in t.iter().enumerate() {
        let pixel = rgb(texel).0;
        let mut best = f32::INFINITY;
        let mut delta = [0.0; 3];
        let mut beta = 0.0;
        let weights = indices.map_or(WEIGHTS, |indices| {
            [WEIGHTS[((indices[i / 4] >> ((i % 4) * 2)) & 3) as usize]; 4]
        });
        for &weight in &weights[..if indices.is_some() || a == b { 1 } else { 4 }] {
            let d = std::array::from_fn::<_, 3, _>(|c| {
                a[c] * (1.0 - weight) + b[c] * weight - pixel[c]
            });
            let squared = d.iter().map(|v| v * v).sum::<f32>();
            if squared < best {
                best = squared;
                delta = d;
                beta = weight;
            }
        }
        let norm = (best + 1e-12).sqrt();
        value += (norm - 1e-6) / 16.0;
        for c in 0..3 {
            let g = delta[c] / (16.0 * norm);
            ga[c] += (1.0 - beta) * g;
            gb[c] += beta * g;
        }
    }
    (value, ga, gb)
}
pub fn refine(t: &Texels, initial: Block, steps: u32, rate: f32) -> Block {
    refine_policy(t, initial, steps, rate, false)
}
pub fn refine_fixed(t: &Texels, initial: Block, steps: u32, rate: f32) -> Block {
    refine_policy(t, initial, steps, rate, true)
}
fn refine_policy(t: &Texels, initial: Block, steps: u32, rate: f32, fixed: bool) -> Block {
    let mut best = initial;
    let mut a = initial.color0.into_f32().0;
    let mut b = initial.color1.into_f32().0;
    for _ in 0..steps {
        let indices = if fixed { Some(initial.indices) } else { None };
        let (value, ga, gb) = relaxed_indices(t, a, b, indices);
        let mut trial_rate = rate;
        let mut accepted = false;
        for _ in 0..9 {
            let na = std::array::from_fn(|c| (a[c] - trial_rate * ga[c]).clamp(0.0, 1.0));
            let nb = std::array::from_fn(|c| (b[c] - trial_rate * gb[c]).clamp(0.0, 1.0));
            if relaxed_indices(t, na, nb, indices).0 < value {
                a = na;
                b = nb;
                accepted = true;
                let candidate = pack(t, [a[0], a[1], a[2], b[0], b[1], b[2]]);
                if error(t, candidate) < error(t, best) {
                    best = candidate;
                }
                break;
            }
            trial_rate *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    best
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuous_gradient_matches_finite_differences() {
        let t = std::array::from_fn(|i| [35 + i as u8 * 9, 70 + i as u8 * 4, 40 + i as u8 * 8]);
        let a = [0.12, 0.25, 0.09];
        let b = [0.83, 0.62, 0.79];
        let (_, ga, gb) = relaxed(&t, a, b);
        let step = 1e-4;
        for i in 0..6 {
            let mut ap = a;
            let mut am = a;
            let mut bp = b;
            let mut bm = b;
            if i < 3 {
                ap[i] += step;
                am[i] -= step;
            } else {
                bp[i - 3] += step;
                bm[i - 3] -= step;
            }
            let numeric = (relaxed(&t, ap, bp).0 - relaxed(&t, am, bm).0) / (2.0 * step);
            let analytic = if i < 3 { ga[i] } else { gb[i - 3] };
            assert!(
                (numeric - analytic).abs() < 0.001,
                "gradient {i}: {numeric} != {analytic}"
            );
        }
    }
    #[test]
    fn quantized_candidate_never_worsens_and_crosses_plateaus() {
        // Small continuous updates accumulate across the RGB565 cell boundary.
        let t = [[134, 134, 134]; 16];
        let initial = pack(&t, [0.5; 6]);
        let one = refine(&t, initial, 1, 0.001);
        let many = refine(&t, initial, 32, 0.001);
        assert_eq!(one, initial);
        assert!(error(&t, many) < error(&t, initial));
        for steps in [4, 8, 16] {
            for rate in [0.02, 0.1, 0.5] {
                let result = refine(&t, initial, steps, rate);
                assert!(error(&t, result) <= error(&t, initial));
            }
        }
    }
    #[test]
    fn fixed_indices_gradient_matches_finite_differences() {
        let t = std::array::from_fn(|i| [35 + i as u8 * 9, 70 + i as u8 * 4, 40 + i as u8 * 8]);
        let a = [0.12, 0.25, 0.09];
        let b = [0.83, 0.62, 0.79];
        let indices = [0xe4; 4];
        let (_, ga, gb) = relaxed_fixed(&t, a, b, indices);
        let step = 1e-4;
        for i in 0..6 {
            let mut ap = a;
            let mut am = a;
            let mut bp = b;
            let mut bm = b;
            if i < 3 {
                ap[i] += step;
                am[i] -= step;
            } else {
                bp[i - 3] += step;
                bm[i - 3] -= step;
            }
            let numeric = (relaxed_fixed(&t, ap, bp, indices).0
                - relaxed_fixed(&t, am, bm, indices).0)
                / (2.0 * step);
            let analytic = if i < 3 { ga[i] } else { gb[i - 3] };
            assert!((numeric - analytic).abs() < 0.001);
        }
    }
    #[test]
    fn ls_then_both_gradient_policies_never_worsen_ls() {
        let mut rng = super::super::Rng(456);
        for i in 0..12 {
            let t = super::super::synthetic(&mut rng, i);
            let start = pack(&t, super::super::baseline(&super::super::canonical(&t)));
            let ls = super::super::refine(&t, start);
            for steps in [1, 2, 4] {
                for rate in [0.02, 0.1, 0.5] {
                    for fixed in [false, true] {
                        let b = refine_policy(&t, ls, steps, rate, fixed);
                        assert!(error(&t, b) <= error(&t, ls));
                        assert_eq!(Block::from_bytes(b.bytes()), b);
                        assert!(b.decode_with_alpha().iter().flatten().all(|c| c.a() == 1.0));
                    }
                }
            }
        }
    }
}
