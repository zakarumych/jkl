use super::*;

// Fixed search budgets bound work independently of the image content.
const FIT_ITERATIONS: usize = 6;
const FINALISTS: usize = 12;
const REFINE_PASSES: usize = 3;

type Pixels = [[u8; 4]; 16];

#[derive(Clone, Copy)]
struct Configuration {
    mode: usize,
    partition: usize,
    rotation: usize,
    selection: usize,
}

#[derive(Clone, Copy)]
struct Candidate {
    config: Configuration,
    endpoints: [[u8; 4]; 6],
    pbits: [u8; 6],
    indices: [[u8; 16]; 2],
    error: u64,
}

#[derive(Clone, Copy)]
struct Fit {
    endpoints: [[u8; 4]; 2],
    pbits: [u8; 2],
    indices: [u8; 16],
    error: u64,
}

pub(super) fn encode(colors: [[(Rgba32F, bool); 4]; 4]) -> Block {
    let mut mask = 0u16;
    let mut pixels = [[0; 4]; 16];
    for i in 0..16 {
        let (color, included) = colors[i / 4][i % 4];
        if included {
            mask |= 1 << i;
            pixels[i] = color.0.map(|v| {
                if v.is_nan() {
                    0
                } else {
                    (v.clamp(0.0, 1.0) * 255.0).round() as u8
                }
            });
        }
    }
    if mask == 0 {
        return Block::TRANSPARENT;
    }
    let mut finalists = Vec::with_capacity(FINALISTS + 1);
    for mode in 0..8 {
        let m = MODES[mode];
        for partition in 0..1 << m.partition_bits {
            for rotation in 0..if m.dual() { 4 } else { 1 } {
                for selection in 0..if mode == 4 { 2 } else { 1 } {
                    let config = Configuration {
                        mode,
                        partition,
                        rotation,
                        selection,
                    };
                    let candidate = fit_configuration(config, &pixels, mask);
                    let insert =
                        finalists.partition_point(|c: &Candidate| c.error <= candidate.error);
                    if insert < FINALISTS {
                        finalists.insert(insert, candidate);
                        finalists.truncate(FINALISTS);
                    }
                }
            }
        }
    }
    let mut best = finalists[0];
    for candidate in &mut finalists {
        refine(candidate, &pixels, mask);
        if candidate.error < best.error {
            best = *candidate;
        }
    }
    pack(best)
}

fn rotated(pixels: &Pixels, rotation: usize) -> Pixels {
    let mut result = *pixels;
    if rotation != 0 {
        for pixel in &mut result {
            pixel.swap(3, rotation - 1);
        }
    }
    result
}

fn subset_mask(config: Configuration, subset: usize, mask: u16) -> u16 {
    let m = MODES[config.mode];
    let (shape, _) = shape(m.subsets, config.partition);
    let mut result = 0;
    for i in 0..16 {
        if mask & (1 << i) != 0 && ((shape >> (i * 2)) & 3) as usize == subset {
            result |= 1 << i;
        }
    }
    result
}

fn fit_configuration(config: Configuration, source: &Pixels, mask: u16) -> Candidate {
    let m = MODES[config.mode];
    let pixels = rotated(source, config.rotation);
    let mut result = Candidate {
        config,
        endpoints: [[0; 4]; 6],
        pbits: [0; 6],
        indices: [[0; 16]; 2],
        error: 0,
    };
    for subset in 0..m.subsets {
        let included = subset_mask(config, subset, mask);
        if m.dual() {
            for stream in 0..2 {
                let alpha = stream != config.selection;
                let (start, end) = if alpha { (3, 4) } else { (0, 3) };
                let bits = if stream == 0 {
                    m.index_bits
                } else {
                    m.secondary_bits
                };
                let fit = fit_group(&pixels, included, m, false, start, end, bits);
                for e in 0..2 {
                    result.endpoints[e][start..end].copy_from_slice(&fit.endpoints[e][start..end]);
                }
                result.indices[stream] = fit.indices;
                result.error += fit.error;
            }
        } else {
            let end = if m.alpha_bits == 0 { 3 } else { 4 };
            let fit = fit_group(&pixels, included, m, config.mode == 1, 0, end, m.index_bits);
            result.endpoints[subset * 2..subset * 2 + 2].copy_from_slice(&fit.endpoints);
            result.pbits[subset * 2..subset * 2 + 2].copy_from_slice(&fit.pbits);
            for i in 0..16 {
                if included & (1 << i) != 0 {
                    result.indices[0][i] = fit.indices[i];
                }
            }
            result.error += fit.error;
        }
    }
    if m.alpha_bits == 0 {
        for i in 0..16 {
            if mask & (1 << i) != 0 {
                result.error += (255 - pixels[i][3] as u64).pow(2);
            }
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn fit_group(
    pixels: &Pixels,
    mask: u16,
    m: Mode,
    shared_pbit: bool,
    start: usize,
    end: usize,
    index_bits: usize,
) -> Fit {
    let mut best = Fit {
        endpoints: [[0; 4]; 2],
        pbits: [0; 2],
        indices: [0; 16],
        error: u64::MAX,
    };
    for seed in seeds(pixels, mask, start, end) {
        let mut target = seed;
        let mut previous = [u8::MAX; 16];
        for _ in 0..FIT_ITERATIONS {
            let fit = quantize_fit(target, pixels, mask, m, shared_pbit, start, end, index_bits);
            if fit.error < best.error {
                best = fit;
            }
            if fit.indices == previous || fit.error == 0 {
                break;
            }
            previous = fit.indices;
            target = least_squares(pixels, mask, start, end, &fit.indices, index_bits);
        }
    }
    best
}

// Axis-aligned extrema, a covariance axis, and the farthest sample pair span
// distinct useful starts without relying on the ordering of the texels.
fn seeds(pixels: &Pixels, mask: u16, start: usize, end: usize) -> [[[f64; 4]; 2]; 3] {
    let mut low = [255.0f64; 4];
    let mut high = [0.0f64; 4];
    let mut mean = [0.0f64; 4];
    let count = mask.count_ones() as f64;
    if count == 0.0 {
        return [[[0.0; 4]; 2]; 3];
    }
    for i in 0..16 {
        if mask & (1 << i) != 0 {
            for c in start..end {
                let value = pixels[i][c] as f64;
                low[c] = low[c].min(value);
                high[c] = high[c].max(value);
                mean[c] += value / count;
            }
        }
    }
    let mut covariance = [[0.0; 4]; 4];
    for i in 0..16 {
        if mask & (1 << i) != 0 {
            for c in start..end {
                for d in start..end {
                    covariance[c][d] +=
                        (pixels[i][c] as f64 - mean[c]) * (pixels[i][d] as f64 - mean[d]);
                }
            }
        }
    }
    let principal = (start..end)
        .max_by(|&a, &b| covariance[a][a].total_cmp(&covariance[b][b]))
        .unwrap();
    let mut axis = [0.0; 4];
    axis[principal] = 1.0;
    for _ in 0..8 {
        let mut next = [0.0f64; 4];
        for c in start..end {
            for d in start..end {
                next[c] += covariance[c][d] * axis[d];
            }
        }
        let length = next.iter().map(|x| x * x).sum::<f64>().sqrt();
        if length < 1e-10 {
            break;
        }
        axis = next.map(|v| v / length);
    }
    let mut projection_min = f64::INFINITY;
    let mut projection_max = f64::NEG_INFINITY;
    let mut farthest = [[0.0; 4]; 2];
    let mut distance_max = -1i32;
    for i in 0..16 {
        if mask & (1 << i) == 0 {
            continue;
        }
        let projection = (start..end)
            .map(|c| (pixels[i][c] as f64 - mean[c]) * axis[c])
            .sum::<f64>();
        projection_min = projection_min.min(projection);
        projection_max = projection_max.max(projection);
        for j in i..16 {
            if mask & (1 << j) == 0 {
                continue;
            }
            let distance = (start..end)
                .map(|c| (pixels[i][c] as i32 - pixels[j][c] as i32).pow(2))
                .sum();
            if distance > distance_max {
                distance_max = distance;
                farthest = [pixels[i].map(f64::from), pixels[j].map(f64::from)];
            }
        }
    }
    let pca = [
        std::array::from_fn(|c| mean[c] + axis[c] * projection_min),
        std::array::from_fn(|c| mean[c] + axis[c] * projection_max),
    ];
    [[low, high], pca, farthest]
}

fn nearest(target: f64, bits: usize, pbit: Option<u8>) -> u8 {
    let target = target.clamp(0.0, 255.0);
    let max = (1usize << bits) - 1;
    // Endpoint expansion is monotone; find the adjacent decoded byte values.
    let decoded = |q: usize| {
        if let Some(p) = pbit {
            expand(((q as u8) << 1) | p, bits + 1)
        } else {
            expand(q as u8, bits)
        }
    };
    let mut lo = 0;
    let mut hi = max;
    while lo < hi {
        let middle = (lo + hi) / 2;
        if (decoded(middle) as f64) < target {
            lo = middle + 1;
        } else {
            hi = middle;
        }
    }
    if lo != 0 && (decoded(lo - 1) as f64 - target).abs() <= (decoded(lo) as f64 - target).abs() {
        (lo - 1) as u8
    } else {
        lo as u8
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_fit(
    target: [[f64; 4]; 2],
    pixels: &Pixels,
    mask: u16,
    m: Mode,
    shared_pbit: bool,
    start: usize,
    end: usize,
    index_bits: usize,
) -> Fit {
    let mut best = Fit {
        endpoints: [[0; 4]; 2],
        pbits: [0; 2],
        indices: [0; 16],
        error: u64::MAX,
    };
    for p0 in 0..if m.pbits { 2 } else { 1 } {
        for p1 in 0..if m.pbits { 2 } else { 1 } {
            if shared_pbit && p0 != p1 {
                continue;
            }
            let mut fit = Fit {
                endpoints: [[0; 4]; 2],
                pbits: [p0, p1],
                indices: [0; 16],
                error: 0,
            };
            for e in 0..2 {
                for c in start..end {
                    fit.endpoints[e][c] =
                        nearest(target[e][c], m.bits(c), m.pbits.then_some(fit.pbits[e]));
                }
            }
            evaluate_fit(&mut fit, pixels, mask, m, start, end, index_bits);
            if fit.error < best.error {
                best = fit;
            }
        }
    }
    best
}

fn evaluate_fit(
    fit: &mut Fit,
    pixels: &Pixels,
    mask: u16,
    m: Mode,
    start: usize,
    end: usize,
    index_bits: usize,
) {
    let a = expand_endpoint(fit.endpoints[0], fit.pbits[0], m);
    let b = expand_endpoint(fit.endpoints[1], fit.pbits[1], m);
    let weights = weights(index_bits);
    let mut palette = [[0i32; 4]; 16];
    for (i, &weight) in weights.iter().enumerate() {
        for c in start..end {
            palette[i][c] = interpolate(a[c], b[c], weight) as i32;
        }
    }
    fit.error = 0;
    fit.indices = [0; 16];
    for i in 0..16 {
        if mask & (1 << i) == 0 {
            continue;
        }
        let mut best_error = u64::MAX;
        for (index, color) in palette[..weights.len()].iter().enumerate() {
            let mut error = 0;
            for c in start..end {
                error += (pixels[i][c] as i32 - color[c]).pow(2) as u64;
            }
            if error < best_error {
                best_error = error;
                fit.indices[i] = index as u8;
            }
        }
        fit.error += best_error;
    }
}

fn least_squares(
    pixels: &Pixels,
    mask: u16,
    start: usize,
    end: usize,
    indices: &[u8; 16],
    index_bits: usize,
) -> [[f64; 4]; 2] {
    let weights = weights(index_bits);
    let mut aa = 0.0;
    let mut ab = 0.0;
    let mut bb = 0.0;
    let mut ay = [0.0; 4];
    let mut by = [0.0; 4];
    for i in 0..16 {
        if mask & (1 << i) == 0 {
            continue;
        }
        let b = weights[indices[i] as usize] as f64 / 64.0;
        let a = 1.0 - b;
        aa += a * a;
        ab += a * b;
        bb += b * b;
        for c in start..end {
            ay[c] += a * pixels[i][c] as f64;
            by[c] += b * pixels[i][c] as f64;
        }
    }
    let determinant = aa * bb - ab * ab;
    let mut result = [[0.0; 4]; 2];
    if determinant > 1e-10 {
        for c in start..end {
            result[0][c] = ((bb * ay[c] - ab * by[c]) / determinant).clamp(0.0, 255.0);
            result[1][c] = ((aa * by[c] - ab * ay[c]) / determinant).clamp(0.0, 255.0);
        }
    } else {
        let count = mask.count_ones().max(1) as f64;
        for c in start..end {
            let mean = (ay[c] + by[c]) / count;
            result[0][c] = mean;
            result[1][c] = mean;
        }
    }
    result
}

fn refine(candidate: &mut Candidate, source: &Pixels, mask: u16) {
    let config = candidate.config;
    let m = MODES[config.mode];
    let pixels = rotated(source, config.rotation);
    for subset in 0..m.subsets {
        let included = subset_mask(config, subset, mask);
        for stream in 0..if m.dual() { 2 } else { 1 } {
            let alpha = m.dual() && stream != config.selection;
            let (start, end) = if alpha {
                (3, 4)
            } else {
                (0, if m.dual() || m.alpha_bits == 0 { 3 } else { 4 })
            };
            let bits = if stream == 0 {
                m.index_bits
            } else {
                m.secondary_bits
            };
            let mut fit = Fit {
                endpoints: [
                    candidate.endpoints[subset * 2],
                    candidate.endpoints[subset * 2 + 1],
                ],
                pbits: [candidate.pbits[subset * 2], candidate.pbits[subset * 2 + 1]],
                indices: candidate.indices[stream],
                error: 0,
            };
            evaluate_fit(&mut fit, &pixels, included, m, start, end, bits);
            let baseline = fit.error;
            for _ in 0..REFINE_PASSES {
                let before = fit.error;
                for e in 0..2 {
                    for c in start..end {
                        let origin = fit;
                        for delta in [-1, 1] {
                            let value = origin.endpoints[e][c] as i32 + delta;
                            if value < 0 || value >= 1 << m.bits(c) {
                                continue;
                            }
                            let mut trial = origin;
                            trial.endpoints[e][c] = value as u8;
                            evaluate_fit(&mut trial, &pixels, included, m, start, end, bits);
                            if trial.error < fit.error {
                                fit = trial;
                            }
                        }
                    }
                }
                let target = least_squares(&pixels, included, start, end, &fit.indices, bits);
                let trial = quantize_fit(
                    target,
                    &pixels,
                    included,
                    m,
                    config.mode == 1,
                    start,
                    end,
                    bits,
                );
                if trial.error < fit.error {
                    fit = trial;
                }
                if fit.error == before {
                    break;
                }
            }
            for e in 0..2 {
                candidate.endpoints[subset * 2 + e][start..end]
                    .copy_from_slice(&fit.endpoints[e][start..end]);
                candidate.pbits[subset * 2 + e] = fit.pbits[e];
            }
            for i in 0..16 {
                if included & (1 << i) != 0 {
                    candidate.indices[stream][i] = fit.indices[i];
                }
            }
            candidate.error = candidate.error - baseline + fit.error;
        }
    }
}

fn pack(mut candidate: Candidate) -> Block {
    let config = candidate.config;
    let m = MODES[config.mode];
    let (subsets, anchors) = shape(m.subsets, config.partition);
    // Each index stream has its own omitted anchor MSB. Dual-index modes can
    // therefore reverse RGB and alpha independently.
    for subset in 0..m.subsets {
        let anchor = (0..16)
            .find(|&i| anchors & (1 << i) != 0 && ((subsets >> (2 * i)) & 3) as usize == subset)
            .unwrap();
        for stream in 0..if m.dual() { 2 } else { 1 } {
            let bits = if stream == 0 {
                m.index_bits
            } else {
                m.secondary_bits
            };
            if candidate.indices[stream][anchor] < 1 << (bits - 1) {
                continue;
            }
            let alpha = m.dual() && stream != config.selection;
            let (start, end) = if alpha {
                (3, 4)
            } else {
                (0, if m.dual() { 3 } else { 4 })
            };
            for c in start..end {
                let a = candidate.endpoints[subset * 2][c];
                candidate.endpoints[subset * 2][c] = candidate.endpoints[subset * 2 + 1][c];
                candidate.endpoints[subset * 2 + 1][c] = a;
            }
            if !m.dual() {
                candidate.pbits.swap(subset * 2, subset * 2 + 1);
            }
            for i in 0..16 {
                if ((subsets >> (2 * i)) & 3) as usize == subset {
                    candidate.indices[stream][i] = ((1 << bits) - 1) - candidate.indices[stream][i];
                }
            }
        }
    }
    let mut output = Bits::new([0; 16]);
    output.write(0, config.mode);
    output.write(1, 1);
    output.write(config.partition as u8, m.partition_bits);
    if m.dual() {
        output.write(config.rotation as u8, 2);
    }
    if config.mode == 4 {
        output.write(config.selection as u8, 1);
    }
    for c in 0..4 {
        for e in 0..m.subsets * 2 {
            output.write(candidate.endpoints[e][c], m.bits(c));
        }
    }
    if config.mode == 1 {
        output.write(candidate.pbits[0], 1);
        output.write(candidate.pbits[2], 1);
    } else if m.pbits {
        for e in 0..m.subsets * 2 {
            output.write(candidate.pbits[e], 1);
        }
    }
    for i in 0..16 {
        output.write(
            candidate.indices[0][i],
            m.index_bits - ((anchors >> i) & 1) as usize,
        );
    }
    if m.dual() {
        for i in 0..16 {
            output.write(
                candidate.indices[1][i],
                m.secondary_bits - usize::from(i == 0),
            );
        }
    }
    debug_assert_eq!(output.position, 128);
    Block::from_bytes(output.raw.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_configuration_preserves_scored_error() {
        let pixels: Pixels = std::array::from_fn(|i| {
            [
                ((i * 67) % 256) as u8,
                ((i * 43) % 256) as u8,
                ((i * 17) % 256) as u8,
                ((i * 89) % 256) as u8,
            ]
        });
        for mask in [u16::MAX, 0x1204, 1] {
            let mut count = 0;
            for mode in 0..8 {
                let m = MODES[mode];
                for partition in 0..1 << m.partition_bits {
                    for rotation in 0..if m.dual() { 4 } else { 1 } {
                        for selection in 0..if mode == 4 { 2 } else { 1 } {
                            let config = Configuration {
                                mode,
                                partition,
                                rotation,
                                selection,
                            };
                            let mut candidate = fit_configuration(config, &pixels, mask);
                            let baseline = candidate.error;
                            refine(&mut candidate, &pixels, mask);
                            assert!(candidate.error <= baseline);
                            let decoded = pack(candidate).decode_bytes();
                            let error: u64 = (0..16)
                                .filter(|&i| mask & (1 << i) != 0)
                                .map(|i| {
                                    (0..4)
                                        .map(|c| {
                                            (pixels[i][c] as i32 - decoded[i][c] as i32).pow(2)
                                                as u64
                                        })
                                        .sum::<u64>()
                                })
                                .sum();
                            assert_eq!(
                                candidate.error, error,
                                "mode {mode}, partition {partition}, rotation {rotation}, selection {selection}, mask {mask:x}"
                            );
                            count += 1;
                        }
                    }
                }
            }
            assert_eq!(count, 285);
        }
    }

    #[test]
    fn quantization_matches_exhaustive_endpoint_search() {
        for bits in 4..=8 {
            for p in [None, Some(0), Some(1)] {
                if p.is_some() && bits == 8 {
                    continue;
                }
                for value in 0..=510 {
                    let target = value as f64 / 2.0;
                    let q = nearest(target, bits, p);
                    let decode = |q: u8| {
                        if let Some(p) = p {
                            expand((q << 1) | p, bits + 1)
                        } else {
                            expand(q, bits)
                        }
                    };
                    let expected = (0..1usize << bits)
                        .min_by(|&a, &b| {
                            (decode(a as u8) as f64 - target)
                                .abs()
                                .total_cmp(&(decode(b as u8) as f64 - target).abs())
                        })
                        .unwrap() as u8;
                    assert_eq!(q, expected);
                }
            }
        }
    }
}
