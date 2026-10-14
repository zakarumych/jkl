//! BC6H compression for linear HDR RGB colors.
//!
//! Blocks contain 4×4 texels in 16 bytes. Signedness belongs to the texture
//! format, not the block: [`Block::decode`] and [`Block::encode`] use UF16,
//! while their signed counterparts use SF16. Encoding and decoding support all
//! 14 modes and all 32 two-subset partitions.
//! Endpoint fitting minimizes squared distances in signed half-float bit space,
//! giving dark and bright HDR values similar relative weight. It does not
//! optimize absolute linear-light error. The bounded multistart search provides
//! high quality, but does not certify the global optimum.
//!
//! Inputs are rounded to binary16 with ties to even. The encoder scores exact
//! finalized palettes, ranks all 324 mode/partition configurations, then
//! refines 64 finalists with independent subset fits, up to 256 sample-pair
//! starts, and discrete endpoint-pair searches. Sparse subsets also use
//! exhaustive ordered cluster assignments; representable palettes have an
//! exact inversion path. This prioritizes quality over encoding throughput.

use std::convert::Infallible;

use crate::math::Rgb32F;

/// A BC6H block. The bytes do not specify whether channels are signed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Block {
    pub data: [u8; 16],
}

impl_fixedcode_struct!(Block { data: [u8; 16] } | Infallible);

impl Block {
    pub const BLACK: Self = Self {
        data: [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    };

    /// Returns the raw 16-byte representation.
    pub const fn bytes(&self) -> [u8; 16] {
        self.data
    }

    /// Constructs a block without interpreting its signedness or mode.
    pub const fn from_bytes(data: [u8; 16]) -> Self {
        Self { data }
    }

    /// Decodes BC6H UF16. Reserved modes produce black texels.
    pub fn decode(self) -> [[Rgb32F; 4]; 4] {
        self.decode_impl(false)
    }

    /// Decodes BC6H SF16. Reserved modes produce black texels.
    pub fn decode_signed(self) -> [[Rgb32F; 4]; 4] {
        self.decode_impl(true)
    }

    /// Encodes BC6H UF16, searching all modes and partitions.
    ///
    /// Only texels with `true` masks participate in fitting. Empty blocks are
    /// black. NaNs become zero; other values are clamped to `[0, 65504]`.
    pub fn encode(colors: [[(Rgb32F, bool); 4]; 4]) -> Self {
        Self::encode_impl(colors, false)
    }

    /// Encodes BC6H SF16, searching all modes and partitions.
    ///
    /// Only texels with `true` masks participate in fitting. Empty blocks are
    /// black. NaNs become zero; other values are clamped to `[-65504, 65504]`.
    pub fn encode_signed(colors: [[(Rgb32F, bool); 4]; 4]) -> Self {
        Self::encode_impl(colors, true)
    }

    fn decode_impl(self, signed: bool) -> [[Rgb32F; 4]; 4] {
        let raw = u128::from_le_bytes(self.data);
        let mode = if raw & 3 < 2 { raw & 3 } else { raw & 31 } as usize;
        let Some(m) = MODES.iter().position(|&v| v == mode) else {
            return [[Rgb32F::BLACK; 4]; 4];
        };
        let bits = ENDPOINT_BITS[m];
        let transformed = mode != 3 && mode != 30;
        let two_subsets = m < 10;
        let mut endpoints = [[0i32; 3]; 4];
        let start = if mode < 2 { 2 } else { 5 };
        for (i, &destination) in ENDPOINT_LAYOUT[m].iter().enumerate() {
            let destination = destination as usize;
            endpoints[destination / 48][destination / 16 % 3] |=
                (((raw >> (start + i)) & 1) as i32) << (destination % 16);
        }
        for c in 0..3 {
            if signed {
                endpoints[0][c] = sign_extend(endpoints[0][c], bits);
            }
            for e in 1..if two_subsets { 4 } else { 2 } {
                if signed || transformed {
                    endpoints[e][c] = sign_extend(endpoints[e][c], DELTA_BITS[m][c]);
                }
                if transformed {
                    endpoints[e][c] = (endpoints[e][c] + endpoints[0][c]) & ((1 << bits) - 1);
                    if signed {
                        endpoints[e][c] = sign_extend(endpoints[e][c], bits);
                    }
                }
            }
        }
        for endpoint in &mut endpoints {
            for value in endpoint {
                *value = unquantize(*value, bits, signed);
            }
        }
        let partition = ((raw >> 77) & 31) as usize;
        let mut offset = if two_subsets { 82 } else { 65 };
        let mut colors = [[Rgb32F::BLACK; 4]; 4];
        for i in 0..16 {
            let subset = if two_subsets {
                (PARTITIONS[partition] >> i) & 1
            } else {
                0
            } as usize;
            let anchor = i == 0 || (two_subsets && i == ANCHORS[partition]);
            let index_bits = if two_subsets { 3 } else { 4 } - usize::from(anchor);
            let index = ((raw >> offset) & ((1 << index_bits) - 1)) as usize;
            offset += index_bits;
            let weight = if two_subsets {
                WEIGHTS3[index]
            } else {
                WEIGHTS4[index]
            };
            colors[i / 4][i % 4] = Rgb32F(std::array::from_fn(|c| {
                let value = ((64 - weight) * endpoints[subset * 2][c]
                    + weight * endpoints[subset * 2 + 1][c]
                    + 32)
                    >> 6;
                finish(value, signed)
            }));
        }
        colors
    }

    fn encode_impl(colors: [[(Rgb32F, bool); 4]; 4], signed: bool) -> Self {
        encode_search(colors, signed, SearchBudget::NORMAL)
    }

    fn encode_baseline(colors: [[(Rgb32F, bool); 4]; 4], signed: bool) -> Self {
        let mut samples = [[0.0f64; 3]; 16];
        let mut valid = [false; 16];
        let mut count = 0;
        for i in 0..16 {
            let (color, include) = colors[i / 4][i % 4];
            valid[i] = include;
            if include {
                count += 1;
                samples[i] = color.0.map(|v| {
                    let v = if v.is_nan() {
                        0.0
                    } else {
                        v.clamp(if signed { -65504.0 } else { 0.0 }, 65504.0)
                    };
                    let sign = if v < 0.0 { -1.0 } else { 1.0 };
                    let h = nearest_half(v.abs());
                    sign * h as f64 * if signed { 32.0 / 31.0 } else { 64.0 / 31.0 }
                });
            }
        }
        if count == 0 {
            return Self::BLACK;
        }
        // A farthest sample pair initializes the line in half-float bit space.
        let first = valid.iter().position(|&v| v).unwrap();
        let mut pair = (first, first);
        let mut distance = -1.0;
        for i in 0..16 {
            for j in 0..i {
                if valid[i] && valid[j] {
                    let d: f64 = (0..3)
                        .map(|c| (samples[i][c] - samples[j][c]).powi(2))
                        .sum();
                    if d > distance {
                        distance = d;
                        pair = (i, j);
                    }
                }
            }
        }
        let mut endpoints = [samples[pair.0], samples[pair.1]];
        let mut best = Self::BLACK;
        let mut best_error = f64::INFINITY;
        // Alternate palette assignment and least-squares endpoint fitting.
        for _ in 0..8 {
            let quantized = endpoints.map(|e| {
                e.map(|v| {
                    nearest_integer(
                        v,
                        if signed { -511 } else { 0 },
                        if signed { 511 } else { 1023 },
                        |n| unquantize(n, 10, signed) as f64,
                    )
                })
            });
            let unquantized = quantized.map(|e| e.map(|v| unquantize(v, 10, signed)));
            let palette: [[f64; 3]; 16] = WEIGHTS4.map(|w| {
                std::array::from_fn(|c| {
                    ((unquantized[0][c] * (64 - w) + unquantized[1][c] * w + 32) >> 6) as f64
                })
            });
            let mut indices = [0usize; 16];
            let mut error = 0.0;
            let mut aa = 0.0;
            let mut ab = 0.0;
            let mut bb = 0.0;
            let mut ax = [0.0; 3];
            let mut bx = [0.0; 3];
            for i in 0..16 {
                if !valid[i] {
                    continue;
                }
                let mut nearest = f64::INFINITY;
                for (index, p) in palette.iter().enumerate() {
                    let d: f64 = (0..3).map(|c| (p[c] - samples[i][c]).powi(2)).sum();
                    if d < nearest {
                        nearest = d;
                        indices[i] = index;
                    }
                }
                error += nearest;
                let b = WEIGHTS4[indices[i]] as f64 / 64.0;
                let a = 1.0 - b;
                aa += a * a;
                ab += a * b;
                bb += b * b;
                for c in 0..3 {
                    ax[c] += a * samples[i][c];
                    bx[c] += b * samples[i][c];
                }
            }
            if error < best_error {
                best_error = error;
                let mut q = quantized;
                if indices[0] >= 8 {
                    q.swap(0, 1);
                    for (i, index) in indices.iter_mut().enumerate() {
                        if valid[i] {
                            *index = 15 - *index;
                        }
                    }
                }
                let mut raw = 3u128;
                for e in 0..2 {
                    for c in 0..3 {
                        raw |= ((q[e][c] & 1023) as u128) << (5 + (e * 3 + c) * 10);
                    }
                }
                let mut offset = 65;
                for (i, index) in indices.iter().enumerate() {
                    raw |= (*index as u128) << offset;
                    offset += if i == 0 { 3 } else { 4 };
                }
                best = Self::from_bytes(raw.to_le_bytes());
            }
            let determinant = aa * bb - ab * ab;
            if determinant <= 1e-10 {
                break;
            }
            for c in 0..3 {
                endpoints[0][c] = (ax[c] * bb - bx[c] * ab) / determinant;
                endpoints[1][c] = (bx[c] * aa - ax[c] * ab) / determinant;
            }
        }
        best
    }
}

#[derive(Clone, Copy)]
struct SearchBudget {
    iterations: usize,
    finalists: usize,
    sweeps: usize,
    cluster_passes: usize,
    sample_starts: usize,
}

impl SearchBudget {
    const NORMAL: Self = Self {
        iterations: 24,
        finalists: 64,
        sweeps: 8,
        cluster_passes: 6,
        sample_starts: 256,
    };
    #[cfg(test)]
    const ORACLE: Self = Self {
        iterations: 48,
        finalists: 128,
        sweeps: 12,
        cluster_passes: 12,
        sample_starts: 512,
    };
}

struct Samples {
    half: [[i32; 3]; 16],
    fit: [[f64; 3]; 16],
    valid: [bool; 16],
    signed: bool,
}

impl Samples {
    fn new(colors: [[(Rgb32F, bool); 4]; 4], signed: bool) -> Self {
        let mut samples = Self {
            half: [[0; 3]; 16],
            fit: [[0.0; 3]; 16],
            valid: [false; 16],
            signed,
        };
        for i in 0..16 {
            let (color, valid) = colors[i / 4][i % 4];
            samples.valid[i] = valid;
            if !valid {
                continue;
            }
            samples.half[i] = color.0.map(|v| {
                let v = if v.is_nan() {
                    0.0
                } else {
                    v.clamp(if signed { -65504.0 } else { 0.0 }, 65504.0)
                };
                let h = nearest_half(v.abs());
                if v < 0.0 { -h } else { h }
            });
            samples.fit[i] =
                samples.half[i].map(|h| h as f64 * if signed { 32.0 / 31.0 } else { 64.0 / 31.0 });
        }
        samples
    }

    fn block_error(&self, block: Block) -> u64 {
        let colors = block.decode_impl(self.signed);
        let mut error = 0;
        for i in 0..16 {
            if !self.valid[i] {
                continue;
            }
            for c in 0..3 {
                let v = colors[i / 4][i % 4].0[c];
                if !v.is_finite() {
                    return u64::MAX;
                }
                let h = nearest_half(v.abs());
                let h = if v < 0.0 { -h } else { h };
                error += (h as i64 - self.half[i][c] as i64).pow(2) as u64;
            }
        }
        error
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    mode: usize,
    partition: usize,
    fields: [[i32; 3]; 4],
    indices: [u8; 16],
    error: u64,
}

fn subset(mode: usize, partition: usize, pixel: usize) -> usize {
    if mode < 10 {
        ((PARTITIONS[partition] >> pixel) & 1) as usize
    } else {
        0
    }
}

fn transformed(mode: usize) -> bool {
    mode != 9 && mode != 10
}

fn field_range(mode: usize, endpoint: usize, channel: usize, signed: bool) -> (i32, i32) {
    let bits = if endpoint == 0 {
        ENDPOINT_BITS[mode]
    } else {
        DELTA_BITS[mode][channel]
    };
    if signed || (endpoint != 0 && transformed(mode)) {
        (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    } else {
        (0, (1 << bits) - 1)
    }
}

impl Candidate {
    fn endpoints(&self, signed: bool) -> [[i32; 3]; 4] {
        let bits = ENDPOINT_BITS[self.mode];
        std::array::from_fn(|e| {
            std::array::from_fn(|c| {
                let mut q = self.fields[e][c];
                if e != 0 && transformed(self.mode) {
                    q = (q + self.fields[0][c]) & ((1 << bits) - 1);
                    if signed {
                        q = sign_extend(q, bits);
                    }
                }
                unquantize(q, bits, signed)
            })
        })
    }

    fn evaluate(mut self, samples: &Samples) -> Self {
        let endpoints = self.endpoints(samples.signed);
        let weights: &[i32] = if self.mode < 10 { &WEIGHTS3 } else { &WEIGHTS4 };
        let mut palette = [[[0i32; 3]; 16]; 2];
        for s in 0..if self.mode < 10 { 2 } else { 1 } {
            for (index, &weight) in weights.iter().enumerate() {
                for c in 0..3 {
                    let value = (endpoints[s * 2][c] * (64 - weight)
                        + endpoints[s * 2 + 1][c] * weight
                        + 32)
                        >> 6;
                    palette[s][index][c] = finish_integer(value, samples.signed);
                }
            }
        }
        self.error = 0;
        self.indices = [0; 16];
        for i in 0..16 {
            if !samples.valid[i] {
                continue;
            }
            let s = subset(self.mode, self.partition, i);
            let anchor = i == 0 || (self.mode < 10 && i == ANCHORS[self.partition]);
            let limit = weights.len() >> usize::from(anchor);
            let mut best_error = u64::MAX;
            for (index, color) in palette[s][..limit].iter().enumerate() {
                if color.iter().any(|&h| h.abs() > 0x7bff) {
                    continue;
                }
                let error = (0..3)
                    .map(|c| (color[c] as i64 - samples.half[i][c] as i64).pow(2) as u64)
                    .sum();
                if error < best_error {
                    best_error = error;
                    self.indices[i] = index as u8;
                }
            }
            self.error = self.error.saturating_add(best_error);
        }
        self
    }

    fn pack(&self) -> Block {
        let mut raw = MODES[self.mode] as u128;
        let start = if self.mode < 2 { 2 } else { 5 };
        for (i, &destination) in ENDPOINT_LAYOUT[self.mode].iter().enumerate() {
            let d = destination as usize;
            raw |= (((self.fields[d / 48][d / 16 % 3] >> (d % 16)) & 1) as u128) << (start + i);
        }
        let mut offset = if self.mode < 10 {
            raw |= (self.partition as u128) << 77;
            82
        } else {
            65
        };
        for i in 0..16 {
            raw |= (self.indices[i] as u128) << offset;
            let anchor = i == 0 || (self.mode < 10 && i == ANCHORS[self.partition]);
            offset += if self.mode < 10 { 3 } else { 4 } - usize::from(anchor);
        }
        Block::from_bytes(raw.to_le_bytes())
    }
}

fn finish_integer(value: i32, signed: bool) -> i32 {
    if signed {
        value.signum() * ((value.abs() * 31) >> 5)
    } else {
        (value * 31) >> 6
    }
}

fn snap_endpoint(value: f64, bits: usize, signed: bool) -> i32 {
    let (low, high) = if signed {
        (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    } else {
        (0, (1 << bits) - 1)
    };
    nearest_integer(value, low, high, |q| unquantize(q, bits, signed) as f64)
}

fn nearest_delta(target: f64, base: i32, bits: usize, delta_bits: usize, signed: bool) -> i32 {
    let period = 1 << bits;
    let low = -(1 << (delta_bits - 1));
    let high = (1 << (delta_bits - 1)) - 1;
    let target_q = snap_endpoint(target, bits, signed);
    let mut best = 0;
    let mut best_error = f64::INFINITY;
    for wrap in -1..=1 {
        let center = (target_q - base + wrap * period).clamp(low, high);
        for delta in [center, (center - 1).max(low), (center + 1).min(high)] {
            let mut q = (base + delta) & (period - 1);
            if signed {
                q = sign_extend(q, bits);
            }
            let error = (unquantize(q, bits, signed) as f64 - target).powi(2);
            if error < best_error {
                best_error = error;
                best = delta;
            }
        }
    }
    best
}

fn quantize_fit(
    target: [[f64; 3]; 4],
    mode: usize,
    partition: usize,
    samples: &Samples,
) -> Candidate {
    let bits = ENDPOINT_BITS[mode];
    let endpoint_count = if mode < 10 { 4 } else { 2 };
    let mut fields = [[0; 3]; 4];
    for c in 0..3 {
        if !transformed(mode) {
            for e in 0..endpoint_count {
                fields[e][c] = snap_endpoint(target[e][c], bits, samples.signed);
            }
            continue;
        }
        let (low, high) = field_range(mode, 0, c, samples.signed);
        let target_q: [i32; 4] = target.map(|e| snap_endpoint(e[c], bits, samples.signed));
        let d_low = -(1 << (DELTA_BITS[mode][c] - 1));
        let d_high = (1 << (DELTA_BITS[mode][c] - 1)) - 1;
        let mut bases = [0i32; 32];
        let mut count = 0;
        let mut add = |q: i32| {
            let q = q.clamp(low, high);
            if !bases[..count].contains(&q) {
                bases[count] = q;
                count += 1;
            }
        };
        for offset in [0, -1, 1, -4, 4, -16, 16] {
            add(target_q[0] + offset);
        }
        let mut interval_low = low;
        let mut interval_high = high;
        for &q in &target_q[1..endpoint_count] {
            add(q - d_low);
            add(q - d_high);
            add(q - (d_low + d_high) / 2);
            interval_low = interval_low.max(q - d_high);
            interval_high = interval_high.min(q - d_low);
        }
        add(interval_low);
        add(interval_high);
        add((interval_low + interval_high) / 2);
        let mut best_error = f64::INFINITY;
        for &base in &bases[..count] {
            let mut channel = [0; 4];
            channel[0] = base;
            let mut error = (unquantize(base, bits, samples.signed) as f64 - target[0][c]).powi(2);
            for e in 1..endpoint_count {
                channel[e] = nearest_delta(
                    target[e][c],
                    base,
                    bits,
                    DELTA_BITS[mode][c],
                    samples.signed,
                );
                let mut q = (base + channel[e]) & ((1 << bits) - 1);
                if samples.signed {
                    q = sign_extend(q, bits);
                }
                error += (unquantize(q, bits, samples.signed) as f64 - target[e][c]).powi(2);
            }
            if error < best_error {
                best_error = error;
                for e in 0..endpoint_count {
                    fields[e][c] = channel[e];
                }
            }
        }
    }
    Candidate {
        mode,
        partition,
        fields,
        indices: [0; 16],
        error: 0,
    }
    .evaluate(samples)
}

fn solve_fit(
    samples: &Samples,
    pixels: &[usize],
    indices: &[u8; 16],
    weights: &[i32],
) -> Option<[[f64; 3]; 2]> {
    let mut aa = 0.0;
    let mut ab = 0.0;
    let mut bb = 0.0;
    let mut ax = [0.0; 3];
    let mut bx = [0.0; 3];
    for &i in pixels {
        let b = weights[indices[i] as usize] as f64 / 64.0;
        let a = 1.0 - b;
        aa += a * a;
        ab += a * b;
        bb += b * b;
        for c in 0..3 {
            ax[c] += a * samples.fit[i][c];
            bx[c] += b * samples.fit[i][c];
        }
    }
    let d = aa * bb - ab * ab;
    if d <= 1e-10 {
        return None;
    }
    Some(std::array::from_fn(|e| {
        std::array::from_fn(|c| {
            if e == 0 {
                (ax[c] * bb - bx[c] * ab) / d
            } else {
                (bx[c] * aa - ax[c] * ab) / d
            }
        })
    }))
}

fn pixels_for(samples: &Samples, mode: usize, partition: usize, s: usize) -> Vec<usize> {
    (0..16)
        .filter(|&i| samples.valid[i] && subset(mode, partition, i) == s)
        .collect()
}

fn initial_fits(samples: &Samples, mode: usize, partition: usize) -> [[[f64; 3]; 4]; 4] {
    let mut starts = [[[0.0; 3]; 4]; 4];
    for s in 0..if mode < 10 { 2 } else { 1 } {
        let pixels = pixels_for(samples, mode, partition, s);
        if pixels.is_empty() {
            continue;
        }
        let mut pair = (pixels[0], pixels[0]);
        let mut distance = -1.0;
        let mut low = [f64::INFINITY; 3];
        let mut high = [f64::NEG_INFINITY; 3];
        let mut mean = [0.0; 3];
        for &i in &pixels {
            for c in 0..3 {
                low[c] = low[c].min(samples.fit[i][c]);
                high[c] = high[c].max(samples.fit[i][c]);
                mean[c] += samples.fit[i][c] / pixels.len() as f64;
            }
            for &j in &pixels {
                let d: f64 = (0..3)
                    .map(|c| (samples.fit[i][c] - samples.fit[j][c]).powi(2))
                    .sum();
                if d > distance {
                    distance = d;
                    pair = (i, j);
                }
            }
        }
        starts[0][s * 2] = samples.fit[pair.0];
        starts[0][s * 2 + 1] = samples.fit[pair.1];
        starts[1][s * 2] = low;
        starts[1][s * 2 + 1] = high;
        let mut covariance = [[0.0; 3]; 3];
        for &i in &pixels {
            for a in 0..3 {
                for b in 0..3 {
                    covariance[a][b] +=
                        (samples.fit[i][a] - mean[a]) * (samples.fit[i][b] - mean[b]);
                }
            }
        }
        let mut axis: [f64; 3] =
            std::array::from_fn(|c| samples.fit[pair.1][c] - samples.fit[pair.0][c]);
        for _ in 0..12 {
            let next: [f64; 3] =
                std::array::from_fn(|a| (0..3).map(|b| covariance[a][b] * axis[b]).sum());
            let length = next.iter().map(|v| v * v).sum::<f64>().sqrt();
            if length < 1e-10 {
                axis = [1.0, 0.0, 0.0];
                break;
            }
            axis = next.map(|v| v / length);
        }
        let project = |i: usize| {
            (0..3)
                .map(|c| (samples.fit[i][c] - mean[c]) * axis[c])
                .sum::<f64>()
        };
        let mut ordered = pixels.clone();
        ordered.sort_by(|&a, &b| project(a).total_cmp(&project(b)));
        let min = project(ordered[0]);
        let max = project(*ordered.last().unwrap());
        starts[2][s * 2] = std::array::from_fn(|c| mean[c] + min * axis[c]);
        starts[2][s * 2 + 1] = std::array::from_fn(|c| mean[c] + max * axis[c]);
        let mut indices = [0u8; 16];
        for (rank, &i) in ordered.iter().enumerate() {
            indices[i] = ((rank * 15) / (ordered.len() - 1).max(1)) as u8;
        }
        let fitted = solve_fit(samples, &ordered, &indices, &WEIGHTS4).unwrap_or([mean; 2]);
        starts[3][s * 2] = fitted[0];
        starts[3][s * 2 + 1] = fitted[1];
    }
    // Empty subsets carry no fitting constraints and use the base endpoint.
    for start in &mut starts {
        for s in 0..if mode < 10 { 2 } else { 1 } {
            if pixels_for(samples, mode, partition, s).is_empty() {
                start[s * 2] = start[0];
                start[s * 2 + 1] = start[0];
            }
        }
    }
    starts
}

fn refine_fit(mut candidate: Candidate, samples: &Samples, iterations: usize) -> Candidate {
    let weights: &[i32] = if candidate.mode < 10 {
        &WEIGHTS3
    } else {
        &WEIGHTS4
    };
    let pixels = [
        pixels_for(samples, candidate.mode, candidate.partition, 0),
        pixels_for(samples, candidate.mode, candidate.partition, 1),
    ];
    let mut best = candidate;
    for _ in 0..iterations {
        let mut target = candidate
            .endpoints(samples.signed)
            .map(|e| e.map(|v| v as f64));
        for s in 0..if candidate.mode < 10 { 2 } else { 1 } {
            if let Some(fit) = solve_fit(samples, &pixels[s], &candidate.indices, weights) {
                target[s * 2] = fit[0];
                target[s * 2 + 1] = fit[1];
            }
        }
        let next = quantize_fit(target, candidate.mode, candidate.partition, samples);
        if next.error < best.error {
            best = next;
        }
        if next.fields == candidate.fields {
            break;
        }
        candidate = next;
    }
    best
}

fn discrete_refine(mut best: Candidate, samples: &Samples, sweeps: usize) -> Candidate {
    for step in [16, 8, 4, 2, 1] {
        for _ in 0..sweeps {
            let previous = best.error;
            for e in 0..if best.mode < 10 { 4 } else { 2 } {
                for c in 0..3 {
                    let (low, high) = field_range(best.mode, e, c, samples.signed);
                    for delta in [-step, step] {
                        let mut trial = best;
                        trial.fields[e][c] = (trial.fields[e][c] + delta).clamp(low, high);
                        trial = trial.evaluate(samples);
                        if trial.error < best.error {
                            best = trial;
                        }
                        if e == 0 && transformed(best.mode) {
                            let mut trial = best;
                            let target = best.endpoints(samples.signed);
                            trial.fields[0][c] = (trial.fields[0][c] + delta).clamp(low, high);
                            for other in 1..if best.mode < 10 { 4 } else { 2 } {
                                trial.fields[other][c] = nearest_delta(
                                    target[other][c] as f64,
                                    trial.fields[0][c],
                                    ENDPOINT_BITS[best.mode],
                                    DELTA_BITS[best.mode][c],
                                    samples.signed,
                                );
                            }
                            trial = trial.evaluate(samples);
                            if trial.error < best.error {
                                best = trial;
                            }
                        }
                    }
                }
            }
            if best.error == previous {
                break;
            }
        }
    }
    best
}

fn ordered_refine(mut best: Candidate, samples: &Samples, passes: usize) -> Candidate {
    let weights: &[i32] = if best.mode < 10 { &WEIGHTS3 } else { &WEIGHTS4 };
    for s in 0..if best.mode < 10 { 2 } else { 1 } {
        let mut pixels = pixels_for(samples, best.mode, best.partition, s);
        if pixels.len() < 2 {
            continue;
        }
        let endpoints = best.endpoints(samples.signed);
        let axis: [f64; 3] =
            std::array::from_fn(|c| (endpoints[s * 2 + 1][c] - endpoints[s * 2][c]) as f64);
        let project = |i: usize| (0..3).map(|c| samples.fit[i][c] * axis[c]).sum::<f64>();
        pixels.sort_by(|&a, &b| project(a).total_cmp(&project(b)));
        let mut assigned = best.indices;
        // Projected cluster boundaries must form a monotone sequence.
        let mut previous = 0;
        for &i in &pixels {
            assigned[i] = assigned[i].max(previous);
            previous = assigned[i];
        }
        for _ in 0..passes {
            let previous = best.error;
            for rank in 0..pixels.len() {
                let i = pixels[rank];
                let low = if rank == 0 {
                    0
                } else {
                    assigned[pixels[rank - 1]] as usize
                };
                let high = if rank + 1 == pixels.len() {
                    weights.len() - 1
                } else {
                    assigned[pixels[rank + 1]] as usize
                };
                for index in low..=high {
                    let mut indices = assigned;
                    indices[i] = index as u8;
                    if let Some(fit) = solve_fit(samples, &pixels, &indices, weights) {
                        let mut target =
                            best.endpoints(samples.signed).map(|e| e.map(|v| v as f64));
                        target[s * 2] = fit[0];
                        target[s * 2 + 1] = fit[1];
                        let trial = quantize_fit(target, best.mode, best.partition, samples);
                        if trial.error < best.error {
                            best = trial;
                            assigned = indices;
                        }
                    }
                }
            }
            if best.error == previous {
                break;
            }
        }
    }
    best
}

fn independent_restarts(mut best: Candidate, samples: &Samples, budget: SearchBudget) -> Candidate {
    let starts = initial_fits(samples, best.mode, best.partition);
    for first in 0..4 {
        for second in 0..if best.mode < 10 { 4 } else { 1 } {
            if best.mode < 10 && first == second {
                continue;
            }
            for orientation in 0..if best.mode < 10 { 4 } else { 2 } {
                let mut target = starts[first];
                if best.mode < 10 {
                    target[2] = starts[second][2];
                    target[3] = starts[second][3];
                }
                for s in 0..if best.mode < 10 { 2 } else { 1 } {
                    if orientation & (1 << s) != 0 {
                        target.swap(s * 2, s * 2 + 1);
                    }
                }
                let trial = refine_fit(
                    quantize_fit(target, best.mode, best.partition, samples),
                    samples,
                    budget.iterations,
                );
                if trial.error < best.error {
                    best = trial;
                }
            }
        }
    }
    {
        // Independent sample-pair starts probe lines outside the PCA basins.
        let mut previous = Vec::new();
        for seed in 0..budget.sample_starts {
            let mut target = best.endpoints(samples.signed).map(|e| e.map(|v| v as f64));
            for s in 0..if best.mode < 10 { 2 } else { 1 } {
                let pixels = pixels_for(samples, best.mode, best.partition, s);
                if !pixels.is_empty() {
                    target[s * 2] = samples.fit[pixels[(seed + s) % pixels.len()]];
                    target[s * 2 + 1] = samples.fit
                        [pixels[(seed / pixels.len() + seed * 7 + 3 + s) % pixels.len()]];
                }
            }
            if previous.contains(&target) {
                continue;
            }
            previous.push(target);
            let trial = refine_fit(
                quantize_fit(target, best.mode, best.partition, samples),
                samples,
                budget.iterations,
            );
            if trial.error < best.error {
                best = trial;
            }
        }
    }
    best
}

fn pair_refine(mut best: Candidate, samples: &Samples, sweeps: usize) -> Candidate {
    for step in [16, 8, 4, 2, 1] {
        for _ in 0..sweeps {
            let previous = best.error;
            for s in 0..if best.mode < 10 { 2 } else { 1 } {
                for c in 0..3 {
                    let e = s * 2;
                    let (alow, ahigh) = field_range(best.mode, e, c, samples.signed);
                    let (blow, bhigh) = field_range(best.mode, e + 1, c, samples.signed);
                    for da in [-step, 0, step] {
                        for db in [-step, 0, step] {
                            if da == 0 && db == 0 {
                                continue;
                            }
                            let mut trial = best;
                            trial.fields[e][c] = (trial.fields[e][c] + da).clamp(alow, ahigh);
                            trial.fields[e + 1][c] =
                                (trial.fields[e + 1][c] + db).clamp(blow, bhigh);
                            trial = trial.evaluate(samples);
                            if trial.error < best.error {
                                best = trial;
                            }
                            if e == 0 && da != 0 && transformed(best.mode) {
                                let endpoints = best.endpoints(samples.signed);
                                for other in 2..if best.mode < 10 { 4 } else { 2 } {
                                    trial.fields[other][c] = nearest_delta(
                                        endpoints[other][c] as f64,
                                        trial.fields[0][c],
                                        ENDPOINT_BITS[best.mode],
                                        DELTA_BITS[best.mode][c],
                                        samples.signed,
                                    );
                                }
                                trial = trial.evaluate(samples);
                                if trial.error < best.error {
                                    best = trial;
                                }
                            }
                        }
                    }
                }
            }
            if best.error == previous {
                break;
            }
        }
    }
    best
}

fn exhaustive_ordered_fit(
    mut best: Candidate,
    samples: &Samples,
    maximum_samples: usize,
) -> Candidate {
    fn visit(
        best: &mut Candidate,
        samples: &Samples,
        pixels: &[usize],
        s: usize,
        weights: &[i32],
        indices: &mut [u8; 16],
        rank: usize,
        minimum: usize,
    ) {
        if rank == pixels.len() {
            if let Some(fit) = solve_fit(samples, pixels, indices, weights) {
                let mut target = best.endpoints(samples.signed).map(|e| e.map(|v| v as f64));
                target[s * 2] = fit[0];
                target[s * 2 + 1] = fit[1];
                let trial = quantize_fit(target, best.mode, best.partition, samples);
                if trial.error < best.error {
                    *best = trial;
                }
            }
            return;
        }
        let i = pixels[rank];
        let anchor = i == 0 || (best.mode < 10 && i == ANCHORS[best.partition]);
        let maximum = weights.len() >> usize::from(anchor);
        for index in minimum..maximum {
            indices[i] = index as u8;
            visit(best, samples, pixels, s, weights, indices, rank + 1, index);
        }
    }
    let weights: &[i32] = if best.mode < 10 { &WEIGHTS3 } else { &WEIGHTS4 };
    for s in 0..if best.mode < 10 { 2 } else { 1 } {
        let mut pixels = pixels_for(samples, best.mode, best.partition, s);
        if pixels.len() < 2 || pixels.len() > maximum_samples {
            continue;
        }
        let endpoints = best.endpoints(samples.signed);
        let axis: [f64; 3] =
            std::array::from_fn(|c| (endpoints[s * 2 + 1][c] - endpoints[s * 2][c]) as f64);
        let project = |i: usize| (0..3).map(|c| samples.fit[i][c] * axis[c]).sum::<f64>();
        pixels.sort_by(|&a, &b| project(a).total_cmp(&project(b)));
        let mut indices = best.indices;
        visit(&mut best, samples, &pixels, s, weights, &mut indices, 0, 0);
    }
    best
}

fn encode_search(colors: [[(Rgb32F, bool); 4]; 4], signed: bool, budget: SearchBudget) -> Block {
    let samples = Samples::new(colors, signed);
    if !samples.valid.iter().any(|&v| v) {
        return Block::BLACK;
    }
    let mut result = Block::encode_baseline(colors, signed);
    let mut error = samples.block_error(result);
    if budget.finalists > SearchBudget::NORMAL.finalists {
        let normal = encode_search(colors, signed, SearchBudget::NORMAL);
        let normal_error = samples.block_error(normal);
        if normal_error < error {
            error = normal_error;
            result = normal;
        }
    }
    let mut candidates = Vec::with_capacity(324);
    for partition in 0..32 {
        let starts = initial_fits(&samples, 0, partition);
        for mode in 0..10 {
            let mut best = Candidate {
                mode,
                partition,
                fields: [[0; 3]; 4],
                indices: [0; 16],
                error: u64::MAX,
            };
            for start in starts {
                for orientation in 0..4 {
                    let mut target = start;
                    for s in 0..2 {
                        if orientation & (1 << s) != 0 {
                            target.swap(s * 2, s * 2 + 1);
                        }
                    }
                    let trial = refine_fit(
                        quantize_fit(target, mode, partition, &samples),
                        &samples,
                        budget.iterations,
                    );
                    if trial.error < best.error {
                        best = trial;
                    }
                }
            }
            if best.error < error {
                error = best.error;
                result = best.pack();
            }
            if error == 0 {
                return result;
            }
            candidates.push(best);
        }
    }
    let starts = initial_fits(&samples, 10, 0);
    for mode in 10..14 {
        let mut best = Candidate {
            mode,
            partition: 0,
            fields: [[0; 3]; 4],
            indices: [0; 16],
            error: u64::MAX,
        };
        for start in starts {
            for orientation in 0..2 {
                let mut target = start;
                if orientation != 0 {
                    target.swap(0, 1);
                }
                let trial = refine_fit(
                    quantize_fit(target, mode, 0, &samples),
                    &samples,
                    budget.iterations,
                );
                if trial.error < best.error {
                    best = trial;
                }
            }
        }
        if best.error < error {
            error = best.error;
            result = best.pack();
        }
        if error == 0 {
            return result;
        }
        candidates.push(best);
    }
    candidates.sort_by_key(|candidate| candidate.error);
    if error == 0 {
        return result;
    }
    if let Some(exact) = exact_six_bit_palette(&samples, &candidates) {
        return exact.pack();
    }
    for candidate in &candidates {
        if let Some(exact) = exact_mode_palette(&samples, *candidate) {
            return exact.pack();
        }
    }
    let mut finalists = Vec::with_capacity(budget.finalists);
    // Preserve every mode when coarse scores tie or favor a different basin.
    for mode in 0..14 {
        if let Some(&candidate) = candidates.iter().find(|candidate| candidate.mode == mode) {
            finalists.push(candidate);
        }
    }
    for candidate in candidates {
        if finalists.len() >= budget.finalists {
            break;
        }
        if !finalists.iter().any(|existing| {
            existing.mode == candidate.mode && existing.partition == candidate.partition
        }) {
            finalists.push(candidate);
        }
    }
    finalists.sort_by_key(|candidate| candidate.error);
    for candidate in finalists {
        let candidate = independent_restarts(candidate, &samples, budget);
        let candidate = discrete_refine(candidate, &samples, budget.sweeps);
        let candidate = pair_refine(candidate, &samples, budget.sweeps);
        let candidate = exhaustive_ordered_fit(
            candidate,
            &samples,
            if budget.finalists > SearchBudget::NORMAL.finalists {
                6
            } else {
                4
            },
        );
        let candidate = ordered_refine(candidate, &samples, budget.cluster_passes);
        let candidate = pair_refine(candidate, &samples, budget.sweeps);
        let candidate = discrete_refine(candidate, &samples, budget.sweeps);
        if candidate.error < error {
            error = candidate.error;
            result = candidate.pack();
        }
    }
    result
}

#[derive(Clone)]
struct ScalarPair {
    endpoints: [i32; 2],
    masks: [u16; 16],
}

fn scalar_six_bit_pairs(
    samples: &Samples,
    pixels: &[usize],
    channel: usize,
    partition: usize,
) -> Vec<ScalarPair> {
    let (low, high) = field_range(9, 0, channel, samples.signed);
    let values: Vec<i32> = (low..=high)
        .map(|q| unquantize(q, 6, samples.signed))
        .collect();
    let minimum = pixels
        .iter()
        .map(|&i| samples.half[i][channel])
        .min()
        .unwrap();
    let maximum = pixels
        .iter()
        .map(|&i| samples.half[i][channel])
        .max()
        .unwrap();
    let mut pairs = Vec::new();
    for (a_index, &a) in values.iter().enumerate() {
        for (b_index, &b) in values.iter().enumerate() {
            let ea = finish_integer(a, samples.signed);
            let eb = finish_integer(b, samples.signed);
            if minimum < ea.min(eb) || maximum > ea.max(eb) {
                continue;
            }
            let palette =
                WEIGHTS3.map(|w| finish_integer((a * (64 - w) + b * w + 32) >> 6, samples.signed));
            let mut masks = [0u16; 16];
            let mut possible = true;
            for &i in pixels {
                let limit = if i == 0 || i == ANCHORS[partition] {
                    4
                } else {
                    8
                };
                for (index, &value) in palette[..limit].iter().enumerate() {
                    if value == samples.half[i][channel] {
                        masks[i] |= 1 << index;
                    }
                }
                if masks[i] == 0 {
                    possible = false;
                    break;
                }
            }
            if possible {
                pairs.push(ScalarPair {
                    endpoints: [low + a_index as i32, low + b_index as i32],
                    masks,
                });
            }
        }
    }
    pairs
}

fn exact_six_bit_subset(samples: &Samples, partition: usize, s: usize) -> Option<[[i32; 3]; 2]> {
    let pixels = pixels_for(samples, 9, partition, s);
    if pixels.is_empty() {
        return Some([[0; 3]; 2]);
    }
    let red = scalar_six_bit_pairs(samples, &pixels, 0, partition);
    if red.is_empty() {
        return None;
    }
    let green = scalar_six_bit_pairs(samples, &pixels, 1, partition);
    if green.is_empty() {
        return None;
    }
    let blue = scalar_six_bit_pairs(samples, &pixels, 2, partition);
    if blue.is_empty() {
        return None;
    }
    for r in &red {
        for g in &green {
            let mut masks = [0u16; 16];
            for &i in &pixels {
                masks[i] = r.masks[i] & g.masks[i];
            }
            if pixels.iter().any(|&i| masks[i] == 0) {
                continue;
            }
            for b in &blue {
                if pixels.iter().all(|&i| masks[i] & b.masks[i] != 0) {
                    return Some(std::array::from_fn(|e| {
                        [r.endpoints[e], g.endpoints[e], b.endpoints[e]]
                    }));
                }
            }
        }
    }
    None
}

fn exact_six_bit_palette(samples: &Samples, ranked: &[Candidate]) -> Option<Candidate> {
    for candidate in ranked.iter().filter(|candidate| candidate.mode == 9) {
        let Some(first) = exact_six_bit_subset(samples, candidate.partition, 0) else {
            continue;
        };
        let Some(second) = exact_six_bit_subset(samples, candidate.partition, 1) else {
            continue;
        };
        let candidate = Candidate {
            fields: [first[0], first[1], second[0], second[1]],
            ..*candidate
        }
        .evaluate(samples);
        if candidate.error == 0 {
            return Some(candidate);
        }
    }
    None
}

fn half_interval(code: i32, signed: bool) -> [f64; 2] {
    let denominator = if signed { 32 } else { 64 };
    let magnitude = code.abs();
    let low = (magnitude * denominator + 30) / 31;
    let high = ((magnitude + 1) * denominator + 30) / 31 - 1;
    if code < 0 {
        [-(high as f64) - 0.5, -(low as f64) + 0.5]
    } else if signed && code == 0 {
        [-(high as f64) - 0.5, high as f64 + 0.5]
    } else {
        [low as f64 - 0.5, high as f64 + 0.5]
    }
}

fn feasible_delta(base: i32, endpoint: i32, bits: usize, delta_bits: usize) -> Option<i32> {
    let delta = sign_extend((endpoint - base) & ((1 << bits) - 1), bits);
    let low = -(1 << (delta_bits - 1));
    let high = (1 << (delta_bits - 1)) - 1;
    (delta >= low && delta <= high).then_some(delta)
}

fn scalar_mode_pairs(
    samples: &Samples,
    candidate: Candidate,
    s: usize,
    channel: usize,
    pixels: &[usize],
) -> Vec<ScalarPair> {
    let bits = ENDPOINT_BITS[candidate.mode];
    let weights: &[i32] = if candidate.mode < 10 {
        &WEIGHTS3
    } else {
        &WEIGHTS4
    };
    let (low, high) = field_range(candidate.mode, 0, channel, samples.signed);
    let minimum = pixels
        .iter()
        .map(|&i| samples.half[i][channel])
        .min()
        .unwrap();
    let maximum = pixels
        .iter()
        .map(|&i| samples.half[i][channel])
        .max()
        .unwrap();
    let mut pairs = Vec::new();
    let mut add_pair = |a: i32, b: i32| {
        if transformed(candidate.mode) {
            if s == 0 && feasible_delta(a, b, bits, DELTA_BITS[candidate.mode][channel]).is_none() {
                return;
            }
            if s != 0 {
                let difference = sign_extend((b - a) & ((1 << bits) - 1), bits);
                if difference.abs() >= 1 << DELTA_BITS[candidate.mode][channel] {
                    return;
                }
            }
        }
        let ua = unquantize(a, bits, samples.signed);
        let ub = unquantize(b, bits, samples.signed);
        let ea = finish_integer(ua, samples.signed);
        let eb = finish_integer(ub, samples.signed);
        if minimum < ea.min(eb) || maximum > ea.max(eb) {
            return;
        }
        let mut palette = [0; 16];
        for (index, &w) in weights.iter().enumerate() {
            palette[index] = finish_integer((ua * (64 - w) + ub * w + 32) >> 6, samples.signed);
        }
        let mut masks = [0u16; 16];
        for &i in pixels {
            let anchor = i == 0 || (candidate.mode < 10 && i == ANCHORS[candidate.partition]);
            let limit = weights.len() >> usize::from(anchor);
            for (index, &value) in palette[..limit].iter().enumerate() {
                if value == samples.half[i][channel] {
                    masks[i] |= 1 << index;
                }
            }
            if masks[i] == 0 {
                return;
            }
        }
        // Independent endpoints need only one witness per index-mask pattern.
        if pairs.iter().any(|pair: &ScalarPair| {
            if transformed(candidate.mode) {
                pair.endpoints == [a, b]
            } else {
                pair.masks == masks
            }
        }) {
            return;
        }
        pairs.push(ScalarPair {
            endpoints: [a, b],
            masks,
        });
    };
    if minimum == maximum {
        let interval = half_interval(minimum, samples.signed);
        let center = snap_endpoint((interval[0] + interval[1]) * 0.5, bits, samples.signed);
        let radius = if transformed(candidate.mode) {
            (1 << DELTA_BITS[candidate.mode][channel]) + 2
        } else {
            high - low
        };
        for a in (center - radius).max(low)..=(center + radius).min(high) {
            let ua = unquantize(a, bits, samples.signed) as f64;
            add_pair(a, a);
            for &w in &weights[1..weights.len() - 1] {
                let qlow = snap_endpoint(
                    (64.0 * interval[0] - (64 - w) as f64 * ua) / w as f64,
                    bits,
                    samples.signed,
                );
                let qhigh = snap_endpoint(
                    (64.0 * interval[1] - (64 - w) as f64 * ua) / w as f64,
                    bits,
                    samples.signed,
                );
                for b in (qlow - 1).max(low)..=(qhigh + 1).min(high) {
                    add_pair(a, b);
                }
            }
        }
        return pairs;
    }
    let low_interval = half_interval(minimum, samples.signed);
    let high_interval = half_interval(maximum, samples.signed);
    for reverse in [false, true] {
        let first = if reverse { high_interval } else { low_interval };
        let second = if reverse { low_interval } else { high_interval };
        for j in 0..weights.len() - 1 {
            for k in j + 1..weights.len() {
                let wj = weights[j] as f64;
                let wk = weights[k] as f64;
                let mut amin = f64::INFINITY;
                let mut amax = f64::NEG_INFINITY;
                let mut bmin = f64::INFINITY;
                let mut bmax = f64::NEG_INFINITY;
                for x in first {
                    for y in second {
                        let a = (wk * x - wj * y) / (wk - wj);
                        let b = ((64.0 - wj) * y - (64.0 - wk) * x) / (wk - wj);
                        amin = amin.min(a);
                        amax = amax.max(a);
                        bmin = bmin.min(b);
                        bmax = bmax.max(b);
                    }
                }
                let alow = (snap_endpoint(amin, bits, samples.signed) - 1).max(low);
                let ahigh = (snap_endpoint(amax, bits, samples.signed) + 1).min(high);
                let blow = (snap_endpoint(bmin, bits, samples.signed) - 1).max(low);
                let bhigh = (snap_endpoint(bmax, bits, samples.signed) + 1).min(high);
                for a in alow..=ahigh {
                    for b in blow..=bhigh {
                        add_pair(a, b);
                    }
                }
            }
        }
    }
    pairs
}

fn compatible_rgb(
    pairs: [&[ScalarPair]; 3],
    pixels: &[usize],
    base: Option<[i32; 3]>,
    mode: usize,
) -> Option<[[i32; 3]; 2]> {
    let feasible = |pair: &ScalarPair, c: usize| {
        base.is_none_or(|base| {
            pair.endpoints.iter().all(|&q| {
                feasible_delta(base[c], q, ENDPOINT_BITS[mode], DELTA_BITS[mode][c]).is_some()
            })
        })
    };
    for r in pairs[0].iter().filter(|pair| feasible(pair, 0)) {
        for g in pairs[1].iter().filter(|pair| feasible(pair, 1)) {
            let mut masks = [0u16; 16];
            for &i in pixels {
                masks[i] = r.masks[i] & g.masks[i];
            }
            if pixels.iter().any(|&i| masks[i] == 0) {
                continue;
            }
            for b in pairs[2].iter().filter(|pair| feasible(pair, 2)) {
                if pixels.iter().all(|&i| masks[i] & b.masks[i] != 0) {
                    return Some(std::array::from_fn(|e| {
                        [r.endpoints[e], g.endpoints[e], b.endpoints[e]]
                    }));
                }
            }
        }
    }
    None
}

fn exact_mode_palette(samples: &Samples, candidate: Candidate) -> Option<Candidate> {
    let pixels = [
        pixels_for(samples, candidate.mode, candidate.partition, 0),
        pixels_for(samples, candidate.mode, candidate.partition, 1),
    ];
    if pixels[0].is_empty() {
        if pixels[1].is_empty() {
            return Some(
                Candidate {
                    fields: [[0; 3]; 4],
                    ..candidate
                }
                .evaluate(samples),
            );
        }
        let pairs: [Vec<ScalarPair>; 3] =
            std::array::from_fn(|c| scalar_mode_pairs(samples, candidate, 1, c, &pixels[1]));
        let second = compatible_rgb(
            [&pairs[0], &pairs[1], &pairs[2]],
            &pixels[1],
            None,
            candidate.mode,
        )?;
        let mut fields = [[0; 3]; 4];
        fields[2] = second[0];
        fields[3] = second[1];
        if transformed(candidate.mode) {
            for c in 0..3 {
                let bits = ENDPOINT_BITS[candidate.mode];
                let delta_bits = DELTA_BITS[candidate.mode][c];
                let dlo = -(1 << (delta_bits - 1));
                let dhi = (1 << (delta_bits - 1)) - 1;
                let mut found = None;
                // A feasible intersection starts at one of the circular
                // delta-interval boundaries for the two observed endpoints.
                for base in [
                    second[0][c] - dhi,
                    second[0][c] - dlo,
                    second[1][c] - dhi,
                    second[1][c] - dlo,
                ] {
                    let mut base = base & ((1 << bits) - 1);
                    if samples.signed {
                        base = sign_extend(base, bits);
                    }
                    if let (Some(a), Some(b)) = (
                        feasible_delta(base, second[0][c], bits, delta_bits),
                        feasible_delta(base, second[1][c], bits, delta_bits),
                    ) {
                        found = Some((base, a, b));
                        break;
                    }
                }
                let (base, a, b) = found?;
                fields[0][c] = base;
                fields[2][c] = a;
                fields[3][c] = b;
            }
        }
        let exact = Candidate {
            fields,
            ..candidate
        }
        .evaluate(samples);
        return (exact.error == 0).then_some(exact);
    }
    let first: [Vec<ScalarPair>; 3] =
        std::array::from_fn(|c| scalar_mode_pairs(samples, candidate, 0, c, &pixels[0]));
    if first.iter().any(|pairs| pairs.is_empty()) {
        return None;
    }
    if candidate.mode >= 10 || pixels[1].is_empty() {
        let first = compatible_rgb(
            [&first[0], &first[1], &first[2]],
            &pixels[0],
            None,
            candidate.mode,
        )?;
        let mut fields = [first[0], first[1], [0; 3], [0; 3]];
        if transformed(candidate.mode) {
            for c in 0..3 {
                fields[1][c] = feasible_delta(
                    fields[0][c],
                    fields[1][c],
                    ENDPOINT_BITS[candidate.mode],
                    DELTA_BITS[candidate.mode][c],
                )?;
            }
        }
        let exact = Candidate {
            fields,
            ..candidate
        }
        .evaluate(samples);
        return (exact.error == 0).then_some(exact);
    }
    let second: [Vec<ScalarPair>; 3] =
        std::array::from_fn(|c| scalar_mode_pairs(samples, candidate, 1, c, &pixels[1]));
    if second.iter().any(|pairs| pairs.is_empty()) {
        return None;
    }
    for r in &first[0] {
        for g in &first[1] {
            let mut masks = [0u16; 16];
            for &i in &pixels[0] {
                masks[i] = r.masks[i] & g.masks[i];
            }
            if pixels[0].iter().any(|&i| masks[i] == 0) {
                continue;
            }
            for b in &first[2] {
                if pixels[0].iter().any(|&i| masks[i] & b.masks[i] == 0) {
                    continue;
                }
                let first: [[i32; 3]; 2] =
                    std::array::from_fn(|e| [r.endpoints[e], g.endpoints[e], b.endpoints[e]]);
                let base = transformed(candidate.mode).then_some(first[0]);
                let Some(second) = compatible_rgb(
                    [&second[0], &second[1], &second[2]],
                    &pixels[1],
                    base,
                    candidate.mode,
                ) else {
                    continue;
                };
                let mut fields = [first[0], first[1], second[0], second[1]];
                if transformed(candidate.mode) {
                    for e in 1..4 {
                        for c in 0..3 {
                            fields[e][c] = feasible_delta(
                                fields[0][c],
                                fields[e][c],
                                ENDPOINT_BITS[candidate.mode],
                                DELTA_BITS[candidate.mode][c],
                            )
                            .unwrap();
                        }
                    }
                }
                let exact = Candidate {
                    fields,
                    ..candidate
                }
                .evaluate(samples);
                if exact.error == 0 {
                    return Some(exact);
                }
            }
        }
    }
    None
}

fn nearest_half(value: f32) -> i32 {
    let h = nearest_integer(value as f64, 0, 0x7bff, |n| half_to_float(n as u16) as f64);
    if h > 0
        && h & 1 != 0
        && (half_to_float(h as u16) as f64 - value as f64).abs()
            == (half_to_float((h - 1) as u16) as f64 - value as f64).abs()
    {
        h - 1
    } else {
        h
    }
}

fn nearest_integer(value: f64, mut low: i32, mut high: i32, decode: impl Fn(i32) -> f64) -> i32 {
    let minimum = low;
    while low < high {
        let middle = low + (high - low) / 2;
        if decode(middle) < value {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if low > minimum && (decode(low - 1) - value).abs() < (decode(low) - value).abs() {
        low - 1
    } else {
        low
    }
}

fn sign_extend(value: i32, bits: usize) -> i32 {
    (value << (32 - bits)) >> (32 - bits)
}

fn unquantize(value: i32, bits: usize, signed: bool) -> i32 {
    if bits >= if signed { 16 } else { 15 } {
        return value;
    }
    let magnitude = value.abs();
    let maximum = (1 << (bits - usize::from(signed))) - 1;
    let result = if magnitude == 0 {
        0
    } else if magnitude >= maximum {
        if signed { 0x7fff } else { 0xffff }
    } else {
        ((magnitude << 15) + 0x4000) >> (bits - 1)
    };
    if value < 0 { -result } else { result }
}

fn finish(value: i32, signed: bool) -> f32 {
    let bits = if signed {
        let magnitude = ((value.abs() * 31) >> 5) as u16;
        // The BC6H finalizer determines the sign after scaling, so tiny
        // negative magnitudes that truncate to zero produce positive zero.
        magnitude
            | if value < 0 && magnitude != 0 {
                0x8000
            } else {
                0
            }
    } else {
        ((value * 31) >> 6) as u16
    };
    half_to_float(bits)
}

fn half_to_float(bits: u16) -> f32 {
    let sign = (bits as u32 & 0x8000) << 16;
    let exponent = bits as u32 >> 10 & 31;
    let mantissa = bits as u32 & 1023;
    if exponent == 0 {
        let value = mantissa as f32 * (1.0 / 16777216.0);
        if sign != 0 { -value } else { value }
    } else {
        f32::from_bits(
            sign | (if exponent == 31 { 255 } else { exponent + 112 } << 23) | mantissa << 13,
        )
    }
}

const MODES: [usize; 14] = [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15];
const ENDPOINT_BITS: [usize; 14] = [10, 7, 11, 11, 11, 9, 8, 8, 8, 6, 10, 11, 12, 16];
const DELTA_BITS: [[usize; 3]; 14] = [
    [5, 5, 5],
    [6, 6, 6],
    [5, 4, 4],
    [4, 5, 4],
    [4, 4, 5],
    [5, 5, 5],
    [6, 5, 5],
    [5, 6, 5],
    [5, 5, 6],
    [6, 6, 6],
    [10, 10, 10],
    [9, 9, 9],
    [8, 8, 8],
    [4, 4, 4],
];
const WEIGHTS3: [i32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
const PARTITIONS: [u16; 32] = [
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800,
    0xffe8, 0xff00, 0xfff0, 0xf000, 0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
];
const ANCHORS: [usize; 32] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2,
];

// Endpoint bit destinations from Khronos DataFormat BC6H block layouts.
// Each entry packs endpoint, channel and bit as endpoint * 48 + channel * 16 + bit.
const ENDPOINT_LAYOUT: [&[u8]; 14] = [
    &[
        116, 132, 180, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 164, 112, 113, 114, 115, 64, 65,
        66, 67, 68, 176, 160, 161, 162, 163, 80, 81, 82, 83, 84, 177, 128, 129, 130, 131, 96, 97,
        98, 99, 100, 178, 144, 145, 146, 147, 148, 179,
    ], // Mode 0
    &[
        117, 164, 165, 0, 1, 2, 3, 4, 5, 6, 176, 177, 132, 16, 17, 18, 19, 20, 21, 22, 133, 178,
        116, 32, 33, 34, 35, 36, 37, 38, 179, 181, 180, 48, 49, 50, 51, 52, 53, 112, 113, 114, 115,
        64, 65, 66, 67, 68, 69, 160, 161, 162, 163, 80, 81, 82, 83, 84, 85, 128, 129, 130, 131, 96,
        97, 98, 99, 100, 101, 144, 145, 146, 147, 148, 149,
    ], // Mode 1
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 10, 112, 113, 114, 115, 64, 65, 66, 67, 26, 176,
        160, 161, 162, 163, 80, 81, 82, 83, 42, 177, 128, 129, 130, 131, 96, 97, 98, 99, 100, 178,
        144, 145, 146, 147, 148, 179,
    ], // Mode 2
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 48, 49, 50, 51, 10, 164, 112, 113, 114, 115, 64, 65, 66, 67, 68, 26,
        160, 161, 162, 163, 80, 81, 82, 83, 42, 177, 128, 129, 130, 131, 96, 97, 98, 99, 176, 178,
        144, 145, 146, 147, 116, 179,
    ], // Mode 6
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 48, 49, 50, 51, 10, 132, 112, 113, 114, 115, 64, 65, 66, 67, 26, 176,
        160, 161, 162, 163, 80, 81, 82, 83, 84, 42, 128, 129, 130, 131, 96, 97, 98, 99, 177, 178,
        144, 145, 146, 147, 180, 179,
    ], // Mode 10
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 132, 16, 17, 18, 19, 20, 21, 22, 23, 24, 116, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 180, 48, 49, 50, 51, 52, 164, 112, 113, 114, 115, 64, 65, 66, 67, 68,
        176, 160, 161, 162, 163, 80, 81, 82, 83, 84, 177, 128, 129, 130, 131, 96, 97, 98, 99, 100,
        178, 144, 145, 146, 147, 148, 179,
    ], // Mode 14
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 164, 132, 16, 17, 18, 19, 20, 21, 22, 23, 178, 116, 32, 33, 34, 35,
        36, 37, 38, 39, 179, 180, 48, 49, 50, 51, 52, 53, 112, 113, 114, 115, 64, 65, 66, 67, 68,
        176, 160, 161, 162, 163, 80, 81, 82, 83, 84, 177, 128, 129, 130, 131, 96, 97, 98, 99, 100,
        101, 144, 145, 146, 147, 148, 149,
    ], // Mode 18
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 176, 132, 16, 17, 18, 19, 20, 21, 22, 23, 117, 116, 32, 33, 34, 35,
        36, 37, 38, 39, 165, 180, 48, 49, 50, 51, 52, 164, 112, 113, 114, 115, 64, 65, 66, 67, 68,
        69, 160, 161, 162, 163, 80, 81, 82, 83, 84, 177, 128, 129, 130, 131, 96, 97, 98, 99, 100,
        178, 144, 145, 146, 147, 148, 179,
    ], // Mode 22
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 177, 132, 16, 17, 18, 19, 20, 21, 22, 23, 133, 116, 32, 33, 34, 35,
        36, 37, 38, 39, 181, 180, 48, 49, 50, 51, 52, 164, 112, 113, 114, 115, 64, 65, 66, 67, 68,
        176, 160, 161, 162, 163, 80, 81, 82, 83, 84, 85, 128, 129, 130, 131, 96, 97, 98, 99, 100,
        178, 144, 145, 146, 147, 148, 179,
    ], // Mode 26
    &[
        0, 1, 2, 3, 4, 5, 164, 176, 177, 132, 16, 17, 18, 19, 20, 21, 117, 133, 178, 116, 32, 33,
        34, 35, 36, 37, 165, 179, 181, 180, 48, 49, 50, 51, 52, 53, 112, 113, 114, 115, 64, 65, 66,
        67, 68, 69, 160, 161, 162, 163, 80, 81, 82, 83, 84, 85, 128, 129, 130, 131, 96, 97, 98, 99,
        100, 101, 144, 145, 146, 147, 148, 149,
    ], // Mode 30
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    ], // Mode 3
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56, 10, 64, 65, 66, 67, 68, 69, 70, 71,
        72, 26, 80, 81, 82, 83, 84, 85, 86, 87, 88, 42,
    ], // Mode 7
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 11, 10, 64, 65, 66, 67, 68, 69, 70, 71,
        27, 26, 80, 81, 82, 83, 84, 85, 86, 87, 43, 42,
    ], // Mode 11
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 48, 49, 50, 51, 15, 14, 13, 12, 11, 10, 64, 65, 66, 67, 31, 30, 29, 28,
        27, 26, 80, 81, 82, 83, 47, 46, 45, 44, 43, 42,
    ], // Mode 15
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::FixedCode;

    #[test]
    fn source_half_rounding_is_ties_even() {
        for lower in [0, 1, 1023, 1024, 15360, 15361, 31742] {
            let value = (half_to_float(lower) + half_to_float(lower + 1)) * 0.5;
            assert_eq!(nearest_half(value), (lower + (lower & 1)) as i32);
        }
        assert_eq!(nearest_half(65504.0), 31743);
    }

    #[test]
    fn candidate_scores_match_packed_decoder_and_delta_wraps() {
        let colors = std::array::from_fn(|y| {
            std::array::from_fn(|x| {
                let i = y * 4 + x;
                (
                    Rgb32F::new(2.0f32.powi(i as i32 - 8), i as f32 * 30.0, -(i as f32)),
                    i != 0 && i != 2 && i != 8 && i != 15,
                )
            })
        });
        let mut random = 0x6bc6u32;
        for signed in [false, true] {
            let samples = Samples::new(colors, signed);
            for mode in 0..14 {
                for partition in 0..if mode < 10 { 32 } else { 1 } {
                    let mut fields = [[0; 3]; 4];
                    for e in 0..if mode < 10 { 4 } else { 2 } {
                        for c in 0..3 {
                            random ^= random << 13;
                            random ^= random >> 17;
                            random ^= random << 5;
                            let (low, high) = field_range(mode, e, c, signed);
                            fields[e][c] = low + (random % (high - low + 1) as u32) as i32;
                        }
                    }
                    let candidate = Candidate {
                        mode,
                        partition,
                        fields,
                        indices: [0; 16],
                        error: 0,
                    }
                    .evaluate(&samples);
                    assert_eq!(
                        candidate.error,
                        samples.block_error(candidate.pack()),
                        "mode {mode}, partition {partition}, signed {signed}"
                    );
                    for i in [0, 2, 8, 15] {
                        assert_eq!(candidate.indices[i], 0);
                    }
                }
            }
        }
        assert_eq!(nearest_delta(0.0, 65535, 16, 4, false), 1);
        assert_eq!(nearest_delta(-32768.0, 32767, 16, 4, true), 1);
    }

    #[test]
    fn exact_inversion_handles_missing_first_subset_and_anchors() {
        let fixtures = include_bytes!("../../../tests/data/bc6-reference.bin");
        for mode in [0, 1, 2, 6, 9] {
            let partition = 17;
            let fixture = &fixtures[(mode * 34 + partition) * 208..];
            for signed in [false, true] {
                let offset = 16 + usize::from(signed) * 96;
                let colors = std::array::from_fn(|y| {
                    std::array::from_fn(|x| {
                        let i = y * 4 + x;
                        let color = Rgb32F(std::array::from_fn(|c| {
                            let p = offset + (i * 3 + c) * 2;
                            half_to_float(u16::from_le_bytes([fixture[p], fixture[p + 1]]))
                        }));
                        (
                            color,
                            subset(mode, partition, i) == 1 && i != ANCHORS[partition],
                        )
                    })
                });
                let samples = Samples::new(colors, signed);
                let initial = Candidate {
                    mode,
                    partition,
                    fields: [[0; 3]; 4],
                    indices: [0; 16],
                    error: u64::MAX,
                };
                let candidate =
                    exact_mode_palette(&samples, initial).expect("representable masked subset");
                assert_eq!(
                    samples.block_error(candidate.pack()),
                    0,
                    "mode {mode}, signed {signed}"
                );
                assert_eq!(candidate.indices[0], 0);
                assert_eq!(candidate.indices[ANCHORS[partition]], 0);
            }
        }
    }

    #[test]
    fn representable_palettes_survive_coarse_ranking_and_ties() {
        let fixtures = include_bytes!("../../../tests/data/bc6-reference.bin");
        for (record, signed) in [(219, false), (226, false), (227, true), (475, true)] {
            let fixture = &fixtures[record * 208..];
            let offset = 16 + usize::from(signed) * 96;
            let colors = std::array::from_fn(|y| {
                std::array::from_fn(|x| {
                    (
                        Rgb32F(std::array::from_fn(|c| {
                            let p = offset + ((y * 4 + x) * 3 + c) * 2;
                            half_to_float(u16::from_le_bytes([fixture[p], fixture[p + 1]]))
                        })),
                        true,
                    )
                })
            });
            let samples = Samples::new(colors, signed);
            let block = Block::encode_impl(colors, signed);
            assert_eq!(
                samples.block_error(block),
                0,
                "record {record}, signed {signed}"
            );
        }
    }

    #[test]
    #[ignore = "full masked palette inversion audit"]
    fn masked_inversion_all_partitions() {
        let fixtures = include_bytes!("../../../tests/data/bc6-reference.bin");
        for mode in 0..10 {
            for partition in 0..32 {
                let fixture = &fixtures[(mode * 34 + partition) * 208..];
                for signed in [false, true] {
                    let offset = 16 + usize::from(signed) * 96;
                    let colors = std::array::from_fn(|y| {
                        std::array::from_fn(|x| {
                            let i = y * 4 + x;
                            let color = Rgb32F(std::array::from_fn(|c| {
                                let p = offset + (i * 3 + c) * 2;
                                half_to_float(u16::from_le_bytes([fixture[p], fixture[p + 1]]))
                            }));
                            (
                                color,
                                subset(mode, partition, i) == 1 && i != ANCHORS[partition],
                            )
                        })
                    });
                    let samples = Samples::new(colors, signed);
                    let initial = Candidate {
                        mode,
                        partition,
                        fields: [[0; 3]; 4],
                        indices: [0; 16],
                        error: u64::MAX,
                    };
                    let candidate =
                        exact_mode_palette(&samples, initial).expect("representable masked subset");
                    assert_eq!(
                        samples.block_error(candidate.pack()),
                        0,
                        "mode {mode}, partition {partition}, signed {signed}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "release timing audit"]
    fn sparse_and_constant_timing() {
        for signed in [false, true] {
            for (name, colors) in [
                ("zero", [[(Rgb32F::BLACK, true); 4]; 4]),
                (
                    "bright",
                    [[(Rgb32F::new(64000.0, 65000.0, 65504.0), true); 4]; 4],
                ),
                (
                    "single",
                    std::array::from_fn(|y| {
                        std::array::from_fn(|x| {
                            (Rgb32F::new(-0.00012, 35.0, 634.0), y == 3 && x == 2)
                        })
                    }),
                ),
                (
                    "masked",
                    std::array::from_fn(|y| {
                        std::array::from_fn(|x| {
                            (
                                Rgb32F::new(
                                    2.0f32.powi((y * 4 + x) as i32 - 5),
                                    2.0f32.powi(x as i32),
                                    -(y as f32),
                                ),
                                (x + y) % 3 == 0,
                            )
                        })
                    }),
                ),
            ] {
                let started = std::time::Instant::now();
                let block = Block::encode_impl(colors, signed);
                let elapsed = started.elapsed();
                let samples = Samples::new(colors, signed);
                println!(
                    "{name} signed {signed} {elapsed:?} error {}",
                    samples.block_error(block)
                );
                assert!(elapsed < std::time::Duration::from_secs(10));
            }
        }
    }

    #[test]
    #[ignore = "search-budget audit"]
    fn search_budget_probe() {
        let record = std::env::var("BC6_CASE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(122);
        let fixtures = include_bytes!("../../../tests/data/bc6-quality.bin");
        let fixture = &fixtures[record * 228..];
        let signed = fixture[0] != 0;
        let colors = std::array::from_fn(|y| {
            std::array::from_fn(|x| {
                (
                    Rgb32F(std::array::from_fn(|c| {
                        let i = 4 + ((y * 4 + x) * 3 + c) * 2;
                        half_to_float(u16::from_le_bytes([fixture[i], fixture[i + 1]]))
                    })),
                    true,
                )
            })
        });
        let samples = Samples::new(colors, signed);
        for (name, budget) in [
            ("normal", SearchBudget::NORMAL),
            (
                "all sample pairs",
                SearchBudget {
                    sample_starts: 256,
                    ..SearchBudget::NORMAL
                },
            ),
            (
                "double finalists",
                SearchBudget {
                    finalists: 64,
                    ..SearchBudget::NORMAL
                },
            ),
            (
                "double iterations",
                SearchBudget {
                    iterations: 48,
                    ..SearchBudget::NORMAL
                },
            ),
            ("oracle", SearchBudget::ORACLE),
        ] {
            let started = std::time::Instant::now();
            let block = encode_search(colors, signed, budget);
            println!(
                "case {record} {name} error {} {:?}",
                samples.block_error(block),
                started.elapsed()
            );
        }
    }

    #[test]
    #[ignore = "quality and release timing audit"]
    fn quality_search_report() {
        let fixtures = include_bytes!("../../../tests/data/bc6-quality.bin");
        let selected = std::env::var("BC6_CASE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let mut totals = [[0u64; 4]; 9];
        let mut zero_count = [0; 2];
        let mut time = [std::time::Duration::ZERO; 2];
        for (record, fixture) in fixtures.chunks_exact(228).enumerate() {
            if selected.is_some_and(|i| i != record) {
                continue;
            }
            let signed = fixture[0] != 0;
            let category = fixture[1] as usize;
            let colors = std::array::from_fn(|y| {
                std::array::from_fn(|x| {
                    let color = Rgb32F(std::array::from_fn(|c| {
                        let i = 4 + ((y * 4 + x) * 3 + c) * 2;
                        half_to_float(u16::from_le_bytes([fixture[i], fixture[i + 1]]))
                    }));
                    (color, true)
                })
            });
            let samples = Samples::new(colors, signed);
            let reference =
                samples.block_error(Block::from_bytes(fixture[100..116].try_into().unwrap()));
            let baseline = samples.block_error(Block::encode_baseline(colors, signed));
            let mut errors = [0; 2];
            for (which, budget) in [SearchBudget::NORMAL, SearchBudget::ORACLE]
                .into_iter()
                .enumerate()
            {
                let started = std::time::Instant::now();
                let block = encode_search(colors, signed, budget);
                time[which] += started.elapsed();
                errors[which] = samples.block_error(block);
                assert!(errors[which] <= baseline);
                if category == 8 && errors[which] == 0 {
                    zero_count[which] += 1;
                }
                if category == 8 {
                    assert_eq!(errors[which], 0, "known global-zero case {record}");
                }
            }
            for (c, value) in [errors[0], reference, errors[1], baseline]
                .into_iter()
                .enumerate()
            {
                totals[category][c] += value;
            }
            assert!(errors[1] <= errors[0]);
            println!(
                "case {record} category {category} signed {signed} normal {} reference {reference} oracle {} baseline {baseline}",
                errors[0], errors[1]
            );
        }
        println!(
            "totals [normal, reference, oracle, baseline] {totals:?}; exact recovered {zero_count:?}; time {time:?}"
        );
    }

    #[test]
    fn decoder_matches_independent_reference() {
        let fixtures = include_bytes!("../../../tests/data/bc6-reference.bin");
        assert_eq!(fixtures.len(), 14 * 34 * 208);
        for (record, fixture) in fixtures.chunks_exact(208).enumerate() {
            let block = Block::from_bytes(fixture[..16].try_into().unwrap());
            for signed in [false, true] {
                let decoded = if signed {
                    block.decode_signed()
                } else {
                    block.decode()
                };
                let expected = &fixture[16 + usize::from(signed) * 96..];
                for i in 0..48 {
                    let half = u16::from_le_bytes([expected[i * 2], expected[i * 2 + 1]]);
                    assert_eq!(
                        decoded[i / 12][i / 3 % 4].0[i % 3].to_bits(),
                        half_to_float(half).to_bits(),
                        "record {record}, signed {signed}, channel {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn raw_and_fixedcode_roundtrip() {
        let bytes = std::array::from_fn(|i| (i * 17) as u8);
        let block = Block::from_bytes(bytes);
        assert_eq!(std::mem::size_of::<Block>(), 16);
        assert_eq!(Block::SIZE, 16);
        assert_eq!(block.bytes(), bytes);
        assert_eq!(block.fix_encode(), bytes);
        assert_eq!(Block::fix_decode(&bytes).unwrap(), block);
    }

    #[test]
    fn black_and_reserved_modes() {
        for signed in [false, true] {
            assert_eq!(Block::BLACK.decode_impl(signed), [[Rgb32F::BLACK; 4]; 4]);
            for mode in [19, 23, 27, 31] {
                let mut bytes = [255; 16];
                bytes[0] = 224 | mode;
                assert_eq!(
                    Block::from_bytes(bytes).decode_impl(signed),
                    [[Rgb32F::BLACK; 4]; 4]
                );
            }
        }
    }

    #[test]
    fn signed_underflow_and_negative_infinity() {
        assert_eq!(finish(-1, true).to_bits(), 0.0f32.to_bits());
        assert_eq!(finish(-32768, true), f32::NEG_INFINITY);
        assert_eq!(half_to_float(1), 2.0f32.powi(-24));
    }

    #[test]
    fn encodes_hdr_constants_and_signed_colors() {
        for signed in [false, true] {
            for color in [
                Rgb32F::BLACK,
                Rgb32F::WHITE,
                Rgb32F::new(0.01, 32.0, 2000.0),
                Rgb32F::new(if signed { -32.0 } else { 32.0 }, 0.5, 65504.0),
            ] {
                let block = Block::encode_impl([[(color, true); 4]; 4], signed);
                let mode = if block.data[0] & 3 < 2 {
                    block.data[0] & 3
                } else {
                    block.data[0] & 31
                };
                assert!(MODES.contains(&(mode as usize)));
                for row in block.decode_impl(signed) {
                    for actual in row {
                        for c in 0..3 {
                            assert!(
                                (actual.0[c] - color.0[c]).abs()
                                    <= color.0[c].abs() * 0.04 + 0.00001,
                                "signed {signed}: {actual:?} != {color:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fits_hdr_gradient_and_anchor() {
        for signed in [false, true] {
            let colors = std::array::from_fn(|y| {
                std::array::from_fn(|x| {
                    let value = 2.0f32.powf((15 - (y * 4 + x)) as f32 / 3.0);
                    (
                        Rgb32F::new(
                            if signed { -value } else { value },
                            value * 4.0,
                            value * 16.0,
                        ),
                        true,
                    )
                })
            });
            let decoded = Block::encode_impl(colors, signed).decode_impl(signed);
            for i in 0..16 {
                for c in 0..3 {
                    let expected = colors[i / 4][i % 4].0.0[c];
                    assert!((decoded[i / 4][i % 4].0[c] / expected - 1.0).abs() < 0.2);
                }
            }
        }
    }

    #[test]
    fn mask_ignores_texels_and_empty_blocks() {
        let empty = [[(
            Rgb32F::new(f32::NAN, f32::INFINITY, f32::NEG_INFINITY),
            false,
        ); 4]; 4];
        assert_eq!(Block::encode(empty), Block::BLACK);
        assert_eq!(Block::encode_signed(empty), Block::BLACK);
        let mut colors = empty;
        colors[2][1] = (Rgb32F::new(1.0, 32.0, 1000.0), true);
        let a = Block::encode(colors);
        for row in &mut colors {
            for (color, valid) in row {
                if !*valid {
                    *color = Rgb32F::new(65504.0, -65504.0, -0.1);
                }
            }
        }
        assert_eq!(a, Block::encode(colors));
        let raw = u128::from_le_bytes(a.data);
        let raw_mode = if raw & 3 < 2 { raw & 3 } else { raw & 31 } as usize;
        let mode = MODES.iter().position(|&m| m == raw_mode).unwrap();
        let partition = ((raw >> 77) & 31) as usize;
        let mut offset = if mode < 10 { 82 } else { 65 };
        for i in 0..16 {
            let anchor = i == 0 || (mode < 10 && i == ANCHORS[partition]);
            let bits = if mode < 10 { 3 } else { 4 } - usize::from(anchor);
            if i != 9 {
                assert_eq!((raw >> offset) & ((1 << bits) - 1), 0);
            }
            offset += bits;
        }
    }

    #[test]
    fn sanitizes_nonfinite_and_unsigned_negative_inputs() {
        for signed in [false, true] {
            let nonfinite = Rgb32F::new(f32::NAN, f32::INFINITY, f32::NEG_INFINITY);
            let finite = Rgb32F::new(0.0, 65504.0, if signed { -65504.0 } else { 0.0 });
            assert_eq!(
                Block::encode_impl([[(nonfinite, true); 4]; 4], signed),
                Block::encode_impl([[(finite, true); 4]; 4], signed)
            );
        }
        assert_eq!(
            Block::encode([[(Rgb32F::new(-1.0, -32.0, -65504.0), true); 4]; 4]),
            Block::BLACK
        );
    }
}
