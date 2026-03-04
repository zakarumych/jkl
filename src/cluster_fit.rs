//! Least-squares cluster-fit quantization.
//!
//! Exhaustively searches all clusterings of samples along their principal axis
//! to find optimal palette endpoints with minimum total error. Used internally
//! by the BC1–BC5 block texture encoders.

use crate::math::{Region2, Region3, Vec2, Vec3, Vector};

/// Result of a cluster-fit quantization pass.
///
/// Contains the two palette endpoints, a per-sample index into the palette,
/// and the total approximation error.
pub struct ClusterFit<T, const N: usize> {
    /// The two optimized palette endpoints.
    pub endpoints: (T, T),
    /// Per-sample index into the palette built from `endpoints`.
    pub indices: [usize; N],
    /// Total accumulated error across all samples.
    pub error: f32,
}

/// A value that can participate in cluster-fit quantization.
///
/// Implementors must support basic arithmetic, projection onto a
/// principal axis, and a fallback heuristic for initial endpoint selection.
pub trait Sample: Vector {
    /// Returns a pair of endpoints derived from the bounding extents of the samples.
    fn fallback_endpoints(samples: impl ExactSizeIterator<Item = Self> + Clone) -> (Self, Self);
}

impl Sample for f32 {
    #[inline]
    fn fallback_endpoints(samples: impl ExactSizeIterator<Item = Self> + Clone) -> (Self, Self) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for s in samples {
            if s < min {
                min = s;
            }
            if s > max {
                max = s;
            }
        }

        (min, max)
    }
}

impl Sample for Vec2 {
    #[inline(always)]
    fn fallback_endpoints(samples: impl ExactSizeIterator<Item = Self> + Clone) -> (Self, Self) {
        let region = Region2::new(samples);
        (region.min, region.max)
    }
}

impl Sample for Vec3 {
    #[inline(always)]
    fn fallback_endpoints(samples: impl ExactSizeIterator<Item = Self> + Clone) -> (Self, Self) {
        let region = Region3::new(samples);
        (region.min, region.max)
    }
}

/// Quantizes `samples` into `I` palette entries by exhaustive cluster-fit search.
///
/// Iterates all possible clusterings of `N` samples into `I` groups along the
/// principal axis, solves for optimal endpoints via least-squares, then keeps
/// the partition with the lowest total error.
///
/// * `samples` – the values to quantize (at most `N`).
/// * `remap_endpoints` – snaps endpoints to the target representation (e.g.
///   rounding to `Rgb565`) and returns the remapped pair.
/// * `error` – distance function between two samples.
pub fn cluster_fit<T, const I: usize, const N: usize>(
    samples: &[T],
    remap_endpoints: impl Fn(T, T) -> (T, T),
    error: impl Fn(T, T) -> f32 + Copy,
) -> ClusterFit<T, N>
where
    T: Sample,
{
    #![allow(clippy::needless_range_loop)]
    assert!(samples.len() <= N);

    let axis = T::principal_axis(samples);

    let mut order = [(0, 0.0f32); N];

    for i in 0..samples.len() {
        let projection = samples[i].project(axis);
        order[i] = (i, projection);
    }

    order[..samples.len()].sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let order = order;

    let mut best_endpoints = T::fallback_endpoints(samples.iter().copied());
    best_endpoints = remap_endpoints(best_endpoints.0, best_endpoints.1);
    let mut best_indices = [0; N];
    let mut best_error = 0.0f32;

    {
        let palette = build_palette::<T, I>(best_endpoints.0, best_endpoints.1);

        for i in 0..samples.len() {
            let (idx, e) = index_error(samples[order[i].0], &palette, error);
            best_indices[order[i].0] = idx;
            best_error += e;
        }
    }

    let mut cuts = [0; I]; // 0th index is unused.
    for i in 1..I {
        cuts[i] = i - 1;
    }

    'a: loop {
        // Loop body

        let mut weights = [0.0f32; N];

        for i in 0..samples.len() {
            let idx: usize = cuts[1..].iter().map(|&c| if i > c { 1 } else { 0 }).sum();
            let t = (idx as f32) / ((I - 1) as f32);
            weights[order[i].0] = t;
        }

        if let Some((c0, c1)) = solve_endpoints(weights, samples) {
            let (c0, c1) = remap_endpoints(c0, c1);

            let palette = build_palette::<T, I>(c0, c1);

            let mut total_error = 0.0f32;
            let mut indices = [0; N];

            for i in 0..samples.len() {
                let (idx, e) = index_error(samples[order[i].0], &palette, error);
                indices[order[i].0] = idx;
                total_error += e;
            }

            if best_error > total_error {
                best_error = total_error;
                best_endpoints = (c0, c1);
                best_indices = indices;
            }
        }

        // Loop increment
        for i in (1..I).rev() {
            let max = samples.len() - (I - i);
            if cuts[i] < max {
                cuts[i] += 1;

                for j in i + 1..I {
                    cuts[j] = cuts[j - 1] + 1;
                }

                continue 'a;
            }
        }

        // All combinations have been tried
        break;
    }

    ClusterFit {
        endpoints: best_endpoints,
        indices: best_indices,
        error: best_error,
    }
}

fn solve_endpoints<T, const N: usize>(weights: [f32; N], samples: &[T]) -> Option<(T, T)>
where
    T: Sample,
{
    #![allow(non_snake_case)]

    assert!(samples.len() <= N);

    let mut A = 0.0f32;
    let mut B = 0.0f32;
    let mut C = 0.0f32;

    let mut X = T::zero();
    let mut Y = T::zero();

    for i in 0..samples.len() {
        let w = weights[i];
        let u = 1.0 - w;
        let s = samples[i];

        A += u * u;
        B += u * w;
        C += w * w;

        X += s * u;
        Y += s * w;
    }

    let D = A * C - B * B;

    if D.abs() < 1e-8 {
        return None;
    }

    let invD = D.recip();

    let C0 = (X * C - Y * B) * invD;
    let C1 = (Y * A - X * B) * invD;

    Some((C0, C1))
}

fn build_palette<T, const I: usize>(c0: T, c1: T) -> [T; I]
where
    T: Sample,
{
    #![allow(clippy::needless_range_loop)]
    let mut palette = [T::zero(); I];

    palette[0] = c0;

    for i in 1..I - 1 {
        let t = (i as f32) / ((I - 1) as f32);
        palette[i] = c0 * (1.0 - t) + c1 * t;
    }

    palette[I - 1] = c1;

    palette
}

fn index_error<T, const I: usize>(
    sample: T,
    palette: &[T; I],
    error: impl Fn(T, T) -> f32,
) -> (usize, f32)
where
    T: Sample,
{
    let mut best_index = 0;
    let mut best_error = f32::MAX;

    for (i, &p) in palette.iter().enumerate() {
        let e = error(sample, p);
        if e < best_error {
            best_index = i;
            best_error = e;
        }
    }

    (best_index, best_error)
}
