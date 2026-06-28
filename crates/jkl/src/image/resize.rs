//! Image resizing via padding or resampling.
//!
//! See [`resize_1d`] and [`resize_2d`] for the main entry points.

use smallvec::SmallVec;

use crate::{
    image::{Image1DMut, Image1DRef, Image2DMut, Image2DRef},
    math::{R8U, R32F, Rg8U, Rg32F, Rgb8U, Rgb32F, Rgba8U, Rgba32F},
};

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Pixel operations required by the resize algorithms.
///
/// All floating-point arithmetic must stay inside the implementations of this
/// trait; callers treat `Self` as an opaque value and never decompose it into
/// individual channels.
pub trait ResizablePixel: Copy {
    /// Fully transparent pixel. For formats without an alpha channel this is
    /// the same as [`black`](Self::black).
    fn transparent() -> Self;

    /// Fully opaque black pixel.
    fn black() -> Self;

    /// Linearly interpolates between `a` and `b` by factor `t ∈ [0, 1]`.
    fn lerp(a: Self, b: Self, t: f32) -> Self;

    /// Computes a weighted sum of `(pixel, weight)` pairs.
    ///
    /// Weights may be negative (Catmull-Rom / Lanczos). Integer-channel types
    /// clamp the accumulated result to the representable range; float-channel
    /// types do not clamp.
    fn weighted_sum(samples: &[(Self, f32)]) -> Self;
}

// ── Trait implementations ─────────────────────────────────────────────────────

impl ResizablePixel for R8U {
    #[inline]
    fn transparent() -> Self {
        R8U::BLACK
    }

    #[inline]
    fn black() -> Self {
        R8U::BLACK
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        R8U::from_f32(R32F::lerp(a.into_f32(), b.into_f32(), t))
    }

    fn weighted_sum(samples: &[(Self, f32)]) -> Self {
        let mut acc = 0.0_f32;
        for &(px, w) in samples {
            acc += px.into_f32().r() * w;
        }
        R8U::from_f32(R32F(acc.clamp(0.0, 1.0)))
    }
}

impl ResizablePixel for R32F {
    #[inline]
    fn transparent() -> Self {
        R32F::BLACK
    }

    #[inline]
    fn black() -> Self {
        R32F::BLACK
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        R32F::lerp(a, b, t)
    }

    fn weighted_sum(samples: &[(Self, f32)]) -> Self {
        let mut acc = 0.0_f32;
        for &(px, w) in samples {
            acc += px.r() * w;
        }
        R32F(acc)
    }
}

impl ResizablePixel for Rg8U {
    #[inline]
    fn transparent() -> Self {
        Rg8U::BLACK
    }

    #[inline]
    fn black() -> Self {
        Rg8U::BLACK
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        Rg8U::from_f32(Rg32F::lerp(a.into_f32(), b.into_f32(), t))
    }

    fn weighted_sum(samples: &[(Self, f32)]) -> Self {
        let mut acc = [0.0_f32; 2];
        for &(px, w) in samples {
            let f = px.into_f32();
            acc[0] += f.r() * w;
            acc[1] += f.g() * w;
        }
        Rg8U::from_f32(Rg32F([acc[0].clamp(0.0, 1.0), acc[1].clamp(0.0, 1.0)]))
    }
}

impl ResizablePixel for Rg32F {
    #[inline]
    fn transparent() -> Self {
        Rg32F::BLACK
    }

    #[inline]
    fn black() -> Self {
        Rg32F::BLACK
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        Rg32F::lerp(a, b, t)
    }

    fn weighted_sum(samples: &[(Self, f32)]) -> Self {
        let mut acc = [0.0_f32; 2];
        for &(px, w) in samples {
            acc[0] += px.r() * w;
            acc[1] += px.g() * w;
        }
        Rg32F(acc)
    }
}

impl ResizablePixel for Rgb8U {
    #[inline]
    fn transparent() -> Self {
        Rgb8U::BLACK
    }

    #[inline]
    fn black() -> Self {
        Rgb8U::BLACK
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        Rgb8U::from_f32(Rgb32F::lerp(a.into_f32(), b.into_f32(), t))
    }

    fn weighted_sum(samples: &[(Self, f32)]) -> Self {
        let mut acc = [0.0_f32; 3];
        for &(px, w) in samples {
            let f = px.into_f32();
            acc[0] += f.r() * w;
            acc[1] += f.g() * w;
            acc[2] += f.b() * w;
        }
        Rgb8U::from_f32(Rgb32F([
            acc[0].clamp(0.0, 1.0),
            acc[1].clamp(0.0, 1.0),
            acc[2].clamp(0.0, 1.0),
        ]))
    }
}

impl ResizablePixel for Rgb32F {
    #[inline]
    fn transparent() -> Self {
        Rgb32F::BLACK
    }

    #[inline]
    fn black() -> Self {
        Rgb32F::BLACK
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        Rgb32F::lerp(a, b, t)
    }

    fn weighted_sum(samples: &[(Self, f32)]) -> Self {
        let mut acc = [0.0_f32; 3];
        for &(px, w) in samples {
            acc[0] += px.r() * w;
            acc[1] += px.g() * w;
            acc[2] += px.b() * w;
        }
        Rgb32F(acc)
    }
}

impl ResizablePixel for Rgba8U {
    #[inline]
    fn transparent() -> Self {
        Rgba8U::TRANSPARENT
    }

    #[inline]
    fn black() -> Self {
        Rgba8U::BLACK
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        Rgba8U::from_f32(Rgba32F::lerp(a.into_f32(), b.into_f32(), t))
    }

    fn weighted_sum(samples: &[(Self, f32)]) -> Self {
        let mut acc = [0.0_f32; 4];
        for &(px, w) in samples {
            let f = px.into_f32();
            acc[0] += f.r() * w;
            acc[1] += f.g() * w;
            acc[2] += f.b() * w;
            acc[3] += f.a() * w;
        }
        Rgba8U::from_f32(Rgba32F([
            acc[0].clamp(0.0, 1.0),
            acc[1].clamp(0.0, 1.0),
            acc[2].clamp(0.0, 1.0),
            acc[3].clamp(0.0, 1.0),
        ]))
    }
}

impl ResizablePixel for Rgba32F {
    #[inline]
    fn transparent() -> Self {
        Rgba32F::TRANSPARENT
    }

    #[inline]
    fn black() -> Self {
        Rgba32F::BLACK
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        Rgba32F::lerp(a, b, t)
    }

    fn weighted_sum(samples: &[(Self, f32)]) -> Self {
        let mut acc = [0.0_f32; 4];
        for &(px, w) in samples {
            acc[0] += px.r() * w;
            acc[1] += px.g() * w;
            acc[2] += px.b() * w;
            acc[3] += px.a() * w;
        }
        Rgba32F(acc)
    }
}

// ── Options types ─────────────────────────────────────────────────────────────

/// Per-side padding amounts, in pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct Padding {
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
    pub right: usize,
}

impl Padding {
    /// Equal padding on all four sides.
    pub fn uniform(n: usize) -> Self {
        Padding {
            top: n,
            bottom: n,
            left: n,
            right: n,
        }
    }

    /// Padding on the left and right sides only.
    pub fn horizontal(left: usize, right: usize) -> Self {
        Padding {
            left,
            right,
            ..Default::default()
        }
    }

    /// Padding on the top and bottom sides only.
    pub fn vertical(top: usize, bottom: usize) -> Self {
        Padding {
            top,
            bottom,
            ..Default::default()
        }
    }
}

/// Fill mode for padded regions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FillMode {
    /// Nearest edge pixel of the source image.
    #[default]
    EdgeCopy,
    /// Fully transparent pixel (zero alpha, or black for opaque formats).
    Transparent,
    /// Opaque black pixel.
    Black,
}

/// Resampling filter to use when scaling.
#[derive(Clone, Copy, Debug, Default)]
pub enum ResampleMethod {
    /// Nearest-neighbor: no filtering, fastest.
    NearestNeighbor,
    /// Box filter: area-weighted average of contributing source pixels.
    Box,
    /// Catmull-Rom bicubic/tricubic filter (separable).
    #[default]
    Bicubic,
    /// Lanczos windowed-sinc filter.
    ///
    /// `lobes` controls the number of lobes (typically 2 or 3).
    Lanczos { lobes: u32 },
}

/// A resize operation applied to an image.
#[derive(Clone, Copy, Debug)]
pub enum ResizeOp {
    /// Add empty border pixels around the image.
    ///
    /// For [`resize_1d`] only `padding.left` and `padding.right` are used;
    /// `top` and `bottom` are ignored.
    Pad { padding: Padding, fill: FillMode },
    /// Scale the image to new dimensions using the given filter.
    ///
    /// For [`resize_1d`] `height` is ignored.
    Resample {
        width: usize,
        height: usize,
        method: ResampleMethod,
    },
}

// ── Kernel helpers ────────────────────────────────────────────────────────────

/// Catmull-Rom weights for fractional offset `t ∈ [0, 1)`.
///
/// The four weights correspond to taps at floor−1, floor, floor+1, floor+2.
#[inline]
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Normalized sinc: sin(πx) / (πx), with sinc(0) = 1.
#[inline]
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        let px = std::f32::consts::PI * x;
        px.sin() / px
    }
}

/// Lanczos kernel: sinc(x) · sinc(x/a) for |x| < a, else 0.
#[inline]
fn lanczos_kernel(x: f32, a: f32) -> f32 {
    if x.abs() < a {
        sinc(x) * sinc(x / a)
    } else {
        0.0
    }
}

// ── 1-D resampling internals ──────────────────────────────────────────────────

fn resample_1d_nearest<T: ResizablePixel>(src: Image1DRef<'_, T>, mut dst: Image1DMut<'_, T>) {
    let src_w = src.width();
    let dst_w = dst.width();
    if src_w == 0 || dst_w == 0 {
        return;
    }
    for ox in 0..dst_w {
        let sx = ((ox as f32 + 0.5) * src_w as f32 / dst_w as f32 - 0.5)
            .round()
            .clamp(0.0, (src_w - 1) as f32) as usize;
        dst.set_pixel(ox, *src.get_pixel(sx));
    }
}

fn resample_1d_box<T: ResizablePixel>(src: Image1DRef<'_, T>, mut dst: Image1DMut<'_, T>) {
    let src_w = src.width();
    let dst_w = dst.width();
    if src_w == 0 || dst_w == 0 {
        return;
    }
    let scale = src_w as f32 / dst_w as f32;
    for ox in 0..dst_w {
        let x0 = ox as f32 * scale;
        let x1 = (ox + 1) as f32 * scale;
        let ix0 = (x0.floor() as usize).min(src_w - 1);
        let ix1 = (x1.ceil() as usize).min(src_w);
        let mut samples: SmallVec<[(T, f32); 8]> =
            SmallVec::with_capacity(ix1.saturating_sub(ix0) + 1);
        for sx in ix0..ix1 {
            let w = (x1.min(sx as f32 + 1.0) - x0.max(sx as f32)).max(0.0);
            if w > 0.0 {
                samples.push((*src.get_pixel(sx), w));
            }
        }
        let total: f32 = samples.iter().map(|(_, w)| w).sum();
        if total > 0.0 {
            for (_, w) in &mut samples {
                *w /= total;
            }
            dst.set_pixel(ox, T::weighted_sum(&samples));
        }
    }
}

fn resample_1d_bicubic<T: ResizablePixel>(src: Image1DRef<'_, T>, mut dst: Image1DMut<'_, T>) {
    let src_w = src.width();
    let dst_w = dst.width();
    if src_w == 0 || dst_w == 0 {
        return;
    }
    let imax = src_w as i64 - 1;
    for ox in 0..dst_w {
        let sx_f = (ox as f32 + 0.5) * src_w as f32 / dst_w as f32 - 0.5;
        let sx_floor = sx_f.floor() as i64;
        let t = sx_f - sx_floor as f32;
        let ws = catmull_rom_weights(t);
        let samples = [
            (
                *src.get_pixel((sx_floor - 1).clamp(0, imax) as usize),
                ws[0],
            ),
            (*src.get_pixel(sx_floor.clamp(0, imax) as usize), ws[1]),
            (
                *src.get_pixel((sx_floor + 1).clamp(0, imax) as usize),
                ws[2],
            ),
            (
                *src.get_pixel((sx_floor + 2).clamp(0, imax) as usize),
                ws[3],
            ),
        ];
        dst.set_pixel(ox, T::weighted_sum(&samples));
    }
}

#[inline]
fn resample_lanczos_slice<T: ResizablePixel>(
    src: &[T],
    src_w: usize,
    src_s: usize,
    lobes: u32,
    dst: &mut [T],
    dst_w: usize,
    dst_s: usize,
) {
    if src_w == 0 || dst_w == 0 {
        return;
    }
    let a = lobes as f32;
    let scale = dst_w as f32 / src_w as f32;
    // Widen support when downscaling to prevent aliasing.
    let support = if scale < 1.0 { a / scale } else { a };
    let imax = src_w as i64 - 1;
    for ox in 0..dst_w {
        let src_center = (ox as f32 + 0.5) * src_w as f32 / dst_w as f32 - 0.5;
        let sx_start = (src_center - support).ceil() as i64;
        let sx_end = (src_center + support).floor() as i64;
        let capacity = (sx_end - sx_start + 1).max(0) as usize;
        let mut samples: SmallVec<[(T, f32); 8]> = SmallVec::with_capacity(capacity);
        for sx in sx_start..=sx_end {
            // Normalize distance to kernel-space so `a` lobes span the source.
            let x = if scale < 1.0 {
                (src_center - sx as f32) * scale
            } else {
                src_center - sx as f32
            };
            let w = lanczos_kernel(x, a);
            if w != 0.0 {
                samples.push((src[(sx.clamp(0, imax) as usize) * src_s], w));
            }
        }
        let total: f32 = samples.iter().map(|(_, w)| w).sum();
        if total.abs() > f32::EPSILON {
            for (_, w) in &mut samples {
                *w /= total;
            }
            dst[ox * dst_s] = T::weighted_sum(&samples);
        }
    }
}

fn resample_1d_lanczos<T: ResizablePixel>(
    src: Image1DRef<'_, T>,
    lobes: u32,
    mut dst: Image1DMut<'_, T>,
) {
    let src_w = src.width();
    let dst_w = dst.width();

    resample_lanczos_slice(src.pixels(), src_w, 1, lobes, dst.pixels_mut(), dst_w, 1);
}

pub fn resample_1d<T: ResizablePixel>(
    src: Image1DRef<'_, T>,
    method: ResampleMethod,
    dst: Image1DMut<'_, T>,
) {
    match method {
        ResampleMethod::NearestNeighbor => resample_1d_nearest(src, dst),
        ResampleMethod::Box => resample_1d_box(src, dst),
        ResampleMethod::Bicubic => resample_1d_bicubic(src, dst),
        ResampleMethod::Lanczos { lobes } => resample_1d_lanczos(src, lobes, dst),
    }
}

// ── 2-D resampling internal ───────────────────────────────────────────────────

fn resample_2d_nearest<T: ResizablePixel>(src: Image2DRef<'_, T>, mut dst: Image2DMut<'_, T>) {
    let src_w = src.width();
    let src_h = src.height();
    let dst_w = dst.width();
    let dst_h = dst.height();

    if dst_w == 0 || dst_h == 0 {
        return;
    }
    if src_w == 0 || src_h == 0 {
        // Fill with transparent/black pixels depending on the format.
        let fill = T::transparent();
        for y in 0..dst_h {
            for x in 0..dst_w {
                dst.set_pixel(x, y, fill);
            }
        }
        return;
    }

    for oy in 0..dst_h {
        for ox in 0..dst_w {
            let sx = ((ox as f32 + 0.5) * src_w as f32 / dst_w as f32 - 0.5)
                .round()
                .clamp(0.0, (src_w - 1) as f32) as usize;
            let sy = ((oy as f32 + 0.5) * src_h as f32 / dst_h as f32 - 0.5)
                .round()
                .clamp(0.0, (src_h - 1) as f32) as usize;
            dst.set_pixel(ox, oy, *src.get_pixel(sx, sy));
        }
    }
}

fn resample_2d_box<T: ResizablePixel>(src: Image2DRef<'_, T>, mut dst: Image2DMut<'_, T>) {
    let src_w = src.width();
    let src_h = src.height();
    let dst_w = dst.width();
    let dst_h = dst.height();

    if dst_w == 0 || dst_h == 0 {
        return;
    }
    if src_w == 0 || src_h == 0 {
        // Fill with transparent/black pixels depending on the format.
        let fill = T::transparent();
        for y in 0..dst_h {
            for x in 0..dst_w {
                dst.set_pixel(x, y, fill);
            }
        }
        return;
    }

    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    for oy in 0..dst_h {
        for ox in 0..dst_w {
            let x0 = ox as f32 * scale_x;
            let x1 = (ox + 1) as f32 * scale_x;
            let ix0 = (x0.floor() as usize).min(src_w - 1);
            let ix1 = (x1.ceil() as usize).min(src_w);

            let y0 = oy as f32 * scale_y;
            let y1 = (oy + 1) as f32 * scale_y;
            let iy0 = (y0.floor() as usize).min(src_h - 1);
            let iy1 = (y1.ceil() as usize).min(src_h);

            let mut samples: SmallVec<[(T, f32); 16]> = SmallVec::with_capacity(
                (ix1.saturating_sub(ix0) + 1) * (iy1.saturating_sub(iy0) + 1),
            );

            for sy in iy0..iy1 {
                for sx in ix0..ix1 {
                    let w = (x1.min(sx as f32 + 1.0) - x0.max(sx as f32)).max(0.0);
                    let h = (y1.min(sy as f32 + 1.0) - y0.max(sy as f32)).max(0.0);
                    if w > 0.0 && h > 0.0 {
                        samples.push((*src.get_pixel(sx, sy), w * h));
                    }
                }
            }
            let total: f32 = samples.iter().map(|(_, w)| w).sum();
            if total > 0.0 {
                for (_, w) in &mut samples {
                    *w /= total;
                }
                dst.set_pixel(ox, oy, T::weighted_sum(&samples));
            }
        }
    }
}

fn resample_2d_bicubic<T: ResizablePixel>(src: Image2DRef<'_, T>, mut dst: Image2DMut<'_, T>) {
    let src_w = src.width();
    let src_h = src.height();
    let dst_w = dst.width();
    let dst_h = dst.height();
    if dst_w == 0 || dst_h == 0 {
        return;
    }
    if src_w == 0 || src_h == 0 {
        // Fill with transparent/black pixels depending on the format.
        let fill = T::transparent();
        for y in 0..dst_h {
            for x in 0..dst_w {
                dst.set_pixel(x, y, fill);
            }
        }
        return;
    }

    let imax = src_w as i64 - 1;
    let jmax = src_h as i64 - 1;
    for oy in 0..dst_h {
        for ox in 0..dst_w {
            let sx_f = (ox as f32 + 0.5) * src_w as f32 / dst_w as f32 - 0.5;
            let sx_floor = sx_f.floor() as i64;
            let sy_f = (oy as f32 + 0.5) * src_h as f32 / dst_h as f32 - 0.5;
            let sy_floor = sy_f.floor() as i64;
            let t = sx_f - sx_floor as f32;
            let ws = catmull_rom_weights(t);
            let samples = [
                (
                    *src.get_pixel(
                        (sx_floor - 1).clamp(0, imax) as usize,
                        (sy_floor - 1).clamp(0, jmax) as usize,
                    ),
                    ws[0] * ws[0],
                ),
                (
                    *src.get_pixel(
                        sx_floor.clamp(0, imax) as usize,
                        (sy_floor - 1).clamp(0, jmax) as usize,
                    ),
                    ws[1] * ws[0],
                ),
                (
                    *src.get_pixel(
                        (sx_floor + 1).clamp(0, imax) as usize,
                        (sy_floor - 1).clamp(0, jmax) as usize,
                    ),
                    ws[2] * ws[0],
                ),
                (
                    *src.get_pixel(
                        (sx_floor + 2).clamp(0, imax) as usize,
                        (sy_floor - 1).clamp(0, jmax) as usize,
                    ),
                    ws[3] * ws[0],
                ),
                (
                    *src.get_pixel(
                        (sx_floor - 1).clamp(0, imax) as usize,
                        sy_floor.clamp(0, jmax) as usize,
                    ),
                    ws[0] * ws[1],
                ),
                (
                    *src.get_pixel(
                        sx_floor.clamp(0, imax) as usize,
                        sy_floor.clamp(0, jmax) as usize,
                    ),
                    ws[1] * ws[1],
                ),
                (
                    *src.get_pixel(
                        (sx_floor + 1).clamp(0, imax) as usize,
                        sy_floor.clamp(0, jmax) as usize,
                    ),
                    ws[2] * ws[1],
                ),
                (
                    *src.get_pixel(
                        (sx_floor + 2).clamp(0, imax) as usize,
                        sy_floor.clamp(0, jmax) as usize,
                    ),
                    ws[3] * ws[1],
                ),
                (
                    *src.get_pixel(
                        (sx_floor - 1).clamp(0, imax) as usize,
                        (sy_floor + 1).clamp(0, jmax) as usize,
                    ),
                    ws[0] * ws[2],
                ),
                (
                    *src.get_pixel(
                        sx_floor.clamp(0, imax) as usize,
                        (sy_floor + 1).clamp(0, jmax) as usize,
                    ),
                    ws[1] * ws[2],
                ),
                (
                    *src.get_pixel(
                        (sx_floor + 1).clamp(0, imax) as usize,
                        (sy_floor + 1).clamp(0, jmax) as usize,
                    ),
                    ws[2] * ws[2],
                ),
                (
                    *src.get_pixel(
                        (sx_floor + 2).clamp(0, imax) as usize,
                        (sy_floor + 1).clamp(0, jmax) as usize,
                    ),
                    ws[3] * ws[2],
                ),
                (
                    *src.get_pixel(
                        (sx_floor - 1).clamp(0, imax) as usize,
                        (sy_floor + 2).clamp(0, jmax) as usize,
                    ),
                    ws[0] * ws[3],
                ),
                (
                    *src.get_pixel(
                        sx_floor.clamp(0, imax) as usize,
                        (sy_floor + 2).clamp(0, jmax) as usize,
                    ),
                    ws[1] * ws[3],
                ),
                (
                    *src.get_pixel(
                        (sx_floor + 1).clamp(0, imax) as usize,
                        (sy_floor + 2).clamp(0, jmax) as usize,
                    ),
                    ws[2] * ws[3],
                ),
                (
                    *src.get_pixel(
                        (sx_floor + 2).clamp(0, imax) as usize,
                        (sy_floor + 2).clamp(0, jmax) as usize,
                    ),
                    ws[3] * ws[3],
                ),
            ];
            dst.set_pixel(ox, oy, T::weighted_sum(&samples));
        }
    }
}

fn resample_2d_lanczos<T: ResizablePixel>(
    src: Image2DRef<'_, T>,
    lobes: u32,
    mut dst: Image2DMut<'_, T>,
) {
    let mut tmp = vec![T::transparent(); dst.width() * src.height()];

    for y in 0..src.height() {
        resample_lanczos_slice(
            src.get_row(y).pixels(),
            src.width(),
            1,
            lobes,
            &mut tmp[dst.width() * y..],
            dst.width(),
            1,
        );
    }

    for x in 0..dst.width() {
        let dst_w = dst.width();
        let dst_h = dst.height();

        resample_lanczos_slice(
            &tmp[x..],
            src.height(),
            dst_w,
            lobes,
            &mut dst.pixels_mut()[x..],
            dst_h,
            dst_w,
        );
    }
}

pub fn resample_2d<T: ResizablePixel>(
    src: Image2DRef<'_, T>,
    method: ResampleMethod,
    dst: Image2DMut<'_, T>,
) {
    match method {
        ResampleMethod::NearestNeighbor => resample_2d_nearest(src, dst),
        ResampleMethod::Box => resample_2d_box(src, dst),
        ResampleMethod::Bicubic => resample_2d_bicubic(src, dst),
        ResampleMethod::Lanczos { lobes } => resample_2d_lanczos(src, lobes, dst),
    }
}

// ── 1-D padding internal ──────────────────────────────────────────────────────

pub fn pad_1d<T: ResizablePixel>(
    src: Image1DRef<'_, T>,
    left: usize,
    right: usize,
    fill: FillMode,
    mut dst: Image1DMut<'_, T>,
) {
    let src_w = src.width();
    let dst_w = left + src_w + right;
    assert_eq!(dst.width(), dst_w, "Output image has wrong width");

    for i in 0..left {
        dst.set_pixel(
            i,
            match fill {
                FillMode::Transparent => T::transparent(),
                FillMode::Black => T::black(),
                FillMode::EdgeCopy if src_w > 0 => *src.get_pixel(0),
                FillMode::EdgeCopy => T::transparent(),
            },
        );
    }
    for x in 0..src_w {
        dst.set_pixel(left + x, *src.get_pixel(x));
    }
    for i in 0..right {
        dst.set_pixel(
            left + src_w + i,
            match fill {
                FillMode::Transparent => T::transparent(),
                FillMode::Black => T::black(),
                FillMode::EdgeCopy if src_w > 0 => *src.get_pixel(src_w - 1),
                FillMode::EdgeCopy => T::transparent(),
            },
        );
    }
}

// ── 2-D padding internal ──────────────────────────────────────────────────────

pub fn pad_2d<T: ResizablePixel>(
    src: Image2DRef<'_, T>,
    padding: Padding,
    fill: FillMode,
    mut dst: Image2DMut<'_, T>,
) {
    let src_w = src.width();
    let src_h = src.height();
    let dst_w = padding.left + src_w + padding.right;
    let dst_h = padding.top + src_h + padding.bottom;

    assert_eq!(dst.width(), dst_w, "Output image has wrong width");
    assert_eq!(dst.height(), dst_h, "Output image has wrong height");

    for oy in 0..dst_h {
        for ox in 0..dst_w {
            let in_x = ox >= padding.left && ox < padding.left + src_w;
            let in_y = oy >= padding.top && oy < padding.top + src_h;

            let p = if in_x && in_y {
                *src.get_pixel(ox - padding.left, oy - padding.top)
            } else {
                match fill {
                    FillMode::Transparent => T::transparent(),
                    FillMode::Black => T::black(),
                    FillMode::EdgeCopy if src_w > 0 && src_h > 0 => {
                        let cx =
                            (ox as i64 - padding.left as i64).clamp(0, src_w as i64 - 1) as usize;
                        let cy =
                            (oy as i64 - padding.top as i64).clamp(0, src_h as i64 - 1) as usize;
                        *src.get_pixel(cx, cy)
                    }
                    FillMode::EdgeCopy => T::transparent(),
                }
            };

            dst.set_pixel(ox, oy, p);
        }
    }
}
