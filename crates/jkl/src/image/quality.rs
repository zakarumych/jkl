//! Image quality metrics for comparing original and lossy-compressed images.
//!
//! All functions operate on pairs of [`ImageRef`] values with the same extent.
//! Callers are responsible for converting to the desired color space before
//! calling (e.g. convert [`Rgb8U`] to [`Yiq32F`] to obtain perceptual metrics).
//!
//! # Per-channel metrics
//!
//! Extract a single-channel image and call the same functions again:
//! ```ignore
//! let r_orig: OwnedImage<R8U> = extract_channel(&original, 0);
//! let r_comp: OwnedImage<R8U> = extract_channel(&compressed, 0);
//! let channel_mse = quality::mse(r_orig.as_ref(), r_comp.as_ref());
//! ```

use crate::image::{Image2DRef, ImageRef, OwnedImage};
use crate::math::{R8U, R32F, Rg8U, Rg32F, Rgb8U, Rgb32F, Rgba8U, Rgba32F, Yiq32F, Yiqa32F};

// ── Trait ──────────────────────────────────────────────────────────────────

/// Implemented by pixel types that can be compared for quality measurement.
///
/// `error_squared`, `error_range`, and `sobel_gm` must all be provided.
/// `error` has a default implementation as `sqrt(error_squared(a, b))`.
pub trait ErrorPixel: Copy {
    /// The pixel type used to represent per-channel absolute error.
    type ChannelError: Copy;

    /// Per-channel absolute error between two pixels.
    ///
    /// For integer types channels are normalised to `[0, 1]`.  Unlike
    /// [`error_squared`](Self::error_squared), channels are not combined
    /// into a single scalar or alpha-premultiplied.
    fn error_channels(a: Self, b: Self) -> Self::ChannelError;

    /// Squared error between two pixels in channel space.
    ///
    /// For types with an alpha channel, computes premultiplied color error.
    fn error_squared(a: Self, b: Self) -> f32;

    /// Upper bound on [`error`](Self::error) for this pixel type.
    ///
    /// Used to normalize histogram bins over the full error range.
    fn error_range() -> f32;

    /// Sobel gradient magnitude at pixel `(x, y)` in a single plane.
    ///
    /// Gradient is computed per channel with a 3×3 Sobel kernel
    /// (replicate-padded at borders), combined across channels as
    /// `sqrt(Σ_c (Gx_c² + Gy_c²))`. Implement by summing `sobel_gm_sq_ch`
    /// results per channel and returning the square root.
    fn sobel_gm(plane: Image2DRef<'_, Self>, x: usize, y: usize) -> f32;

    /// Error magnitude between two pixels.
    ///
    /// Default: `sqrt(error_squared(a, b))`.
    #[inline]
    fn error(a: Self, b: Self) -> f32 {
        Self::error_squared(a, b).sqrt()
    }
}

/// Subtrait for pixel types that carry an alpha channel.
///
/// The [`ErrorPixel`] implementation for these types computes color error on
/// alpha-premultiplied channels only. Use the `alpha_*` metric functions to
/// measure error in the alpha channel separately.
pub trait ErrorPixelAlpha: ErrorPixel {
    /// Returns the alpha value in `[0, 1]`.
    fn alpha(self) -> f32;
}

impl ErrorPixel for R8U {
    type ChannelError = R32F;
    #[inline]
    fn error_channels(a: Self, b: Self) -> R32F {
        R32F((a.0 as f32 - b.0 as f32).abs() / 255.0)
    }
    #[inline]
    fn error_squared(a: Self, b: Self) -> f32 {
        R8U::distance_squared(a, b) / (255.0 * 255.0)
    }
    #[inline]
    fn error_range() -> f32 {
        1.0
    }
    #[inline]
    fn error(a: Self, b: Self) -> f32 {
        R8U::distance(a, b) / 255.0
    }
    #[inline]
    fn sobel_gm(plane: Image2DRef<'_, Self>, x: usize, y: usize) -> f32 {
        sobel_gm_sq_ch(plane, x, y, |p| p.0 as f32 / 255.0).sqrt()
    }
}

impl ErrorPixel for Rg8U {
    type ChannelError = Rg32F;
    #[inline]
    fn error_channels(a: Self, b: Self) -> Rg32F {
        Rg32F([
            (a.0[0] as f32 - b.0[0] as f32).abs() / 255.0,
            (a.0[1] as f32 - b.0[1] as f32).abs() / 255.0,
        ])
    }
    #[inline]
    fn error_squared(a: Self, b: Self) -> f32 {
        Rg8U::distance_squared(a, b) / (255.0 * 255.0)
    }
    #[inline]
    fn error_range() -> f32 {
        2.0f32.sqrt()
    }
    #[inline]
    fn error(a: Self, b: Self) -> f32 {
        Rg8U::distance(a, b) / 255.0
    }
    #[inline]
    fn sobel_gm(plane: Image2DRef<'_, Self>, x: usize, y: usize) -> f32 {
        (sobel_gm_sq_ch(plane, x, y, |p| p.0[0] as f32 / 255.0)
            + sobel_gm_sq_ch(plane, x, y, |p| p.0[1] as f32 / 255.0))
        .sqrt()
    }
}

impl ErrorPixel for Rgb8U {
    type ChannelError = Rgb32F;
    #[inline]
    fn error_channels(a: Self, b: Self) -> Rgb32F {
        Rgb32F([
            (a.0[0] as f32 - b.0[0] as f32).abs() / 255.0,
            (a.0[1] as f32 - b.0[1] as f32).abs() / 255.0,
            (a.0[2] as f32 - b.0[2] as f32).abs() / 255.0,
        ])
    }
    #[inline]
    fn error_squared(a: Self, b: Self) -> f32 {
        Rgb8U::distance_squared(a, b) / (255.0 * 255.0)
    }
    #[inline]
    fn error_range() -> f32 {
        3.0f32.sqrt()
    }
    #[inline]
    fn error(a: Self, b: Self) -> f32 {
        Rgb8U::distance(a, b) / 255.0
    }
    #[inline]
    fn sobel_gm(plane: Image2DRef<'_, Self>, x: usize, y: usize) -> f32 {
        (sobel_gm_sq_ch(plane, x, y, |p| p.0[0] as f32 / 255.0)
            + sobel_gm_sq_ch(plane, x, y, |p| p.0[1] as f32 / 255.0)
            + sobel_gm_sq_ch(plane, x, y, |p| p.0[2] as f32 / 255.0))
        .sqrt()
    }
}

impl ErrorPixel for Rgba8U {
    type ChannelError = Rgba32F;
    #[inline]
    fn error_channels(a: Self, b: Self) -> Rgba32F {
        Rgba32F([
            (a.0[0] as f32 - b.0[0] as f32).abs() / 255.0,
            (a.0[1] as f32 - b.0[1] as f32).abs() / 255.0,
            (a.0[2] as f32 - b.0[2] as f32).abs() / 255.0,
            (a.0[3] as f32 - b.0[3] as f32).abs() / 255.0,
        ])
    }
    #[inline]
    fn error_squared(a: Self, b: Self) -> f32 {
        // Alpha-premultiplied: transparent pixels contribute no color error.
        let mut sum = 0.0f32;
        for i in 0..3 {
            let d = a.0[i] as f32 / 255.0 * a.0[3] as f32 / 255.0
                - b.0[i] as f32 / 255.0 * b.0[3] as f32 / 255.0;
            sum += d * d;
        }
        sum
    }
    #[inline]
    fn error_range() -> f32 {
        3.0f32.sqrt() // 3 premultiplied channels, each in [0, 1]
    }
    #[inline]
    fn sobel_gm(plane: Image2DRef<'_, Self>, x: usize, y: usize) -> f32 {
        (sobel_gm_sq_ch(plane, x, y, |p| {
            p.0[0] as f32 / 255.0 * p.0[3] as f32 / 255.0
        }) + sobel_gm_sq_ch(plane, x, y, |p| {
            p.0[1] as f32 / 255.0 * p.0[3] as f32 / 255.0
        }) + sobel_gm_sq_ch(plane, x, y, |p| {
            p.0[2] as f32 / 255.0 * p.0[3] as f32 / 255.0
        }))
        .sqrt()
    }
}

impl ErrorPixelAlpha for Rgba8U {
    #[inline]
    fn alpha(self) -> f32 {
        self.0[3] as f32 / 255.0
    }
}

impl ErrorPixel for Yiq32F {
    type ChannelError = Yiq32F;
    #[inline]
    fn error_channels(a: Self, b: Self) -> Yiq32F {
        Yiq32F([
            (a.0[0] - b.0[0]).abs(),
            (a.0[1] - b.0[1]).abs(),
            (a.0[2] - b.0[2]).abs(),
        ])
    }
    #[inline]
    fn error_squared(a: Self, b: Self) -> f32 {
        Yiq32F::distance_squared(a, b)
    }
    #[inline]
    fn error_range() -> f32 {
        3.0f32.sqrt()
    }
    #[inline]
    fn error(a: Self, b: Self) -> f32 {
        Yiq32F::distance(a, b)
    }
    #[inline]
    fn sobel_gm(plane: Image2DRef<'_, Self>, x: usize, y: usize) -> f32 {
        (sobel_gm_sq_ch(plane, x, y, |p| p.0[0])
            + sobel_gm_sq_ch(plane, x, y, |p| p.0[1])
            + sobel_gm_sq_ch(plane, x, y, |p| p.0[2]))
        .sqrt()
    }
}

impl ErrorPixel for Yiqa32F {
    type ChannelError = Yiqa32F;
    #[inline]
    fn error_channels(a: Self, b: Self) -> Yiqa32F {
        Yiqa32F([
            (a.0[0] - b.0[0]).abs(),
            (a.0[1] - b.0[1]).abs(),
            (a.0[2] - b.0[2]).abs(),
            (a.0[3] - b.0[3]).abs(),
        ])
    }
    #[inline]
    fn error_squared(a: Self, b: Self) -> f32 {
        // Alpha-premultiplied color error.
        let mut sum = 0.0f32;
        for i in 0..3 {
            let d = a.0[i] * a.0[3] - b.0[i] * b.0[3];
            sum += d * d;
        }
        sum
    }
    #[inline]
    fn error_range() -> f32 {
        3.0f32.sqrt()
    }
    #[inline]
    fn sobel_gm(plane: Image2DRef<'_, Self>, x: usize, y: usize) -> f32 {
        (sobel_gm_sq_ch(plane, x, y, |p| p.0[0] * p.0[3])
            + sobel_gm_sq_ch(plane, x, y, |p| p.0[1] * p.0[3])
            + sobel_gm_sq_ch(plane, x, y, |p| p.0[2] * p.0[3]))
        .sqrt()
    }
}

impl ErrorPixelAlpha for Yiqa32F {
    #[inline]
    fn alpha(self) -> f32 {
        self.0[3]
    }
}

impl ErrorPixel for R32F {
    type ChannelError = R32F;
    #[inline]
    fn error_channels(a: Self, b: Self) -> R32F {
        R32F((a.0 - b.0).abs())
    }
    #[inline]
    fn error_squared(a: Self, b: Self) -> f32 {
        R32F::distance_squared(a, b)
    }
    #[inline]
    fn error_range() -> f32 {
        1.0
    }
    #[inline]
    fn error(a: Self, b: Self) -> f32 {
        R32F::distance(a, b)
    }
    #[inline]
    fn sobel_gm(plane: Image2DRef<'_, Self>, x: usize, y: usize) -> f32 {
        sobel_gm_sq_ch(plane, x, y, |p| p.0).sqrt()
    }
}

// ── Output types ───────────────────────────────────────────────────────────

/// Distribution of per-pixel errors across uniform bins.
pub struct ErrorHistogram {
    /// Pixel counts per bin. `bins[i]` counts pixels whose error falls in
    /// the range `[i * bin_width, (i+1) * bin_width)`.
    pub bins: Vec<u64>,
    /// Width of each bin in error units: `T::error_range() / bins.len()`.
    pub bin_width: f64,
}

// ── Internal helpers ───────────────────────────────────────────────────────

/// Total pixel count of an image (all planes × all rows × all columns).
fn pixel_count(image: &ImageRef<'_, impl ErrorPixel>) -> usize {
    let [w, h, d] = image.raw_extent();
    w * h * d
}

/// Assert that two images have identical dimensionality and extent.
fn assert_same_extent<T: ErrorPixel>(a: &ImageRef<'_, T>, b: &ImageRef<'_, T>) {
    assert_eq!(a.raw_extent(), b.raw_extent(), "image extents must match");
    assert_eq!(
        a.dimensions(),
        b.dimensions(),
        "image dimensions must match"
    );
}

/// Squared Sobel gradient magnitude for a single scalar channel extracted from
/// each pixel by `ch`. Uses replicate (clamp-to-border) padding at image edges.
#[inline]
fn sobel_gm_sq_ch<P: Copy>(
    plane: Image2DRef<'_, P>,
    x: usize,
    y: usize,
    ch: impl Fn(P) -> f32,
) -> f32 {
    let w = plane.width() as i32;
    let h = plane.height() as i32;
    let xi = x as i32;
    let yi = y as i32;
    let p = |dx: i32, dy: i32| -> f32 {
        let nx = (xi + dx).max(0).min(w - 1) as usize;
        let ny = (yi + dy).max(0).min(h - 1) as usize;
        ch(*plane.get_pixel(nx, ny))
    };
    let gx = -p(-1, -1) + p(1, -1) - 2.0 * p(-1, 0) + 2.0 * p(1, 0) - p(-1, 1) + p(1, 1);
    let gy = -p(-1, -1) - 2.0 * p(0, -1) - p(1, -1) + p(-1, 1) + 2.0 * p(0, 1) + p(1, 1);
    gx * gx + gy * gy
}

fn extract_alpha<T: ErrorPixelAlpha>(img: &ImageRef<'_, T>) -> OwnedImage<R32F> {
    let pixels: Box<[R32F]> = img.iter_pixels().map(|p| R32F(p.alpha())).collect();
    OwnedImage::new(img.dimensions(), img.raw_extent(), pixels)
}

// ── Public metric functions ────────────────────────────────────────────────

/// Mean Squared Error between two images.
///
/// Averages `error_squared(a, b)` over every pixel. Uses `f64` accumulators to
/// avoid catastrophic cancellation on large images.
///
/// # Panics
///
/// Panics if the images have different extents or dimensionality.
pub fn mse<T: ErrorPixel>(a: ImageRef<'_, T>, b: ImageRef<'_, T>) -> f64 {
    assert_same_extent(&a, &b);
    let n = pixel_count(&a);
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = a
        .iter_pixels()
        .zip(b.iter_pixels())
        .map(|(pa, pb)| T::error_squared(*pa, *pb) as f64)
        .sum();
    sum / n as f64
}

/// Peak Signal-to-Noise Ratio derived from a pre-computed MSE.
///
/// Uses `T::error_range()²` as the peak signal power, consistent with
/// [`psnr`]. Returns `f64::INFINITY` when `mse == 0`.
#[inline]
pub fn psnr_from_mse<T: ErrorPixel>(mse: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY
    } else {
        let peak_sq = (T::error_range() as f64).powi(2);
        10.0 * (peak_sq / mse).log10()
    }
}

/// Peak Signal-to-Noise Ratio between two images.
///
/// Equivalent to `psnr_from_mse::<T>(mse(a, b))`.
///
/// # Panics
///
/// Panics if the images have different extents or dimensionality.
pub fn psnr<T: ErrorPixel>(a: ImageRef<'_, T>, b: ImageRef<'_, T>) -> f64 {
    psnr_from_mse::<T>(mse(a, b))
}

/// Maximum per-pixel error across all pixels.
///
/// # Panics
///
/// Panics if the images have different extents or dimensionality.
pub fn max_error<T: ErrorPixel>(a: ImageRef<'_, T>, b: ImageRef<'_, T>) -> f64 {
    assert_same_extent(&a, &b);
    a.iter_pixels()
        .zip(b.iter_pixels())
        .map(|(pa, pb)| T::error(*pa, *pb) as f64)
        .fold(0.0f64, f64::max)
}

/// Distribution of per-pixel errors across `bins` uniform bins spanning
/// `[0, T::error_range()]`.
///
/// # Panics
///
/// Panics if `bins == 0` or if the images have different extents or
/// dimensionality.
pub fn error_histogram<T: ErrorPixel>(
    a: ImageRef<'_, T>,
    b: ImageRef<'_, T>,
    bins: usize,
) -> ErrorHistogram {
    assert!(bins > 0, "histogram bin count must be non-zero");
    assert_same_extent(&a, &b);

    let mut counts = vec![0u64; bins];
    let bins_f = bins as f32;
    let max_err = T::error_range();
    for (pa, pb) in a.iter_pixels().zip(b.iter_pixels()) {
        let e = T::error(*pa, *pb);
        let idx = ((e / max_err * bins_f) as usize).min(bins - 1);
        counts[idx] += 1;
    }
    ErrorHistogram {
        bin_width: max_err as f64 / bins as f64,
        bins: counts,
    }
}

/// Spatial map of per-channel per-pixel errors as an [`OwnedImage<T::ChannelError>`].
///
/// Each pixel in the returned image holds the absolute per-channel difference
/// between the corresponding input pixels.  For integer types channels are
/// normalised to `[0, 1]`; for float types the raw absolute difference is
/// returned.  Unlike [`error_heatmap`], channels are **not** combined into a
/// single scalar or alpha-premultiplied.
///
/// # Panics
///
/// Panics if the images have different extents or dimensionality.
pub fn error_map<T: ErrorPixel>(
    a: ImageRef<'_, T>,
    b: ImageRef<'_, T>,
) -> OwnedImage<T::ChannelError> {
    assert_same_extent(&a, &b);
    let pixels: Box<[T::ChannelError]> = a
        .iter_pixels()
        .zip(b.iter_pixels())
        .map(|(pa, pb)| T::error_channels(*pa, *pb))
        .collect();
    OwnedImage::new(a.dimensions(), a.raw_extent(), pixels)
}

/// Spatial map of per-pixel errors as an [`OwnedImage<f32>`].
///
/// The returned image has the same extent and dimensionality as the inputs.
/// Each pixel value is `T::error(a_px, b_px)`. For types with an alpha
/// channel this is the premultiplied color error; use [`alpha_error_heatmap`]
/// for alpha error.
///
/// # Panics
///
/// Panics if the images have different extents or dimensionality.
pub fn error_heatmap<T: ErrorPixel>(a: ImageRef<'_, T>, b: ImageRef<'_, T>) -> OwnedImage<f32> {
    assert_same_extent(&a, &b);
    let errors: Box<[f32]> = a
        .iter_pixels()
        .zip(b.iter_pixels())
        .map(|(pa, pb)| T::error(*pa, *pb))
        .collect::<Vec<f32>>()
        .into_boxed_slice();
    OwnedImage::new(a.dimensions(), a.raw_extent(), errors)
}

/// Gradient Magnitude Similarity Deviation (GMSD) between two images.
///
/// GMSD measures structural fidelity, with lower values indicating better
/// preservation of edges and fine detail. A value of `0` means perfect
/// structural similarity.
///
/// Implements the multi-channel extension of Xue et al. (2014): gradient
/// magnitude per pixel is the L2 norm of the per-channel Sobel responses.
/// The GMS map is `(2·gm_a·gm_b + C) / (gm_a² + gm_b² + C)` with
/// `C = 0.0026²`, and GMSD is the standard deviation of that map.
///
/// # Panics
///
/// Panics if the images have different extents or dimensionality.
pub fn gmsd<T: ErrorPixel>(a: ImageRef<'_, T>, b: ImageRef<'_, T>) -> f64 {
    assert_same_extent(&a, &b);

    const C: f64 = 0.0026 * 0.0026;

    let mut sum_gms = 0.0f64;
    let mut sum_gms_sq = 0.0f64;
    let mut n = 0u64;

    for (plane_a, plane_b) in a.iter_planes().zip(b.iter_planes()) {
        let w = plane_a.width();
        let h = plane_a.height();
        for y in 0..h {
            for x in 0..w {
                let gm_a = T::sobel_gm(plane_a, x, y) as f64;
                let gm_b = T::sobel_gm(plane_b, x, y) as f64;
                let gms = (2.0 * gm_a * gm_b + C) / (gm_a * gm_a + gm_b * gm_b + C);
                sum_gms += gms;
                sum_gms_sq += gms * gms;
                n += 1;
            }
        }
    }

    if n == 0 {
        return 0.0;
    }
    let mean = sum_gms / n as f64;
    let mean_sq = sum_gms_sq / n as f64;
    // Variance may be slightly negative due to floating-point error; clamp to 0.
    (mean_sq - mean * mean).max(0.0).sqrt()
}

// ── Alpha channel metric functions ───────────────────────────────────────────

/// MSE of the alpha channel between two images.
pub fn alpha_mse<T: ErrorPixelAlpha>(a: ImageRef<'_, T>, b: ImageRef<'_, T>) -> f64 {
    let a_ch = extract_alpha(&a);
    let b_ch = extract_alpha(&b);
    mse(a_ch.as_ref(), b_ch.as_ref())
}

/// Maximum alpha error across all pixels.
pub fn alpha_max_error<T: ErrorPixelAlpha>(a: ImageRef<'_, T>, b: ImageRef<'_, T>) -> f64 {
    let a_ch = extract_alpha(&a);
    let b_ch = extract_alpha(&b);
    max_error(a_ch.as_ref(), b_ch.as_ref())
}

/// Distribution of alpha errors across `bins` uniform bins in `[0, 1]`.
pub fn alpha_error_histogram<T: ErrorPixelAlpha>(
    a: ImageRef<'_, T>,
    b: ImageRef<'_, T>,
    bins: usize,
) -> ErrorHistogram {
    let a_ch = extract_alpha(&a);
    let b_ch = extract_alpha(&b);
    error_histogram(a_ch.as_ref(), b_ch.as_ref(), bins)
}

/// Spatial map of per-pixel alpha errors as an [`OwnedImage<f32>`].
pub fn alpha_error_heatmap<T: ErrorPixelAlpha>(
    a: ImageRef<'_, T>,
    b: ImageRef<'_, T>,
) -> OwnedImage<f32> {
    let a_ch = extract_alpha(&a);
    let b_ch = extract_alpha(&b);
    error_heatmap(a_ch.as_ref(), b_ch.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    fn solid_rgb(w: usize, h: usize, r: u8, g: u8, b: u8) -> OwnedImage<Rgb8U> {
        let pixels = vec![Rgb8U::new(r, g, b); w * h].into_boxed_slice();
        Image::new_2d(w, h, pixels)
    }

    #[test]
    fn same_image_zero_error() {
        let img = solid_rgb(8, 8, 128, 64, 200);
        assert_eq!(mse(img.as_ref(), img.as_ref()), 0.0);
        assert_eq!(max_error(img.as_ref(), img.as_ref()), 0.0);
        assert_eq!(psnr_from_mse::<Rgb8U>(0.0), f64::INFINITY);
        let hist = error_histogram(img.as_ref(), img.as_ref(), 256);
        assert_eq!(hist.bins[0], 64); // all 64 pixels in bin 0
        assert!(hist.bins[1..].iter().all(|&c| c == 0));
        assert_eq!(gmsd(img.as_ref(), img.as_ref()), 0.0);
    }

    #[test]
    fn black_vs_white_max_error() {
        let black = solid_rgb(4, 4, 0, 0, 0);
        let white = solid_rgb(4, 4, 255, 255, 255);
        // l2_squared for RGB = (1-0)^2 + (1-0)^2 + (1-0)^2 = 3; mse = mean over pixels.
        let m = mse(black.as_ref(), white.as_ref());
        assert!((m - 3.0).abs() < 1e-6, "mse={m}");
        // max per-pixel L2 = sqrt(3) for (1,1,1) difference.
        let me = max_error(black.as_ref(), white.as_ref());
        let expected = (3.0f64).sqrt();
        assert!((me - expected).abs() < 1e-6, "max_error={me}");
        // psnr_from_mse::<Rgb8U>(3) = 10*log10(3/3) = 0 dB
        assert!(psnr_from_mse::<Rgb8U>(m).is_finite());
    }

    #[test]
    fn heatmap_shape_matches_input() {
        let a = solid_rgb(5, 7, 10, 20, 30);
        let b = solid_rgb(5, 7, 50, 60, 70);
        let hm = error_heatmap(a.as_ref(), b.as_ref());
        assert_eq!(hm.raw_extent(), [5, 7, 1]);
    }

    #[test]
    fn gmsd_uniform_image_is_zero() {
        // Flat image has zero gradient → gm_a = gm_b = 0 everywhere.
        // GMS(p) = C / C = 1 for all p, so std dev = 0.
        let a = solid_rgb(8, 8, 128, 128, 128);
        let b = solid_rgb(8, 8, 200, 200, 200);
        assert_eq!(gmsd(a.as_ref(), b.as_ref()), 0.0);
    }
}
