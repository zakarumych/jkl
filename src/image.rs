//! Provides immutable (`ImageRef`) and mutable (`ImageMut`) 2D image views over flat pixel buffers.
//!
//! These types represent strided, non-owning references into pixel data, supporting sub-region
//! extraction, block matching, residual computation, and patch-based initialization. They are
//! generic over the pixel type `T`.
//!
//! # Layout
//!
//! Pixels are stored in row-major order in a flat slice. The `stride` field indicates the number
//! of `T` elements between the start of consecutive rows, which may be larger than `width` when
//! the view is a sub-region of a larger image.
//!
//! # Examples
//!
//! ```
//! let pixels = vec![0u8; 640 * 480];
//! let image = ImageRef {
//!     width: 640,
//!     height: 480,
//!     pixels: &pixels,
//!     stride: 640,
//! };
//! let pixel = image.get(10, 20);
//! ```

use std::ops;

use crate::math::Zero;

/// An immutable, non-owning 2D view into a pixel buffer.
///
/// `ImageRef` borrows a flat slice of pixel data and interprets it as a 2D image
/// with the given `width`, `height`, and row `stride`. It is `Copy` and `Clone`,
/// making it cheap to pass around.
///
/// # Type Parameters
///
/// * `T` — The pixel type. Must be `Copy` for most operations.
pub struct ImageRef<'a, T> {
    /// The width of the image view in pixels.
    width: usize,
    /// The height of the image view in pixels.
    height: usize,
    /// The underlying pixel data. Length must be at least `(height - 1) * stride + width`.
    pixels: &'a [T],
    /// The number of `T` elements between the start of consecutive rows.
    stride: usize,
}

impl<T> Copy for ImageRef<'_, T> {}
impl<T> Clone for ImageRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> ImageRef<'a, T> {
    /// Creates a new `ImageRef` with contiguous row storage (stride equals width).
    ///
    /// # Parameters
    ///
    /// * `width` — The width of the image in pixels.
    /// * `height` — The height of the image in pixels.
    /// * `pixels` — The underlying pixel data slice.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() < height * width`.
    pub fn new(width: usize, height: usize, pixels: &'a [T]) -> Self {
        assert!(pixels.len() >= height * width);
        ImageRef {
            width,
            height,
            pixels,
            stride: width,
        }
    }

    /// Creates a new `ImageRef` with a custom row stride.
    ///
    /// Use this when the pixel data has padding between rows, e.g. when
    /// the view represents a sub-region of a larger image.
    ///
    /// # Parameters
    ///
    /// * `width` — The width of the image in pixels.
    /// * `height` — The height of the image in pixels.
    /// * `pixels` — The underlying pixel data slice.
    /// * `stride` — The number of `T` elements between the start of consecutive rows.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() < (height - 1) * stride + width`.
    pub fn with_stride(width: usize, height: usize, pixels: &'a [T], stride: usize) -> Self {
        assert!(pixels.len() >= (height - 1) * stride + width);
        ImageRef {
            width,
            height,
            pixels,
            stride: width,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn as_ref(&self) -> ImageRef<'_, T> {
        *self
    }

    /// Returns a reference to the pixel at coordinates (`x`, `y`).
    ///
    /// # Panics
    ///
    /// Panics if the computed index `y * stride + x` is out of bounds.
    pub fn get(&self, x: usize, y: usize) -> &'a T {
        assert!(x < self.width);
        assert!(y < self.height);

        &self.pixels[y * self.stride + x]
    }

    /// Returns a reference to the row at vertical coordinate `y`.
    pub fn row(&self, y: usize) -> &'a [T] {
        assert!(y < self.height);
        &self.pixels[y * self.stride..][..self.width]
    }

    /// Returns a sub-region of this image as a new `ImageRef`.
    ///
    /// The returned view starts at (`x`, `y`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range(&self, x: usize, y: usize, w: usize, h: usize) -> ImageRef<'a, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);

        ImageRef {
            width: w,
            height: h,
            pixels: &self.pixels[y * self.stride + x..],
            stride: self.stride,
        }
    }

    /// Returns the raw underlying pixel slice.
    pub fn pixels(&self) -> &'a [T] {
        &self.pixels
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &'a T> {
        let width = self.width;
        self.pixels
            .chunks(self.stride)
            .take(self.height)
            .flat_map(move |row| &row[..width])
    }

    /// Calculates total error between the reference map patch and the given block using the provided error function.
    ///
    /// Compares each pixel in `block` against the corresponding pixel in `self` starting
    /// at offset (`x`, `y`). The per-pixel errors are accumulated via addition.
    ///
    /// # Parameters
    ///
    /// * `x` — Left offset into `self` where the comparison starts.
    /// * `y` — Top offset into `self` where the comparison starts.
    /// * `block` — The block to compare against this image region.
    /// * `error` — A function that computes the error between two pixels.
    ///
    /// # Panics
    ///
    /// Panics if the block placed at (`x`, `y`) extends beyond the image bounds.
    pub fn match_block<'b, U>(
        &self,
        x: usize,
        y: usize,
        block: ImageRef<'b, T>,
        mut error: impl FnMut(T, T) -> U,
    ) -> U
    where
        T: Copy + 'a + 'b,
        U: ops::Add<Output = U> + Zero,
    {
        assert!(x + block.width() <= self.width());
        assert!(y + block.height() <= self.height());

        let mut acc = U::zero();

        for j in 0..block.height() {
            for i in 0..block.width() {
                let r = *self.get(x + i, y + j);
                let b = *block.get(i, j);

                acc = acc + error(r, b);
            }
        }

        acc
    }

    /// Calculates total error between the reference map patch and the given block using the provided error function, but returns `None` if the error exceeds the given upper bound.
    ///
    /// This is an early-exit variant of [`match_block`](Self::match_block). If the accumulated
    /// error exceeds `upper_bound` at any point during iteration, the function immediately
    /// returns `None`, avoiding unnecessary computation.
    ///
    /// # Parameters
    ///
    /// * `x` — Left offset into `self` where the comparison starts.
    /// * `y` — Top offset into `self` where the comparison starts.
    /// * `block` — The block to compare against this image region.
    /// * `upper_bound` — Maximum acceptable total error. If exceeded, returns `None`.
    /// * `error` — A function that computes the error between two pixels.
    ///
    /// # Returns
    ///
    /// `Some(total_error)` if the total error is within the upper bound, or `None` otherwise.
    /// Also returns `None` if the image has zero width or height.
    ///
    /// # Panics
    ///
    /// Panics if the block placed at (`x`, `y`) extends beyond the image bounds.
    pub fn match_block_upper_bound<'b, U>(
        &self,
        x: usize,
        y: usize,
        block: ImageRef<'b, T>,
        upper_bound: U,
        mut error: impl FnMut(T, T) -> U,
    ) -> Option<U>
    where
        T: Copy + 'a + 'b,
        U: ops::Add<Output = U> + PartialOrd + Zero,
    {
        assert!(x + block.width() <= self.width());
        assert!(y + block.height() <= self.height());

        let mut acc = U::zero();

        for j in 0..block.height() {
            for i in 0..block.width() {
                let a = *self.get(x + i, y + j);
                let b = *block.get(i, j);

                acc = acc + error(a, b);
                if acc > upper_bound {
                    return None;
                }
            }
        }

        Some(acc)
    }

    /// Finds the position within this image where the given block best matches (lowest error).
    ///
    /// Performs a strided exhaustive search over all valid placements of `block` within `self`,
    /// stepping by `step_x` horizontally and `step_y` vertically. Uses early-exit upper-bound
    /// pruning via [`match_block_upper_bound`](Self::match_block_upper_bound) to skip
    /// positions that cannot improve on the current best.
    ///
    /// The search starts at position (0, 0) as the initial candidate, then scans the remainder
    /// of the first row before proceeding to subsequent rows.
    ///
    /// # Parameters
    ///
    /// * `step_x` — Horizontal step size between candidate positions.
    /// * `step_y` — Vertical step size between candidate positions.
    /// * `block` — The block to search for within this image.
    /// * `error` — A function that computes the per-pixel error between two pixels.
    ///
    /// # Returns
    ///
    /// A tuple of `(best_error, (x, y))` where `(x, y)` is the top-left position of the
    /// best-matching region.
    ///
    /// # Panics
    ///
    /// Panics if the block dimensions exceed the image dimensions.
    pub fn find_best_match<'b, U>(
        &self,
        step_x: usize,
        step_y: usize,
        block: ImageRef<'b, T>,
        mut error: impl FnMut(T, T) -> U,
    ) -> (U, (usize, usize))
    where
        T: Copy + 'a + 'b,
        U: ops::Add<Output = U> + PartialOrd + Copy + Zero,
    {
        assert!(block.width() <= self.width());
        assert!(block.height() <= self.height());

        if self.width() == 0 || self.height() == 0 {
            return (U::zero(), (0, 0));
        }

        // Calculate error for the starting point.
        let mut best_error = self.match_block(0, 0, block.as_ref(), &mut error);
        let mut best = (0, 0);

        for y in 0..1 {
            for x in (1..self.width() - block.width() + 1).step_by(step_x) {
                if let Some(error) =
                    self.match_block_upper_bound(x, y, block.as_ref(), best_error, &mut error)
                {
                    best_error = error;
                    best = (x, y);
                }
            }
        }

        for y in (1..(self.height() - block.height() + 1)).step_by(step_y) {
            for x in (0..self.width() - block.width() + 1).step_by(step_x) {
                if let Some(error) =
                    self.match_block_upper_bound(x, y, block.as_ref(), best_error, &mut error)
                {
                    best_error = error;
                    best = (x, y);
                }
            }
        }

        (best_error, best)
    }

    /// Returns residuals between the reference map patch and the given block using the provided residual function.
    ///
    /// For each pixel in `input`, computes `residual(self_pixel, input_pixel)` and writes
    /// the result into the corresponding position of `output`. The reference patch in `self`
    /// starts at offset (`x`, `y`).
    ///
    /// # Parameters
    ///
    /// * `x` — Left offset into `self` for the reference patch.
    /// * `y` — Top offset into `self` for the reference patch.
    /// * `input` — The input block to compute residuals against.
    /// * `output` — The mutable image to write residual values into. Must have the same
    ///   dimensions as `input`.
    /// * `residual` — A function that computes the residual between a reference pixel and
    ///   an input pixel.
    ///
    /// # Panics
    ///
    /// Panics if `input` and `output` dimensions do not match.
    pub fn residual<'b, 'c>(
        &self,
        x: usize,
        y: usize,
        input: ImageRef<'b, T>,
        mut output: ImageMut<'c, T>,
        mut residual: impl FnMut(T, T) -> T,
    ) where
        T: Copy + 'a + 'b + 'c,
    {
        if self.width() == 0 || self.height() == 0 {
            return;
        }

        assert_eq!(input.width(), output.width());
        assert_eq!(input.height(), output.height());

        for j in 0..input.height() {
            for i in 0..input.width() {
                let a = *self.get(x + i, y + j);
                let b = *input.get(i, j);

                output.set(i, j, residual(a, b));
            }
        }
    }

    pub fn into_matrix<const W: usize, const H: usize>(self) -> [[T; W]; H]
    where
        T: Copy,
    {
        assert_eq!(self.width, W);
        assert_eq!(self.height, H);

        let mut colors = [[self.pixels[0]; W]; H];

        for y in 0..H {
            for x in 0..W {
                colors[y][x] = *self.get(x, y);
            }
        }

        colors
    }
}

/// A mutable, non-owning 2D view into a pixel buffer.
///
/// `ImageMut` borrows a flat slice of pixel data mutably and interprets it as a 2D image
/// with the given `width`, `height`, and row `stride`. It supports reading, writing,
/// sub-region extraction, copying from an `ImageRef`, and random/patch-based initialization.
///
/// # Type Parameters
///
/// * `T` — The pixel type. Must be `Copy` for most operations.
pub struct ImageMut<'a, T> {
    /// The width of the image view in pixels.
    width: usize,
    /// The height of the image view in pixels.
    height: usize,
    /// The underlying mutable pixel data. Length must be at least `(height - 1) * stride + width`.
    pixels: &'a mut [T],
    /// The number of `T` elements between the start of consecutive rows.
    stride: usize,
}

impl<'a, T> ImageMut<'a, T> {
    /// Creates a new `ImageMut` with contiguous row storage (stride equals width).
    ///
    /// # Parameters
    ///
    /// * `width` — The width of the image in pixels.
    /// * `height` — The height of the image in pixels.
    /// * `pixels` — The underlying mutable pixel data slice.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() < height * width`.
    pub fn new(width: usize, height: usize, pixels: &'a mut [T]) -> Self {
        assert!(pixels.len() >= height * width);
        ImageMut {
            width,
            height,
            pixels,
            stride: width,
        }
    }

    /// Creates a new `ImageMut` with a custom row stride.
    ///
    /// Use this when the pixel data has padding between rows, e.g. when
    /// the view represents a sub-region of a larger image.
    ///
    /// # Parameters
    ///
    /// * `width` — The width of the image in pixels.
    /// * `height` — The height of the image in pixels.
    /// * `pixels` — The underlying mutable pixel data slice.
    /// * `stride` — The number of `T` elements between the start of consecutive rows.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() < (height - 1) * stride + width`.
    pub fn with_stride(width: usize, height: usize, pixels: &'a mut [T], stride: usize) -> Self {
        assert!(pixels.len() >= (height - 1) * stride + width);
        ImageMut {
            width,
            height,
            pixels,
            stride: width,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn as_ref(&self) -> ImageRef<'_, T> {
        ImageRef {
            width: self.width,
            height: self.height,
            pixels: &*self.pixels,
            stride: self.stride,
        }
    }

    pub fn as_mut(&mut self) -> ImageMut<'_, T> {
        ImageMut {
            width: self.width,
            height: self.height,
            pixels: &mut *self.pixels,
            stride: self.stride,
        }
    }

    /// Returns a reference to the pixel at coordinates (`x`, `y`).
    ///
    /// # Panics
    ///
    /// Panics if the computed index `y * stride + x` is out of bounds.
    pub fn get(&self, x: usize, y: usize) -> &T {
        &self.pixels[y * self.stride + x]
    }

    /// Returns a reference to the row at vertical coordinate `y`.
    pub fn row(&self, y: usize) -> &'_ [T] {
        assert!(y < self.height);
        &self.pixels[y * self.stride..][..self.width]
    }

    /// Returns a reference to the pixel at coordinates (`x`, `y`).
    ///
    /// # Panics
    ///
    /// Panics if the computed index `y * stride + x` is out of bounds.
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.pixels[y * self.stride + x]
    }

    /// Returns a reference to the row at vertical coordinate `y`.
    pub fn row_mut(&mut self, y: usize) -> &'_ mut [T] {
        assert!(y < self.height);
        &mut self.pixels[y * self.stride..][..self.width]
    }

    /// Sets the pixel at coordinates (`x`, `y`) to `value`.
    ///
    /// # Panics
    ///
    /// Panics if the computed index `y * stride + x` is out of bounds.
    pub fn set(&mut self, x: usize, y: usize, value: T) {
        self.pixels[y * self.stride + x] = value;
    }

    /// Returns the raw underlying pixel slice.
    pub fn pixels(&self) -> &[T] {
        &*self.pixels
    }

    /// Returns the raw underlying pixel slice.
    pub fn pixels_mut(&mut self) -> &mut [T] {
        &mut *self.pixels
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &'_ T> {
        let width = self.width;
        self.pixels
            .chunks(self.stride)
            .take(self.height)
            .flat_map(move |row| &row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut T> {
        let width = self.width;
        self.pixels
            .chunks_mut(self.stride)
            .take(self.height)
            .flat_map(move |row| &mut row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn into_iter(self) -> impl Iterator<Item = &'a mut T> {
        let width = self.width;
        self.pixels
            .chunks_mut(self.stride)
            .take(self.height)
            .flat_map(move |row| &mut row[..width])
    }

    /// Returns a immutable sub-region of this image as a new `ImageRef`.
    ///
    /// The returned view starts at (`x`, `y`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range(&mut self, x: usize, y: usize, w: usize, h: usize) -> ImageRef<'_, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);

        ImageRef {
            width: w,
            height: h,
            pixels: &self.pixels[y * self.stride + x..],
            stride: self.stride,
        }
    }

    /// Returns a mutable sub-region of this image as a new `ImageMut`.
    ///
    /// The returned view starts at (`x`, `y`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range_mut(&mut self, x: usize, y: usize, w: usize, h: usize) -> ImageMut<'_, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);

        ImageMut {
            width: w,
            height: h,
            pixels: &mut self.pixels[y * self.stride + x..],
            stride: self.stride,
        }
    }

    /// Copies pixel data from `src` into this image.
    ///
    /// # Panics
    ///
    /// Panics if `src` and `self` have different dimensions.
    pub fn copy_from(&mut self, src: ImageRef<'_, T>)
    where
        T: Copy,
    {
        assert_eq!(src.width, self.width);
        assert_eq!(src.height, self.height);

        for j in 0..src.height {
            for i in 0..src.width {
                self.set(i, j, *src.get(i, j));
            }
        }
    }

    pub fn copy_from_matrix<const W: usize, const H: usize>(&mut self, matrix: &[[T; W]; H])
    where
        T: Copy,
    {
        assert_eq!(self.width, W);
        assert_eq!(self.height, H);

        for y in 0..H {
            for x in 0..W {
                self.set(x, y, matrix[y][x]);
            }
        }
    }
}

/// An owning 2D image type that manages its own pixel buffer.
pub struct Image<T> {
    width: usize,
    height: usize,
    pixels: Vec<T>,
}

impl<T> Image<T> {
    pub fn new(width: usize, height: usize, pixels: Vec<T>) -> Self {
        assert!(pixels.len() >= height * width);
        Image {
            width,
            height,
            pixels,
        }
    }

    pub fn as_ref(&self) -> ImageRef<'_, T> {
        ImageRef::new(self.width, self.height, &self.pixels)
    }

    pub fn as_mut(&mut self) -> ImageMut<'_, T> {
        ImageMut::new(self.width, self.height, &mut self.pixels)
    }
}
