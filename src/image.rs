//! Provides immutable (`Image2DRef`) and mutable (`Image2DMut`) 2D image views over flat pixel buffers.
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

/// An immutable, non-owning 2D rectangular view into a pixel buffer.
///
/// `Image2DRef` borrows a flat slice of pixel data and interprets it as a 2D image
/// with the given `width`, `height`, and row `stride`. It is `Copy` and `Clone`,
/// making it cheap to pass around.
///
/// # Type Parameters
///
/// * `T` — The pixel type. Must be `Copy` for most operations.
pub struct Image2DRef<'a, T> {
    /// The width of the image view in pixels.
    width: usize,
    /// The height of the image view in pixels.
    height: usize,
    /// The number of `T` elements between the start of consecutive rows.
    stride: usize,
    /// The underlying pixel data. Length must be at least `(height - 1) * stride + width`.
    pixels: &'a [T],
}

impl<T> Copy for Image2DRef<'_, T> {}
impl<T> Clone for Image2DRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Image2DRef<'a, T> {
    /// Creates a new `Image2DRef` with contiguous row storage (stride equals width).
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
        Image2DRef {
            width,
            height,
            pixels,
            stride: width,
        }
    }

    /// Creates a new `Image2DRef` with a custom row stride.
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
    pub fn with_stride(width: usize, height: usize, stride: usize, pixels: &'a [T]) -> Self {
        let len = (height != 0)
            .then(|| (height - 1) * stride + width)
            .unwrap_or(0);

        assert!(pixels.len() >= len);
        Image2DRef {
            width,
            height,
            stride,
            pixels,
        }
    }

    /// Creates a new `Image2DRef` from a single row of pixels.
    pub fn from_row(pixels: &'a [T]) -> Self {
        Image2DRef {
            width: pixels.len(),
            height: 1,
            stride: pixels.len(),
            pixels,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn as_ref(&self) -> Image2DRef<'_, T> {
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

    /// Returns a sub-region of this image as a new `Image2DRef`.
    ///
    /// The returned view starts at (`x`, `y`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range(&self, x: usize, y: usize, w: usize, h: usize) -> Image2DRef<'a, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);

        Image2DRef {
            width: w,
            height: h,
            stride: self.stride,
            pixels: &self.pixels[y * self.stride + x..],
        }
    }

    /// Returns the raw underlying pixel slice.
    pub fn pixels(&self) -> &'a [T] {
        &self.pixels
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn iter_rows(&self) -> impl DoubleEndedIterator<Item = &'a [T]> {
        let Self {
            width,
            height,
            stride,
            pixels,
        } = *self;

        pixels[..stride * height]
            .chunks_exact(stride)
            .map(move |row| &row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn iter_pixels(&self) -> impl DoubleEndedIterator<Item = &'a T> {
        let Self {
            width,
            height,
            stride,
            pixels,
        } = *self;

        pixels[..stride * height]
            .chunks_exact(stride)
            .flat_map(move |row| &row[..width])
    }

    pub fn into_matrix<const W: usize, const H: usize>(self) -> [[T; W]; H]
    where
        T: Copy,
    {
        assert_eq!(self.width, W);
        assert_eq!(self.height, H);

        let mut colors = [[self.pixels[0]; W]; H];

        for y in 0..H {
            colors[y].copy_from_slice(self.row(y));
        }

        colors
    }
}

/// A mutable, non-owning 2D rectangular view into a pixel buffer.
///
/// `Image2DMut` borrows a flat slice of pixel data mutably and interprets it as a 2D image
/// with the given `width`, `height`, and row `stride`. It supports reading, writing,
/// sub-region extraction, copying from an `Image2DRef`, and random/patch-based initialization.
///
/// # Type Parameters
///
/// * `T` — The pixel type. Must be `Copy` for most operations.
pub struct Image2DMut<'a, T> {
    /// The width of the image view in pixels.
    width: usize,
    /// The height of the image view in pixels.
    height: usize,
    /// The number of `T` elements between the start of consecutive rows.
    stride: usize,
    /// The underlying mutable pixel data. Length must be at least `(height - 1) * stride + width`.
    pixels: &'a mut [T],
}

impl<'a, T> Image2DMut<'a, T> {
    /// Creates a new `Image2DMut` with contiguous row storage (stride equals width).
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
        Image2DMut {
            width,
            height,
            stride: width,
            pixels,
        }
    }

    /// Creates a new `Image2DMut` with a custom row stride.
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
    pub fn with_stride(width: usize, height: usize, stride: usize, pixels: &'a mut [T]) -> Self {
        let len = (height != 0)
            .then(|| (height - 1) * stride + width)
            .unwrap_or(0);
        assert!(pixels.len() >= len);
        Image2DMut {
            width,
            height,
            stride,
            pixels,
        }
    }

    /// Creates a new `Image2DMut` from a single row of pixels.
    pub fn from_row(pixels: &'a mut [T]) -> Self {
        Image2DMut {
            width: pixels.len(),
            height: 1,
            stride: pixels.len(),
            pixels,
        }
    }

    /// Creates a new `Image2DMut` by reborrowing its pixel data mutably.
    ///
    /// The returned view has the same dimensions and stride as the input reference.
    pub fn reborrow(&mut self) -> Image2DMut<'_, T> {
        Image2DMut {
            width: self.width,
            height: self.height,
            stride: self.stride,
            pixels: &mut *self.pixels,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn as_ref(&self) -> Image2DRef<'_, T> {
        Image2DRef {
            width: self.width,
            height: self.height,
            stride: self.stride,
            pixels: &*self.pixels,
        }
    }

    pub fn as_mut(&mut self) -> Image2DMut<'_, T> {
        Image2DMut {
            width: self.width,
            height: self.height,
            stride: self.stride,
            pixels: &mut *self.pixels,
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
    pub fn iter_rows(&self) -> impl Iterator<Item = &'_ [T]> {
        let Self {
            width,
            height,
            stride,
            ref pixels,
        } = *self;

        pixels[..stride * height]
            .chunks_exact(stride)
            .map(move |row| &row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn iter_rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        let Self {
            width,
            height,
            stride,
            ref mut pixels,
        } = *self;

        pixels[..stride * height]
            .chunks_exact_mut(stride)
            .map(move |row| &mut row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn into_iter_rows(self) -> impl Iterator<Item = &'a mut [T]> {
        let Self {
            width,
            height,
            stride,
            pixels,
        } = self;

        pixels[..stride * height]
            .chunks_exact_mut(stride)
            .map(move |row| &mut row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn iter_pixels(&self) -> impl Iterator<Item = &T> {
        let Self {
            width,
            height,
            stride,
            ref pixels,
        } = *self;

        pixels[..stride * height]
            .chunks_exact(stride)
            .flat_map(move |row| &row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn iter_pixels_mut(&mut self) -> impl Iterator<Item = &'_ mut T> {
        let Self {
            width,
            height,
            stride,
            ref mut pixels,
        } = *self;

        pixels[..stride * height]
            .chunks_exact_mut(stride)
            .flat_map(move |row| &mut row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    pub fn into_iter_pixels(self) -> impl Iterator<Item = &'a mut T> {
        let Self {
            width,
            height,
            stride,
            pixels,
        } = self;

        pixels[..stride * height]
            .chunks_exact_mut(stride)
            .flat_map(move |row| &mut row[..width])
    }

    /// Returns a immutable sub-region of this image as a new `Image2DRef`.
    ///
    /// The returned view starts at (`x`, `y`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range(&mut self, x: usize, y: usize, w: usize, h: usize) -> Image2DRef<'_, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);

        Image2DRef {
            width: w,
            height: h,
            stride: self.stride,
            pixels: &self.pixels[y * self.stride + x..],
        }
    }

    /// Returns a mutable sub-region of this image as a new `Image2DMut`.
    ///
    /// The returned view starts at (`x`, `y`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range_mut(&mut self, x: usize, y: usize, w: usize, h: usize) -> Image2DMut<'_, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);

        Image2DMut {
            width: w,
            height: h,
            stride: self.stride,
            pixels: &mut self.pixels[y * self.stride + x..],
        }
    }

    /// Copies pixel data from `src` into this image.
    ///
    /// # Panics
    ///
    /// Panics if `src` and `self` have different dimensions.
    pub fn copy_from(&mut self, src: Image2DRef<'_, T>)
    where
        T: Copy,
    {
        assert_eq!(src.width, self.width);
        assert_eq!(src.height, self.height);

        for j in 0..src.height {
            self.row_mut(j).copy_from_slice(src.row(j));
        }
    }

    pub fn copy_from_matrix<const W: usize, const H: usize>(&mut self, matrix: &[[T; W]; H])
    where
        T: Copy,
    {
        assert_eq!(self.width, W);
        assert_eq!(self.height, H);

        for y in 0..H {
            self.row_mut(y).copy_from_slice(&matrix[y]);
        }
    }
}

/// An immutable, non-owning 3D view into a pixel buffer.
///
/// `Image3DRef` borrows a flat slice of pixel data and interprets it as a 3D image
/// with the given `width`, `height`, and row `stride`. It is `Copy` and `Clone`,
/// making it cheap to pass around.
///
/// # Type Parameters
///
/// * `T` — The pixel type. Must be `Copy` for most operations.
pub struct Image3DRef<'a, T> {
    /// The width of the image view in pixels.
    width: usize,
    /// The height of the image view in pixels.
    height: usize,
    /// The depth of the image view in pixels.
    depth: usize,
    /// The number of `T` elements between the start of consecutive rows.
    row_stride: usize,
    /// The number of `T` elements between the start of consecutive planes.
    plane_stride: usize,
    /// The underlying pixel data. Length must be at least `(depth - 1) * plane_stride + (height - 1) * row_stride + width`.
    pixels: &'a [T],
}

impl<T> Copy for Image3DRef<'_, T> {}
impl<T> Clone for Image3DRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Image3DRef<'a, T> {
    /// Creates a new `Image3DRef` with contiguous row storage (row stride equals width, plane stride equals width * height).
    ///
    /// # Parameters
    ///
    /// * `width` — The width of the image in pixels.
    /// * `height` — The height of the image in pixels.
    /// * `depth` — The depth of the image in pixels.
    /// * `pixels` — The underlying pixel data slice.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() < height * width`.
    pub fn new(width: usize, height: usize, depth: usize, pixels: &'a [T]) -> Self {
        assert!(pixels.len() >= depth * height * width);
        Image3DRef {
            width,
            height,
            depth,
            row_stride: width,
            plane_stride: width * height,
            pixels,
        }
    }

    /// Creates a new `Image3DRef` with a custom row stride.
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
    /// Panics if `pixels.len() < (depth - 1) * plane_stride + (height - 1) * row_stride + width`.
    pub fn with_stride(
        width: usize,
        height: usize,
        depth: usize,
        row_stride: usize,
        plane_stride: usize,
        pixels: &'a [T],
    ) -> Self {
        let len = (depth != 0 && height != 0)
            .then(|| (depth - 1) * plane_stride + (height - 1) * row_stride + width)
            .unwrap_or(0);
        assert!(pixels.len() >= len);
        Image3DRef {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            pixels,
        }
    }

    /// Creates a new `Image3DRef` from a single row of pixels.
    pub fn from_row(pixels: &'a [T]) -> Self {
        Image3DRef {
            width: pixels.len(),
            height: 1,
            depth: 1,
            row_stride: pixels.len(),
            plane_stride: pixels.len(),
            pixels,
        }
    }

    /// Creates a new `Image3DRef` from a single XY plane of pixels.
    pub fn from_plane(plane: Image2DRef<'a, T>) -> Self {
        Image3DRef {
            width: plane.width,
            height: plane.height,
            depth: 1,
            row_stride: plane.stride,
            plane_stride: plane.stride * plane.height,
            pixels: plane.pixels,
        }
    }

    /// Creates a new `Image3DRef` by reborrowing its pixel data.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Creates a new `Image3DRef` by reborrowing its pixel data.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Creates a new `Image3DRef` by reborrowing its pixel data.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns a reference to the pixel at coordinates (`x`, `y`, `z`).
    ///
    /// # Panics
    ///
    /// Panics if the computed index `z * plane_stride + y * row_stride + x` is out of bounds.
    pub fn get(&self, x: usize, y: usize, z: usize) -> &'a T {
        assert!(x < self.width);
        assert!(y < self.height);
        assert!(z < self.depth);

        &self.pixels[z * self.plane_stride + y * self.row_stride + x]
    }

    /// Returns a reference to the entire row at coordinates (`y`, `z`).
    pub fn row(&self, y: usize, z: usize) -> &'a [T] {
        assert!(y < self.height);
        assert!(z < self.depth);
        &self.pixels[z * self.plane_stride + y * self.row_stride..][..self.width]
    }

    /// Returns a reference to the entire XY plane at coordinate `z`.
    pub fn get_plane_xy(&self, z: usize) -> Image2DRef<'a, T> {
        assert!(z < self.depth);

        Image2DRef {
            width: self.width,
            height: self.height,
            stride: self.row_stride,
            pixels: &self.pixels[z * self.plane_stride..],
        }
    }

    /// Returns a reference to the entire XZ plane at coordinate `y`.
    pub fn get_plane_xz(&self, y: usize) -> Image2DRef<'a, T> {
        assert!(y < self.height);

        Image2DRef {
            width: self.width,
            height: self.depth,
            stride: self.plane_stride,
            pixels: &self.pixels[y * self.row_stride..],
        }
    }

    /// Returns a reference to a sub-region of XY plane of this image as a new `Image2DRef`.
    ///
    /// The returned view starts at (`x`, `y`, `z`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range_xy(
        &self,
        x: usize,
        y: usize,
        z: usize,
        w: usize,
        h: usize,
    ) -> Image2DRef<'a, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);
        assert!(z < self.depth);

        Image2DRef {
            width: w,
            height: h,
            stride: self.row_stride,
            pixels: &self.pixels[z * self.plane_stride + y * self.row_stride + x..],
        }
    }

    /// Returns a reference to a sub-region of XZ plane of this image as a new `Image2DRef`.
    ///
    /// The returned view starts at (`x`, `y`, `z`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range_xz(
        &self,
        x: usize,
        y: usize,
        z: usize,
        w: usize,
        h: usize,
    ) -> Image2DRef<'a, T> {
        assert!(x + w <= self.width);
        assert!(y < self.height);
        assert!(z + h <= self.depth);

        Image2DRef {
            width: w,
            height: h,
            stride: self.plane_stride,
            pixels: &self.pixels[z * self.plane_stride + y * self.row_stride + x..],
        }
    }

    /// Returns a sub-region of this image as a new `Image3DRef`.
    ///
    /// The returned view starts at (`x`, `y`, `z`) and has dimensions `w` × `h` × `d`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range(
        &self,
        x: usize,
        y: usize,
        z: usize,
        w: usize,
        h: usize,
        d: usize,
    ) -> Image3DRef<'a, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);
        assert!(z + d <= self.depth);

        Image3DRef {
            width: w,
            height: h,
            depth: d,
            row_stride: self.row_stride,
            plane_stride: self.plane_stride,
            pixels: &self.pixels[z * self.plane_stride + y * self.row_stride + x..],
        }
    }

    /// Returns the raw underlying pixel slice.
    pub fn pixels(&self) -> &'a [T] {
        &self.pixels
    }

    /// Returns an iterator over all planes in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over planes in an image with arbitrary strides,
    /// use `(0..self.depth()).map(|z| self.get_plane_xy(z))` instead.
    pub fn iter_planes(&self) -> impl DoubleEndedIterator<Item = Image2DRef<'a, T>> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            pixels,
        } = *self;

        pixels[..plane_stride * depth]
            .chunks_exact(plane_stride)
            .map(move |plane| Image2DRef {
                width,
                height,
                stride: row_stride,
                pixels: plane,
            })
    }

    /// Returns an iterator over all rows in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over rows in an image with arbitrary strides,
    /// use `(0..self.depth()).flat_map(|z| self.get_plane_xy(z).iter_rows())` instead.
    pub fn iter_rows(&self) -> impl DoubleEndedIterator<Item = &'a [T]> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            pixels,
        } = *self;

        pixels[..plane_stride * depth]
            .chunks_exact(plane_stride)
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact(row_stride))
            .map(move |row| &row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over pixels in an image with arbitrary strides,
    /// use `(0..self.depth()).flat_map(|z| self.get_plane_xy(z).iter_pixels())` instead.
    pub fn iter_pixels(&self) -> impl DoubleEndedIterator<Item = &'a T> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            pixels,
        } = *self;

        pixels[..plane_stride * depth]
            .chunks_exact(plane_stride)
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact(row_stride))
            .flat_map(move |row| &row[..width])
    }
}

/// A mutable, non-owning 3D view into a pixel buffer.
///
/// `Image3DMut` borrows a flat slice of pixel data mutably and interprets it as a 3D image
/// with the given `width`, `height`, and row `stride`. It supports reading, writing,
/// sub-region extraction, copying from an `Image2DRef`, and random/patch-based initialization.
///
/// # Type Parameters
///
/// * `T` — The pixel type. Must be `Copy` for most operations.
pub struct Image3DMut<'a, T> {
    /// The width of the image view in pixels.
    width: usize,
    /// The height of the image view in pixels.
    height: usize,
    /// The depth of the image view in pixels.
    depth: usize,
    /// The number of `T` elements between the start of consecutive rows.
    row_stride: usize,
    /// The number of `T` elements between the start of consecutive planes.
    plane_stride: usize,
    /// The underlying mutable pixel data. Length must be at least `(depth - 1) * plane_stride + (height - 1) * row_stride + width`.
    pixels: &'a mut [T],
}

impl<'a, T> Image3DMut<'a, T> {
    /// Creates a new `Image3DMut` with contiguous row storage (stride equals width).
    ///
    /// # Parameters
    ///
    /// * `width` — The width of the image in pixels.
    /// * `height` — The height of the image in pixels.
    /// * `depth` — The depth of the image in pixels.
    /// * `pixels` — The underlying mutable pixel data slice.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() < height * width * depth`.
    pub fn new(width: usize, height: usize, depth: usize, pixels: &'a mut [T]) -> Self {
        assert!(pixels.len() >= height * width * depth);
        Image3DMut {
            width,
            height,
            depth,
            row_stride: width,
            plane_stride: width * height,
            pixels,
        }
    }

    /// Creates a new `Image3DMut` with a custom row stride.
    ///
    /// Use this when the pixel data has padding between rows, e.g. when
    /// the view represents a sub-region of a larger image.
    ///
    /// # Parameters
    ///
    /// * `width` — The width of the image in pixels.
    /// * `height` — The height of the image in pixels.
    /// * `depth` — The depth of the image in pixels.
    /// * `pixels` — The underlying mutable pixel data slice.
    /// * `row_stride` — The number of `T` elements between the start of consecutive rows.
    /// * `plane_stride` — The number of `T` elements between the start of consecutive planes.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() < (depth - 1) * plane_stride + (height - 1) * row_stride + width`.
    pub fn with_stride(
        width: usize,
        height: usize,
        depth: usize,
        row_stride: usize,
        plane_stride: usize,
        pixels: &'a mut [T],
    ) -> Self {
        let len = (depth != 0 && height != 0)
            .then(|| (depth - 1) * plane_stride + (height - 1) * row_stride + width)
            .unwrap_or(0);
        assert!(pixels.len() >= len);
        Image3DMut {
            width,
            height,
            depth,
            pixels,
            row_stride,
            plane_stride,
        }
    }

    /// Creates a new `Image3DMut` from a single row of pixels.
    pub fn from_row(pixels: &'a mut [T]) -> Self {
        Image3DMut {
            width: pixels.len(),
            height: 1,
            depth: 1,
            row_stride: pixels.len(),
            plane_stride: pixels.len(),
            pixels,
        }
    }

    /// Creates a new `Image3DMut` from a single XY plane of pixels.
    pub fn from_plane(plane: Image2DMut<'a, T>) -> Self {
        Image3DMut {
            width: plane.width,
            height: plane.height,
            depth: 1,
            row_stride: plane.stride,
            plane_stride: plane.stride * plane.height,
            pixels: plane.pixels,
        }
    }

    /// Creates a new `Image3DMut` by reborrowing its pixel data mutably.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Creates a new `Image3DMut` by reborrowing its pixel data mutably.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Creates a new `Image3DMut` by reborrowing its pixel data mutably.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Creates a new `Image3DMut` by reborrowing its pixel data immutably.
    pub fn as_ref(&self) -> Image3DRef<'_, T> {
        Image3DRef {
            width: self.width,
            height: self.height,
            depth: self.depth,
            row_stride: self.row_stride,
            plane_stride: self.plane_stride,
            pixels: &*self.pixels,
        }
    }

    /// Creates a new `Image3DMut` by reborrowing its pixel data mutably.
    pub fn as_mut(&mut self) -> Image3DMut<'_, T> {
        Image3DMut {
            width: self.width,
            height: self.height,
            depth: self.depth,
            row_stride: self.row_stride,
            plane_stride: self.plane_stride,
            pixels: &mut *self.pixels,
        }
    }

    /// Returns an immutable reference to the pixel at coordinates (`x`, `y`, `z`).
    ///
    /// # Panics
    ///
    /// Panics if the computed index `z * plane_stride + y * row_stride + x` is out of bounds.
    pub fn get(&self, x: usize, y: usize, z: usize) -> &T {
        assert!(x < self.width);
        assert!(y < self.height);
        assert!(z < self.depth);

        &self.pixels[z * self.plane_stride + y * self.row_stride + x]
    }

    /// Returns a mutable reference to the pixel at coordinates (`x`, `y`, `z`).
    ///
    /// # Panics
    ///
    /// Panics if the computed index `z * plane_stride + y * row_stride + x` is out of bounds.
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut T {
        assert!(x < self.width);
        assert!(y < self.height);
        assert!(z < self.depth);

        &mut self.pixels[z * self.plane_stride + y * self.row_stride + x]
    }

    /// Returns an immutable reference to the entire row at coordinates (`y`, `z`).
    pub fn row(&self, y: usize, z: usize) -> &[T] {
        assert!(y < self.height);
        assert!(z < self.depth);
        &self.pixels[z * self.plane_stride + y * self.row_stride..][..self.width]
    }

    /// Returns a mutable reference to the entire row at coordinates (`y`, `z`).
    pub fn row_mut(&mut self, y: usize, z: usize) -> &mut [T] {
        assert!(y < self.height);
        assert!(z < self.depth);
        &mut self.pixels[z * self.plane_stride + y * self.row_stride..][..self.width]
    }

    /// Returns an immutable reference to the entire XY plane at coordinate `z`.
    pub fn get_plane_xy(&self, z: usize) -> Image2DRef<'_, T> {
        assert!(z < self.depth);

        Image2DRef {
            width: self.width,
            height: self.height,
            stride: self.row_stride,
            pixels: &self.pixels[z * self.plane_stride..],
        }
    }

    /// Returns a mutable reference to the entire XY plane at coordinate `z`.
    pub fn get_plane_xy_mut(&mut self, z: usize) -> Image2DMut<'_, T> {
        assert!(z < self.depth);

        Image2DMut {
            width: self.width,
            height: self.height,
            stride: self.row_stride,
            pixels: &mut self.pixels[z * self.plane_stride..],
        }
    }

    /// Returns an immutable reference to the entire XZ plane at coordinate `y`.
    pub fn get_plane_xz(&self, y: usize) -> Image2DRef<'_, T> {
        assert!(y < self.height);

        Image2DRef {
            width: self.width,
            height: self.depth,
            stride: self.plane_stride,
            pixels: &self.pixels[y * self.row_stride..],
        }
    }

    /// Returns a mutable reference to the entire XZ plane at coordinate `y`.
    pub fn get_plane_xz_mut(&mut self, y: usize) -> Image2DMut<'_, T> {
        assert!(y < self.height);

        Image2DMut {
            width: self.width,
            height: self.depth,
            stride: self.plane_stride,
            pixels: &mut self.pixels[y * self.row_stride..],
        }
    }

    /// Returns an immutable reference to a sub-region of XY plane of this image as a new `Image2DRef`.
    ///
    /// The returned view starts at (`x`, `y`, `z`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range_xy(
        &self,
        x: usize,
        y: usize,
        z: usize,
        w: usize,
        h: usize,
    ) -> Image2DRef<'_, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);
        assert!(z < self.depth);

        Image2DRef {
            width: w,
            height: h,
            stride: self.row_stride,
            pixels: &self.pixels[z * self.plane_stride + y * self.row_stride + x..],
        }
    }

    /// Returns a mutable reference to a sub-region of XY plane of this image as a new `Image2DMut`.
    ///
    /// The returned view starts at (`x`, `y`, `z`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range_xy_mut(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        w: usize,
        h: usize,
    ) -> Image2DMut<'_, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);
        assert!(z < self.depth);

        Image2DMut {
            width: w,
            height: h,
            stride: self.row_stride,
            pixels: &mut self.pixels[z * self.plane_stride + y * self.row_stride + x..],
        }
    }

    /// Returns an immutable reference to a sub-region of XZ plane of this image as a new `Image2DRef`.
    ///
    /// The returned view starts at (`x`, `y`, `z`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range_xz(
        &self,
        x: usize,
        y: usize,
        z: usize,
        w: usize,
        h: usize,
    ) -> Image2DRef<'_, T> {
        assert!(x + w <= self.width);
        assert!(y < self.height);
        assert!(z + h <= self.depth);

        Image2DRef {
            width: w,
            height: h,
            stride: self.plane_stride,
            pixels: &self.pixels[z * self.plane_stride + y * self.row_stride + x..],
        }
    }

    /// Returns a mutable reference to a sub-region of XZ plane of this image as a new `Image2DMut`.
    ///
    /// The returned view starts at (`x`, `y`, `z`) and has dimensions `w` × `h`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range_xz_mut(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        w: usize,
        h: usize,
    ) -> Image2DMut<'_, T> {
        assert!(x + w <= self.width);
        assert!(y < self.height);
        assert!(z + h <= self.depth);

        Image2DMut {
            width: w,
            height: h,
            stride: self.plane_stride,
            pixels: &mut self.pixels[z * self.plane_stride + y * self.row_stride + x..],
        }
    }

    /// Returns a sub-region of this image as a new `Image3DRef`.
    ///
    /// The returned view starts at (`x`, `y`, `z`) and has dimensions `w` × `h` × `d`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range(
        &self,
        x: usize,
        y: usize,
        z: usize,
        w: usize,
        h: usize,
        d: usize,
    ) -> Image3DRef<'_, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);
        assert!(z + d <= self.depth);

        Image3DRef {
            width: w,
            height: h,
            depth: d,
            row_stride: self.row_stride,
            plane_stride: self.plane_stride,
            pixels: &self.pixels[z * self.plane_stride + y * self.row_stride + x..],
        }
    }

    /// Returns a sub-region of this image as a new `Image3DMut`.
    ///
    /// The returned view starts at (`x`, `y`, `z`) and has dimensions `w` × `h` × `d`.
    /// It shares the same underlying pixel data and stride.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region extends beyond the image bounds.
    pub fn get_range_mut(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        w: usize,
        h: usize,
        d: usize,
    ) -> Image3DMut<'_, T> {
        assert!(x + w <= self.width);
        assert!(y + h <= self.height);
        assert!(z + d <= self.depth);

        Image3DMut {
            width: w,
            height: h,
            depth: d,
            row_stride: self.row_stride,
            plane_stride: self.plane_stride,
            pixels: &mut self.pixels[z * self.plane_stride + y * self.row_stride + x..],
        }
    }

    /// Returns the raw underlying pixel slice.
    pub fn pixels(&self) -> &[T] {
        &self.pixels
    }

    /// Returns the raw underlying pixel slice.
    pub fn pixels_mut(&mut self) -> &mut [T] {
        &mut self.pixels
    }

    /// Returns an iterator over all planes in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over planes in an image with arbitrary strides,
    /// use `(0..self.depth()).map(|z| self.get_plane_xy(z))` instead.
    pub fn iter_planes(&self) -> impl DoubleEndedIterator<Item = Image2DRef<'_, T>> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            ref pixels,
        } = *self;

        pixels[..plane_stride * depth]
            .chunks_exact(plane_stride)
            .map(move |plane| Image2DRef {
                width,
                height,
                stride: row_stride,
                pixels: plane,
            })
    }

    /// Returns an iterator over all planes in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    pub fn iter_planes_mut(&mut self) -> impl DoubleEndedIterator<Item = Image2DMut<'_, T>> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            ref mut pixels,
        } = *self;

        pixels[..plane_stride * depth]
            .chunks_exact_mut(plane_stride)
            .map(move |plane| Image2DMut {
                width,
                height,
                stride: row_stride,
                pixels: plane,
            })
    }

    /// Returns an iterator over all planes in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    pub fn into_iter_planes(self) -> impl DoubleEndedIterator<Item = Image2DMut<'a, T>> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            pixels,
        } = self;

        pixels[..plane_stride * depth]
            .chunks_exact_mut(plane_stride)
            .map(move |plane| Image2DMut {
                width,
                height,
                stride: row_stride,
                pixels: plane,
            })
    }

    /// Returns an iterator over all rows in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    /// In case you want to iterate over rows in an image with arbitrary strides,
    /// use `(0..self.depth()).flat_map(|z| self.get_plane_xy(z).iter_rows())` instead.
    pub fn iter_rows(&self) -> impl DoubleEndedIterator<Item = &[T]> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            ref pixels,
        } = *self;

        pixels[..plane_stride * depth]
            .chunks_exact(plane_stride)
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact(row_stride))
            .map(move |row| &row[..width])
    }

    /// Returns an iterator over all rows in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    pub fn iter_rows_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut [T]> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            ref mut pixels,
        } = *self;

        pixels[..plane_stride * depth]
            .chunks_exact_mut(plane_stride)
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact_mut(row_stride))
            .map(move |row| &mut row[..width])
    }

    /// Returns an iterator over all rows in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    pub fn into_iter_rows(self) -> impl DoubleEndedIterator<Item = &'a mut [T]> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            pixels,
        } = self;

        pixels[..plane_stride * depth]
            .chunks_exact_mut(plane_stride)
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact_mut(row_stride))
            .map(move |row| &mut row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    /// In case you want to iterate over pixels in an image with arbitrary strides,
    /// use `(0..self.depth()).flat_map(|z| self.get_plane_xy(z).iter_pixels())` instead.
    pub fn iter_pixels(&self) -> impl DoubleEndedIterator<Item = &T> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            ref pixels,
        } = *self;

        pixels[..plane_stride * depth]
            .chunks_exact(plane_stride)
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact(row_stride))
            .flat_map(move |row| &row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    pub fn iter_pixels_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut T> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            ref mut pixels,
        } = *self;

        pixels[..plane_stride * depth]
            .chunks_exact_mut(plane_stride)
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact_mut(row_stride))
            .flat_map(move |row| &mut row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    pub fn into_iter_pixels(self) -> impl DoubleEndedIterator<Item = &'a mut T> {
        let Self {
            width,
            height,
            depth,
            row_stride,
            plane_stride,
            pixels,
        } = self;

        pixels[..plane_stride * depth]
            .chunks_exact_mut(plane_stride)
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact_mut(row_stride))
            .flat_map(move |row| &mut row[..width])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extent {
    D1 {
        width: usize,
    },
    D2 {
        width: usize,
        height: usize,
    },
    D3 {
        width: usize,
        height: usize,
        depth: usize,
    },
    D1Array {
        width: usize,
        layers: usize,
    },
    D2Array {
        width: usize,
        height: usize,
        layers: usize,
    },
}

impl Extent {
    pub fn width(&self) -> usize {
        match *self {
            Extent::D1 { width } => width,
            Extent::D2 { width, .. } => width,
            Extent::D3 { width, .. } => width,
            Extent::D1Array { width, .. } => width,
            Extent::D2Array { width, .. } => width,
        }
    }

    pub fn height(&self) -> usize {
        match *self {
            Extent::D1 { .. } => 1,
            Extent::D2 { height, .. } => height,
            Extent::D3 { height, .. } => height,
            Extent::D1Array { .. } => 1,
            Extent::D2Array { height, .. } => height,
        }
    }

    pub fn depth(&self) -> usize {
        match *self {
            Extent::D1 { .. } => 1,
            Extent::D2 { .. } => 1,
            Extent::D3 { depth, .. } => depth,
            Extent::D1Array { .. } => 1,
            Extent::D2Array { .. } => 1,
        }
    }

    pub fn layers(&self) -> usize {
        match *self {
            Extent::D1 { .. } => 1,
            Extent::D2 { .. } => 1,
            Extent::D3 { .. } => 1,
            Extent::D1Array { layers, .. } => layers,
            Extent::D2Array { layers, .. } => layers,
        }
    }

    pub fn dimensions(self) -> Dimensions {
        match self {
            Extent::D1 { .. } => Dimensions::D1,
            Extent::D2 { .. } => Dimensions::D2,
            Extent::D3 { .. } => Dimensions::D3,
            Extent::D1Array { .. } => Dimensions::D1,
            Extent::D2Array { .. } => Dimensions::D2,
        }
    }

    pub fn raw_size(self) -> [usize; 3] {
        match self {
            Extent::D1 { width } => [width, 1, 1],
            Extent::D2 { width, height } => [width, height, 1],
            Extent::D3 {
                width,
                height,
                depth,
            } => [width, height, depth],
            Extent::D1Array { width, layers } => [width, 1, layers],
            Extent::D2Array {
                width,
                height,
                layers,
            } => [width, height, layers],
        }
    }

    pub fn from_raw_size(value: [usize; 3], dimensions: Dimensions) -> Option<Self> {
        match dimensions {
            Dimensions::D1 => {
                if value[1] != 1 || value[2] != 1 {
                    return None;
                }
                Some(Extent::D1 { width: value[0] })
            }
            Dimensions::D2 => {
                if value[2] != 1 {
                    return None;
                }
                Some(Extent::D2 {
                    width: value[0],
                    height: value[1],
                })
            }
            Dimensions::D3 => Some(Extent::D3 {
                width: value[0],
                height: value[1],
                depth: value[2],
            }),
            Dimensions::D1Array => {
                if value[1] != 1 {
                    return None;
                }
                Some(Extent::D1Array {
                    width: value[0],
                    layers: value[2],
                })
            }
            Dimensions::D2Array => Some(Extent::D2Array {
                width: value[0],
                height: value[1],
                layers: value[2],
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dimensions {
    D1,
    D2,
    D3,
    D1Array,
    D2Array,
}

/// A non-owning view into a pixel buffer interpreted as an image of certain dimensions and extent.
pub struct ImageRef<'a, T> {
    dimensions: Dimensions,
    extent: [usize; 3],
    stride: [usize; 2],
    pixels: &'a [T],
}

impl<'a, T> Copy for ImageRef<'a, T> {}
impl<'a, T> Clone for ImageRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> ImageRef<'a, T> {
    pub fn new(dimensions: Dimensions, extent: [usize; 3], pixels: &'a [T]) -> Self {
        let len = match dimensions {
            Dimensions::D1 => {
                assert_eq!(extent[1], 1);
                assert_eq!(extent[2], 1);
                extent[0]
            }
            Dimensions::D2 => {
                assert_eq!(extent[2], 1);
                extent[0] * extent[1]
            }
            Dimensions::D3 => extent[0] * extent[1] * extent[2],
            Dimensions::D1Array => {
                assert_eq!(extent[2], 1);
                extent[0] * extent[1]
            }
            Dimensions::D2Array => extent[0] * extent[1] * extent[2],
        };

        assert_eq!(pixels.len(), len);

        ImageRef {
            dimensions,
            extent,
            stride: [extent[0], extent[0] * extent[1]],
            pixels,
        }
    }

    pub fn with_stride(
        dimensions: Dimensions,
        extent: [usize; 3],
        stride: [usize; 2],
        pixels: &'a [T],
    ) -> Self {
        let len = match dimensions {
            Dimensions::D1 => {
                assert_eq!(extent[1], 1);
                assert_eq!(extent[2], 1);
                extent[0]
            }
            Dimensions::D2 | Dimensions::D1Array => {
                assert_eq!(extent[2], 1);
                (extent[1] != 0 && extent[0] != 0)
                    .then(|| (extent[1] - 1) * stride[0] + extent[0])
                    .unwrap_or(0)
            }
            Dimensions::D3 | Dimensions::D2Array => {
                (extent[2] != 0 && extent[1] != 0 && extent[0] != 0)
                    .then(|| (extent[2] - 1) * stride[1] + (extent[1] - 1) * stride[0] + extent[0])
                    .unwrap_or(0)
            }
        };

        assert_eq!(pixels.len(), len);

        ImageRef {
            dimensions,
            extent,
            stride,
            pixels,
        }
    }

    pub fn new_1d(width: usize, pixels: &'a [T]) -> Self {
        ImageRef::new(Dimensions::D1, [width, 1, 1], pixels)
    }

    pub fn new_2d(width: usize, height: usize, pixels: &'a [T]) -> Self {
        ImageRef::new(Dimensions::D2, [width, height, 1], pixels)
    }

    pub fn new_3d(width: usize, height: usize, depth: usize, pixels: &'a [T]) -> Self {
        ImageRef::new(Dimensions::D3, [width, height, depth], pixels)
    }

    pub fn new_1d_array(width: usize, layers: usize, pixels: &'a [T]) -> Self {
        ImageRef::new(Dimensions::D1Array, [width, layers, 1], pixels)
    }

    pub fn new_2d_array(width: usize, height: usize, layers: usize, pixels: &'a [T]) -> Self {
        ImageRef::new(Dimensions::D2Array, [width, height, layers], pixels)
    }

    pub fn with_stride_2d(width: usize, height: usize, stride: usize, pixels: &'a [T]) -> Self {
        ImageRef::with_stride(Dimensions::D2, [width, height, 1], [stride, 0], pixels)
    }

    pub fn with_stride_3d(
        width: usize,
        height: usize,
        depth: usize,
        row_stride: usize,
        plane_stride: usize,
        pixels: &'a [T],
    ) -> Self {
        ImageRef::with_stride(
            Dimensions::D3,
            [width, height, depth],
            [row_stride, plane_stride],
            pixels,
        )
    }

    pub fn with_stride_1d_array(
        width: usize,
        layers: usize,
        stride: usize,
        pixels: &'a [T],
    ) -> Self {
        ImageRef::with_stride(Dimensions::D1Array, [width, layers, 1], [stride, 0], pixels)
    }

    pub fn with_stride_2d_array(
        width: usize,
        height: usize,
        layers: usize,
        row_stride: usize,
        layer_stride: usize,
        pixels: &'a [T],
    ) -> Self {
        ImageRef::with_stride(
            Dimensions::D2Array,
            [width, height, layers],
            [row_stride, layer_stride],
            pixels,
        )
    }

    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    pub fn extent(&self) -> Extent {
        match self.dimensions {
            Dimensions::D1 => Extent::D1 {
                width: self.extent[0],
            },
            Dimensions::D2 => Extent::D2 {
                width: self.extent[0],
                height: self.extent[1],
            },
            Dimensions::D3 => Extent::D3 {
                width: self.extent[0],
                height: self.extent[1],
                depth: self.extent[2],
            },
            Dimensions::D1Array => Extent::D1Array {
                width: self.extent[0],
                layers: self.extent[1],
            },
            Dimensions::D2Array => Extent::D2Array {
                width: self.extent[0],
                height: self.extent[1],
                layers: self.extent[2],
            },
        }
    }

    pub fn raw_extent(&self) -> [usize; 3] {
        self.extent
    }

    pub fn width(&self) -> usize {
        self.extent[0]
    }

    pub fn height(&self) -> usize {
        match self.dimensions {
            Dimensions::D1 | Dimensions::D1Array => 1,
            _ => self.extent[1],
        }
    }

    pub fn depth(&self) -> usize {
        match self.dimensions {
            Dimensions::D1 | Dimensions::D1Array | Dimensions::D2 | Dimensions::D2Array => 1,
            Dimensions::D3 => self.extent[2],
        }
    }

    /// Returns the number of layers in this image. For non-array types, this will return 1.
    pub fn layers(&self) -> usize {
        match self.dimensions {
            Dimensions::D1 | Dimensions::D2 | Dimensions::D3 => 1,
            Dimensions::D1Array => self.extent[1],
            Dimensions::D2Array => self.extent[2],
        }
    }

    /// Returns a reference to the specified layer of this image. For non-array types, `layer` must be 0.
    /// The returned image reference is a non-array type.
    pub fn layer_ref(&self, layer: usize) -> ImageRef<'a, T> {
        match self.dimensions {
            Dimensions::D1 => {
                assert_eq!(layer, 0);
                ImageRef::new_1d(self.extent[0], self.pixels)
            }
            Dimensions::D2 => {
                assert_eq!(layer, 0);
                ImageRef::new_2d(self.extent[0], self.extent[1], self.pixels)
            }
            Dimensions::D3 => {
                assert_eq!(layer, 0);
                ImageRef::new_3d(self.extent[0], self.extent[1], self.extent[2], self.pixels)
            }
            Dimensions::D1Array => {
                assert!(layer < self.extent[1]);
                ImageRef::new_1d(
                    self.extent[0],
                    &self.pixels[layer * self.extent[0]..][..self.extent[0]],
                )
            }
            Dimensions::D2Array => {
                assert!(layer < self.extent[2]);
                ImageRef::new_2d(
                    self.extent[0],
                    self.extent[1],
                    &self.pixels[layer * self.extent[0] * self.extent[1]..]
                        [..self.extent[0] * self.extent[1]],
                )
            }
        }
    }

    /// Reinterpret this image as a 3D image, regardless of its original dimensions.
    /// 1D images will be treated as having height and depth equal to 1.
    /// 1D array images will be treated as having height equal to the number of layers and depth 1.
    /// 2D images will be treated as having depth equal to 1.
    /// 2D array images will be treated as having depth equal to the number of layers.
    ///
    /// This is useful to generically treat any image as 3D for algorithms that operate on 3D data,
    /// where the original dimensions may be ignored.
    ///
    /// Layers will be treated as depth dimension,
    /// and missing dimensions will be treated as having extent 1.
    pub fn as_ref_3d(&self) -> Image3DRef<'a, T> {
        Image3DRef {
            width: self.extent[0],
            height: self.extent[1],
            depth: self.extent[2],
            row_stride: self.stride[0],
            plane_stride: self.stride[1],
            pixels: self.pixels,
        }
    }

    /// Reinterpret this image as layered 2D image, regardless of its original dimensions
    /// and returns a reference to the specified layer.
    ///
    /// 1D, both array and non-array images will be treated as having height equal to 1.
    /// 3D images will be treated as having layers equal to the depth dimension.
    /// Which is different than reinterpretation as 3D and taking a plane.
    ///
    /// The main difference from `as_ref_3d` is that layers of D1 are interpreted as layers, not height.
    ///
    /// `self.depth() * self.layers()` must be greater than `layer`, otherwise this method will panic.
    pub fn plane_ref(&self, layer: usize) -> Image2DRef<'a, T> {
        match self.dimensions {
            Dimensions::D1 | Dimensions::D1Array => {
                assert!(layer < self.extent[1]);
                Image2DRef {
                    width: self.extent[0],
                    height: 1,
                    stride: self.stride[0],
                    pixels: &self.pixels[layer * self.stride[0]..],
                }
            }
            Dimensions::D2 | Dimensions::D2Array => {
                assert!(layer < self.extent[2]);
                Image2DRef {
                    width: self.extent[0],
                    height: self.extent[1],
                    stride: self.stride[0],
                    pixels: &self.pixels[layer * self.stride[1]..],
                }
            }
            Dimensions::D3 => {
                assert!(layer < self.extent[2]);
                Image2DRef {
                    width: self.extent[0],
                    height: self.extent[1],
                    stride: self.stride[0],
                    pixels: &self.pixels[layer * self.stride[1]..],
                }
            }
        }
    }

    /// Returns an iterator over all planes in this image in row-major order.
    /// If image is 1D or 1D array, this will return a single-row planes.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over planes in an image with arbitrary strides,
    /// use `(0..self.depth()).map(|z| self.get_plane_xy(z))` instead.
    pub fn iter_planes(&self) -> impl DoubleEndedIterator<Item = Image2DRef<'a, T>> {
        let [width, height, depth, layers] = total_extent(self.dimensions, self.extent);
        let [row_stride, plane_stride, layer_stride] = total_stride(self.dimensions, self.stride);

        self.pixels[..layer_stride * layers]
            .chunks_exact(layer_stride)
            .flat_map(move |layer| layer[..plane_stride * depth].chunks_exact(plane_stride))
            .map(move |plane| Image2DRef {
                width,
                height,
                stride: row_stride,
                pixels: plane,
            })
    }

    /// Returns an iterator over all rows in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over rows in an image with arbitrary strides,
    /// use `(0..self.depth()).flat_map(|z| self.get_plane_xy(z).iter_rows())` instead.
    pub fn iter_rows(&self) -> impl DoubleEndedIterator<Item = &'a [T]> {
        let [width, height, depth, layers] = total_extent(self.dimensions, self.extent);
        let [row_stride, plane_stride, layer_stride] = total_stride(self.dimensions, self.stride);

        self.pixels[..layer_stride * layers]
            .chunks_exact(layer_stride)
            .flat_map(move |layer| layer[..plane_stride * depth].chunks_exact(plane_stride))
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact(row_stride))
            .map(move |row| &row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over pixels in an image with arbitrary strides,
    /// use `(0..self.depth()).flat_map(|z| self.get_plane_xy(z).iter_pixels())` instead.
    pub fn iter_pixels(&self) -> impl DoubleEndedIterator<Item = &'a T> {
        let [width, height, depth, layers] = total_extent(self.dimensions, self.extent);
        let [row_stride, plane_stride, layer_stride] = total_stride(self.dimensions, self.stride);

        self.pixels[..layer_stride * layers]
            .chunks_exact(layer_stride)
            .flat_map(move |layer| layer[..plane_stride * depth].chunks_exact(plane_stride))
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact(row_stride))
            .flat_map(move |row| &row[..width])
    }
}

/// A non-owning view into a pixel buffer interpreted as an image of certain dimensions and extent.
pub struct ImageMut<'a, T> {
    dimensions: Dimensions,
    extent: [usize; 3],
    stride: [usize; 2],
    pixels: &'a mut [T],
}

impl<'a, T> ImageMut<'a, T> {
    pub fn new(dimensions: Dimensions, extent: [usize; 3], pixels: &'a mut [T]) -> Self {
        let len = match dimensions {
            Dimensions::D1 => {
                assert_eq!(extent[1], 1);
                assert_eq!(extent[2], 1);
                extent[0]
            }
            Dimensions::D2 => {
                assert_eq!(extent[2], 1);
                extent[0] * extent[1]
            }
            Dimensions::D3 => extent[0] * extent[1] * extent[2],
            Dimensions::D1Array => {
                assert_eq!(extent[2], 1);
                extent[0] * extent[1]
            }
            Dimensions::D2Array => extent[0] * extent[1] * extent[2],
        };

        assert_eq!(pixels.len(), len);

        ImageMut {
            dimensions,
            extent,
            stride: [extent[0], extent[0] * extent[1]],
            pixels,
        }
    }

    pub fn with_stride(
        dimensions: Dimensions,
        extent: [usize; 3],
        stride: [usize; 2],
        pixels: &'a mut [T],
    ) -> Self {
        let len = match dimensions {
            Dimensions::D1 => {
                assert_eq!(extent[1], 1);
                assert_eq!(extent[2], 1);
                extent[0]
            }
            Dimensions::D2 | Dimensions::D1Array => {
                assert_eq!(extent[2], 1);
                (extent[1] != 0 && extent[0] != 0)
                    .then(|| (extent[1] - 1) * stride[0] + extent[0])
                    .unwrap_or(0)
            }
            Dimensions::D3 | Dimensions::D2Array => {
                (extent[2] != 0 && extent[1] != 0 && extent[0] != 0)
                    .then(|| (extent[2] - 1) * stride[1] + (extent[1] - 1) * stride[0] + extent[0])
                    .unwrap_or(0)
            }
        };

        assert_eq!(pixels.len(), len);

        ImageMut {
            dimensions,
            extent,
            stride,
            pixels,
        }
    }

    pub fn new_1d(width: usize, pixels: &'a mut [T]) -> Self {
        ImageMut::new(Dimensions::D1, [width, 1, 1], pixels)
    }

    pub fn new_2d(width: usize, height: usize, pixels: &'a mut [T]) -> Self {
        ImageMut::new(Dimensions::D2, [width, height, 1], pixels)
    }

    pub fn new_3d(width: usize, height: usize, depth: usize, pixels: &'a mut [T]) -> Self {
        ImageMut::new(Dimensions::D3, [width, height, depth], pixels)
    }

    pub fn new_1d_array(width: usize, layers: usize, pixels: &'a mut [T]) -> Self {
        ImageMut::new(Dimensions::D1Array, [width, layers, 1], pixels)
    }

    pub fn new_2d_array(width: usize, height: usize, layers: usize, pixels: &'a mut [T]) -> Self {
        ImageMut::new(Dimensions::D2Array, [width, height, layers], pixels)
    }

    pub fn with_stride_2d(width: usize, height: usize, stride: usize, pixels: &'a mut [T]) -> Self {
        ImageMut::with_stride(Dimensions::D2, [width, height, 1], [stride, 0], pixels)
    }

    pub fn with_stride_3d(
        width: usize,
        height: usize,
        depth: usize,
        row_stride: usize,
        plane_stride: usize,
        pixels: &'a mut [T],
    ) -> Self {
        ImageMut::with_stride(
            Dimensions::D3,
            [width, height, depth],
            [row_stride, plane_stride],
            pixels,
        )
    }

    pub fn with_stride_1d_array(
        width: usize,
        layers: usize,
        stride: usize,
        pixels: &'a mut [T],
    ) -> Self {
        ImageMut::with_stride(Dimensions::D1Array, [width, layers, 1], [stride, 0], pixels)
    }

    pub fn with_stride_2d_array(
        width: usize,
        height: usize,
        layers: usize,
        row_stride: usize,
        layer_stride: usize,
        pixels: &'a mut [T],
    ) -> Self {
        ImageMut::with_stride(
            Dimensions::D2Array,
            [width, height, layers],
            [row_stride, layer_stride],
            pixels,
        )
    }

    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    pub fn extent(&self) -> [usize; 3] {
        self.extent
    }

    /// Creates a new `ImageRef` by reborrowing its pixel data immutably.
    pub fn as_ref(&self) -> ImageRef<'_, T> {
        ImageRef {
            dimensions: self.dimensions,
            extent: self.extent,
            stride: self.stride,
            pixels: &*self.pixels,
        }
    }

    /// Creates a new `ImageMut` by reborrowing its pixel data mutably.
    pub fn as_mut(&mut self) -> ImageMut<'_, T> {
        ImageMut {
            dimensions: self.dimensions,
            extent: self.extent,
            stride: self.stride,
            pixels: &mut *self.pixels,
        }
    }

    /// Returns the number of layers in this image. For non-array types, this will return 1.
    pub fn layers(&self) -> usize {
        match self.dimensions {
            Dimensions::D1 | Dimensions::D2 | Dimensions::D3 => 1,
            Dimensions::D1Array => self.extent[1],
            Dimensions::D2Array => self.extent[2],
        }
    }

    /// Returns a reference to the specified layer of this image. For non-array types, `layer` must be 0.
    /// The returned image reference is a non-array type.
    pub fn layer_ref(&self, layer: usize) -> ImageRef<'_, T> {
        match self.dimensions {
            Dimensions::D1 => {
                assert_eq!(layer, 0);
                ImageRef::new_1d(self.extent[0], self.pixels)
            }
            Dimensions::D2 => {
                assert_eq!(layer, 0);
                ImageRef::new_2d(self.extent[0], self.extent[1], self.pixels)
            }
            Dimensions::D3 => {
                assert_eq!(layer, 0);
                ImageRef::new_3d(self.extent[0], self.extent[1], self.extent[2], self.pixels)
            }
            Dimensions::D1Array => {
                assert!(layer < self.extent[1]);
                ImageRef::new_1d(
                    self.extent[0],
                    &self.pixels[layer * self.extent[0]..][..self.extent[0]],
                )
            }
            Dimensions::D2Array => {
                assert!(layer < self.extent[2]);
                ImageRef::new_2d(
                    self.extent[0],
                    self.extent[1],
                    &self.pixels[layer * self.extent[0] * self.extent[1]..]
                        [..self.extent[0] * self.extent[1]],
                )
            }
        }
    }

    /// Returns a mutable reference to the specified layer of this image. For non-array types, `layer` must be 0.
    /// The returned image reference is a non-array type.
    pub fn layer_mut(&mut self, layer: usize) -> ImageMut<'_, T> {
        match self.dimensions {
            Dimensions::D1 => {
                assert_eq!(layer, 0);
                ImageMut::new_1d(self.extent[0], &mut *self.pixels)
            }
            Dimensions::D2 => {
                assert_eq!(layer, 0);
                ImageMut::new_2d(self.extent[0], self.extent[1], &mut *self.pixels)
            }
            Dimensions::D3 => {
                assert_eq!(layer, 0);
                ImageMut::new_3d(
                    self.extent[0],
                    self.extent[1],
                    self.extent[2],
                    &mut *self.pixels,
                )
            }
            Dimensions::D1Array => {
                assert!(layer < self.extent[1]);
                ImageMut::new_1d(
                    self.extent[0],
                    &mut self.pixels[layer * self.extent[0]..][..self.extent[0]],
                )
            }
            Dimensions::D2Array => {
                assert!(layer < self.extent[2]);
                ImageMut::new_2d(
                    self.extent[0],
                    self.extent[1],
                    &mut self.pixels[layer * self.extent[0] * self.extent[1]..]
                        [..self.extent[0] * self.extent[1]],
                )
            }
        }
    }

    /// Reinterpret this image as a 3D image immutable reference, regardless of its original dimensions.
    ///
    /// This is useful to generically treat any image as 3D for algorithms that operate on 3D data,
    /// where the original dimensions may be ignored.
    ///
    /// Layers will be treated as next dimension,
    /// and missing dimensions will be treated as having extent 1.
    pub fn as_ref_3d(&self) -> Image3DRef<'_, T> {
        Image3DRef {
            width: self.extent[0],
            height: self.extent[1],
            depth: self.extent[2],
            row_stride: self.stride[0],
            plane_stride: self.stride[1],
            pixels: &*self.pixels,
        }
    }

    /// Reinterpret this image as a 3D image mutable reference, regardless of its original dimensions.
    ///
    /// This is useful to generically treat any image as 3D for algorithms that operate on 3D data,
    /// where the original dimensions may be ignored.
    ///
    /// Layers will be treated as next dimension,
    /// and missing dimensions will be treated as having extent 1.
    pub fn as_mut_3d(&mut self) -> Image3DMut<'_, T> {
        Image3DMut {
            width: self.extent[0],
            height: self.extent[1],
            depth: self.extent[2],
            row_stride: self.stride[0],
            plane_stride: self.stride[1],
            pixels: &mut *self.pixels,
        }
    }

    /// Returns an iterator over all planes in this image in row-major order.
    /// If image is 1D or 1D array, this will return a single-row planes.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over planes in an image with arbitrary strides,
    /// use `(0..self.depth()).map(|z| self.get_plane_xy(z))` instead.
    pub fn iter_planes(&self) -> impl DoubleEndedIterator<Item = Image2DRef<'_, T>> {
        let [width, height, depth, layers] = total_extent(self.dimensions, self.extent);
        let [row_stride, plane_stride, layer_stride] = total_stride(self.dimensions, self.stride);

        self.pixels[..layer_stride * layers]
            .chunks_exact(layer_stride)
            .flat_map(move |layer| layer[..plane_stride * depth].chunks_exact(plane_stride))
            .map(move |plane| Image2DRef {
                width,
                height,
                stride: row_stride,
                pixels: plane,
            })
    }

    /// Returns an iterator over all planes in this image in row-major order.
    /// If image is 1D or 1D array, this will return a single-row planes.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over planes in an image with arbitrary strides,
    /// use `(0..self.depth()).map(|z| self.get_plane_xy(z))` instead.
    pub fn iter_planes_mut(&mut self) -> impl DoubleEndedIterator<Item = Image2DMut<'_, T>> {
        let [width, height, depth, layers] = total_extent(self.dimensions, self.extent);
        let [row_stride, plane_stride, layer_stride] = total_stride(self.dimensions, self.stride);

        self.pixels[..layer_stride * layers]
            .chunks_exact_mut(layer_stride)
            .flat_map(move |layer| layer[..plane_stride * depth].chunks_exact_mut(plane_stride))
            .map(move |plane| Image2DMut {
                width,
                height,
                stride: row_stride,
                pixels: plane,
            })
    }

    /// Returns an iterator over all planes in this image in row-major order.
    /// If image is 1D or 1D array, this will return a single-row planes.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over planes in an image with arbitrary strides,
    /// use `(0..self.depth()).map(|z| self.get_plane_xy(z))` instead.
    pub fn into_iter_planes(self) -> impl DoubleEndedIterator<Item = Image2DMut<'a, T>> {
        let [width, height, depth, layers] = total_extent(self.dimensions, self.extent);
        let [row_stride, plane_stride, layer_stride] = total_stride(self.dimensions, self.stride);

        self.pixels[..layer_stride * layers]
            .chunks_exact_mut(layer_stride)
            .flat_map(move |layer| layer[..plane_stride * depth].chunks_exact_mut(plane_stride))
            .map(move |plane| Image2DMut {
                width,
                height,
                stride: row_stride,
                pixels: plane,
            })
    }

    /// Returns an iterator over all rows in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over rows in an image with arbitrary strides,
    /// use `(0..self.depth()).flat_map(|z| self.get_plane_xy(z).iter_rows())` instead.
    pub fn iter_rows(&self) -> impl DoubleEndedIterator<Item = &'_ [T]> {
        let [width, height, depth, layers] = total_extent(self.dimensions, self.extent);
        let [row_stride, plane_stride, layer_stride] = total_stride(self.dimensions, self.stride);

        self.pixels[..layer_stride * layers]
            .chunks_exact(layer_stride)
            .flat_map(move |layer| layer[..plane_stride * depth].chunks_exact(plane_stride))
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact(row_stride))
            .map(move |row| &row[..width])
    }

    /// Returns an iterator over all pixels in this image in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if this image reference was constructed with `plane_stride < row_stride * height`.
    /// This is niche case, so this method is focused on performance instead of handling all possible stride configurations.
    ///
    /// In case you want to iterate over pixels in an image with arbitrary strides,
    /// use `(0..self.depth()).flat_map(|z| self.get_plane_xy(z).iter_pixels())` instead.
    pub fn iter_pixels(&self) -> impl DoubleEndedIterator<Item = &'_ T> {
        let [width, height, depth, layers] = total_extent(self.dimensions, self.extent);
        let [row_stride, plane_stride, layer_stride] = total_stride(self.dimensions, self.stride);

        self.pixels[..layer_stride * layers]
            .chunks_exact(layer_stride)
            .flat_map(move |layer| layer[..plane_stride * depth].chunks_exact(plane_stride))
            .flat_map(move |plane| plane[..row_stride * height].chunks_exact(row_stride))
            .flat_map(move |row| &row[..width])
    }
}

fn total_extent(dimensions: Dimensions, extent: [usize; 3]) -> [usize; 4] {
    match dimensions {
        Dimensions::D1 => [extent[0], 1, 1, 1],
        Dimensions::D2 => [extent[0], extent[1], 1, 1],
        Dimensions::D3 => [extent[0], extent[1], extent[2], 1],
        Dimensions::D1Array => [extent[0], 1, 1, extent[1]],
        Dimensions::D2Array => [extent[0], extent[1], 1, extent[2]],
    }
}

fn total_stride(dimensions: Dimensions, stride: [usize; 2]) -> [usize; 3] {
    match dimensions {
        Dimensions::D1 => [stride[0], stride[0], stride[0]],
        Dimensions::D2 => [stride[0], stride[1], stride[1]],
        Dimensions::D3 => [stride[0], stride[1], stride[1]],
        Dimensions::D1Array => [stride[0], stride[0], stride[1]],
        Dimensions::D2Array => [stride[0], stride[1], stride[2]],
    }
}

// impl<'a, T> ImageRef<'a, T> {
//     /// Calculates total error between the reference map patch and the given block using the provided error function.
//     ///
//     /// Compares each pixel in `block` against the corresponding pixel in `self` starting
//     /// at offset (`x`, `y`). The per-pixel errors are accumulated via addition.
//     ///
//     /// # Parameters
//     ///
//     /// * `x` — Left offset into `self` where the comparison starts.
//     /// * `y` — Top offset into `self` where the comparison starts.
//     /// * `block` — The block to compare against this image region.
//     /// * `error` — A function that computes the error between two pixels.
//     ///
//     /// # Panics
//     ///
//     /// Panics if the block placed at (`x`, `y`) extends beyond the image bounds.
//     pub fn match_block<'b, U>(
//         &self,
//         x: usize,
//         y: usize,
//         block: Image2DRef<'b, T>,
//         mut error: impl FnMut(T, T) -> U,
//     ) -> U
//     where
//         T: Copy + 'a + 'b,
//         U: ops::Add<Output = U> + Zero,
//     {
//         assert!(x + block.width() <= self.width());
//         assert!(y + block.height() <= self.height());

//         let mut acc = U::zero();

//         for j in 0..block.height() {
//             for i in 0..block.width() {
//                 let r = *self.get(x + i, y + j);
//                 let b = *block.get(i, j);

//                 acc = acc + error(r, b);
//             }
//         }

//         acc
//     }

//     /// Calculates total error between the reference map patch and the given block using the provided error function, but returns `None` if the error exceeds the given upper bound.
//     ///
//     /// This is an early-exit variant of [`match_block`](Self::match_block). If the accumulated
//     /// error exceeds `upper_bound` at any point during iteration, the function immediately
//     /// returns `None`, avoiding unnecessary computation.
//     ///
//     /// # Parameters
//     ///
//     /// * `x` — Left offset into `self` where the comparison starts.
//     /// * `y` — Top offset into `self` where the comparison starts.
//     /// * `block` — The block to compare against this image region.
//     /// * `upper_bound` — Maximum acceptable total error. If exceeded, returns `None`.
//     /// * `error` — A function that computes the error between two pixels.
//     ///
//     /// # Returns
//     ///
//     /// `Some(total_error)` if the total error is within the upper bound, or `None` otherwise.
//     /// Also returns `None` if the image has zero width or height.
//     ///
//     /// # Panics
//     ///
//     /// Panics if the block placed at (`x`, `y`) extends beyond the image bounds.
//     pub fn match_block_upper_bound<'b, U>(
//         &self,
//         x: usize,
//         y: usize,
//         block: Image2DRef<'b, T>,
//         upper_bound: U,
//         mut error: impl FnMut(T, T) -> U,
//     ) -> Option<U>
//     where
//         T: Copy + 'a + 'b,
//         U: ops::Add<Output = U> + PartialOrd + Zero,
//     {
//         assert!(x + block.width() <= self.width());
//         assert!(y + block.height() <= self.height());

//         let mut acc = U::zero();

//         for j in 0..block.height() {
//             for i in 0..block.width() {
//                 let a = *self.get(x + i, y + j);
//                 let b = *block.get(i, j);

//                 acc = acc + error(a, b);
//                 if acc > upper_bound {
//                     return None;
//                 }
//             }
//         }

//         Some(acc)
//     }

//     /// Finds the position within this image where the given block best matches (lowest error).
//     ///
//     /// Performs a strided exhaustive search over all valid placements of `block` within `self`,
//     /// stepping by `step_x` horizontally and `step_y` vertically. Uses early-exit upper-bound
//     /// pruning via [`match_block_upper_bound`](Self::match_block_upper_bound) to skip
//     /// positions that cannot improve on the current best.
//     ///
//     /// The search starts at position (0, 0) as the initial candidate, then scans the remainder
//     /// of the first row before proceeding to subsequent rows.
//     ///
//     /// # Parameters
//     ///
//     /// * `step_x` — Horizontal step size between candidate positions.
//     /// * `step_y` — Vertical step size between candidate positions.
//     /// * `block` — The block to search for within this image.
//     /// * `error` — A function that computes the per-pixel error between two pixels.
//     ///
//     /// # Returns
//     ///
//     /// A tuple of `(best_error, (x, y))` where `(x, y)` is the top-left position of the
//     /// best-matching region.
//     ///
//     /// # Panics
//     ///
//     /// Panics if the block dimensions exceed the image dimensions.
//     pub fn find_best_match<'b, U>(
//         &self,
//         step_x: usize,
//         step_y: usize,
//         block: Image2DRef<'b, T>,
//         mut error: impl FnMut(T, T) -> U,
//     ) -> (U, (usize, usize))
//     where
//         T: Copy + 'a + 'b,
//         U: ops::Add<Output = U> + PartialOrd + Copy + Zero,
//     {
//         assert!(block.width() <= self.width());
//         assert!(block.height() <= self.height());

//         if self.width() == 0 || self.height() == 0 {
//             return (U::zero(), (0, 0));
//         }

//         // Calculate error for the starting point.
//         let mut best_error = self.match_block(0, 0, block.as_ref(), &mut error);
//         let mut best = (0, 0);

//         for y in 0..1 {
//             for x in (1..self.width() - block.width() + 1).step_by(step_x) {
//                 if let Some(error) =
//                     self.match_block_upper_bound(x, y, block.as_ref(), best_error, &mut error)
//                 {
//                     best_error = error;
//                     best = (x, y);
//                 }
//             }
//         }

//         for y in (1..(self.height() - block.height() + 1)).step_by(step_y) {
//             for x in (0..self.width() - block.width() + 1).step_by(step_x) {
//                 if let Some(error) =
//                     self.match_block_upper_bound(x, y, block.as_ref(), best_error, &mut error)
//                 {
//                     best_error = error;
//                     best = (x, y);
//                 }
//             }
//         }

//         (best_error, best)
//     }

//     /// Returns residuals between the reference map patch and the given block using the provided residual function.
//     ///
//     /// For each pixel in `input`, computes `residual(self_pixel, input_pixel)` and writes
//     /// the result into the corresponding position of `output`. The reference patch in `self`
//     /// starts at offset (`x`, `y`).
//     ///
//     /// # Parameters
//     ///
//     /// * `x` — Left offset into `self` for the reference patch.
//     /// * `y` — Top offset into `self` for the reference patch.
//     /// * `input` — The input block to compute residuals against.
//     /// * `output` — The mutable image to write residual values into. Must have the same
//     ///   dimensions as `input`.
//     /// * `residual` — A function that computes the residual between a reference pixel and
//     ///   an input pixel.
//     ///
//     /// # Panics
//     ///
//     /// Panics if `input` and `output` dimensions do not match.
//     pub fn residual<'b, 'c>(
//         &self,
//         x: usize,
//         y: usize,
//         input: Image2DRef<'b, T>,
//         mut output: Image2DMut<'c, T>,
//         mut residual: impl FnMut(T, T) -> T,
//     ) where
//         T: Copy + 'a + 'b + 'c,
//     {
//         if self.width() == 0 || self.height() == 0 {
//             return;
//         }

//         assert_eq!(input.width(), output.width());
//         assert_eq!(input.height(), output.height());

//         for j in 0..input.height() {
//             for i in 0..input.width() {
//                 let a = *self.get(x + i, y + j);
//                 let b = *input.get(i, j);

//                 output.set(i, j, residual(a, b));
//             }
//         }
//     }
// }
