use std::ops;

use crate::math::Zero;

pub struct ImageRef<'a, T> {
    pub width: usize,
    pub height: usize,
    pub pixels: &'a [T],
    pub stride: usize,
}

impl<T> Copy for ImageRef<'_, T> {}
impl<T> Clone for ImageRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> ImageRef<'a, T>
where
    T: Copy,
{
    pub fn get(&self, x: usize, y: usize) -> &'a T {
        &self.pixels[y * self.stride + x]
    }

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

    pub fn pixels(&self) -> &'a [T] {
        &self.pixels
    }

    /// Calculates total error between the reference map patch and the given block using the provided error function.
    pub fn match_block<U>(
        &self,
        l: usize,
        t: usize,
        block: ImageRef<'_, T>,
        mut error: impl FnMut(T, T) -> U,
    ) -> U
    where
        U: ops::Add<Output = U> + Zero,
    {
        assert!(l + block.width <= self.width);
        assert!(t + block.height <= self.height);

        let mut acc = U::zero();

        for j in 0..block.height {
            for i in 0..block.width {
                let x = i + l;
                let y = j + t;

                let r = *self.get(x, y);
                let b = *block.get(i, j);

                acc = acc + error(r, b);
            }
        }

        acc
    }

    /// Calculates total error between the reference map patch and the given block using the provided error function, but returns `None` if the error exceeds the given upper bound.
    pub fn match_block_upper_bound<U>(
        &self,
        l: usize,
        t: usize,
        block: ImageRef<'_, T>,
        upper_bound: U,
        mut error: impl FnMut(T, T) -> U,
    ) -> Option<U>
    where
        U: ops::Add<Output = U> + PartialOrd + Zero,
    {
        assert!(l + block.width <= self.width);
        assert!(t + block.height <= self.height);

        if self.width == 0 || self.height == 0 {
            return None;
        }

        let mut acc = U::zero();

        for j in 0..block.height {
            for i in 0..block.width {
                let x = i + l;
                let y = j + t;

                let a = *self.get(x, y);
                let b = *block.get(i, j);

                acc = acc + error(a, b);
                if acc > upper_bound {
                    return None;
                }
            }
        }

        Some(acc)
    }

    pub fn find_best_match<U>(
        &self,
        step_x: usize,
        step_y: usize,
        block: ImageRef<'_, T>,
        mut error: impl FnMut(T, T) -> U,
    ) -> (U, (usize, usize))
    where
        U: ops::Add<Output = U> + PartialOrd + Copy + Zero,
    {
        assert!(block.width <= self.width);
        assert!(block.height <= self.height);

        if self.width == 0 || self.height == 0 {
            return (U::zero(), (0, 0));
        }

        // Calculate error for the starting point.
        let mut best_error = self.match_block(0, 0, block, &mut error);
        let mut best = (0, 0);

        for t in 0..1 {
            for l in (1..self.width - block.width + 1).step_by(step_x) {
                if let Some(error) =
                    self.match_block_upper_bound(l, t, block, best_error, &mut error)
                {
                    best_error = error;
                    best = (l, t);
                }
            }
        }

        for t in (1..(self.height - block.height + 1)).step_by(step_y) {
            for l in (0..self.width - block.width + 1).step_by(step_x) {
                if let Some(error) =
                    self.match_block_upper_bound(l, t, block, best_error, &mut error)
                {
                    best_error = error;
                    best = (l, t);
                }
            }
        }

        (best_error, best)
    }

    /// Returns residuals between the reference map patch and the given block using the provided residual function.
    pub fn residual(
        &self,
        l: usize,
        t: usize,
        mut block: ImageMut<'_, T>,
        mut residual: impl FnMut(T, T) -> T,
    ) {
        if self.width == 0 || self.height == 0 {
            return;
        }

        for j in 0..block.height {
            for i in 0..block.width {
                let x = i + l;
                let y = j + t;

                let a = *self.get(x, y);
                let b = *block.get(i, j);

                block.set(i, j, residual(a, b));
            }
        }
    }
}

pub struct ImageMut<'a, T> {
    pub width: usize,
    pub height: usize,
    pub pixels: &'a mut [T],
    pub stride: usize,
}

impl<T> ImageMut<'_, T>
where
    T: Copy,
{
    pub fn as_ref(&self) -> ImageRef<'_, T> {
        ImageRef {
            width: self.width,
            height: self.height,
            pixels: &self.pixels,
            stride: self.stride,
        }
    }

    pub fn copy_from(&mut self, src: ImageRef<'_, T>) {
        assert_eq!(src.width, self.width);
        assert_eq!(src.height, self.height);

        for j in 0..src.height {
            for i in 0..src.width {
                self.set(i, j, *src.get(i, j));
            }
        }
    }

    pub fn random_initialize(&mut self, mut rand: impl FnMut() -> T) {
        for pixel in &mut *self.pixels {
            *pixel = rand();
        }
    }

    pub fn initialize_patches(&mut self, input: ImageRef<'_, T>, block_size: usize) {
        if self.width == 0
            || self.height == 0
            || input.width == 0
            || input.height == 0
            || block_size == 0
        {
            return;
        }

        for y in (0..self.height).step_by(block_size) {
            for x in (0..self.width).step_by(block_size) {
                let j = rand::random_range(0..input.height);
                let i = rand::random_range(0..input.width);

                let j = (j / block_size) * block_size;
                let i = (i / block_size) * block_size;

                // Copy a block.
                for bj in 0..block_size {
                    for bi in 0..block_size {
                        let mx = x + bi;
                        let my = y + bj;

                        if mx >= self.width || my >= self.height {
                            continue;
                        }

                        let bx = (j + bj).min(input.height - 1);
                        let by = (i + bi).min(input.width - 1);

                        self.set(mx, my, *input.get(by, bx));
                    }
                }
            }
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &T {
        &self.pixels[y * self.stride + x]
    }

    pub fn pixels(&self) -> &[T] {
        &self.pixels
    }

    pub fn set(&mut self, x: usize, y: usize, value: T) {
        self.pixels[y * self.stride + x] = value;
    }
}

/// Reference map is a 2D image constructed to be used as lookup table for other image blocks.
pub struct ReferenceMap<T> {
    width: usize,
    height: usize,
    pixels: Vec<T>,
}

impl<T> ReferenceMap<T>
where
    T: Copy,
{
    pub fn new(width: usize, height: usize, init: T) -> Self {
        let pixels = vec![init; width * height];
        ReferenceMap {
            width,
            height,
            pixels,
        }
    }

    pub fn as_mut(&mut self) -> ImageMut<'_, T> {
        ImageMut {
            width: self.width,
            height: self.height,
            pixels: &mut self.pixels,
            stride: self.width,
        }
    }

    pub fn as_ref(&self) -> ImageRef<'_, T> {
        ImageRef {
            width: self.width,
            height: self.height,
            pixels: &self.pixels,
            stride: self.width,
        }
    }

    /// Updates the reference map using the given block and the provided learn function.
    pub fn learn<U>(
        &mut self,
        l: usize,
        t: usize,
        error: U,
        block: ImageRef<'_, T>,
        mut learn: impl FnMut(T, T, U) -> T,
    ) where
        U: Copy,
    {
        if self.width == 0 || self.height == 0 {
            return;
        }

        for j in 0..block.height {
            for i in 0..block.width {
                let x = i + l;
                let y = j + t;

                let a = *self.as_ref().get(x, y);
                let b = *block.get(i, j);

                self.as_mut().set(x, y, learn(a, b, error));
            }
        }
    }

    pub fn train<U>(
        &mut self,
        input: ImageRef<'_, T>,
        block_size: usize,
        batch_size: usize,
        mut error: impl FnMut(T, T) -> U,
        mut learn_forward: impl FnMut(T, T, U) -> T,
        mut learn_backward: impl FnMut(T, T, U) -> T,
    ) where
        U: ops::Add<Output = U> + PartialOrd + Copy + Zero,
    {
        if self.width == 0
            || self.height == 0
            || input.width == 0
            || input.height == 0
            || block_size == 0
            || batch_size == 0
        {
            return;
        }

        for _ in 0..batch_size {
            if rand::random_bool(0.5) {
                // Train random patch on matching block.

                // Get a random patch index.
                let i = rand::random_range(0..self.width - block_size + 1);
                let j = rand::random_range(0..self.height - block_size + 1);

                // Get the patch from the reference map.
                let patch = self.as_ref().get_range(i, j, block_size, block_size);

                let (error, (l, t)) =
                    input.find_best_match(block_size, block_size, patch, &mut error);

                let block = input.get_range(l, t, block_size, block_size);

                self.learn(i, j, error, block, &mut learn_forward);
            } else {
                // Train matching patch on random block.

                // Get a random block index.
                let i = rand::random_range(0..input.width - block_size + 1);
                let j = rand::random_range(0..input.height - block_size + 1);

                // Round down to the nearest block boundary.
                let i = (i / block_size) * block_size;
                let j = (j / block_size) * block_size;

                // Get the block from the input image.
                let block = input.get_range(i, j, block_size, block_size);

                let (error, (l, t)) = self.as_ref().find_best_match(1, 1, block, &mut error);

                self.learn(l, t, error, block, &mut learn_backward);
            }
        }
    }
}
