//! Trainable 2D lookup table for patch-based image compression.
//!
//! [`ReferenceMap`] stores a pixel buffer that is iteratively refined by matching
//! patches against an input image. It supports random initialization, block-copy
//! seeding, and several training strategies.

use std::ops;

use hashbrown::HashSet;

use crate::{
    image::{Image2DMut, Image2DRef},
    math::Zero,
};

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

    pub fn as_mut(&mut self) -> Image2DMut<'_, T> {
        Image2DMut::new(self.width, self.height, &mut self.pixels)
    }

    pub fn as_ref(&self) -> Image2DRef<'_, T> {
        Image2DRef::new(self.width, self.height, &self.pixels)
    }

    pub fn set(&mut self, x: usize, y: usize, value: T) {
        assert!(x < self.width);
        assert!(y < self.height);
        self.pixels[y * self.width + x] = value;
    }

    /// Fills every pixel in the underlying buffer with values produced by `rand`.
    ///
    /// Note: this initializes the entire backing slice, not just the `width × height` region,
    /// which may include stride padding.
    pub fn random_initialize(&mut self, mut rand: impl FnMut() -> T) {
        for pixel in &mut *self.pixels {
            *pixel = rand();
        }
    }

    /// Fills this image with randomly selected `block_size × block_size` patches from `input`.
    ///
    /// The image is divided into a grid of non-overlapping blocks of size `block_size`.
    /// Each block is filled by copying pixels from a randomly chosen block-aligned position
    /// in `input`. Blocks at the edges are clamped to the input bounds.
    ///
    /// Does nothing if either image has zero dimensions or `block_size` is zero.
    ///
    /// # Parameters
    ///
    /// * `input` — The source image to sample patches from.
    /// * `block_size` — The side length of each square patch.
    pub fn initialize_patches(&mut self, input: Image2DRef<'_, T>, block_size: usize) {
        if self.width == 0
            || self.height == 0
            || input.width() == 0
            || input.height() == 0
            || block_size == 0
        {
            return;
        }

        for y in (0..self.height).step_by(block_size) {
            for x in (0..self.width).step_by(block_size) {
                let j = rand::random_range(0..input.height());
                let i = rand::random_range(0..input.width());

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

                        let bx = (j + bj).min(input.height() - 1);
                        let by = (i + bi).min(input.width() - 1);

                        self.set(mx, my, *input.get(by, bx));
                    }
                }
            }
        }
    }

    /// Updates the reference map using the given block and the provided learn function.
    pub fn learn<U>(
        &mut self,
        l: usize,
        t: usize,
        error: U,
        block: Image2DRef<'_, T>,
        mut learn: impl FnMut(T, T, U) -> T,
    ) where
        U: Copy,
    {
        if self.width == 0 || self.height == 0 {
            return;
        }

        for j in 0..block.height() {
            for i in 0..block.width() {
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
        input: Image2DRef<'_, T>,
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
            || input.width() == 0
            || input.height() == 0
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
                let i = rand::random_range(0..input.width() - block_size + 1);
                let j = rand::random_range(0..input.height() - block_size + 1);

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

    pub fn train2<U>(
        &mut self,
        input: Image2DRef<'_, T>,
        block_size: usize,
        batch_size: usize,
        mut error: impl FnMut(T, T) -> U,
        mut learn: impl FnMut(T, T, U) -> T,
    ) where
        U: ops::Add<Output = U> + PartialOrd + Copy + Zero,
    {
        if self.width == 0
            || self.height == 0
            || input.width() == 0
            || input.height() == 0
            || block_size == 0
            || batch_size == 0
        {
            return;
        }

        let mut block_claims = (0..(input.height() - block_size + 1) / block_size)
            .flat_map(|by| {
                (0..(input.width() - block_size + 1) / block_size).map(move |bx| (bx, by))
            })
            .collect::<HashSet<_>>();

        for _ in 0..batch_size {
            if block_claims.is_empty() {
                break;
            }

            // Train random patch on matching block.

            // Get a random patch index.
            let i = rand::random_range(0..self.width - block_size + 1);
            let j = rand::random_range(0..self.height - block_size + 1);

            // Get the patch from the reference map.
            let patch = self.as_ref().get_range(i, j, block_size, block_size);

            // Calculate error for the starting point.
            let mut best_error = None;
            let mut best = None;

            for &(bx, by) in &block_claims {
                match best_error {
                    None => {
                        let error =
                            input.match_block(bx * block_size, by * block_size, patch, &mut error);
                        best_error = Some(error);
                        best = Some((bx, by));
                    }
                    Some(upper_bound) => {
                        if let Some(new_error) = input.match_block_upper_bound(
                            bx * block_size,
                            by * block_size,
                            patch,
                            upper_bound,
                            &mut error,
                        ) {
                            best_error = Some(new_error);
                            best = Some((bx, by));
                        }
                    }
                }
            }

            let best_error = best_error.expect("There's at least one unclaimed block");
            let (bx, by) = best.expect("There's at least one unclaimed block");

            block_claims.remove(&(bx, by));

            let block = input.get_range(bx * block_size, by * block_size, block_size, block_size);

            self.learn(i, j, best_error, block, &mut learn);
        }
    }

    pub fn train3<U>(
        &mut self,
        input: Image2DRef<'_, T>,
        block_size: usize,
        batch_size: usize,
        mut error: impl FnMut(T, T) -> U,
        mut learn: impl FnMut(T, T, U) -> T,
    ) where
        U: ops::Add<Output = U> + PartialOrd + Copy + Zero,
    {
        if self.width == 0
            || self.height == 0
            || input.width() == 0
            || input.height() == 0
            || block_size == 0
            || batch_size == 0
        {
            return;
        }

        for _ in 0..batch_size {
            // Get a random patch index.
            let pi = rand::random_range(0..self.width - block_size + 1);
            let pj = rand::random_range(0..self.height - block_size + 1);

            // Get a random block index.
            let bi = rand::random_range(0..input.width() - block_size + 1);
            let bi = (bi / block_size) * block_size;
            let bj = rand::random_range(0..input.height() - block_size + 1);
            let bj = (bj / block_size) * block_size;

            let block = input.get_range(bi, bj, block_size, block_size);

            let error = self.as_ref().match_block(pi, pj, block, &mut error);

            self.learn(pi, pj, error, block, &mut learn);
        }
    }
}
