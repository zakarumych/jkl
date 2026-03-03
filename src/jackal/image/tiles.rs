use std::convert::Infallible;

use crate::{
    encode::FixedCode,
    image::{Extent, Image2DRef, ImageRef},
    jackal::image::header,
};

/// Size of the tile in number of blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSize {
    pub width: u16,
    pub height: u16,
}

impl FixedCode for TileSize {
    const SIZE: usize = 1;
    type Array = [u8; 1];
    type Error = Infallible;

    fn fix_encode(&self) -> [u8; 1] {
        debug_assert!(self.width.is_power_of_two());
        debug_assert!(self.height.is_power_of_two());

        let w = self.width.trailing_zeros();
        let h = self.height.trailing_zeros();

        debug_assert!(w < 16);
        debug_assert!(h < 16);

        [((w << 4) | h) as u8]
    }

    fn fix_decode(input: &[u8; 1]) -> Result<Self, Infallible> {
        let byte = input[0];
        let w = byte >> 4;
        let h = byte & 0x0F;

        let width = 1 << w;
        let height = 1 << h;

        Ok(TileSize { width, height })
    }
}

impl TileSize {
    /// Finds the optimal tile size for the given extent and block size.
    ///
    /// The optimal tile size is the one that minimizes the cost function:
    /// cost = (flat_cost + size_cost * tile_size) * ceil(number_of_tiles / 64) * 64
    /// i.e. tile cost is linear to size plus a float amount and it is multiplied by the number of tiles rounded up to the nearest multiple of 64.
    /// This cost function models the GPU decompression cost, which runs thread per tile, each thread has a fixed overhead of dispatching and a cost proportional to the tile size,
    /// and the GPU executes threads in warps of 32 or 64 threads, so the number of tiles is rounded up to the nearest multiple of 64.
    pub fn find_optimal(
        extent: Extent,
        block_width: u16,
        block_height: u16,
        flat_cost: f32,
        size_cost: f32,
    ) -> Self {
        assert!(block_width.is_power_of_two());
        assert!(block_height.is_power_of_two());

        // The max tile size to consider is either enough to cover the entire extent,
        // or 32768 blocks, which is the maximum tile size.
        let max_tile_width = u16::try_from(extent.width().next_power_of_two()).unwrap_or(1 << 15);
        let max_tile_height = u16::try_from(extent.height().next_power_of_two()).unwrap_or(1 << 15);

        let mut min_cost = f32::INFINITY;
        let mut best_candidate = TileSize {
            width: block_width,
            height: block_height,
        };

        // Initial tile dimensions are the block size.
        let mut tile_width = usize::from(block_width);
        let mut tile_height = usize::from(block_height);

        loop {
            let row_size = (extent.width() + tile_width - 1) / tile_width;
            let column_size = (extent.height() + tile_height - 1) / tile_height;
            let planes = extent.depth() * extent.layers();

            let tiles_count = row_size * column_size * planes;
            let warps_count = (tiles_count + 63) / 64;

            let cost = (flat_cost + size_cost * (tile_width as f32 * tile_height as f32))
                * warps_count as f32;

            if cost < min_cost {
                min_cost = cost;
                best_candidate = TileSize {
                    width: tile_width as u16,
                    height: tile_height as u16,
                };
            }

            if tile_width == usize::from(max_tile_width) {
                if tile_height == usize::from(max_tile_height) {
                    break;
                } else {
                    tile_height *= 2;
                    tile_width = usize::from(block_width);
                }
            } else {
                tile_width *= 2;
            }
        }

        best_candidate
    }

    pub fn iter_tiles<'a, T>(&self, image: ImageRef<'a, T>) -> TilesIter<'a, T> {
        TilesIter {
            image,
            tile_size: *self,
            x: 0,
            y: 0,
            z: 0,
        }
    }

    #[inline]
    pub fn tiles_count(&self, extent: Extent) -> usize {
        let [width, height, depth] = self.tiles(extent);
        width * height * depth
    }

    #[inline]
    pub fn tiles(&self, extent: Extent) -> [usize; 3] {
        let width = extent.width();
        let height = extent.height();
        let planes = extent.layers() * extent.layers();

        let tiles_x = (width + usize::from(self.width) - 1) / usize::from(self.width);
        let tiles_y = (height + usize::from(self.height) - 1) / usize::from(self.height);

        [tiles_x, tiles_y, planes]
    }

    pub fn tile_pos(&self, extent: Extent, index: usize) -> [usize; 3] {
        let [tiles_x, tiles_y, planes] = self.tiles(extent);

        let plane = index / (tiles_x * tiles_y);
        assert!(plane < planes);

        let tile_index = index % (tiles_x * tiles_y);

        let tile_y = tile_index / tiles_x;
        let tile_x = tile_index % tiles_x;

        let x = tile_x * usize::from(self.width);
        let y = tile_y * usize::from(self.height);

        [x, y, plane]
    }
}

pub struct TilesIter<'a, T> {
    image: ImageRef<'a, T>,
    tile_size: TileSize,
    x: usize,
    y: usize,
    z: usize,
}

impl<'a, T> Clone for TilesIter<'a, T> {
    fn clone(&self) -> Self {
        TilesIter {
            image: self.image,
            tile_size: self.tile_size,
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl<'a, T> Iterator for TilesIter<'a, T> {
    type Item = Image2DRef<'a, T>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn next(&mut self) -> Option<Image2DRef<'a, T>> {
        let w = usize::min(
            usize::from(self.tile_size.width),
            self.image.width() - self.x,
        );
        let h = usize::min(
            usize::from(self.tile_size.height),
            self.image.height() - self.y,
        );

        if self.x >= self.image.width() {
            self.x = 0;
            self.y += h;
        }

        if self.y >= self.image.height() {
            self.y = 0;
            self.z += 1;
        }

        if self.z >= self.image.raw_extent()[2] {
            return None;
        }

        let plane = self.image.plane_ref(self.z);

        let tile = plane.get_range(self.x, self.y, w, h);

        self.x += w;

        Some(tile)
    }
}

impl<'a, T> ExactSizeIterator for TilesIter<'a, T> {
    fn len(&self) -> usize {
        let extent = self.image.raw_extent();

        let rows =
            (extent[0] + usize::from(self.tile_size.width) - 1) / usize::from(self.tile_size.width);
        let columns = (extent[1] + usize::from(self.tile_size.height) - 1)
            / usize::from(self.tile_size.height);
        let planes = extent[2];

        planes * columns * rows - self.z * columns * rows - self.y * rows - self.x
    }
}
