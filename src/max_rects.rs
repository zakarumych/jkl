use std::cmp::{max, min};
use std::f32;

use crate::math::Rect;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Heuristic {
    #[default]
    BestAreaFit,
    BestShortSideFit,
    BestLongSideFit,
    BottomLeft,
}

/// 2d bin packer that uses maximal rectangles algorithm.
pub struct MaximalRectangles {
    bin_w: usize,
    bin_h: usize,
    used: Vec<Rect<usize>>,
    free: Vec<Rect<usize>>,
    heuristic: Heuristic,
    allow_rotation: bool,
    quantization: (usize, usize),
}

impl MaximalRectangles {
    /// Create new maximal rectangles packer with single empty region with specified dimensions
    pub fn new(width: usize, height: usize) -> Self {
        let mut free = Vec::new();
        if width > 0 && height > 0 {
            free.push(Rect {
                x: 0,
                y: 0,
                w: width,
                h: height,
            });
        }
        MaximalRectangles {
            bin_w: width,
            bin_h: height,
            used: Vec::new(),
            free,
            heuristic: Heuristic::default(),
            allow_rotation: false,
            quantization: (1, 1),
        }
    }

    /// Configure packer to allow or disallow rotations of following rectangles.
    pub fn with_allow_rotation(&mut self, allow_rotation: bool) -> &mut Self {
        self.allow_rotation = allow_rotation;
        self
    }

    /// Configure packer to use specified scoring heuristic to place following rectangles.
    pub fn with_heuristic(&mut self, heuristic: Heuristic) -> &mut Self {
        self.heuristic = heuristic;
        self
    }

    /// Configure packer to quantization all sizes to specified values.
    ///
    /// # Panics
    ///
    /// This function panics if either value is 0.
    pub fn with_quantization(&mut self, x: usize, y: usize) -> &mut Self {
        assert_ne!(x, 0);
        assert_ne!(y, 0);
        self.quantization = (x, y);
        self
    }

    /// Returns occupancy - ratio between used area and total area.
    pub fn occupancy(&self) -> f32 {
        let used_area: usize = self.used.iter().map(|r| r.area()).sum();
        used_area as f32 / (self.bin_w * self.bin_h) as f32
    }

    /// Returns bin width
    pub fn width(&self) -> usize {
        self.bin_w
    }

    /// Returns bin height
    pub fn height(&self) -> usize {
        self.bin_h
    }

    /// Returns whether rotation of rectangles is allowed.
    pub fn allow_rotation(&self) -> bool {
        self.allow_rotation
    }

    /// Returns current heuristic for rectangle placement.
    pub fn heuristic(&self) -> Heuristic {
        self.heuristic
    }

    /// Returns current quantization for rectangle placement.
    pub fn quantization(&mut self) -> (usize, usize) {
        self.quantization
    }

    /// Returns iterator over used rectangles.
    pub fn used(&self) -> impl Iterator<Item = Rect<usize>> + '_ {
        self.used.iter().copied()
    }

    /// Returns iterator over free rectangles.
    pub fn free(&self) -> impl Iterator<Item = Rect<usize>> + '_ {
        self.free.iter().copied()
    }

    /// Inserts rectangle with specified size.
    pub fn insert(&mut self, w: usize, h: usize) -> Option<Rect<usize>> {
        let r = self.find_position(w, h);
        match r {
            None => None,
            Some(r) => {
                self.place_rect(r);
                Some(r)
            }
        }
    }

    /// Core logic to search for best candidate.
    fn find_position(&self, w: usize, h: usize) -> Option<Rect<usize>> {
        let (qx, qy) = self.quantization;
        // let w = round_up(w, qx);
        // let h = round_up(h, qy);
        let mut best_placement = None;
        let mut best1 = usize::MAX;
        let mut best2 = usize::MAX;

        for &fr in self.free.iter() {
            let fr = fr.round_in(qx, qy);
            // no rotation
            if w <= fr.w && h <= fr.h {
                let (s1, s2) = self.score(fr, w, h);
                if s1 < best1 || (s1 == best1 && s2 < best2) {
                    best_placement = Some((fr.x, fr.y, false));
                    best1 = s1;
                    best2 = s2;
                }
            }

            // rotated
            if self.allow_rotation && h != w && h <= fr.w && w <= fr.h {
                let (s1, s2) = self.score(fr, h, w);
                if s1 < best1 || (s1 == best1 && s2 < best2) {
                    best_placement = Some((fr.x, fr.y, true));
                    best1 = s1;
                    best2 = s2;
                }
            }
        }

        best_placement.map(|(x, y, rotated)| Rect {
            x: x,
            y: y,
            w: if rotated { h } else { w },
            h: if rotated { w } else { h },
        })
    }

    fn score(&self, f: Rect<usize>, w: usize, h: usize) -> (usize, usize) {
        debug_assert!(f.w >= w);
        debug_assert!(f.h >= h);

        let leftover_h = f.h - h;
        let leftover_w = f.w - w;
        let short_side = min(leftover_h, leftover_w);
        let long_side = max(leftover_h, leftover_w);
        let area_fit = (f.w * f.h) - (w * h);

        match self.heuristic {
            Heuristic::BestShortSideFit => (short_side, long_side),
            Heuristic::BestLongSideFit => (long_side, short_side),
            Heuristic::BestAreaFit => (area_fit, short_side),
            Heuristic::BottomLeft => (f.y, f.x),
        }
    }

    /// Emplaces rect and splits all free rectangles.
    fn place_rect(&mut self, used: Rect<usize>) {
        let mut i = 0;

        while i < self.free.len() {
            if self.split_free_node(i, used) {
                // free rect removed
            } else {
                i += 1;
            }
        }

        self.prune_free_list();
        self.used.push(used.clone());
    }

    fn split_free_node(&mut self, idx: usize, used: Rect<usize>) -> bool {
        let fr = &self.free[idx];

        if used.x >= fr.x + fr.w
            || used.x + used.w <= fr.x
            || used.y >= fr.y + fr.h
            || used.y + used.h <= fr.y
        {
            return false;
        }

        let (qx, qy) = self.quantization;

        let fr = self.free[idx].clone();
        self.free.remove(idx);

        // left split
        if used.x >= fr.x + qx {
            self.free.push(Rect {
                x: fr.x,
                y: fr.y,
                w: used.x - fr.x,
                h: fr.h,
            });
        }

        // right split
        if used.x + used.w + qx <= fr.x + fr.w {
            self.free.push(Rect {
                x: used.x + used.w,
                y: fr.y,
                w: fr.x + fr.w - (used.x + used.w),
                h: fr.h,
            });
        }

        // top
        if used.y >= fr.y + qy {
            self.free.push(Rect {
                x: fr.x,
                y: fr.y,
                w: fr.w,
                h: used.y - fr.y,
            });
        }

        // bottom
        if used.y + used.h + qy <= fr.y + fr.h {
            self.free.push(Rect {
                x: fr.x,
                y: used.y + used.h,
                w: fr.w,
                h: fr.y + fr.h - (used.y + used.h),
            });
        }

        true
    }

    /// Remove any free rectangle if one is fully contained within the other.
    fn prune_free_list(&mut self) {
        let mut i = 0;
        while i < self.free.len() {
            let mut j = i + 1;
            let mut removed = false;

            while j < self.free.len() {
                let a = self.free[i];
                let b = self.free[j];

                if b.contains(&a) {
                    self.free.remove(i);
                    removed = true;
                    break;
                }
                if a.contains(&b) {
                    self.free.remove(j);
                    continue;
                }
                j += 1;
            }

            if !removed {
                i += 1;
            }
        }
    }
}
