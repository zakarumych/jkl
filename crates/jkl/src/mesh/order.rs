//! Hilbert ordering and bounded open-path refinement of quantized positions.

use std::{error::Error, fmt};

const MAX_PASSES: usize = 4;
const MAX_SEGMENT: usize = 32;

/// Invalid input to [`order_quantized`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderError {
    /// Coordinates require between 1 and 20 bits.
    InvalidBits,
    /// A coordinate exceeds the unsigned range specified by `bits`.
    CoordinateOutOfRange { vertex: usize, component: usize },
}

impl fmt::Display for OrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBits => write!(f, "coordinate bits must be between 1 and 20"),
            Self::CoordinateOutOfRange { vertex, component } => write!(
                f,
                "vertex {vertex} coordinate {component} exceeds the specified bit range"
            ),
        }
    }
}

impl Error for OrderError {}

/// Two permutations of the input vertices, with open-path costs.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchOrder {
    /// Ascending 3D Hilbert key, with input indices breaking ties.
    pub hilbert: Vec<usize>,
    /// Hilbert order refined by short segment reversals.
    pub refined: Vec<usize>,
    /// Sum of `log2(1 + abs(delta))` over components and consecutive vertices.
    pub hilbert_cost: f64,
    /// The same cost measured on `refined`.
    pub refined_cost: f64,
    /// Number of strictly improving reversals accepted.
    pub accepted_moves: usize,
    /// Number of refinement passes performed; zero for fewer than three vertices.
    pub passes: usize,
}

/// Orders quantized positions using a 3D Hilbert curve, then bounded 2-opt.
///
/// Each returned index addresses the input slice; vertices and topology are not
/// mutated. Duplicate coordinates retain input-index order in the Hilbert seed.
/// Refinement deterministically scans segment starts and lengths 2 through 32,
/// accepting strictly cheaper boundary connections. Prefix and suffix reversals
/// participate because this is an open path. A complete path reversal is skipped.
/// At most four passes run, stopping after a pass without an improvement.
///
/// Cost is the sum of `log2(1 + abs(delta))` for all three integer components of
/// each edge. The first absolute position has no cost here. This is a compression
/// heuristic, not an entropy estimate or a guarantee of the shortest path.
///
/// Uses O(n) memory and O(n log n + p n w²) time including segment reversals,
/// with p <= 4 and w <= 32. Hilbert keys use at most 60 bits.
///
/// # Errors
///
/// Rejects `bits` outside 1..=20, then the first out-of-range coordinate in input
/// order. Empty and singleton inputs are valid and have zero path cost.
pub fn order_quantized(points: &[[u32; 3]], bits: u8) -> Result<PatchOrder, OrderError> {
    if !(1..=20).contains(&bits) {
        return Err(OrderError::InvalidBits);
    }
    let maximum = (1u32 << bits) - 1;
    for (vertex, point) in points.iter().enumerate() {
        for (component, &coordinate) in point.iter().enumerate() {
            if coordinate > maximum {
                return Err(OrderError::CoordinateOutOfRange { vertex, component });
            }
        }
    }
    let mut keyed: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(index, &point)| (hilbert_key(point, bits), index))
        .collect();
    keyed.sort_unstable();
    let hilbert: Vec<_> = keyed.into_iter().map(|(_, index)| index).collect();
    let hilbert_cost = path_cost(points, &hilbert);
    let mut refined = hilbert.clone();
    let (accepted_moves, passes) = refine(points, &mut refined);
    let refined_cost = path_cost(points, &refined);
    Ok(PatchOrder {
        hilbert,
        refined,
        hilbert_cost,
        refined_cost,
        accepted_moves,
        passes,
    })
}

// Skilling's axes-to-transpose transform, followed by MSB-first interleaving.
// Reference: https://github.com/davidmoten/hilbert-curve/blob/master/src/main/java/org/davidmoten/hilbert/HilbertCurve.java
fn hilbert_key(mut axes: [u32; 3], bits: u8) -> u64 {
    let mut q = 1u32 << (bits - 1);
    while q > 1 {
        let p = q - 1;
        for i in 0..3 {
            if axes[i] & q != 0 {
                axes[0] ^= p;
            } else {
                let t = (axes[0] ^ axes[i]) & p;
                axes[0] ^= t;
                axes[i] ^= t;
            }
        }
        q >>= 1;
    }
    axes[1] ^= axes[0];
    axes[2] ^= axes[1];
    let mut t = 0;
    q = 1u32 << (bits - 1);
    while q > 1 {
        if axes[2] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for axis in &mut axes {
        *axis ^= t;
    }
    let mut key = 0;
    for bit in (0..bits).rev() {
        for axis in axes {
            key = (key << 1) | u64::from((axis >> bit) & 1);
        }
    }
    key
}

fn edge_cost(a: [u32; 3], b: [u32; 3]) -> f64 {
    (0..3)
        .map(|axis| (1.0 + f64::from(a[axis].abs_diff(b[axis]))).log2())
        .sum()
}

fn path_cost(points: &[[u32; 3]], order: &[usize]) -> f64 {
    order
        .windows(2)
        .map(|edge| edge_cost(points[edge[0]], points[edge[1]]))
        .sum()
}

fn boundary_costs(points: &[[u32; 3]], order: &[usize], start: usize, end: usize) -> (f64, f64) {
    let mut old = 0.0;
    let mut new = 0.0;
    if start > 0 {
        let before = points[order[start - 1]];
        old += edge_cost(before, points[order[start]]);
        new += edge_cost(before, points[order[end]]);
    }
    if end + 1 < order.len() {
        let after = points[order[end + 1]];
        old += edge_cost(points[order[end]], after);
        new += edge_cost(points[order[start]], after);
    }
    (old, new)
}

fn refine(points: &[[u32; 3]], order: &mut [usize]) -> (usize, usize) {
    if order.len() < 3 {
        return (0, 0);
    }
    let mut accepted = 0;
    let mut passes = 0;
    for _ in 0..MAX_PASSES {
        passes += 1;
        let previous = accepted;
        for start in 0..order.len() - 1 {
            for len in 2..=MAX_SEGMENT.min(order.len() - start) {
                if len == order.len() {
                    continue;
                }
                let end = start + len - 1;
                let (old, new) = boundary_costs(points, order, start, end);
                if new < old {
                    order[start..=end].reverse();
                    accepted += 1;
                }
            }
        }
        if previous == accepted {
            break;
        }
    }
    (accepted, passes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversal_boundary_cost_matches_complete_path_including_endpoints() {
        let points = [[0, 0, 0], [11, 5, 7], [3, 1, 0], [0, 9, 15], [6, 8, 1]];
        let order = [0, 1, 2, 3, 4];
        let original = path_cost(&points, &order);
        for start in 0..order.len() - 1 {
            for end in start + 1..order.len() {
                let (old, new) = boundary_costs(&points, &order, start, end);
                let mut reversed = order;
                reversed[start..=end].reverse();
                assert!((path_cost(&points, &reversed) - original - (new - old)).abs() < 1e-12);
            }
        }
    }
}
