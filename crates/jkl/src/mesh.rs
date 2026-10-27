//! Patch-based mesh compression algorithms and attribute codecs.

use std::{error::Error, fmt};

pub mod codec;
pub mod compress;
pub mod data;
pub mod fit;
pub mod gpu;
pub mod layout;
pub mod normal;
pub mod order;
pub mod spatial;
pub mod tangent;

/// Invalid input to [`split_positions`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitPositionsError {
    /// The maximum number of vertices per patch must be positive.
    ZeroMaxVertices,
    /// A position contains NaN or infinity.
    NonFiniteCoordinate {
        /// Index of the position in the input slice.
        vertex: usize,
        /// Coordinate index: 0 for X, 1 for Y, or 2 for Z.
        component: usize,
    },
}

impl fmt::Display for SplitPositionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMaxVertices => write!(f, "maximum vertices per patch must be positive"),
            Self::NonFiniteCoordinate { vertex, component } => {
                write!(
                    f,
                    "vertex {vertex} has a non-finite coordinate at component {component}"
                )
            }
        }
    }
}

impl Error for SplitPositionsError {}

/// Splits positions into spatially compact, balanced groups of original indices.
///
/// For `n` nonempty input positions, returns exactly `k = ceil(n / max_vertices)`
/// nonempty patches. Each patch contains either `floor(n / k)` or `ceil(n / k)`
/// vertices, never more than `max_vertices`. Every input index occurs exactly
/// once. Empty input returns no patches.
///
/// The heuristic recursively partitions by rank along the longest axis of each
/// group's axis-aligned bounding box. Equal spans prefer X, then Y, then Z;
/// coordinate comparisons use total floating-point order and original indices
/// break ties. Results are deterministic for the same ordered input. This is a
/// spatial heuristic, not an optimal compactness partition or a connectivity
/// guarantee. The order inside each patch is not a short route.
///
/// Positions and topology are not modified. To reorder a mesh, apply the returned
/// index permutation to every vertex attribute and remap triangle indices using
/// its inverse. Vertices shared by triangles are not duplicated.
///
/// Uses O(n * (1 + log k)) time, O(n) index workspace plus O(n + k) output,
/// and O(1 + log k) recursion depth. No transform or quantization is performed.
///
/// # Errors
///
/// Returns [`SplitPositionsError::ZeroMaxVertices`] for a zero cap, including
/// empty input. Otherwise rejects the first non-finite coordinate in input order.
pub fn split_positions(
    positions: &[[f64; 3]],
    max_vertices: usize,
) -> Result<Vec<Vec<usize>>, SplitPositionsError> {
    if max_vertices == 0 {
        return Err(SplitPositionsError::ZeroMaxVertices);
    }
    for (vertex, position) in positions.iter().enumerate() {
        for (component, coordinate) in position.iter().enumerate() {
            if !coordinate.is_finite() {
                return Err(SplitPositionsError::NonFiniteCoordinate { vertex, component });
            }
        }
    }
    if positions.is_empty() {
        return Ok(Vec::new());
    }

    let count = positions.len().div_ceil(max_vertices);
    let base_size = positions.len() / count;
    let larger_count = positions.len() % count;
    let mut indices: Vec<_> = (0..positions.len()).collect();
    let mut patches = Vec::with_capacity(count);
    split_recursive(
        positions,
        &mut indices,
        count,
        base_size,
        larger_count,
        &mut patches,
    );
    Ok(patches)
}

fn split_recursive(
    positions: &[[f64; 3]],
    indices: &mut [usize],
    count: usize,
    base_size: usize,
    larger_count: usize,
    patches: &mut Vec<Vec<usize>>,
) {
    if count == 1 {
        patches.push(indices.to_vec());
        return;
    }

    let mut min = positions[indices[0]];
    let mut max = min;
    for &index in indices.iter().skip(1) {
        for axis in 0..3 {
            min[axis] = min[axis].min(positions[index][axis]);
            max[axis] = max[axis].max(positions[index][axis]);
        }
    }
    let mut axis = 0;
    for candidate in 1..3 {
        if span_key(min[candidate], max[candidate]) > span_key(min[axis], max[axis]) {
            axis = candidate;
        }
    }

    let left_count = count / 2;
    let left_larger = larger_count.min(left_count);
    let left_len = base_size * left_count + left_larger;
    indices.select_nth_unstable_by(left_len, |&a, &b| {
        positions[a][axis]
            .total_cmp(&positions[b][axis])
            .then_with(|| a.cmp(&b))
    });
    let (left, right) = indices.split_at_mut(left_len);
    split_recursive(positions, left, left_count, base_size, left_larger, patches);
    split_recursive(
        positions,
        right,
        count - left_count,
        base_size,
        larger_count - left_larger,
        patches,
    );
}

fn span_key(min: f64, max: f64) -> (bool, f64) {
    let span = max - min;
    if span.is_infinite() {
        // Overflowing spans outrank finite spans; their half-spans remain finite.
        (true, max * 0.5 - min * 0.5)
    } else {
        // Direct subtraction preserves subnormal spans without rescaling.
        (false, span)
    }
}
