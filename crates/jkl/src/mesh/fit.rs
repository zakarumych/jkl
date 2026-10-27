//! Deterministic affine cube fitting with bounded quantization-error refinement.
//!
//! Matrices are row-major and multiply column homogeneous vectors. Both matrices
//! have final row `[0, 0, 0, 1]`. Fitting uses f64 matrices; the mesh container
//! stores only the inverse matrix, converted to f32 for reconstruction.

use std::{error::Error, fmt};

/// Accepted normalized-coordinate roundoff beyond either cube boundary.
pub const CONTAINMENT_TOLERANCE: f64 = 1e-9;

/// Invalid input or an unrepresentable affine fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitError {
    /// Quantization supports 1 through 20 bits per coordinate.
    InvalidBits,
    /// A patch must contain at least one position.
    EmptyPatch,
    /// The indicated input coordinate is not finite.
    NonFiniteCoordinate { vertex: usize, component: usize },
    /// Finite matrices, containment, or finite error statistics cannot be obtained.
    NumericalRange,
    /// A point lies outside the fitted cube beyond the roundoff tolerance.
    OutsideCube,
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBits => write!(f, "quantization bits must be between 1 and 20"),
            Self::EmptyPatch => write!(f, "cannot fit an empty patch"),
            Self::NonFiniteCoordinate { vertex, component } => write!(
                f,
                "vertex {vertex} has a non-finite coordinate at component {component}"
            ),
            Self::NumericalRange => {
                write!(f, "patch exceeds the numerical range of the affine fitter")
            }
            Self::OutsideCube => write!(f, "point lies outside the fitted cube"),
        }
    }
}

impl Error for FitError {}

/// World-space reconstruction error measured through the public decode path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorStats {
    pub mean_distance: f64,
    pub mean_squared_distance: f64,
    pub max_distance: f64,
}

/// One fitted affine transform and its quantized input positions.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedFit {
    pub forward: [[f64; 4]; 4],
    pub inverse: [[f64; 4]; 4],
    pub quantized: Vec<[u32; 3]>,
    pub errors: ErrorStats,
}

impl QuantizedFit {
    /// Applies the world-to-cube transform without clipping.
    pub fn normalized_point(&self, point: [f64; 3]) -> [f64; 3] {
        transform_point(&self.forward, point)
    }

    /// Quantizes a point using nearest integer, ties-to-even rounding and the supplied bit depth.
    /// Only boundary roundoff within [`CONTAINMENT_TOLERANCE`] may be clipped.
    pub fn encode_point(&self, point: [f64; 3], bits: u8) -> Result<[u32; 3], FitError> {
        let q = levels(bits)?;
        let normalized = self.normalized_point(point);
        let mut encoded = [0; 3];
        for axis in 0..3 {
            let value = normalized[axis];
            if !value.is_finite() {
                return Err(FitError::NumericalRange);
            }
            if !(-CONTAINMENT_TOLERANCE..=1.0 + CONTAINMENT_TOLERANCE).contains(&value) {
                return Err(FitError::OutsideCube);
            }
            encoded[axis] = (value.clamp(0.0, 1.0) * q).round_ties_even() as u32;
        }
        Ok(encoded)
    }

    /// Decodes integer coordinates using `q / (2^bits - 1)` and the inverse matrix.
    /// `bits` must match the fit and be in `1..=20`; coordinates must be in range.
    pub fn decode_point(&self, point: [u32; 3], bits: u8) -> [f64; 3] {
        assert!((1..=20).contains(&bits), "invalid quantization bit depth");
        let q = (1u32 << bits) - 1;
        assert!(
            point.iter().all(|&value| value <= q),
            "coordinate exceeds quantization range"
        );
        transform_point(&self.inverse, point.map(|value| value as f64 / q as f64))
    }
}

/// Seed fits and the best result of bounded local refinement.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchFit {
    pub bits: u8,
    pub axis_seed: QuantizedFit,
    pub diagonal_seed: QuantizedFit,
    pub refined: QuantizedFit,
    /// Candidate evaluations, including rejected numerical/shape/margin attempts.
    pub evaluations: usize,
}

/// Applies an affine row-major matrix to a column-vector position.
pub fn transform_point(matrix: &[[f64; 4]; 4], point: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        dot([matrix[axis][0], matrix[axis][1], matrix[axis][2]], point) + matrix[axis][3]
    })
}

/// Fits positions into `[0, 1]^3` and minimizes world-space quantization MSE.
///
/// Seeds use a full PCA frame and a diagonal mixing of that frame. Coordinate
/// search refines all nine shape entries and six boundary margins (0..=2 grid
/// cells), with at most 1,600 attempts per seed plus the two seed evaluations.
/// Shape rows have unit length and absolute determinant at least 1e-4. This
/// restriction applies before anisotropic bound scaling. Small spans receive
/// a relative thickness of 1e-8; exact constant coordinates occupy grid level
/// zero in seed fits instead of a nonrepresentable cube midpoint.
///
/// This deterministic local heuristic does not guarantee a global optimum.
/// `refined` is never worse in MSE than either returned seed. Every original
/// position is contained within [`CONTAINMENT_TOLERANCE`]; no meaningful clipping
/// is used. Reported errors use the returned matrices and the public decode path.
///
/// Work is O(n) per candidate plus O(n) PCA preprocessing. Output preserves input
/// order. Extreme finite input can return [`FitError::NumericalRange`] when a
/// usable flattened affine transform or finite error statistics are unavailable.
pub fn fit_patch(positions: &[[f64; 3]], bits: u8) -> Result<PatchFit, FitError> {
    let q = levels(bits)?;
    if positions.is_empty() {
        return Err(FitError::EmptyPatch);
    }
    for (vertex, point) in positions.iter().enumerate() {
        for (component, value) in point.iter().enumerate() {
            if !value.is_finite() {
                return Err(FitError::NonFiniteCoordinate { vertex, component });
            }
        }
    }
    let mut min = positions[0];
    let mut max = min;
    for point in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    let center: [f64; 3] = std::array::from_fn(|i| min[i] * 0.5 + max[i] * 0.5);
    let scale = (0..3)
        .map(|i| (max[i] - center[i]).abs().max((min[i] - center[i]).abs()))
        .fold(0.0, f64::max);
    let scale = if scale == 0.0 { 1.0 } else { scale };
    if !scale.is_finite() || !(1.0 / scale).is_finite() {
        return Err(FitError::NumericalRange);
    }
    let centered: Vec<[f64; 3]> = positions
        .iter()
        .map(|p| std::array::from_fn(|i| (p[i] - center[i]) / scale))
        .collect();
    let frame = pca_frame(&centered);
    let axis = Candidate {
        shape: frame,
        low: [0.0; 3],
        high: [0.0; 3],
    };
    // Columns of D are (1,1,1), (1,-1,-1), (-1,1,-1).
    let diagonal = Candidate {
        shape: multiply(
            [[1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [1.0, -1.0, -1.0]],
            frame,
        ),
        low: [0.0; 3],
        high: [0.0; 3],
    };
    let context = Context {
        positions,
        centered: &centered,
        scale,
        bits,
        q,
    };
    let axis = axis.normalized().ok_or(FitError::NumericalRange)?;
    let diagonal = diagonal.normalized().ok_or(FitError::NumericalRange)?;
    let axis_seed = context.evaluate(&axis).ok_or(FitError::NumericalRange)?;
    let diagonal_seed = context
        .evaluate(&diagonal)
        .ok_or(FitError::NumericalRange)?;
    let mut evaluations = 2;
    let axis_refined = context.refine(axis, axis_seed.clone(), &mut evaluations);
    let diagonal_refined = context.refine(diagonal, diagonal_seed.clone(), &mut evaluations);
    let refined = if diagonal_refined.errors.mean_squared_distance
        < axis_refined.errors.mean_squared_distance
    {
        diagonal_refined
    } else {
        axis_refined
    };
    Ok(PatchFit {
        bits,
        axis_seed,
        diagonal_seed,
        refined,
        evaluations,
    })
}

fn levels(bits: u8) -> Result<f64, FitError> {
    if !(1..=20).contains(&bits) {
        return Err(FitError::InvalidBits);
    }
    Ok(((1u32 << bits) - 1) as f64)
}

#[derive(Clone, Copy)]
struct Candidate {
    shape: [[f64; 3]; 3],
    low: [f64; 3],
    high: [f64; 3],
}

impl Candidate {
    fn normalized(mut self) -> Option<Self> {
        for row in &mut self.shape {
            let length = dot(*row, *row).sqrt();
            if !length.is_finite() || length < 1e-12 {
                return None;
            }
            for value in row {
                *value /= length;
            }
        }
        if determinant(self.shape).abs() < 1e-4
            || self
                .low
                .iter()
                .chain(&self.high)
                .any(|m| !(0.0..=2.0).contains(m))
        {
            return None;
        }
        Some(self)
    }
}

struct Context<'a> {
    positions: &'a [[f64; 3]],
    centered: &'a [[f64; 3]],
    scale: f64,
    bits: u8,
    q: f64,
}

impl Context<'_> {
    fn evaluate(&self, candidate: &Candidate) -> Option<QuantizedFit> {
        let shape_inverse = inverse(candidate.shape)?;
        let mut forward = [[0.0; 4]; 4];
        let mut inverse = [[0.0; 4]; 4];
        forward[3][3] = 1.0;
        inverse[3][3] = 1.0;
        for axis in 0..3 {
            let row = candidate.shape[axis];
            let mut low = f64::INFINITY;
            let mut high = f64::NEG_INFINITY;
            for &point in self.centered {
                let projection = dot(row, point);
                low = low.min(projection);
                high = high.max(projection);
            }
            let width = (high - low).max(1e-8)
                * (1.0 + (candidate.low[axis] + candidate.high[axis]) / self.q)
                * (1.0 + 32.0 * f64::EPSILON);
            let world_width = width * self.scale;
            if !world_width.is_finite() || world_width <= 0.0 {
                return None;
            }
            let linear = row.map(|v| v / world_width);
            // Bounds use the exact flattened evaluation arithmetic, including its rounding.
            let low_flat = self
                .positions
                .iter()
                .map(|&p| dot(linear, p))
                .fold(f64::INFINITY, f64::min);
            for component in 0..3 {
                forward[axis][component] = linear[component];
                inverse[component][axis] = shape_inverse[component][axis] * world_width;
            }
            forward[axis][3] = -low_flat
                + candidate.low[axis] / (self.q + candidate.low[axis] + candidate.high[axis]);
        }
        for axis in 0..3 {
            inverse[axis][3] = -dot(
                [inverse[axis][0], inverse[axis][1], inverse[axis][2]],
                [forward[0][3], forward[1][3], forward[2][3]],
            );
        }
        if forward
            .iter()
            .flatten()
            .chain(inverse.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return None;
        }
        let mut fit = QuantizedFit {
            forward,
            inverse,
            quantized: Vec::with_capacity(self.positions.len()),
            errors: ErrorStats {
                mean_distance: 0.0,
                mean_squared_distance: 0.0,
                max_distance: 0.0,
            },
        };
        let count = self.positions.len() as f64;
        for &point in self.positions {
            let quantized = fit.encode_point(point, self.bits).ok()?;
            let decoded = fit.decode_point(quantized, self.bits);
            let distance = (decoded[0] - point[0])
                .hypot(decoded[1] - point[1])
                .hypot(decoded[2] - point[2]);
            let squared = distance * distance;
            if !squared.is_finite() {
                return None;
            }
            fit.errors.mean_distance += distance / count;
            fit.errors.mean_squared_distance += squared / count;
            fit.errors.max_distance = fit.errors.max_distance.max(distance);
            fit.quantized.push(quantized);
        }
        if !fit.errors.mean_distance.is_finite() || !fit.errors.mean_squared_distance.is_finite() {
            return None;
        }
        Some(fit)
    }

    fn refine(
        &self,
        mut candidate: Candidate,
        mut best: QuantizedFit,
        evaluations: &mut usize,
    ) -> QuantizedFit {
        let start = *evaluations;
        for (shape_step, margin_step) in [
            (0.3, 1.0),
            (0.1, 0.5),
            (0.03, 0.25),
            (0.01, 0.125),
            (0.003, 0.0625),
            (0.001, 0.03125),
        ] {
            for _ in 0..8 {
                let mut improved = false;
                for parameter in 0..15 {
                    for direction in [-1.0, 1.0] {
                        if *evaluations - start >= 1600 {
                            return best;
                        }
                        *evaluations += 1;
                        let mut neighbor = candidate;
                        if parameter < 9 {
                            neighbor.shape[parameter / 3][parameter % 3] += direction * shape_step;
                        } else if parameter < 12 {
                            neighbor.low[parameter - 9] += direction * margin_step;
                        } else {
                            neighbor.high[parameter - 12] += direction * margin_step;
                        }
                        let Some(neighbor) = neighbor.normalized() else {
                            continue;
                        };
                        let Some(fit) = self.evaluate(&neighbor) else {
                            continue;
                        };
                        if fit.errors.mean_squared_distance < best.errors.mean_squared_distance {
                            candidate = neighbor;
                            best = fit;
                            improved = true;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }
        best
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn determinant(a: [[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}
fn inverse(a: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = determinant(a);
    if !det.is_finite() || det.abs() < 1e-4 {
        return None;
    }
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (a[(j + 1) % 3][(i + 1) % 3] * a[(j + 2) % 3][(i + 2) % 3]
                - a[(j + 1) % 3][(i + 2) % 3] * a[(j + 2) % 3][(i + 1) % 3])
                / det
        })
    }))
}
fn multiply(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| dot(a[i], [b[0][j], b[1][j], b[2][j]])))
}

fn pca_frame(points: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let count = points.len() as f64;
    let mut mean = [0.0; 3];
    for p in points {
        for i in 0..3 {
            mean[i] += p[i] / count;
        }
    }
    let mut covariance = [[0.0; 3]; 3];
    for p in points {
        let d: [f64; 3] = std::array::from_fn(|i| p[i] - mean[i]);
        for i in 0..3 {
            for j in 0..3 {
                covariance[i][j] += d[i] * d[j] / count;
            }
        }
    }
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let mut pair = (0, 1);
        for candidate in [(0, 2), (1, 2)] {
            if covariance[candidate.0][candidate.1].abs() > covariance[pair.0][pair.1].abs() {
                pair = candidate;
            }
        }
        let (p, q) = pair;
        if covariance[p][q].abs()
            <= f64::EPSILON
                * (covariance[0][0].abs() + covariance[1][1].abs() + covariance[2][2].abs())
        {
            break;
        }
        let angle = 0.5 * (2.0 * covariance[p][q]).atan2(covariance[q][q] - covariance[p][p]);
        let (s, c) = angle.sin_cos();
        let mut rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        rotation[p][p] = c;
        rotation[q][q] = c;
        rotation[p][q] = s;
        rotation[q][p] = -s;
        let transpose = std::array::from_fn(|i| std::array::from_fn(|j| rotation[j][i]));
        covariance = multiply(multiply(transpose, covariance), rotation);
        vectors = multiply(vectors, rotation);
    }
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| {
        covariance[b][b]
            .total_cmp(&covariance[a][a])
            .then(a.cmp(&b))
    });
    std::array::from_fn(|i| std::array::from_fn(|j| vectors[j][order[i]]))
}
