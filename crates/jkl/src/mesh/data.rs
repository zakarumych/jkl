//! Owned mesh attributes, validation, and attribute-aware vertex welding.
use super::{fit::FitError, order::OrderError};
use std::{collections::HashMap, error::Error, fmt};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub tangents: Option<Vec<[f32; 4]>>,
    pub tex_coords: Vec<Vec<[f32; 2]>>,
    pub colors: Option<Vec<[u8; 4]>>,
    /// One primitive ID per triangle; an empty array means primitive zero.
    pub primitive_ids: Vec<u32>,
    /// Material ID indexed by primitive ID; empty means unspecified.
    pub materials: Vec<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncodingOptions {
    pub max_vertices: usize,
    pub position_bits: u8,
    pub uv_bits: u8,
    pub normal_bits: u8,
    pub tangent_bits: u8,
    pub tangent_uv_set: usize,
    pub weld_epsilon: Option<f32>,
}
impl Default for EncodingOptions {
    fn default() -> Self {
        Self {
            max_vertices: 256,
            position_bits: 10,
            uv_bits: 10,
            normal_bits: 10,
            tangent_bits: 10,
            tangent_uv_set: 0,
            weld_epsilon: Some(1e-6),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshError {
    InvalidInput(&'static str),
    InvalidData(&'static str),
    Fit(FitError),
    Order(OrderError),
}
impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid mesh input: {s}"),
            Self::InvalidData(s) => write!(f, "invalid compressed mesh: {s}"),
            Self::Fit(e) => e.fmt(f),
            Self::Order(e) => e.fmt(f),
        }
    }
}
impl Error for MeshError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Fit(e) => Some(e),
            Self::Order(e) => Some(e),
            _ => None,
        }
    }
}
impl From<FitError> for MeshError {
    fn from(e: FitError) -> Self {
        Self::Fit(e)
    }
}
impl From<OrderError> for MeshError {
    fn from(e: OrderError) -> Self {
        Self::Order(e)
    }
}

impl MeshData {
    /// Checks attribute lengths, finite values, index bounds, and primitive metadata.
    /// Zero normal/tangent directions are accepted for decoder fallback handling.
    pub fn validate(&self) -> Result<(), MeshError> {
        let n = self.positions.len();
        if n > u32::MAX as usize || self.triangles.len() > u32::MAX as usize {
            return Err(MeshError::InvalidInput("mesh exceeds u32 index range"));
        }
        fn attr<const N: usize>(a: &[[f32; N]], n: usize) -> Result<(), MeshError> {
            if a.len() != n {
                return Err(MeshError::InvalidInput("attribute length mismatch"));
            }
            if a.iter().flatten().any(|v| !v.is_finite()) {
                return Err(MeshError::InvalidInput("non-finite attribute"));
            }
            Ok(())
        }
        attr(&self.positions, n)?;
        if let Some(a) = &self.normals {
            attr(a, n)?;
        }
        if let Some(a) = &self.tangents {
            attr(a, n)?;
            if self.normals.is_none() {
                return Err(MeshError::InvalidInput("tangents require normals"));
            }
            if a.iter().any(|t| t[3] != 1.0 && t[3] != -1.0) {
                return Err(MeshError::InvalidInput(
                    "tangent handedness must be -1 or 1",
                ));
            }
        }
        for a in &self.tex_coords {
            attr(a, n)?;
        }
        if self.colors.as_ref().is_some_and(|a| a.len() != n) {
            return Err(MeshError::InvalidInput("color length mismatch"));
        }
        if self.triangles.iter().flatten().any(|&v| v as usize >= n) {
            return Err(MeshError::InvalidInput("triangle index out of range"));
        }
        if !self.primitive_ids.is_empty() && self.primitive_ids.len() != self.triangles.len() {
            return Err(MeshError::InvalidInput("primitive ID length mismatch"));
        }
        if !self.materials.is_empty()
            && self
                .primitive_ids
                .iter()
                .any(|&p| p as usize >= self.materials.len())
        {
            return Err(MeshError::InvalidInput(
                "primitive material index out of range",
            ));
        }
        Ok(())
    }
}
impl EncodingOptions {
    pub fn validate(&self, mesh: &MeshData) -> Result<(), MeshError> {
        mesh.validate()?;
        if !(1..=256).contains(&self.max_vertices) {
            return Err(MeshError::InvalidInput(
                "patch vertex count must be 1 through 256",
            ));
        }
        if [
            self.position_bits,
            self.uv_bits,
            self.normal_bits,
            self.tangent_bits,
        ]
        .into_iter()
        .any(|b| !(2..=20).contains(&b))
        {
            return Err(MeshError::InvalidInput(
                "quantization bits must be 2 through 20",
            ));
        }
        if self
            .weld_epsilon
            .is_some_and(|e| !e.is_finite() || e <= 0.0)
        {
            return Err(MeshError::InvalidInput(
                "weld epsilon must be finite and positive",
            ));
        }
        if mesh.tangents.is_some() && self.tangent_uv_set >= mesh.tex_coords.len() {
            return Err(MeshError::InvalidInput(
                "tangent prediction UV set is absent",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeldedMesh {
    pub mesh: MeshData,
    /// Source vertex chosen as each welded vertex's representative.
    pub source_vertices: Vec<u32>,
    pub old_to_new: Vec<u32>,
}

/// Welds to the earliest compatible representative, preserving attribute seams.
/// Positions use Euclidean distance; other float components use absolute distance.
/// Colors must match exactly. Triangle primitive metadata is retained independently.
/// Clusters are not transitive.
pub fn weld(mesh: &MeshData, epsilon: f32) -> Result<WeldedMesh, MeshError> {
    mesh.validate()?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(MeshError::InvalidInput(
            "weld epsilon must be finite and positive",
        ));
    }
    let e = epsilon as f64;
    fn close<const N: usize>(a: [f32; N], b: [f32; N], e: f64) -> bool {
        a.into_iter()
            .zip(b)
            .all(|(a, b)| (a as f64 - b as f64).abs() <= e)
    }
    let compatible = |a: usize, b: usize| {
        mesh.positions[a]
            .into_iter()
            .zip(mesh.positions[b])
            .map(|(a, b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>()
            <= e * e
            && mesh.normals.as_ref().is_none_or(|x| close(x[a], x[b], e))
            && mesh.tangents.as_ref().is_none_or(|x| close(x[a], x[b], e))
            && mesh.tex_coords.iter().all(|x| close(x[a], x[b], e))
            && mesh.colors.as_ref().is_none_or(|x| x[a] == x[b])
    };
    fn key(c: [f64; 3]) -> [u64; 3] {
        c.map(|v| if v == 0.0 { 0 } else { v.to_bits() })
    }
    let mut buckets: HashMap<[u64; 3], Vec<u32>> = HashMap::new();
    let mut source_vertices = Vec::<u32>::new();
    let mut old_to_new = Vec::with_capacity(mesh.positions.len());
    for (i, p) in mesh.positions.iter().enumerate() {
        let cell = p.map(|v| (v as f64 / e).floor());
        let mut found = None;
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if let Some(candidates) = buckets.get(&key([
                        cell[0] + x as f64,
                        cell[1] + y as f64,
                        cell[2] + z as f64,
                    ])) {
                        for &j in candidates {
                            if found.is_none_or(|best| j < best)
                                && compatible(i, source_vertices[j as usize] as usize)
                            {
                                found = Some(j);
                            }
                        }
                    }
                }
            }
        }
        let index = found.unwrap_or_else(|| {
            let j = source_vertices.len() as u32;
            source_vertices.push(i as u32);
            buckets.entry(key(cell)).or_default().push(j);
            j
        });
        old_to_new.push(index);
    }
    fn gather<T: Copy>(a: &[T], ids: &[u32]) -> Vec<T> {
        ids.iter().map(|&i| a[i as usize]).collect()
    }
    let result = MeshData {
        positions: gather(&mesh.positions, &source_vertices),
        triangles: mesh
            .triangles
            .iter()
            .map(|t| t.map(|v| old_to_new[v as usize]))
            .collect(),
        normals: mesh.normals.as_ref().map(|a| gather(a, &source_vertices)),
        tangents: mesh.tangents.as_ref().map(|a| gather(a, &source_vertices)),
        tex_coords: mesh
            .tex_coords
            .iter()
            .map(|a| gather(a, &source_vertices))
            .collect(),
        colors: mesh.colors.as_ref().map(|a| gather(a, &source_vertices)),
        primitive_ids: mesh.primitive_ids.clone(),
        materials: mesh.materials.clone(),
    };
    Ok(WeldedMesh {
        mesh: result,
        source_vertices,
        old_to_new,
    })
}
