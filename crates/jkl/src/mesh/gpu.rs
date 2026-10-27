//! Backend-independent GPU vertex layout and decoder metadata packing.
use super::{
    codec::{AttributeSchema, PatchInfo},
    data::MeshError,
};
pub const DECODER_WGSL: &str = include_str!("shaders/decode.wgsl");
/// Interleaved GPU attributes. Offsets and stride are measured in bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexLayout {
    pub stride: u32,
    pub position: u32,
    pub normal: Option<u32>,
    pub tangent: Option<u32>,
    pub tex_coords: Vec<u32>,
    pub color: Option<u32>,
}
impl VertexLayout {
    pub fn new(schema: &AttributeSchema) -> Result<Self, MeshError> {
        schema.validate()?;
        let mut words = 3u32;
        let normal = schema.has_normals.then(|| {
            let p = words * 4;
            words += 3;
            p
        });
        let tangent = schema.has_tangents.then(|| {
            let p = words * 4;
            words += 4;
            p
        });
        let mut tex_coords = Vec::new();
        for _ in 0..schema.uv_sets {
            tex_coords.push(
                words
                    .checked_mul(4)
                    .ok_or(MeshError::InvalidData("GPU vertex stride overflow"))?,
            );
            words = words
                .checked_add(2)
                .ok_or(MeshError::InvalidData("GPU vertex stride overflow"))?;
        }
        let color = if schema.has_colors {
            let p = words
                .checked_mul(4)
                .ok_or(MeshError::InvalidData("GPU vertex stride overflow"))?;
            words = words
                .checked_add(1)
                .ok_or(MeshError::InvalidData("GPU vertex stride overflow"))?;
            Some(p)
        } else {
            None
        };
        Ok(Self {
            stride: words
                .checked_mul(4)
                .ok_or(MeshError::InvalidData("GPU vertex stride overflow"))?,
            position: 0,
            normal,
            tangent,
            tex_coords,
            color,
        })
    }
}
/// Packs algorithm metadata, followed by patch transforms and entropy descriptors.
/// Each stream descriptor is `[payload_offset,payload_len,table_offset,table_len,symbol_count]`.
/// No attribute symbols are decoded on the host.
pub fn metadata(
    schema: &AttributeSchema,
    patches: &[PatchInfo],
    streams: &[[u32; 5]],
    vertex_count: u32,
    triangle_count: u32,
) -> Result<(Vec<u32>, VertexLayout), MeshError> {
    let layout = VertexLayout::new(schema)?;
    let count = u32::try_from(schema.stream_kinds().len())
        .map_err(|_| MeshError::InvalidData("GPU stream count overflow"))?;
    if patches.len().checked_mul(count as usize) != Some(streams.len()) {
        return Err(MeshError::InvalidData("GPU stream descriptor count"));
    }
    let patch_stride = 22u32
        .checked_add(
            schema
                .uv_sets
                .checked_mul(4)
                .ok_or(MeshError::InvalidData("GPU metadata overflow"))?,
        )
        .and_then(|v| v.checked_add(count.checked_mul(5)?))
        .ok_or(MeshError::InvalidData("GPU metadata overflow"))?;
    let patch_count = u32::try_from(patches.len())
        .map_err(|_| MeshError::InvalidData("GPU patch count overflow"))?;
    patch_count
        .checked_mul(patch_stride)
        .and_then(|n| n.checked_add(16))
        .ok_or(MeshError::InvalidData("GPU metadata overflow"))?;
    let flags = u32::from(schema.has_normals)
        | (u32::from(schema.has_tangents) << 1)
        | (u32::from(schema.has_colors) << 2);
    let mut out = vec![
        patch_count,
        vertex_count,
        triangle_count,
        schema.uv_sets,
        flags,
        schema.tangent_uv_set,
        schema.position_bits as u32,
        schema.uv_bits as u32,
        schema.normal_bits as u32,
        schema.tangent_bits as u32,
        layout.stride / 4,
        patch_stride,
        count,
        0,
        0,
        0,
    ];
    for (i, p) in patches.iter().enumerate() {
        if p.uv_bounds.len() != schema.uv_sets as usize {
            return Err(MeshError::InvalidData("GPU UV bounds count"));
        }
        out.extend([
            p.vertex_base,
            p.triangle_base,
            p.internal_triangle_count,
            p.triangle_count,
            p.internal_vertex_count,
            p.vertex_count,
        ]);
        out.extend(p.inverse.iter().flatten().map(|x| x.to_bits()));
        out.extend(p.uv_bounds.iter().flatten().map(|x| x.to_bits()));
        for s in &streams[i * count as usize..(i + 1) * count as usize] {
            out.extend(s);
        }
    }
    Ok((out, layout))
}
