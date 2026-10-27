//! Patch preparation and independent CPU reconstruction.
use super::{
    data::{EncodingOptions, MeshData, MeshError, weld},
    fit,
    layout::build_layout,
    normal, spatial, tangent,
};

/// Attributes and quantization settings shared by every patch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttributeSchema {
    pub uv_sets: u32,
    pub has_normals: bool,
    pub has_tangents: bool,
    pub has_colors: bool,
    pub tangent_uv_set: u32,
    pub position_bits: u8,
    pub uv_bits: u8,
    pub normal_bits: u8,
    pub tangent_bits: u8,
}
/// One entropy context per role, shared across all patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    Indices,
    Primitive,
    Position,
    Uv(u32),
    Normal,
    TangentAngle,
    Handedness,
    Color,
}
impl AttributeSchema {
    pub fn stream_kinds(&self) -> Vec<StreamKind> {
        let mut kinds = vec![
            StreamKind::Indices,
            StreamKind::Primitive,
            StreamKind::Position,
        ];
        kinds.extend((0..self.uv_sets).map(StreamKind::Uv));
        if self.has_normals {
            kinds.push(StreamKind::Normal);
        }
        if self.has_tangents {
            kinds.extend([StreamKind::TangentAngle, StreamKind::Handedness]);
        }
        if self.has_colors {
            kinds.push(StreamKind::Color);
        }
        kinds
    }
    pub fn validate(&self) -> Result<(), MeshError> {
        if self.uv_sets.checked_add(7).is_none()
            || self.uv_sets as u64 + 7 > (usize::MAX / std::mem::size_of::<StreamKind>()) as u64
        {
            return Err(MeshError::InvalidData("attribute stream count overflow"));
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
            return Err(MeshError::InvalidData("attribute precision"));
        }
        if self.has_tangents && (!self.has_normals || self.tangent_uv_set >= self.uv_sets) {
            return Err(MeshError::InvalidData("tangent dependencies"));
        }
        Ok(())
    }
}
/// Disjoint output ranges and serialized patch transforms.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchInfo {
    pub vertex_base: u32,
    pub triangle_base: u32,
    pub internal_triangle_count: u32,
    pub triangle_count: u32,
    pub internal_vertex_count: u32,
    pub vertex_count: u32,
    pub inverse: [[f32; 4]; 4],
    pub uv_bounds: Vec<[f32; 4]>,
}
#[derive(Clone, Debug)]
pub struct SymbolPatch {
    pub info: PatchInfo,
    pub streams: Vec<Vec<u32>>,
}
#[derive(Clone, Debug)]
pub struct PreparedMesh {
    pub schema: AttributeSchema,
    pub patches: Vec<SymbolPatch>,
    pub materials: Vec<i32>,
    pub vertex_count: u32,
    pub triangle_count: u32,
}
/// Owned patch attributes; triangle indices refer to the assembled global mesh.
#[derive(Clone, Debug)]
pub struct DecodedPatch {
    pub vertex_base: u32,
    pub triangle_base: u32,
    pub data: MeshData,
}

/// Encode topology slot deltas, converting the predecessor to global space at the tail.
pub fn encode_indices(
    head: &[[u32; 3]],
    tail: &[[u32; 3]],
    base: u32,
) -> Result<Vec<u32>, MeshError> {
    let mut previous = [0i64; 3];
    let mut out = Vec::with_capacity((head.len() + tail.len()) * 3);
    for &t in head {
        for k in 0..3 {
            out.push(normal::zigzag(t[k] as i64 - previous[k])?);
        }
        previous = t.map(i64::from);
    }
    if !head.is_empty() {
        for p in &mut previous {
            *p += base as i64;
        }
    }
    for &t in tail {
        for k in 0..3 {
            out.push(normal::zigzag(t[k] as i64 - previous[k])?);
        }
        previous = t.map(i64::from);
    }
    Ok(out)
}
/// Decode stored triangles without undoing odd-triangle reversal.
pub fn decode_indices(
    symbols: &[u32],
    head_count: usize,
    base: u32,
) -> Result<(Vec<[u32; 3]>, Vec<[u32; 3]>), MeshError> {
    if symbols.len() % 3 != 0 || head_count > symbols.len() / 3 {
        return Err(MeshError::InvalidData("index stream length"));
    }
    let mut previous = [0i64; 3];
    let mut triangles = Vec::with_capacity(symbols.len() / 3);
    for (i, s) in symbols.chunks_exact(3).enumerate() {
        if i == head_count && head_count != 0 {
            for p in &mut previous {
                *p += base as i64;
            }
        }
        let mut t = [0; 3];
        for k in 0..3 {
            previous[k] += normal::unzigzag(s[k]);
            t[k] = u32::try_from(previous[k])
                .map_err(|_| MeshError::InvalidData("index delta overflow"))?;
        }
        triangles.push(t);
    }
    let tail = triangles.split_off(head_count);
    Ok((triangles, tail))
}

/// Prepare quantized symbols with every predictor evaluated from decoded attributes.
///
/// ```
/// use jkl::mesh::{codec, data::{MeshData, EncodingOptions}};
/// let mesh = MeshData {
///     positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
///     triangles: vec![[0, 1, 2]],
///     ..MeshData::default()
/// };
/// let prepared = codec::prepare(&mesh, &EncodingOptions::default())?;
/// // Patches can be decoded independently and in any order.
/// let patches = prepared.patches.iter().rev()
///     .map(|p| codec::decode_patch(&prepared.schema, &p.info, &p.streams))
///     .collect::<Result<Vec<_>, _>>()?;
/// let decoded = codec::assemble(&prepared.schema, &patches, prepared.materials)?;
/// assert_eq!(decoded.positions.len(), 3);
/// assert_eq!(decoded.triangles.len(), 1);
/// # Ok::<(), jkl::mesh::data::MeshError>(())
/// ```
pub fn prepare(input: &MeshData, options: &EncodingOptions) -> Result<PreparedMesh, MeshError> {
    input.validate()?;
    options.validate(input)?;
    let welded = if let Some(epsilon) = options.weld_epsilon {
        Some(weld(input, epsilon)?)
    } else {
        None
    };
    let mesh = welded.as_ref().map_or(input, |w| &w.mesh);
    let layout = build_layout(mesh, options.max_vertices)?;
    let schema = AttributeSchema {
        uv_sets: mesh.tex_coords.len() as u32,
        has_normals: mesh.normals.is_some(),
        has_tangents: mesh.tangents.is_some(),
        has_colors: mesh.colors.is_some(),
        tangent_uv_set: options.tangent_uv_set as u32,
        position_bits: options.position_bits,
        uv_bits: options.uv_bits,
        normal_bits: options.normal_bits,
        tangent_bits: options.tangent_bits,
    };
    schema.validate()?;
    let mut patches = Vec::with_capacity(layout.patches.len());
    for p in &layout.patches {
        let source: Vec<_> = p
            .vertices
            .iter()
            .map(|&v| mesh.positions[v as usize].map(f64::from))
            .collect();
        let fitted = fit::fit_patch(&source, options.position_bits)?.refined;
        let inverse = fitted.inverse.map(|row| row.map(|x| x as f32));
        if inverse.iter().flatten().any(|x| !x.is_finite()) {
            return Err(MeshError::InvalidInput(
                "position transform exceeds f32 range",
            ));
        }
        let scale = ((1u32 << options.position_bits) - 1) as f64;
        let quantized: Vec<_> = source
            .iter()
            .map(|&v| {
                fitted
                    .normalized_point(v)
                    .map(|c| (c.clamp(0., 1.) * scale).round_ties_even() as u32)
            })
            .collect();
        let pos_symbols = spatial::encode(&p.head, &quantized, options.position_bits)?;
        let (q, order) =
            spatial::decode::<3>(&p.head, source.len(), options.position_bits, &pos_symbols)?;
        let positions = spatial::positions(&q, &inverse, options.position_bits)?;
        if positions.iter().flatten().any(|x| !x.is_finite()) {
            return Err(MeshError::InvalidInput(
                "reconstructed position exceeds f32 range",
            ));
        }
        let mut primitive = Vec::with_capacity(p.head.len() + p.tail.len());
        for &face in p.head_face_ids.iter().chain(&p.tail_face_ids) {
            primitive.push(mesh.primitive_ids.get(face as usize).copied().unwrap_or(0));
        }
        let mut streams = vec![
            encode_indices(&p.head, &p.tail, p.vertex_base)?,
            primitive,
            pos_symbols,
        ];
        let mut uv_bounds = Vec::new();
        let mut tangent_uv = Vec::new();
        for (set, values) in mesh.tex_coords.iter().enumerate() {
            let local: Vec<_> = p.vertices.iter().map(|&v| values[v as usize]).collect();
            let (q, bounds) = spatial::quantize_uv(&local, options.uv_bits)?;
            if bounds.iter().any(|x| !x.is_finite()) {
                return Err(MeshError::InvalidInput("UV transform exceeds f32 range"));
            }
            streams.push(spatial::encode(&p.head, &q, options.uv_bits)?);
            if set == options.tangent_uv_set {
                tangent_uv = q;
            }
            uv_bounds.push(bounds);
        }
        let decoded_normals = if let Some(normals) = &mesh.normals {
            let local: Vec<_> = p.vertices.iter().map(|&v| normals[v as usize]).collect();
            let predictions = normal::geometric(&positions, &p.head)?;
            let (symbols, decoded) = normal::stream(
                Some(&local),
                &predictions,
                &order,
                p.internal_vertex_count as usize,
                options.normal_bits,
                &[],
            )?;
            streams.push(symbols);
            Some(decoded)
        } else {
            None
        };
        if let Some(tangents) = &mesh.tangents {
            let normals = decoded_normals
                .as_ref()
                .ok_or(MeshError::InvalidInput("tangent normals missing"))?;
            let predictions =
                tangent::predictors_quantized(&positions, &tangent_uv, normals, &p.head)?;
            let mut angles = Vec::new();
            let mut signs = Vec::new();
            for (i, &v) in p.vertices.iter().enumerate() {
                let (angle, sign) = tangent::encode(
                    tangents[v as usize],
                    normals[i],
                    predictions[i],
                    options.tangent_bits,
                )?;
                angles.push(angle);
                signs.push(sign);
            }
            streams.push(angles);
            streams.push(signs);
        }
        if let Some(colors) = &mesh.colors {
            streams.push(
                p.vertices
                    .iter()
                    .map(|&v| u32::from_le_bytes(colors[v as usize]))
                    .collect(),
            );
        }
        patches.push(SymbolPatch {
            info: PatchInfo {
                vertex_base: p.vertex_base,
                triangle_base: p.triangle_base,
                internal_triangle_count: p.head.len() as u32,
                triangle_count: (p.head.len() + p.tail.len()) as u32,
                internal_vertex_count: p.internal_vertex_count,
                vertex_count: p.vertices.len() as u32,
                inverse,
                uv_bounds,
            },
            streams,
        });
    }
    Ok(PreparedMesh {
        schema,
        patches,
        materials: mesh.materials.clone(),
        vertex_count: mesh.positions.len() as u32,
        triangle_count: mesh.triangles.len() as u32,
    })
}

/// Reconstruct one patch without reading attributes owned by another patch.
pub fn decode_patch(
    schema: &AttributeSchema,
    info: &PatchInfo,
    streams: &[Vec<u32>],
) -> Result<DecodedPatch, MeshError> {
    schema.validate()?;
    if info.internal_triangle_count > info.triangle_count
        || info.internal_vertex_count > info.vertex_count
        || info.vertex_count == 0
        || info.vertex_count > 256
        || info.uv_bounds.len() != schema.uv_sets as usize
        || info.inverse.iter().flatten().any(|x| !x.is_finite())
        || info.uv_bounds.iter().flatten().any(|x| !x.is_finite())
        || info.uv_bounds.iter().any(|b| b[2] < 0. || b[3] < 0.)
    {
        return Err(MeshError::InvalidData("patch metadata"));
    }
    info.vertex_base
        .checked_add(info.vertex_count)
        .ok_or(MeshError::InvalidData("vertex range overflow"))?;
    info.triangle_base
        .checked_add(info.triangle_count)
        .ok_or(MeshError::InvalidData("triangle range overflow"))?;
    if streams.len() != schema.stream_kinds().len() {
        return Err(MeshError::InvalidData("attribute stream count"));
    }
    if streams[0].len() != info.triangle_count as usize * 3
        || streams[1].len() != info.triangle_count as usize
    {
        return Err(MeshError::InvalidData("triangle stream length"));
    }
    let (head, tail) = decode_indices(
        &streams[0],
        info.internal_triangle_count as usize,
        info.vertex_base,
    )?;
    let count = info.vertex_count as usize;
    let mut used = vec![false; count];
    for t in &head {
        for &v in t {
            if v as usize >= count {
                return Err(MeshError::InvalidData("local index ownership"));
            }
            used[v as usize] = true;
        }
    }
    if used
        .iter()
        .enumerate()
        .any(|(i, &used)| used != (i < info.internal_vertex_count as usize))
    {
        return Err(MeshError::InvalidData("internal vertex count"));
    }
    let (q, order) = spatial::decode::<3>(&head, count, schema.position_bits, &streams[2])?;
    if order.iter().enumerate().any(|(i, &id)| i != id) {
        return Err(MeshError::InvalidData("vertex first appearance order"));
    }
    let positions = spatial::positions(&q, &info.inverse, schema.position_bits)?;
    let mut cursor = 3;
    let mut tex_coords = Vec::new();
    let mut tangent_uv = Vec::new();
    for (set, bounds) in info.uv_bounds.iter().enumerate() {
        let (q, _) = spatial::decode::<2>(&head, count, schema.uv_bits, &streams[cursor])?;
        tex_coords.push(spatial::tex_coords(&q, bounds, schema.uv_bits)?);
        if set == schema.tangent_uv_set as usize {
            tangent_uv = q;
        }
        cursor += 1;
    }
    let normals = if schema.has_normals {
        let predictions = normal::geometric(&positions, &head)?;
        let (_, normals) = normal::stream(
            None,
            &predictions,
            &order,
            info.internal_vertex_count as usize,
            schema.normal_bits,
            &streams[cursor],
        )?;
        cursor += 1;
        Some(normals)
    } else {
        None
    };
    let tangents = if schema.has_tangents {
        let normals = normals.as_ref().unwrap();
        let predictions = tangent::predictors_quantized(&positions, &tangent_uv, normals, &head)?;
        if streams[cursor].len() != count || streams[cursor + 1].len() != count {
            return Err(MeshError::InvalidData("tangent stream length"));
        }
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            values.push(tangent::decode(
                streams[cursor][i],
                streams[cursor + 1][i],
                normals[i],
                predictions[i],
                schema.tangent_bits,
            )?);
        }
        cursor += 2;
        Some(values)
    } else {
        None
    };
    let colors = if schema.has_colors {
        if streams[cursor].len() != count {
            return Err(MeshError::InvalidData("color stream length"));
        }
        Some(streams[cursor].iter().map(|&c| c.to_le_bytes()).collect())
    } else {
        None
    };
    let mut triangles: Vec<_> = head
        .into_iter()
        .map(|t| t.map(|v| v + info.vertex_base))
        .chain(tail)
        .collect();
    for (i, t) in triangles.iter_mut().enumerate() {
        if i & 1 != 0 {
            t.swap(1, 2);
        }
    }
    if positions
        .iter()
        .flatten()
        .chain(tex_coords.iter().flatten().flatten())
        .any(|x| !x.is_finite())
    {
        return Err(MeshError::InvalidData("nonfinite reconstructed attribute"));
    }
    Ok(DecodedPatch {
        vertex_base: info.vertex_base,
        triangle_base: info.triangle_base,
        data: MeshData {
            positions,
            triangles,
            normals,
            tangents,
            tex_coords,
            colors,
            primitive_ids: streams[1].clone(),
            materials: Vec::new(),
        },
    })
}

/// Assemble independently decoded patches, checking contiguous disjoint ranges.
pub fn assemble(
    schema: &AttributeSchema,
    patches: &[DecodedPatch],
    materials: Vec<i32>,
) -> Result<MeshData, MeshError> {
    schema.validate()?;
    let mut vertex_order: Vec<_> = patches.iter().collect();
    vertex_order.sort_by_key(|p| p.vertex_base);
    let mut mesh = MeshData {
        normals: schema.has_normals.then(Vec::new),
        tangents: schema.has_tangents.then(Vec::new),
        colors: schema.has_colors.then(Vec::new),
        tex_coords: vec![Vec::new(); schema.uv_sets as usize],
        materials,
        ..MeshData::default()
    };
    for p in vertex_order {
        let d = &p.data;
        if p.vertex_base as usize != mesh.positions.len()
            || d.tex_coords.len() != schema.uv_sets as usize
            || d.normals.is_some() != schema.has_normals
            || d.tangents.is_some() != schema.has_tangents
            || d.colors.is_some() != schema.has_colors
        {
            return Err(MeshError::InvalidData("patch attribute ranges"));
        }
        let n = d.positions.len();
        if d.normals.as_ref().is_some_and(|v| v.len() != n)
            || d.tangents.as_ref().is_some_and(|v| v.len() != n)
            || d.colors.as_ref().is_some_and(|v| v.len() != n)
            || d.tex_coords.iter().any(|v| v.len() != n)
        {
            return Err(MeshError::InvalidData("patch attribute lengths"));
        }
        mesh.positions.extend_from_slice(&d.positions);
        if let (Some(out), Some(input)) = (&mut mesh.normals, &d.normals) {
            out.extend_from_slice(input);
        }
        if let (Some(out), Some(input)) = (&mut mesh.tangents, &d.tangents) {
            out.extend_from_slice(input);
        }
        if let (Some(out), Some(input)) = (&mut mesh.colors, &d.colors) {
            out.extend_from_slice(input);
        }
        for (out, input) in mesh.tex_coords.iter_mut().zip(&d.tex_coords) {
            out.extend_from_slice(input);
        }
    }
    let mut triangle_order: Vec<_> = patches.iter().collect();
    triangle_order.sort_by_key(|p| p.triangle_base);
    for p in triangle_order {
        if p.triangle_base as usize != mesh.triangles.len()
            || p.data.primitive_ids.len() != p.data.triangles.len()
        {
            return Err(MeshError::InvalidData("patch triangle ranges"));
        }
        mesh.triangles.extend_from_slice(&p.data.triangles);
        mesh.primitive_ids.extend_from_slice(&p.data.primitive_ids);
    }
    mesh.validate()?;
    Ok(mesh)
}
