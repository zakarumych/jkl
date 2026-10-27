//! Versioned, seekable mesh container with shared entropy tables.
//!
//! Version 1 stores little-endian `JKLM`, version, attribute flags, UV-set count,
//! tangent UV set, four precision bytes, vertex/triangle/patch/material/stream
//! counts, then table/directory/payload byte lengths (`u64`). The 68-byte header
//! is followed by material IDs (`i32`), frequency tables, patch records, payload.
//! Each table is an entry count followed by sorted (symbol, occurrence count)
//! pairs. Patch records contain six counts/bases, a row-major inverse matrix,
//! UV bounds, and stream descriptors (symbol count, byte offset, word count).
//! Offsets are relative to payload start. Nonempty rANS streams store emitted
//! words followed by final-state low/high words; decoding consumes backwards.
//!
//! GPU preparation validates structure, not compressed symbol content. GPU
//! execution requires trusted payloads or prior CPU decoding validation.
use crate::algos::ans;
use crate::mesh::{
    codec::{self, AttributeSchema, DecodedPatch, PatchInfo, PreparedMesh, StreamKind},
    compress::{self, CompressedMesh, EncodedStream, Frequency},
    data::{EncodingOptions, MeshData},
};
use std::io::{self, Read, Seek, SeekFrom, Write};

pub use crate::mesh::data::EncodingOptions as Options;
const HEADER_SIZE: u64 = 68;
pub const VERSION: u32 = 1;

fn invalid(s: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s)
}
fn mesh_error(e: crate::mesh::data::MeshError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}
fn read32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn put32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn put64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn len32(n: usize) -> io::Result<u32> {
    u32::try_from(n).map_err(|_| invalid("mesh exceeds u32 addressing"))
}
fn add(a: u64, b: u64) -> io::Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("mesh size overflow"))
}

#[derive(Clone, Copy, Debug)]
pub struct Stream {
    pub symbol_count: u32,
    pub payload_offset: u64,
    pub word_count: u32,
}
#[derive(Clone, Debug)]
pub struct Patch {
    pub info: PatchInfo,
    pub streams: Vec<Stream>,
}

/// Normalized rANS entry; all fields map directly to shader `u32` values.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct GpuEntry {
    pub freq: u32,
    pub cumul: u32,
    pub symbol: u32,
}
/// Patch-major stream descriptor. Payload offsets count words, table offsets entries.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct GpuStream {
    pub payload_offset: u32,
    pub payload_len: u32,
    pub table_offset: u32,
    pub table_len: u32,
    pub symbol_count: u32,
}
/// Upload preparation, without CPU decompression. Each patch owns independent streams.
#[derive(Clone, Debug)]
pub struct PreparedGpuMesh {
    pub schema: AttributeSchema,
    pub patches: Vec<PatchInfo>,
    pub streams: Vec<GpuStream>,
    pub table: Vec<GpuEntry>,
    pub payload: Vec<u32>,
    pub materials: Vec<i32>,
    pub vertex_count: u32,
    pub triangle_count: u32,
}

/// Preprocesses and encodes mesh attributes, then writes the versioned container.
pub fn write_mesh(
    mesh: &MeshData,
    options: &EncodingOptions,
    writer: &mut impl Write,
) -> io::Result<()> {
    let prepared = codec::prepare(mesh, options).map_err(mesh_error)?;
    write_prepared(&prepared, writer)
}
pub fn write_prepared(mesh: &PreparedMesh, writer: &mut impl Write) -> io::Result<()> {
    let compressed = compress::compress(mesh).map_err(mesh_error)?;
    write_compressed(&compressed, writer)
}

/// Serializes independently compressed patch streams and shared frequency tables.
pub fn write_compressed(mesh: &CompressedMesh, writer: &mut impl Write) -> io::Result<()> {
    validate_schema(&mesh.schema)?;
    let streams = mesh.schema.stream_kinds();
    if mesh.tables.len() != streams.len() {
        return Err(invalid("entropy table count mismatch"));
    }
    let mut tables = Vec::new();
    for (kind, table) in streams.iter().zip(&mesh.tables) {
        compress::context(table).map_err(mesh_error)?;
        if table.iter().map(|e| u64::from(e.count)).sum::<u64>()
            != mesh_symbols(kind, mesh.vertex_count, mesh.triangle_count)
        {
            return Err(invalid("entropy frequency total mismatch"));
        }
        put32(&mut tables, len32(table.len())?)?;
        for e in table {
            put32(&mut tables, e.symbol)?;
            put32(&mut tables, e.count)?;
        }
    }
    let mut directory = Vec::new();
    let mut payload = Vec::new();
    let (mut vertices, mut triangles) = (0u32, 0u32);
    for patch in &mesh.patches {
        validate_patch(&mesh.schema, &patch.info, vertices, triangles)?;
        vertices = vertices
            .checked_add(patch.info.vertex_count)
            .ok_or_else(|| invalid("vertex count overflow"))?;
        triangles = triangles
            .checked_add(patch.info.triangle_count)
            .ok_or_else(|| invalid("triangle count overflow"))?;
        if patch.streams.len() != streams.len() {
            return Err(invalid("patch stream count mismatch"));
        }
        write_info(&mut directory, &patch.info)?;
        for (kind, stream) in streams.iter().zip(&patch.streams) {
            validate_stream(
                kind,
                &patch.info,
                stream.symbol_count,
                len32(stream.words.len())?,
            )?;
            put32(&mut directory, stream.symbol_count)?;
            put64(&mut directory, payload.len() as u64)?;
            put32(&mut directory, len32(stream.words.len())?)?;
            for &word in &stream.words {
                put32(&mut payload, word)?;
            }
        }
    }
    if vertices != mesh.vertex_count || triangles != mesh.triangle_count {
        return Err(invalid("mesh output counts disagree"));
    }
    writer.write_all(b"JKLM")?;
    put32(writer, VERSION)?;
    let s = &mesh.schema;
    put32(
        writer,
        u32::from(s.has_normals) | u32::from(s.has_tangents) << 1 | u32::from(s.has_colors) << 2,
    )?;
    put32(writer, s.uv_sets)?;
    put32(writer, s.tangent_uv_set)?;
    writer.write_all(&[s.position_bits, s.uv_bits, s.normal_bits, s.tangent_bits])?;
    for n in [
        mesh.vertex_count,
        mesh.triangle_count,
        len32(mesh.patches.len())?,
        len32(mesh.materials.len())?,
        len32(streams.len())?,
    ] {
        put32(writer, n)?;
    }
    for n in [tables.len(), directory.len(), payload.len()] {
        put64(writer, n as u64)?;
    }
    for &material in &mesh.materials {
        put32(writer, material as u32)?;
    }
    writer.write_all(&tables)?;
    writer.write_all(&directory)?;
    writer.write_all(&payload)
}
fn validate_schema(s: &AttributeSchema) -> io::Result<()> {
    s.validate().map_err(mesh_error)
}
fn validate_patch(s: &AttributeSchema, p: &PatchInfo, v: u32, t: u32) -> io::Result<()> {
    if p.vertex_base != v
        || p.triangle_base != t
        || !(1..=256).contains(&p.vertex_count)
        || p.internal_triangle_count > p.triangle_count
        || p.internal_vertex_count > p.vertex_count
        || (p.internal_triangle_count == 0) != (p.internal_vertex_count == 0)
        || p.uv_bounds.len() != s.uv_sets as usize
        || p.inverse.iter().flatten().any(|v| !v.is_finite())
        || p.uv_bounds.iter().flatten().any(|v| !v.is_finite())
        || p.uv_bounds.iter().any(|b| b[2] < 0.0 || b[3] < 0.0)
    {
        return Err(invalid("invalid patch metadata"));
    }
    Ok(())
}
fn expected_symbols(kind: &StreamKind, p: &PatchInfo) -> u64 {
    mesh_symbols(kind, p.vertex_count, p.triangle_count)
}
fn mesh_symbols(kind: &StreamKind, vertices: u32, triangles: u32) -> u64 {
    let n = u64::from(vertices);
    match kind {
        StreamKind::Indices => u64::from(triangles) * 3,
        StreamKind::Primitive => u64::from(triangles),
        StreamKind::Position => n * 3,
        StreamKind::Uv(_) | StreamKind::Normal => n * 2,
        StreamKind::TangentAngle | StreamKind::Handedness | StreamKind::Color => n,
    }
}
fn validate_stream(kind: &StreamKind, p: &PatchInfo, n: u32, words: u32) -> io::Result<()> {
    if u64::from(n) != expected_symbols(kind, p)
        || (n == 0 && words != 0)
        || (n != 0 && (words < 2 || u64::from(words) > u64::from(n) + 2))
    {
        return Err(invalid("invalid patch stream length"));
    }
    Ok(())
}
fn write_info(w: &mut impl Write, p: &PatchInfo) -> io::Result<()> {
    for v in [
        p.vertex_base,
        p.triangle_base,
        p.internal_triangle_count,
        p.triangle_count,
        p.internal_vertex_count,
        p.vertex_count,
    ] {
        put32(w, v)?;
    }
    for &v in p.inverse.iter().flatten() {
        put32(w, v.to_bits())?;
    }
    for &v in p.uv_bounds.iter().flatten() {
        put32(w, v.to_bits())?;
    }
    Ok(())
}

/// Opening validates directories and contexts. Reading a patch validates its symbols.
pub struct JackalMeshReader<R> {
    reader: R,
    schema: AttributeSchema,
    patches: Vec<Patch>,
    materials: Vec<i32>,
    contexts: Vec<ans::Context<u32>>,
    vertex_count: u32,
    triangle_count: u32,
    payload_start: u64,
}
impl<R: Read + Seek> JackalMeshReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        let start = reader.stream_position()?;
        let end = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(start))?;
        if end.checked_sub(start).is_none_or(|size| size < HEADER_SIZE) {
            return Err(invalid("truncated mesh header"));
        }
        let mut magic = [0; 4];
        reader.read_exact(&mut magic)?;
        if magic != *b"JKLM" || read32(&mut reader)? != VERSION {
            return Err(invalid("unsupported mesh header"));
        }
        let flags = read32(&mut reader)?;
        if flags & !7 != 0 {
            return Err(invalid("unknown mesh flags"));
        }
        let uv_sets = read32(&mut reader)?;
        let tangent_uv_set = read32(&mut reader)?;
        let mut bits = [0; 4];
        reader.read_exact(&mut bits)?;
        let schema = AttributeSchema {
            uv_sets,
            tangent_uv_set,
            has_normals: flags & 1 != 0,
            has_tangents: flags & 2 != 0,
            has_colors: flags & 4 != 0,
            position_bits: bits[0],
            uv_bits: bits[1],
            normal_bits: bits[2],
            tangent_bits: bits[3],
        };
        validate_schema(&schema)?;
        let vertex_count = read32(&mut reader)?;
        let triangle_count = read32(&mut reader)?;
        let patch_count = read32(&mut reader)?;
        let material_count = read32(&mut reader)?;
        let stream_count = read32(&mut reader)?;
        let table_bytes = read64(&mut reader)?;
        let directory_bytes = read64(&mut reader)?;
        let payload_bytes = read64(&mut reader)?;
        let expected_streams = 3u64
            + u64::from(uv_sets)
            + u64::from(schema.has_normals)
            + 2 * u64::from(schema.has_tangents)
            + u64::from(schema.has_colors);
        let patch_bytes = 88u64 + 16 * u64::from(uv_sets) + 16 * u64::from(stream_count);
        let table_start = add(add(start, HEADER_SIZE)?, 4 * u64::from(material_count))?;
        let directory_start = add(table_start, table_bytes)?;
        let payload_start = add(directory_start, directory_bytes)?;
        if add(payload_start, payload_bytes)? > end
            || u64::from(stream_count) != expected_streams
            || Some(directory_bytes) != patch_bytes.checked_mul(u64::from(patch_count))
            || table_bytes < 4 * u64::from(stream_count)
            || patch_count > vertex_count
            || (patch_count == 0) != (vertex_count == 0)
            || (vertex_count == 0 && triangle_count != 0)
        {
            return Err(invalid("invalid mesh section bounds"));
        }
        let mut materials = Vec::new();
        for _ in 0..material_count {
            materials.push(read32(&mut reader)? as i32);
        }
        let kinds = schema.stream_kinds();
        let mut contexts = Vec::new();
        for kind in &kinds {
            let remaining = directory_start
                .checked_sub(reader.stream_position()?)
                .ok_or_else(|| invalid("table exceeds section"))?;
            if remaining < 4 {
                return Err(invalid("truncated table count"));
            }
            let count = read32(&mut reader)?;
            let max_symbols = mesh_symbols(kind, vertex_count, triangle_count);
            if 8 * u64::from(count) > remaining - 4 || u64::from(count) > max_symbols {
                return Err(invalid("invalid entropy table size"));
            }
            let mut table = Vec::new();
            let mut total = 0u64;
            for _ in 0..count {
                let e = Frequency {
                    symbol: read32(&mut reader)?,
                    count: read32(&mut reader)?,
                };
                total += u64::from(e.count);
                table.push(e);
            }
            if total != max_symbols {
                return Err(invalid("entropy frequency total mismatch"));
            }
            contexts.push(compress::context(&table).map_err(mesh_error)?);
        }
        if reader.stream_position()? != directory_start {
            return Err(invalid("extra entropy table bytes"));
        }
        let mut patches = Vec::new();
        let (mut vertices, mut triangles, mut offset) = (0u32, 0u32, 0u64);
        for _ in 0..patch_count {
            let mut info = PatchInfo {
                vertex_base: read32(&mut reader)?,
                triangle_base: read32(&mut reader)?,
                internal_triangle_count: read32(&mut reader)?,
                triangle_count: read32(&mut reader)?,
                internal_vertex_count: read32(&mut reader)?,
                vertex_count: read32(&mut reader)?,
                inverse: [[0.0; 4]; 4],
                uv_bounds: Vec::new(),
            };
            for v in info.inverse.iter_mut().flatten() {
                *v = f32::from_bits(read32(&mut reader)?);
            }
            for _ in 0..uv_sets {
                let mut b = [0.0; 4];
                for v in &mut b {
                    *v = f32::from_bits(read32(&mut reader)?);
                }
                info.uv_bounds.push(b);
            }
            validate_patch(&schema, &info, vertices, triangles)?;
            vertices = vertices
                .checked_add(info.vertex_count)
                .ok_or_else(|| invalid("vertex count overflow"))?;
            triangles = triangles
                .checked_add(info.triangle_count)
                .ok_or_else(|| invalid("triangle count overflow"))?;
            let mut streams = Vec::new();
            for kind in &kinds {
                let stream = Stream {
                    symbol_count: read32(&mut reader)?,
                    payload_offset: read64(&mut reader)?,
                    word_count: read32(&mut reader)?,
                };
                validate_stream(kind, &info, stream.symbol_count, stream.word_count)?;
                if stream.payload_offset != offset {
                    return Err(invalid("noncontiguous mesh payload"));
                }
                offset = add(offset, 4 * u64::from(stream.word_count))?;
                if offset > payload_bytes {
                    return Err(invalid("stream exceeds payload"));
                }
                streams.push(stream);
            }
            patches.push(Patch { info, streams });
        }
        if vertices != vertex_count
            || triangles != triangle_count
            || offset != payload_bytes
            || reader.stream_position()? != payload_start
        {
            return Err(invalid("inconsistent mesh output ranges"));
        }
        Ok(Self {
            reader,
            schema,
            patches,
            materials,
            contexts,
            vertex_count,
            triangle_count,
            payload_start,
        })
    }
    pub fn schema(&self) -> &AttributeSchema {
        &self.schema
    }
    pub fn patches(&self) -> &[Patch] {
        &self.patches
    }
    pub fn materials(&self) -> &[i32] {
        &self.materials
    }
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
    pub fn triangle_count(&self) -> u32 {
        self.triangle_count
    }
    fn words(&mut self, s: Stream) -> io::Result<Vec<u32>> {
        self.reader
            .seek(SeekFrom::Start(add(self.payload_start, s.payload_offset)?))?;
        let mut words = Vec::new();
        for _ in 0..s.word_count {
            words.push(read32(&mut self.reader)?);
        }
        Ok(words)
    }
    pub fn read_patch(&mut self, index: usize) -> io::Result<DecodedPatch> {
        let patch = self
            .patches
            .get(index)
            .ok_or_else(|| invalid("patch index out of range"))?
            .clone();
        let mut streams = Vec::new();
        for (i, &s) in patch.streams.iter().enumerate() {
            let encoded = EncodedStream {
                symbol_count: s.symbol_count,
                words: self.words(s)?,
            };
            streams.push(compress::decode_stream(&self.contexts[i], &encoded).map_err(mesh_error)?);
        }
        let decoded =
            codec::decode_patch(&self.schema, &patch.info, &streams).map_err(mesh_error)?;
        if decoded
            .data
            .triangles
            .iter()
            .flatten()
            .any(|&v| v >= self.vertex_count)
            || (!self.materials.is_empty()
                && decoded
                    .data
                    .primitive_ids
                    .iter()
                    .any(|&p| p as usize >= self.materials.len()))
        {
            return Err(invalid("decoded index exceeds mesh bounds"));
        }
        Ok(decoded)
    }
    pub fn read_mesh(&mut self) -> io::Result<MeshData> {
        let mut patches = Vec::with_capacity(self.patches.len());
        for i in 0..self.patches.len() {
            patches.push(self.read_patch(i)?);
        }
        codec::assemble(&self.schema, &patches, self.materials.clone()).map_err(mesh_error)
    }
    pub fn prepare_gpu(&mut self) -> io::Result<PreparedGpuMesh> {
        let mut table = Vec::new();
        let mut tables = Vec::new();
        for context in &self.contexts {
            let base = len32(table.len())?;
            for e in context.table() {
                table.push(GpuEntry {
                    freq: e.freq.get(),
                    cumul: e.cumul,
                    symbol: e.symbol,
                });
            }
            tables.push((base, len32(context.table().len())?));
        }
        len32(table.len())?;
        let mut payload = Vec::new();
        let mut streams = Vec::new();
        for patch in self.patches.clone() {
            for (i, s) in patch.streams.into_iter().enumerate() {
                let payload_offset = len32(payload.len())?;
                let words = self.words(s)?;
                payload.extend(words.into_iter().rev());
                len32(payload.len())?;
                streams.push(GpuStream {
                    payload_offset,
                    payload_len: s.word_count,
                    table_offset: tables[i].0,
                    table_len: tables[i].1,
                    symbol_count: s.symbol_count,
                });
            }
        }
        len32(streams.len())?;
        Ok(PreparedGpuMesh {
            schema: self.schema.clone(),
            patches: self.patches.iter().map(|p| p.info.clone()).collect(),
            streams,
            table,
            payload,
            materials: self.materials.clone(),
            vertex_count: self.vertex_count,
            triangle_count: self.triangle_count,
        })
    }
}
