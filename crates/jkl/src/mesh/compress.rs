//! Shared mesh-wide entropy tables with independently decodable patch streams.
use super::{
    codec::{AttributeSchema, PatchInfo, PreparedMesh},
    data::MeshError,
};
use crate::algos::ans;
use std::{collections::BTreeMap, num::NonZero};

#[derive(Clone, Copy, Debug)]
pub struct Frequency {
    pub symbol: u32,
    pub count: u32,
}
#[derive(Clone, Debug)]
pub struct EncodedStream {
    pub symbol_count: u32,
    pub words: Vec<u32>,
}
#[derive(Clone, Debug)]
pub struct CompressedPatch {
    pub info: PatchInfo,
    pub streams: Vec<EncodedStream>,
}
#[derive(Clone, Debug)]
pub struct CompressedMesh {
    pub schema: AttributeSchema,
    pub patches: Vec<CompressedPatch>,
    pub materials: Vec<i32>,
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub tables: Vec<Vec<Frequency>>,
}
fn invalid(s: &'static str) -> MeshError {
    MeshError::InvalidData(s)
}

/// Validates sorted occurrence frequencies before constructing a normalized ANS table.
pub fn context(table: &[Frequency]) -> Result<ans::Context<u32>, MeshError> {
    let mut total = 0u32;
    let mut previous = None;
    for entry in table {
        if entry.count == 0 || previous.is_some_and(|s| s >= entry.symbol) {
            return Err(invalid("invalid entropy frequency table"));
        }
        total = total
            .checked_add(entry.count)
            .ok_or(invalid("frequency overflow"))?;
        previous = Some(entry.symbol);
    }
    Ok(ans::Context::from_frequencies(
        table
            .iter()
            .map(|e| (e.symbol, NonZero::new(e.count).unwrap())),
    ))
}

pub fn compress(mesh: &PreparedMesh) -> Result<CompressedMesh, MeshError> {
    mesh.schema.validate()?;
    let count = mesh.schema.stream_kinds().len();
    let mut frequencies = vec![BTreeMap::<u32, u32>::new(); count];
    let mut totals = vec![0u32; count];
    for patch in &mesh.patches {
        if patch.streams.len() != count {
            return Err(invalid("stream count mismatch"));
        }
        for (index, stream) in patch.streams.iter().enumerate() {
            for &symbol in stream {
                totals[index] = totals[index]
                    .checked_add(1)
                    .ok_or(invalid("frequency overflow"))?;
                *frequencies[index].entry(symbol).or_default() += 1;
            }
        }
    }
    let tables: Vec<Vec<_>> = frequencies
        .into_iter()
        .map(|t| {
            t.into_iter()
                .map(|(symbol, count)| Frequency { symbol, count })
                .collect()
        })
        .collect();
    let contexts = tables
        .iter()
        .map(|t| context(t))
        .collect::<Result<Vec<_>, _>>()?;
    let mut patches = Vec::with_capacity(mesh.patches.len());
    for patch in &mesh.patches {
        let mut streams = Vec::with_capacity(count);
        for (symbols, ctx) in patch.streams.iter().zip(&contexts) {
            let symbol_count =
                u32::try_from(symbols.len()).map_err(|_| invalid("too many symbols"))?;
            let mut words = Vec::new();
            if !symbols.is_empty() {
                let mut encoder = ans::Encoder::new(ctx);
                for &symbol in symbols.iter().rev() {
                    if let Some(word) = encoder.encode(symbol) {
                        words.push(word);
                    }
                }
                words.extend(encoder.finish());
            }
            streams.push(EncodedStream {
                symbol_count,
                words,
            });
        }
        patches.push(CompressedPatch {
            info: patch.info.clone(),
            streams,
        });
    }
    Ok(CompressedMesh {
        schema: mesh.schema.clone(),
        patches,
        materials: mesh.materials.clone(),
        vertex_count: mesh.vertex_count,
        triangle_count: mesh.triangle_count,
        tables,
    })
}

/// Checked rANS decoding; malformed states cannot index beyond the table.
pub fn decode_stream(
    ctx: &ans::Context<u32>,
    stream: &EncodedStream,
) -> Result<Vec<u32>, MeshError> {
    if stream.symbol_count == 0 {
        return if stream.words.is_empty() {
            Ok(Vec::new())
        } else {
            Err(invalid("nonempty zero-symbol stream"))
        };
    }
    if ctx.table().is_empty()
        || stream.words.len() < 2
        || stream.words.len() as u64 > u64::from(stream.symbol_count) + 2
    {
        return Err(invalid("invalid entropy stream length"));
    }
    let mut words = stream.words.iter().rev();
    let mut state = 0u64;
    let mut output = Vec::with_capacity(stream.symbol_count as usize);
    for _ in 0..stream.symbol_count {
        while state <= u32::MAX as u64 {
            let word = words.next().ok_or(invalid("truncated entropy stream"))?;
            state = (state << 32) | u64::from(*word);
        }
        let bucket = state as u32;
        let index = ctx.table().partition_point(|e| e.cumul <= bucket);
        let entry = ctx
            .table()
            .get(index.wrapping_sub(1))
            .ok_or(invalid("invalid entropy bucket"))?;
        if u64::from(bucket) >= u64::from(entry.cumul) + u64::from(entry.freq.get()) {
            return Err(invalid("invalid entropy bucket"));
        }
        state = (state >> 32) * u64::from(entry.freq.get()) + u64::from(bucket - entry.cumul);
        output.push(entry.symbol);
    }
    if state != u32::MAX as u64 || words.next().is_some() {
        return Err(invalid("unfinished entropy stream"));
    }
    Ok(output)
}
