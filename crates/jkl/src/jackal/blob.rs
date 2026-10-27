//! Independently compressed, seekable byte chunks.
//!
//! ```
//! use std::io::Cursor;
//! use jkl::jackal::blob::{write_blob, JackalBlobReader, Options};
//! let mut encoded = Vec::new();
//! write_blob(b"some bytes", &Options::default(), &mut encoded)?;
//! let mut reader = JackalBlobReader::open(Cursor::new(encoded))?;
//! assert_eq!(reader.read_blob()?, b"some bytes");
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! Version 1 uses little-endian integers: `JKLB`, version (`u32`), decoded
//! length (`u64`), chunk count (`u32`), then 32-byte chunk records (codec and
//! decoded length as `u32`, payload offset and length as `u64`, context length
//! as `u64`). Offsets are relative to the start of the payload area. Context
//! bytes precede token bytes in each payload. Chunks have a 16 MiB decoded
//! size limit. GPU preparation validates structure, not compressed content;
//! GPU callers should use trusted payloads.
//!
//! Non-ANS tokens are pairs of `u32`: LZ77 (literal, 0) or (zero-based
//! distance, match length), RLE (byte, count), and LZ78 (prefix index, byte).
//! LZ78 prefix zero denotes an empty prefix; other indices are one-based.
//! LZ77 uses a zero-filled 1024-byte initial window. ANS contexts contain a
//! `u32` entry count followed by (symbol, length, frequency) triples of `u32`,
//! sorted by (symbol, length), with unnormalized occurrence frequencies.
//! ANS byte symbols have length zero; combined codecs use their token pairs.
//! ANS encodes symbols in reverse and stores emitted words followed by final
//! state low/high words. Decoding consumes words backwards. GPU preparation
//! reverses those words into consumption order.

use crate::algos::{ans, lz77, lz78, rle};
use std::{
    collections::BTreeMap,
    io::{self, Read, Seek, SeekFrom, Write},
    num::NonZero,
};

/// Largest supported decoded chunk, in bytes.
pub const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024;
const HEADER: u64 = 20;
const RECORD: u64 = 32;

/// General-purpose lossless codecs. Texture codecs are not byte codecs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Compression {
    None = 0,
    Lz77 = 1,
    Ans = 2,
    Lz77Ans = 3,
    RleAns = 4,
    Rle = 5,
    Lz78 = 6,
}
impl Compression {
    /// Whether the GPU backends implement this codec.
    pub const fn supports_gpu(self) -> bool {
        matches!(self, Self::None | Self::Ans | Self::Lz77Ans)
    }
    fn parse(value: u32) -> io::Result<Self> {
        Ok(match value {
            0 => Self::None,
            1 => Self::Lz77,
            2 => Self::Ans,
            3 => Self::Lz77Ans,
            4 => Self::RleAns,
            5 => Self::Rle,
            6 => Self::Lz78,
            _ => return Err(invalid("unknown blob codec")),
        })
    }
    fn entropy(self) -> bool {
        matches!(self, Self::Ans | Self::Lz77Ans | Self::RleAns)
    }
}
/// Chunking and codec selection. Equal sizes prefer the first allowed codec.
#[derive(Clone, Debug)]
pub struct Options {
    /// Maximum decoded bytes per chunk, from 1 to 16 MiB.
    pub chunk_size: usize,
    /// Nonempty candidates. Raw storage is used only when `None` is allowed.
    pub allowed_compressions: Vec<Compression>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            chunk_size: 64 * 1024,
            allowed_compressions: vec![
                Compression::None,
                Compression::Lz77,
                Compression::Ans,
                Compression::Lz77Ans,
                Compression::RleAns,
                Compression::Rle,
                Compression::Lz78,
            ],
        }
    }
}
/// On-disk chunk descriptor. Offsets include the context and are relative to payload start.
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    pub compression: Compression,
    pub decoded_len: u32,
    pub payload_offset: u64,
    pub payload_len: u64,
    pub context_len: u64,
}
/// GPU descriptor. Payload offsets/lengths count words; table offsets/lengths count entries.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct GpuChunk {
    pub compression: u32,
    pub output_offset: u32,
    pub output_len: u32,
    pub payload_offset: u32,
    pub payload_len: u32,
    pub table_offset: u32,
    pub table_len: u32,
}
/// Normalized ANS entry. Length zero denotes a literal; otherwise symbol is LZ77 distance.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct GpuEntry {
    pub freq: u32,
    pub cumul: u32,
    pub symbol: u32,
    pub length: u32,
}
/// CPU-side upload data. Preparing this data performs no decompression.
#[derive(Clone, Debug)]
pub struct PreparedGpuBlob {
    pub len: u32,
    pub chunks: Vec<GpuChunk>,
    pub payload: Vec<u32>,
    pub table: Vec<GpuEntry>,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn u32_len(value: usize) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid("blob exceeds GPU addressing limits"))
}
fn read32(reader: &mut impl Read) -> io::Result<u32> {
    let mut b = [0; 4];
    reader.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read64(reader: &mut impl Read) -> io::Result<u64> {
    let mut b = [0; 8];
    reader.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn push32(output: &mut Vec<u8>, value: u32) {
    output.extend(value.to_le_bytes());
}
// A symbol is (literal/distance/prefix, length). Zero length is a literal.
type Symbol = (u32, u32);
fn symbols(input: &[u8], codec: Compression) -> Vec<Symbol> {
    match codec {
        Compression::Lz77 | Compression::Lz77Ans => {
            let mut encoder = lz77::Encoder::new(0u8, 1024);
            let mut tokens = Vec::new();
            for &b in input {
                encoder.encode(b, &mut tokens);
            }
            encoder.finish(&mut tokens);
            tokens
                .into_iter()
                .map(|t| match t {
                    lz77::Token::Literal { symbol } => (u32::from(symbol), 0),
                    lz77::Token::Reference {
                        distance,
                        length_minus_2,
                    } => (u32::from(distance), u32::from(length_minus_2) + 2),
                })
                .collect()
        }
        Compression::Rle | Compression::RleAns => rle::rle(input.iter().copied())
            .map(|r| (u32::from(r.value), r.count.get() as u32))
            .collect(),
        Compression::Lz78 => lz78::Encoder::new()
            .stream(input.iter().copied())
            .map(|t| (t.prefix as u32, u32::from(t.literal)))
            .collect(),
        _ => input.iter().map(|&b| (u32::from(b), 0)).collect(),
    }
}
fn encode_chunk(input: &[u8], codec: Compression) -> (Vec<u8>, u64) {
    if codec == Compression::None {
        return (input.to_vec(), 0);
    }
    let tokens = symbols(input, codec);
    let mut bytes = Vec::new();
    if !codec.entropy() {
        for (a, b) in tokens {
            push32(&mut bytes, a);
            push32(&mut bytes, b);
        }
        return (bytes, 0);
    }
    let mut counts = BTreeMap::new();
    for &symbol in &tokens {
        *counts.entry(symbol).or_insert(0u32) += 1;
    }
    push32(&mut bytes, counts.len() as u32);
    for (&(a, b), &count) in &counts {
        push32(&mut bytes, a);
        push32(&mut bytes, b);
        push32(&mut bytes, count);
    }
    let context_len = bytes.len() as u64;
    let mut context = ans::Context::from_frequencies(
        counts
            .into_iter()
            .map(|(s, c)| (s, NonZero::new(c).unwrap())),
    );
    context.build_encoder_index();
    let mut encoder = ans::Encoder::new(&context);
    // ANS is a stack; reversed symbols make decompression proceed forwards.
    for &symbol in tokens.iter().rev() {
        if let Some(word) = encoder.encode(symbol) {
            push32(&mut bytes, word);
        }
    }
    for word in encoder.finish() {
        push32(&mut bytes, word);
    }
    (bytes, context_len)
}
/// Compresses each chunk with every allowed codec and writes the smallest result.
pub fn write_blob(input: &[u8], options: &Options, mut writer: impl Write) -> io::Result<()> {
    if options.chunk_size == 0
        || options.chunk_size > MAX_CHUNK_SIZE
        || options.allowed_compressions.is_empty()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid blob options",
        ));
    }
    let count = input.len().div_ceil(options.chunk_size);
    let count = u32_len(count)?;
    let mut chunks = Vec::new();
    let mut payload = Vec::new();
    for data in input.chunks(options.chunk_size) {
        let mut best = None;
        for &codec in &options.allowed_compressions {
            let (bytes, context_len) = encode_chunk(data, codec);
            if best
                .as_ref()
                .is_none_or(|(_, b, _): &(Compression, Vec<u8>, u64)| bytes.len() < b.len())
            {
                best = Some((codec, bytes, context_len));
            }
        }
        let (compression, bytes, context_len) = best.unwrap();
        chunks.push(Chunk {
            compression,
            decoded_len: data.len() as u32,
            payload_offset: payload.len() as u64,
            payload_len: bytes.len() as u64,
            context_len,
        });
        payload.extend(bytes);
    }
    writer.write_all(b"JKLB")?;
    writer.write_all(&1u32.to_le_bytes())?;
    writer.write_all(&(input.len() as u64).to_le_bytes())?;
    writer.write_all(&count.to_le_bytes())?;
    for chunk in chunks {
        writer.write_all(&(chunk.compression as u32).to_le_bytes())?;
        writer.write_all(&chunk.decoded_len.to_le_bytes())?;
        writer.write_all(&chunk.payload_offset.to_le_bytes())?;
        writer.write_all(&chunk.payload_len.to_le_bytes())?;
        writer.write_all(&chunk.context_len.to_le_bytes())?;
    }
    writer.write_all(&payload)
}
/// Seekable reader. Opening validates metadata; reading validates codec data.
pub struct JackalBlobReader<R> {
    reader: R,
    len: u64,
    chunks: Vec<Chunk>,
    payload_start: u64,
}
impl<R: Read + Seek> JackalBlobReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        let start = reader.stream_position()?;
        let end = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(start))?;
        let mut magic = [0; 4];
        reader.read_exact(&mut magic)?;
        if magic != *b"JKLB" || read32(&mut reader)? != 1 {
            return Err(invalid("unsupported blob header"));
        }
        let len = read64(&mut reader)?;
        let count = u64::from(read32(&mut reader)?);
        let payload_start = start
            .checked_add(HEADER)
            .and_then(|s| s.checked_add(count * RECORD))
            .ok_or_else(|| invalid("blob offset overflow"))?;
        if payload_start > end || (count == 0) != (len == 0) || count > len {
            return Err(invalid("invalid blob chunk count"));
        }
        let mut chunks = Vec::new();
        let mut decoded = 0u64;
        let mut offset = 0u64;
        for _ in 0..count {
            let chunk = Chunk {
                compression: Compression::parse(read32(&mut reader)?)?,
                decoded_len: read32(&mut reader)?,
                payload_offset: read64(&mut reader)?,
                payload_len: read64(&mut reader)?,
                context_len: read64(&mut reader)?,
            };
            if chunk.decoded_len == 0
                || chunk.decoded_len as usize > MAX_CHUNK_SIZE
                || chunk.payload_offset != offset
                || chunk.context_len > chunk.payload_len
                || chunk.payload_len > u64::from(chunk.decoded_len) * 24 + 16
            {
                return Err(invalid("invalid blob chunk bounds"));
            }
            if chunk.compression.entropy() {
                if chunk.context_len < 16
                    || chunk.payload_len - chunk.context_len < 8
                    || (chunk.payload_len - chunk.context_len) % 4 != 0
                {
                    return Err(invalid("invalid ANS chunk layout"));
                }
            } else if chunk.context_len != 0 {
                return Err(invalid("unexpected codec context"));
            }
            if chunk.compression == Compression::None
                && chunk.payload_len != u64::from(chunk.decoded_len)
            {
                return Err(invalid("invalid raw chunk length"));
            }
            offset = offset
                .checked_add(chunk.payload_len)
                .ok_or_else(|| invalid("blob offset overflow"))?;
            decoded = decoded
                .checked_add(u64::from(chunk.decoded_len))
                .ok_or_else(|| invalid("blob length overflow"))?;
            chunks.push(chunk);
        }
        if decoded != len || offset > end - payload_start {
            return Err(invalid("truncated blob or inconsistent decoded size"));
        }
        Ok(Self {
            reader,
            len,
            chunks,
            payload_start,
        })
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }
    fn payload(&mut self, index: usize) -> io::Result<(Chunk, Vec<u8>)> {
        let chunk = *self.chunks.get(index).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "chunk index out of range")
        })?;
        self.reader
            .seek(SeekFrom::Start(self.payload_start + chunk.payload_offset))?;
        let mut bytes = vec![0; chunk.payload_len as usize];
        self.reader.read_exact(&mut bytes)?;
        Ok((chunk, bytes))
    }
    pub fn read_chunk(&mut self, index: usize) -> io::Result<Vec<u8>> {
        let (chunk, bytes) = self.payload(index)?;
        decode_chunk(chunk, &bytes)
    }
    pub fn read_blob(&mut self) -> io::Result<Vec<u8>> {
        let mut output = Vec::new();
        for i in 0..self.chunks.len() {
            output.extend(self.read_chunk(i)?);
        }
        Ok(output)
    }
    /// Preflights every codec before preparing upload data. Unsupported codecs
    /// return `Unsupported`; no CPU decompression or fallback takes place.
    pub fn prepare_gpu(&mut self) -> io::Result<PreparedGpuBlob> {
        if self.chunks.iter().any(|c| !c.compression.supports_gpu()) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "blob codec has no GPU decoder",
            ));
        }
        let mut output = PreparedGpuBlob {
            len: u32::try_from(self.len).map_err(|_| invalid("GPU blob exceeds 4 GiB"))?,
            chunks: Vec::new(),
            payload: Vec::new(),
            table: Vec::new(),
        };
        let mut output_offset = 0;
        for i in 0..self.chunks.len() {
            let (chunk, bytes) = self.payload(i)?;
            let table_offset = u32_len(output.table.len())?;
            let payload_offset = u32_len(output.payload.len())?;
            if chunk.compression.entropy() {
                let context = read_context(chunk, &bytes[..chunk.context_len as usize])?;
                output
                    .table
                    .extend(context.table().iter().map(|e| GpuEntry {
                        freq: e.freq.get(),
                        cumul: e.cumul,
                        symbol: e.symbol.0,
                        length: e.symbol.1,
                    }));
            }
            for word in bytes[chunk.context_len as usize..].chunks(4) {
                let mut padded = [0; 4];
                padded[..word.len()].copy_from_slice(word);
                output.payload.push(u32::from_le_bytes(padded));
            }
            if chunk.compression.entropy() {
                output.payload[payload_offset as usize..].reverse();
            }
            output.chunks.push(GpuChunk {
                compression: chunk.compression as u32,
                output_offset,
                output_len: chunk.decoded_len,
                payload_offset,
                payload_len: u32_len(output.payload.len())? - payload_offset,
                table_offset,
                table_len: u32_len(output.table.len())? - table_offset,
            });
            output_offset += chunk.decoded_len;
        }
        Ok(output)
    }
}
fn valid_symbol(codec: Compression, (a, b): Symbol, decoded: u32) -> bool {
    match codec {
        Compression::Ans => a < 256 && b == 0,
        Compression::Lz77 | Compression::Lz77Ans => {
            (b == 0 && a < 256) || (b >= 2 && b <= 65535 && b <= decoded && a < 1024)
        }
        Compression::Rle | Compression::RleAns => a < 256 && b > 0 && b <= decoded,
        Compression::Lz78 => a <= decoded && b < 256,
        _ => false,
    }
}
fn read_context(chunk: Chunk, mut bytes: &[u8]) -> io::Result<ans::Context<Symbol>> {
    let count = read32(&mut bytes)? as usize;
    if count == 0
        || count > chunk.decoded_len as usize
        || count.checked_mul(12) != Some(bytes.len())
    {
        return Err(invalid("invalid ANS table count"));
    }
    let mut frequencies = Vec::new();
    let mut total = 0u32;
    let mut previous = None;
    for _ in 0..count {
        let symbol = (read32(&mut bytes)?, read32(&mut bytes)?);
        let frequency = read32(&mut bytes)?;
        if !valid_symbol(chunk.compression, symbol, chunk.decoded_len)
            || previous.is_some_and(|p| p >= symbol)
            || frequency == 0
        {
            return Err(invalid("invalid ANS table entry"));
        }
        total = total
            .checked_add(frequency)
            .ok_or_else(|| invalid("ANS frequency overflow"))?;
        if total > chunk.decoded_len {
            return Err(invalid("ANS frequency exceeds chunk size"));
        }
        previous = Some(symbol);
        frequencies.push((symbol, NonZero::new(frequency).unwrap()));
    }
    Ok(ans::Context::from_frequencies(frequencies))
}
fn decode_chunk(chunk: Chunk, bytes: &[u8]) -> io::Result<Vec<u8>> {
    if chunk.compression == Compression::None {
        return Ok(bytes.to_vec());
    }
    let mut output = Vec::new();
    let mut dictionary: Vec<(usize, usize)> = Vec::new();
    let mut emit = |symbol: Symbol| -> io::Result<()> {
        if !valid_symbol(chunk.compression, symbol, chunk.decoded_len) {
            return Err(invalid("invalid blob token"));
        }
        let (a, b) = symbol;
        match chunk.compression {
            Compression::Ans => output.push(a as u8),
            Compression::Lz77 | Compression::Lz77Ans => {
                if b == 0 {
                    output.push(a as u8);
                } else {
                    if b as usize > chunk.decoded_len as usize - output.len() {
                        return Err(invalid("LZ77 output exceeds chunk"));
                    }
                    for _ in 0..b {
                        let value = if output.len() > a as usize {
                            output[output.len() - a as usize - 1]
                        } else {
                            0
                        };
                        output.push(value);
                    }
                }
            }
            Compression::Rle | Compression::RleAns => {
                if b as usize > chunk.decoded_len as usize - output.len() {
                    return Err(invalid("RLE output exceeds chunk"));
                }
                output.resize(output.len() + b as usize, a as u8);
            }
            Compression::Lz78 => {
                let start = output.len();
                if a != 0 {
                    let &(from, to) = dictionary
                        .get(a as usize - 1)
                        .ok_or_else(|| invalid("invalid LZ78 prefix"))?;
                    if to - from >= chunk.decoded_len as usize - output.len() {
                        return Err(invalid("LZ78 output exceeds chunk"));
                    }
                    output.extend_from_within(from..to);
                }
                output.push(b as u8);
                dictionary.push((start, output.len()));
            }
            _ => unreachable!(),
        }
        if output.len() > chunk.decoded_len as usize {
            return Err(invalid("decoded chunk too long"));
        }
        Ok(())
    };
    if chunk.compression.entropy() {
        let context = read_context(chunk, &bytes[..chunk.context_len as usize])?;
        let mut words = bytes[chunk.context_len as usize..]
            .chunks_exact(4)
            .rev()
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
        let mut state = 0u64;
        // Bound symbol count by decoded size, including malicious terminal states.
        for _ in 0..chunk.decoded_len {
            if state <= u32::MAX as u64 {
                let Some(w) = words.next() else { break };
                state = (state << 32) | u64::from(w);
            }
            if state <= u32::MAX as u64 {
                let w = words.next().ok_or_else(|| invalid("truncated ANS state"))?;
                state = (state << 32) | u64::from(w);
            }
            let bucket = state as u32;
            let table = context.table();
            let index = table.partition_point(|e| e.cumul <= bucket);
            let entry = table
                .get(index.wrapping_sub(1))
                .ok_or_else(|| invalid("invalid ANS bucket"))?;
            if bucket - entry.cumul >= entry.freq.get() {
                return Err(invalid("invalid ANS bucket"));
            }
            state = (state >> 32) * u64::from(entry.freq.get()) + u64::from(bucket - entry.cumul);
            emit(entry.symbol)?;
            if state == u32::MAX as u64 && words.len() == 0 {
                break;
            }
        }
        if state != u32::MAX as u64 || words.len() != 0 {
            return Err(invalid("incomplete ANS stream"));
        }
    } else {
        if bytes.len() % 8 != 0 {
            return Err(invalid("truncated blob token"));
        }
        for token in bytes.chunks_exact(8) {
            emit((
                u32::from_le_bytes(token[..4].try_into().unwrap()),
                u32::from_le_bytes(token[4..].try_into().unwrap()),
            ))?;
        }
    }
    if output.len() != chunk.decoded_len as usize {
        return Err(invalid("decoded chunk length mismatch"));
    }
    Ok(output)
}
