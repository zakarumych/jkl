// One invocation per independent chunk. Output must be zero initialized.
// Payloads must come from a trusted encoder; malformed streams have no status output.
struct Chunk {
    compression: u32,
    output_offset: u32,
    output_len: u32,
    payload_offset: u32,
    payload_len: u32,
    table_offset: u32,
    table_len: u32,
}
struct Entry { freq: u32, cumul: u32, symbol: u32, length: u32 }
@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> chunks: array<Chunk>;
@group(0) @binding(2) var<storage, read> table: array<Entry>;
@group(0) @binding(3) var<storage, read_write> output: array<atomic<u32>>;

fn umuladd64(a: u32, b: u32, c: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let lo0 = p00 + c;
    var carry = select(0u, 1u, lo0 < p00);
    let mid = p01 + p10;
    carry += select(0u, 1u << 16u, mid < p01);
    let low = lo0 + (mid << 16u);
    carry += select(0u, 1u, low < lo0);
    return vec2<u32>(low, p11 + (mid >> 16u) + carry);
}

var<private> window: array<u32, 1024>;

@compute @workgroup_size(64)
fn decompress_blob(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= arrayLength(&chunks) { return; }
    let chunk = chunks[gid.x];
    if chunk.output_len == 0u { return; }
    var cursor = chunk.payload_offset;
    let end = cursor + chunk.payload_len;
    var state = vec2<u32>(0u);
    if chunk.compression != 0u {
        if chunk.table_len == 0u || cursor >= end { return; }
        state.x = payload[cursor];
        cursor += 1u;
    }
    var remaining = 0u;
    var distance = 0u;
    var packed = 0u;
    var word_index = chunk.output_offset >> 2u;
    for (var i = 0u; i < chunk.output_len; i += 1u) {
        var value = 0u;
        if chunk.compression == 0u {
            if (i >> 2u) >= chunk.payload_len { return; }
            value = (payload[chunk.payload_offset + (i >> 2u)] >> ((i & 3u) * 8u)) & 255u;
        } else if remaining > 0u {
            value = window[(i + 1023u - distance) & 1023u];
            remaining -= 1u;
        } else {
            if state.y == 0u && cursor < end {
                state = vec2<u32>(payload[cursor], state.x);
                cursor += 1u;
            }
            var low = chunk.table_offset;
            var high = low + chunk.table_len;
            loop {
                if low + 1u >= high { break; }
                let mid = low + ((high - low) >> 1u);
                if table[mid].cumul <= state.x { low = mid; } else { high = mid; }
            }
            let entry = table[low];
            if entry.freq == 0u || state.x < entry.cumul || state.x - entry.cumul >= entry.freq { return; }
            state = umuladd64(state.y, entry.freq, state.x - entry.cumul);
            if entry.length > 0u {
                if chunk.compression != 3u || entry.symbol >= 1024u || entry.length > chunk.output_len - i { return; }
                distance = entry.symbol;
                remaining = entry.length - 1u;
                value = window[(i + 1023u - distance) & 1023u];
            } else {
                if entry.symbol > 255u { return; }
                value = entry.symbol;
            }
        }
        window[i & 1023u] = value;
        let shift = ((chunk.output_offset + i) & 3u) * 8u;
        packed |= value << shift;
        if shift == 24u || i + 1u == chunk.output_len {
            atomicOr(&output[word_index], packed);
            word_index += 1u;
            packed = 0u;
        }
    }
}
