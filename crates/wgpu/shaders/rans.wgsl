// RGB8U + rANS tile decompression kernel - byte-stream version.
//
// Dispatch model:
// - one invocation per tile (`global_invocation_id.x == tile_index`)

alias U64 = vec2<u32>;

fn umuladd64(a: u32, b: u32, c: u32) -> U64 {
    let mask: u32 = 0xffffu;

    let a0 = a & mask;
    let a1 = a >> 16u;
    let b0 = b & mask;
    let b1 = b >> 16u;

    // partial products
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;

    // add c into lowest part
    let lo0 = p00 + c;
    var carry = select(0u, 1u, lo0 < p00);

    // mid terms and their carry
    let mid = p01 + p10;
    carry += select(0u, 1u << 16u, mid < p01);

    // final low word
    let lo = lo0 + (mid << 16u);
    carry += select(0u, 1u, lo < lo0);

    // final high word
    let hi = p11 + (mid >> 16u) + carry;

    return vec2<u32>(lo, hi);
}

fn lo(a: U64) -> u32 {
    return a.x;
}

fn hi(a: U64) -> u32 {
    return a.y;
}

fn lo2hi(a: U64) -> U64 {
    return U64(0u, a.x);
}

fn shift_add(a: U64, b: u32) -> U64 {
    return U64(b, a.x);
}

struct BC1 {
    colors: u32,
    indices: u32,
}

struct Tile {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
};

struct Entry {
    freq: u32,
    cumul: u32,
};

struct Params {
    // Number of entries in table1
    table0_count: u32,

    // Number of entries in table2
    // Some formats use 2 independent tables.
    table1_count: u32,

    // Number of tiles to process.
    // Threads with `global_invocation_id.x >= tile_count` do nothing.
    tile_count: u32,

    // Output image dimensions and stride in texel blocks.
    width: u32,

    // Output image dimensions and stride in texel blocks.
    height: u32,

    // Output buffer stride, in words (not bytes or texels).
    // This is the distance between the start of one texel block row and the start of the next texel block row.
    stride: u32,
};

// Compressed data in 32-bit words.
@group(0) @binding(0)
var<storage, read> payload_words: array<u32>;

// Offsets into `payload_words` for the start of each tile's compressed data, in words.
@group(0) @binding(1)
var<storage, read> offsets: array<u32>;
@group(0) @binding(2)
var<storage, read> tiles: array<Tile>;
@group(0) @binding(3)
var<storage, read> table: array<Entry>;
@group(0) @binding(4)
var<storage, read> symbols: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out_buf: array<u32>;
@group(0) @binding(6)
var<uniform> params: Params;

fn get_u16_symbol(index: u32) -> u32 {
    let word = symbols[index >> 1u];
    return (word >> ((index & 1u) * 16u)) & 0xFFFFu;
}

fn get_u8_symbol(index: u32) -> u32 {
    let word = symbols[index >> 2u];
    return (word >> ((index & 3u) * 8u)) & 0xFFu;
}

var<private> write_word_buffer: u32 = 0u;

fn write_r8(x: u32, y: u32, stride: u32, r: u32) {
    switch (x & 3u) {
        case 0: {
            write_word_buffer = r;
        }
        case 1: {
            write_word_buffer = write_word_buffer | (r << 8u);
        }
        case 2: {
            write_word_buffer = write_word_buffer | (r << 16u);
        }
        case 3, default {
            let rgba = (r << 24u) | write_word_buffer;
            out_buf[(y * stride) + (x >> 2u)] = rgba;
        }
    }
}

fn write_rg8(x: u32, y: u32, stride: u32, rg: u32) {
    if ((x & 1u) == 0u) {
        write_word_buffer = rg;
    } else {
        let rgba = (rg << 16u) | write_word_buffer;
        out_buf[(y * stride) + (x >> 1u)] = rgba;
    }
}

fn write_rgb8(x: u32, y: u32, stride: u32, rgbx: u32) {
    out_buf[(y * stride) + x] = (0xFF000000u | rgbx);
}

fn write_rgba8(x: u32, y: u32, stride: u32, rgba: u32) {
    out_buf[(y * stride) + x] = rgba;
}

fn write_bc1_colors(x: u32, y: u32, stride: u32, rgb565_0: u32, rgb565_1: u32) {
    let rgb565_x2 = (rgb565_1 << 16u) | rgb565_0;
    out_buf[(y * stride) + (x << 1u)] = rgb565_x2;
}

fn write_bc1_indices(x: u32, y: u32, stride: u32, indices: u32) {
    out_buf[(y * stride) + (x << 1u) + 1u] = indices;
}

fn find_symbol_index(bucket: u32, start: u32, end: u32) -> u32 {
    var lo = start;
    var hi = end;

    loop {
        if (lo + 1u >= hi) {
            break;
        }

        let mid = (lo + hi) >> 1u;
        if (table[mid].cumul <= bucket) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    return lo;
}

fn renorm_state(state: ptr<function, U64>, cursor: ptr<function, u32>, end: u32) {
    if (hi(*state) == 0u && *cursor < end) {
        // For streams produced from 32-bit token chunks.
        *state = shift_add(*state, payload_words[*cursor]);
        *cursor += 1u;
    }
}

fn decode_symbol(state: ptr<function, U64>, cursor: ptr<function, u32>, end: u32, range: u32) -> u32 {
    renorm_state(state, cursor, end);

    let range_start = select(0u, params.table0_count, range > 0u);
    let range_end = select(params.table0_count, params.table0_count + params.table1_count, range > 0u);

    let bucket = lo(*state);
    let q = hi(*state);
    let index = find_symbol_index(bucket, range_start, range_end);
    let entry = table[index];

    let freq = entry.freq;
    let cumul = entry.cumul;

    *state = umuladd64(q, freq, bucket - cumul);
    return index - range_start;
}

var<private> lz77_head: u32 = 0u;
var<private> lz77_window: array<u32, 1024>;
var<private> lz77_length: u32 = 0u;
var<private> lz77_distance: u32 = 0u;

fn lz77_window_get(idx: u32) -> u32 {
    return lz77_window[(lz77_head + 1023u - idx) & 1023u];
}

fn lz77_window_push(value: u32) {
    lz77_window[lz77_head] = value;
    lz77_head = (lz77_head + 1u) & 1023u;
}

fn lz77_rans_decode_u32(state: ptr<function, U64>, cursor: ptr<function, u32>, end: u32, range: u32, symbol_offset: u32) -> u32 {
    var value: u32;

    if (lz77_length > 0) {
        lz77_length -= 1u;
        value = lz77_window_get(lz77_distance);
    } else {
        let index = decode_symbol(state, cursor, end, range);
        let length = symbols[symbol_offset + (index << 1u)];
        let symbol = symbols[symbol_offset + (index << 1u) + 1u];
        if (length > 0u) {
            // Reference token
            lz77_length = length - 1;
            lz77_distance = symbol;
            value = lz77_window_get(lz77_distance);
        } else {
            // Literal token
            value = symbol;
        }
    }

    lz77_window_push(value);
    return value;
}

fn lz77_rans_decode_u16(state: ptr<function, U64>, cursor: ptr<function, u32>, end: u32, range: u32, symbol_offset: u32) -> u32 {
    var value: u32;

    if (lz77_length > 0) {
        lz77_length -= 1u;
        value = lz77_window_get(lz77_distance);
    } else {
        let index = decode_symbol(state, cursor, end, range);
        let length = get_u16_symbol(symbol_offset + (index << 1u));
        let symbol = get_u16_symbol(symbol_offset + (index << 1u) + 1u);
        if (length > 0u) {
            // Reference token
            lz77_length = length - 1;
            lz77_distance = symbol;
            value = lz77_window_get(lz77_distance);
        } else {
            // Literal token
            value = symbol;
        }
    }

    lz77_window_push(value);
    return value;
}

@compute @workgroup_size(64)
fn decompress_rgb8_rans(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_index = gid.x;
    if (tile_index >= params.tile_count) {
        return;
    }

    if (params.table0_count == 0u) {
        return;
    }

    let tile = tiles[tile_index];

    var cursor = offsets[tile_index];
    var end = offsets[tile_index + 1u];
    var state: U64 = U64(0u, 0u);

    // Prime the decoder state.
    renorm_state(&state, &cursor, end);

    let pixel_count = tile.w * tile.h;

    for (var i = 0u; i < pixel_count; i += 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let index = decode_symbol(&state, &cursor, end, 0);
        let rgbx = symbols[index];

        write_rgb8(x, y, params.stride, rgbx);
    }
}

@compute @workgroup_size(64)
fn decompress_rgb8_lz77_rans(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_index = gid.x;
    if (tile_index >= params.tile_count) {
        return;
    }

    if (params.table0_count == 0u) {
        return;
    }

    let tile = tiles[tile_index];

    var cursor = offsets[tile_index];
    var end = offsets[tile_index + 1u];
    var state: U64 = U64(0u, 0u);

    // Prime the decoder state.
    renorm_state(&state, &cursor, end);

    let pixel_count = tile.w * tile.h;

    for (var i = 0u; i < pixel_count; i += 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let rgbx = lz77_rans_decode_u32(&state, &cursor, end, 0, 0);

        write_rgb8(x, y, params.stride, rgbx);
    }
}

@compute @workgroup_size(64)
fn decompress_rgba8_rans(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_index = gid.x;
    if (tile_index >= params.tile_count) {
        return;
    }

    if (params.table0_count == 0u) {
        return;
    }

    let tile = tiles[tile_index];

    var cursor = offsets[tile_index];
    var end = offsets[tile_index + 1u];
    var state: U64 = U64(0u, 0u);

    // Prime the decoder state.
    renorm_state(&state, &cursor, end);

    let pixel_count = tile.w * tile.h;

    for (var i = 0u; i < pixel_count; i += 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let index = decode_symbol(&state, &cursor, end, 0);
        let rgba = symbols[index];

        write_rgba8(x, y, params.stride, rgba);
    }
}

@compute @workgroup_size(64)
fn decompress_bc1_rans(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_index = gid.x;
    if (tile_index >= params.tile_count) {
        return;
    }

    if (params.table0_count == 0u || params.table1_count == 0u) {
        return;
    }

    let tile = tiles[tile_index];

    var cursor = offsets[tile_index];
    var end = offsets[tile_index + 1u];

    // Prime the decoder state.
    var state: U64 = U64(0u, 0u);
    renorm_state(&state, &cursor, end);

    let pixel_count = tile.w * tile.h;

    for (var i = 0u; i < pixel_count; i += 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let color0 = get_u16_symbol(decode_symbol(&state, &cursor, end, 0));
        let color1 = get_u16_symbol(decode_symbol(&state, &cursor, end, 0));

        // let color0 = (31u << 11u);
        // let color1 = (31u << 6u);

        write_bc1_colors(x, y, params.stride, color0, color1);
    }

    // Prime the decoder state.
    state = U64(0u, 0u);
    renorm_state(&state, &cursor, end);

    for (var i = 0u; i < pixel_count; i += 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let a = get_u8_symbol(params.table0_count * 2 + decode_symbol(&state, &cursor, end, 1));
        let b = get_u8_symbol(params.table0_count * 2 + decode_symbol(&state, &cursor, end, 1));
        let c = get_u8_symbol(params.table0_count * 2 + decode_symbol(&state, &cursor, end, 1));
        let d = get_u8_symbol(params.table0_count * 2 + decode_symbol(&state, &cursor, end, 1));

        write_bc1_indices(x, y, params.stride, (d << 24u) | (c << 16u) | (b << 8u) | a);
    }
}

@compute @workgroup_size(64)
fn decompress_bc1_lz77_rans(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_index = gid.x;
    if (tile_index >= params.tile_count) {
        return;
    }

    if (params.table0_count == 0u || params.table1_count == 0u) {
        return;
    }

    let tile = tiles[tile_index];

    var cursor = offsets[tile_index];
    var end = offsets[tile_index + 1u];

    // Prime the decoder state.
    var state: U64 = U64(0u, 0u);
    renorm_state(&state, &cursor, end);

    let pixel_count = tile.w * tile.h;

    for (var i = 0u; i < pixel_count; i += 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let color0 = lz77_rans_decode_u16(&state, &cursor, end, 0, 0);
        let color1 = lz77_rans_decode_u16(&state, &cursor, end, 0, 0);

        write_bc1_colors(x, y, params.stride, color0, color1);
    }

    // Prime the decoder state.
    state = U64(0u, 0u);
    renorm_state(&state, &cursor, end);

    lz77_length = 0u;
    for (var i = 0u; i < 1024u; i += 1u) {
        lz77_head = 0u;
        lz77_window[i] = 0u;
    }

    for (var i = 0u; i < pixel_count; i += 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let a = lz77_rans_decode_u16(&state, &cursor, end, 1, params.table0_count * 2);
        let b = lz77_rans_decode_u16(&state, &cursor, end, 1, params.table0_count * 2);
        let c = lz77_rans_decode_u16(&state, &cursor, end, 1, params.table0_count * 2);
        let d = lz77_rans_decode_u16(&state, &cursor, end, 1, params.table0_count * 2);

        write_bc1_indices(x, y, params.stride, (d << 24u) | (c << 16u) | (b << 8u) | a);
    }
}
