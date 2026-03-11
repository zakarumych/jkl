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

struct Symbol {
    sym: u32,
    freq: u32,
    cumul: u32,
    pad: u32,
};

struct Params {
    symbol_count: u32,
    symbol2_count: u32,
    tile_count: u32,
    width: u32,
    height: u32,
    stride: u32, // in words, not bytes or texels
};

@group(0) @binding(0)
var<storage, read> payload_words: array<u32>;
@group(0) @binding(1)
var<storage, read> offsets: array<u32>;
@group(0) @binding(2)
var<storage, read> tiles: array<Tile>;
@group(0) @binding(3)
var<storage, read> symbol_table: array<Symbol>;
@group(0) @binding(4)
var<storage, read_write> out_buf: array<u32>;
@group(0) @binding(5)
var<uniform> params: Params;

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
        if (symbol_table[mid].cumul <= bucket) {
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
        *cursor = *cursor + 1u;
    }
}

fn decode_symbol(state: ptr<function, U64>, cursor: ptr<function, u32>, end: u32) -> u32 {
    renorm_state(state, cursor, end);

    let bucket = lo(*state);
    let q = hi(*state);
    let symbol_index = find_symbol_index(bucket, 0, params.symbol_count);
    let entry = symbol_table[symbol_index];

    let freq = entry.freq;
    let cumul = entry.cumul;

    *state = umuladd64(q, freq, bucket - cumul);
    return entry.sym;
}

fn decode_symbol2(state: ptr<function, U64>, cursor: ptr<function, u32>, end: u32) -> u32 {
    renorm_state(state, cursor, end);

    let bucket = lo(*state);
    let q = hi(*state);
    let symbol_index = find_symbol_index(bucket, params.symbol_count, params.symbol_count + params.symbol2_count);
    let entry = symbol_table[symbol_index];

    let freq = entry.freq;
    let cumul = entry.cumul;

    *state = umuladd64(q, freq, bucket - cumul);
    return entry.sym;
}

@compute @workgroup_size(64)
fn decompress_rgb8_rans(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_index = gid.x;
    if (tile_index >= params.tile_count) {
        return;
    }

    if (params.symbol_count == 0u) {
        return;
    }

    let tile = tiles[tile_index];

    var cursor = offsets[tile_index] / 4u; // Convert byte offset to word offset.
    var end = offsets[tile_index + 1u] / 4u; // Convert byte offset to word offset.
    var state: U64 = U64(0u, 0u);

    // Prime the decoder state.
    renorm_state(&state, &cursor, end);

    let pixel_count = tile.w * tile.h;

    for (var i = 0u; i < pixel_count; i = i + 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let rgbx = decode_symbol(&state, &cursor, end);

        write_rgb8(x, y, params.stride, rgbx);
    }
}


@compute @workgroup_size(64)
fn decompress_rgba8_rans(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_index = gid.x;
    if (tile_index >= params.tile_count) {
        return;
    }

    if (params.symbol_count == 0u) {
        return;
    }

    let tile = tiles[tile_index];

    var cursor = offsets[tile_index] / 4u; // Convert byte offset to word offset.
    var end = offsets[tile_index + 1u] / 4u; // Convert byte offset to word offset.
    var state: U64 = U64(0u, 0u);

    // Prime the decoder state.
    renorm_state(&state, &cursor, end);

    let pixel_count = tile.w * tile.h;

    for (var i = 0u; i < pixel_count; i = i + 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let rgba = decode_symbol(&state, &cursor, end);

        write_rgba8(x, y, params.stride, rgba);
    }
}

@compute @workgroup_size(64)
fn decompress_bc1_rans(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_index = gid.x;
    if (tile_index >= params.tile_count) {
        return;
    }

    if (params.symbol_count == 0u) {
        return;
    }

    let tile = tiles[tile_index];

    var cursor = offsets[tile_index] / 4u; // Convert byte offset to word offset.
    var end = offsets[tile_index + 1u] / 4u; // Convert byte offset to word offset.

    // Prime the decoder state.
    var state: U64 = U64(0u, 0u);
    renorm_state(&state, &cursor, end);

    let pixel_count = tile.w * tile.h;

    for (var i = 0u; i < pixel_count; i = i + 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let color0 = decode_symbol(&state, &cursor, end);
        let color1 = decode_symbol(&state, &cursor, end);

        // let color0 = (31u << 11u);
        // let color1 = (31u << 6u);

        write_bc1_colors(x, y, params.stride, color0, color1);
    }

    // Prime the decoder state.
    state = U64(0u, 0u);
    renorm_state(&state, &cursor, end);

    for (var i = 0u; i < pixel_count; i = i + 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.width || y >= params.height) {
            continue;
        }

        let a = decode_symbol2(&state, &cursor, end);
        let b = decode_symbol2(&state, &cursor, end);
        let c = decode_symbol2(&state, &cursor, end);
        let d = decode_symbol2(&state, &cursor, end);

        write_bc1_indices(x, y, params.stride, (d << 24u) | (c << 16u) | (b << 8u) | a);
    }
}
