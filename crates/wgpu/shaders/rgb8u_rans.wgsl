// RGB8U + rANS tile decompression kernel - byte-stream version.
//
// Dispatch model:
// - one invocation per tile (`global_invocation_id.x == tile_index`)

struct Tile {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
};

struct Symbol {
    xrgb: u32, // 0x00RRGGBB
    freq: u32,
    cumul: u32,
    pad: u32,
};

struct Params {
    tile_count: u32,
    output_width: u32,
    padded_output_width: u32,
    output_height: u32,
    symbol_count: u32,
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

fn find_symbol_index(bucket: u32) -> u32 {
    var lo = 0u;
    var hi = params.symbol_count;

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

fn renorm_state(state: ptr<function, u64>, cursor: ptr<function, u32>, end: u32) {
    if (*state <= u64(0xFFFFFFFFu) && *cursor < end) {
        // For streams produced from 32-bit token chunks.
        *state = ((*state) << 32u) | u64(payload_words[*cursor]);
        *cursor = *cursor + 1u;
    }
}

@compute @workgroup_size(64)
fn decompress_rgb8_rans(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_index = gid.x;
    if (tile_index >= params.tile_count) {
        return;
    }

    let tile = tiles[tile_index];

    var cursor = offsets[tile_index] / 4u; // Convert byte offset to word offset.
    var end = offsets[tile_index + 1u] / 4u; // Convert byte offset to word offset.
    var state: u64 = u64(0u);

    // Prime the decoder state.
    renorm_state(&state, &cursor, end);

    let pixel_count = tile.w * tile.h;

    for (var i = 0u; i < pixel_count; i = i + 1u) {
        let local_x = i % tile.w;
        let local_y = i / tile.w;

        let x = tile.x + local_x;
        let y = tile.y + local_y;

        if (x >= params.output_width || y >= params.output_height) {
            continue;
        }

        renorm_state(&state, &cursor, end);
        if (state <= u64(0xFFFFFFFFu) || params.symbol_count == 0u) {
            break;
        }

        let bucket = u32(state);
        let q = state >> 32u;
        let symbol_index = find_symbol_index(bucket);
        let symbol = symbol_table[symbol_index];

        let freq = symbol.freq;
        let cumul = symbol.cumul;

        state = q * u64(freq) + u64(bucket - cumul);

        out_buf[y * params.padded_output_width + x] = (0xFF000000u | symbol.xrgb);
    }
}

