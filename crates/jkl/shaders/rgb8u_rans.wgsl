// RGB8U + rANS tile decompression kernel.
//
// Dispatch model:
// - one invocation per tile (`global_invocation_id.x == tile_index`)
//
// Expected buffers:
// - `payload_words`: concatenated token stream for all tiles (u32 words)
// - `tile_word_offsets`: start word offset for each tile in payload_words
// - `symbol_cumul`: sorted cumulative frequency starts per symbol
// - `symbol_freq`: per-symbol frequency
// - `symbol_rgb8`: packed 0x00RRGGBB value per symbol index
// - `tile_meta`: 2 u32 values per tile:
//     * meta0 = (origin_y << 16) | origin_x
//     * meta1 = (tile_h   << 16) | tile_w
//
// Notes:
// - This shader uses native u64 math (requires SHADER_INT64 support in the device).
// - It is intended as a direct GPU-side decompression kernel for RGB8 JKLI tiles.

struct Params {
    tile_count: u32,
    output_width: u32,
    output_height: u32,
    ans_total: u32,
    symbol_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> payload_words: array<u32>;

@group(0) @binding(1)
var<storage, read> tile_word_offsets: array<u32>;

@group(0) @binding(3)
var<storage, read> symbol_cumul: array<u32>;

@group(0) @binding(4)
var<storage, read> symbol_freq: array<u32>;

@group(0) @binding(5)
var<storage, read> symbol_rgb8: array<u32>;

@group(0) @binding(7)
var<storage, read> tile_meta: array<u32>;

@group(0) @binding(8)
var out_image: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(9)
var<uniform> params: Params;

const RANS_L: u32 = 0x80000000u;
const RANS_L_U64: u64 = u64(RANS_L);

fn unpack_rgb8(packed: u32) -> vec3<u32> {
    let r = (packed >> 16u) & 0xFFu;
    let g = (packed >> 8u) & 0xFFu;
    let b = packed & 0xFFu;
    return vec3<u32>(r, g, b);
}

fn find_symbol_index(bucket: u32) -> u32 {
    var lo = 0u;
    var hi = params.symbol_count;

    loop {
        if (lo + 1u >= hi) {
            break;
        }

        let mid = (lo + hi) >> 1u;
        if (symbol_cumul[mid] <= bucket) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    return lo;
}

fn renorm_state(state: ptr<function, u64>, cursor: ptr<function, u32>, end: u32) {
    if (*state < RANS_L_U64 && *cursor < end) {
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

    let begin = tile_word_offsets[tile_index];
    let end = tile_word_offsets[tile_index + 1u];

    var cursor = begin;
    var state: u64 = u64(0u);

    // Prime the decoder state.
    renorm_state(&state, &cursor, end);

    let meta_index = tile_index * 2u;
    let meta0 = tile_meta[meta_index + 0u];
    let meta1 = tile_meta[meta_index + 1u];

    let origin_x = meta0 & 0xFFFFu;
    let origin_y = (meta0 >> 16u) & 0xFFFFu;

    let tile_w = meta1 & 0xFFFFu;
    let tile_h = (meta1 >> 16u) & 0xFFFFu;

    let pixel_count = tile_w * tile_h;

    for (var i = 0u; i < pixel_count; i = i + 1u) {
        renorm_state(&state, &cursor, end);
        if (state == u64(0u) || params.ans_total == 0u || params.symbol_count == 0u) {
            break;
        }

        let ans_total_u64 = u64(params.ans_total);
        let bucket = u32(state % ans_total_u64);
        let q = state / ans_total_u64;
        let symbol_index = find_symbol_index(bucket);

        let freq = symbol_freq[symbol_index];
        let cumul = symbol_cumul[symbol_index];

        // Corrupt stream guard.
        if (freq == 0u || bucket < cumul) {
            break;
        }

        state = q * u64(freq) + u64(bucket - cumul);

        let local_x = i % tile_w;
        let local_y = i / tile_w;

        let x = origin_x + local_x;
        let y = origin_y + local_y;

        if (x >= params.output_width || y >= params.output_height) {
            continue;
        }

        let rgb = unpack_rgb8(symbol_rgb8[symbol_index]);
        let out = vec4<f32>(
            f32(rgb.x) / 255.0,
            f32(rgb.y) / 255.0,
            f32(rgb.z) / 255.0,
            1.0,
        );

        textureStore(out_image, vec2<i32>(i32(x), i32(y)), out);
    }
}
