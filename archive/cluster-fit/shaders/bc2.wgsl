struct Bc2Block {
    alpha_lo: u32,   // explicit 4-bit alpha nibbles for pixel rows 0–1
    alpha_hi: u32,   // explicit 4-bit alpha nibbles for pixel rows 2–3
    color01: u32,
    indices: u32,
}

fn rgb565(color: vec3<f32>) -> u32 {
    let r = u32(color.r * 31.0 + 0.5);
    let g = u32(color.g * 63.0 + 0.5);
    let b = u32(color.b * 31.0 + 0.5);

    return (r << 11) | (g << 5) | b;
}

fn make_bc2_block(alpha_lo: u32, alpha_hi: u32, color0: u32, color1: u32, indices: u32) -> Bc2Block {
    let color01 = color0 | (color1 << 16);
    return Bc2Block(alpha_lo, alpha_hi, color01, indices);
}

@group(0) @binding(0)
var<storage, read> image_input: array<u32>;

@group(0) @binding(1)
var<storage, read_write> blocks_output: array<Bc2Block>;

struct Params {
    in_width: u32,
    in_height: u32,
    layers: u32,
    in_row_stride: u32,
    in_plane_stride: u32,
    out_row_stride: u32,
    out_plane_stride: u32,
    alpha_threshold: f32,   // unused by BC2; present for layout compatibility
    x_offset: u32,
    y_offset: u32,
    z_offset: u32,
}

var<immediate> params: Params;

// Workgroup shared memory
var<workgroup> shared_sample_count: u32;
var<workgroup> shared_samples: array<vec3<f32>, 16>;
var<workgroup> shared_pixel_valid: array<bool, 16>;
var<workgroup> shared_alpha_nibbles: array<u32, 16>;

@compute @workgroup_size(1, 1, 64)
fn compress_bc2(
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
) {
    let block_x = wg_id.x + params.x_offset;
    let block_y = wg_id.y + params.y_offset;
    let block_z = wg_id.z + params.z_offset;

    if block_x * 4u >= params.in_width {
        return;
    }
    if block_y * 4u >= params.in_height {
        return;
    }
    if block_z >= params.layers {
        return;
    }

    let block_idx = block_z * params.out_plane_stride + block_y * params.out_row_stride + block_x;

    // Threads 0–15 each load one pixel; threads 16–63 contribute 0 to prefix sums
    var color = vec3<f32>(0.0);
    var is_valid = false;

    if local_idx < 16u {
        shared_pixel_valid[local_idx] = false;
        shared_alpha_nibbles[local_idx] = 0u;

        let px = local_idx % 4u;
        let py = local_idx / 4u;
        let x = block_x * 4u + px;
        let y = block_y * 4u + py;

        if x < params.in_width && y < params.in_height {
            let c = unpack4x8unorm(image_input[block_z * params.in_plane_stride + y * params.in_row_stride + x]);
            color = c.rgb;
            // Quantise alpha to 4 bits (same rounding as bc2.rs)
            shared_alpha_nibbles[local_idx] = u32(c.a * 15.0 + 0.5);
            shared_pixel_valid[local_idx] = true;
            is_valid = true;
        }
    }

    let valid_u = select(0u, 1u, is_valid);

    if sg_size >= 16u {
        // Subgroup prefix sum assigns compacted write slots in order
        let my_slot = subgroupExclusiveAdd(valid_u);

        if is_valid {
            shared_samples[my_slot] = color;
        }

        // Thread 15's exclusive sum + own value = total valid count across threads 0–15
        if local_idx == 15u {
            shared_sample_count = my_slot + valid_u;
        }
    } else {
        // Fallback for sg_size < 16: store uncompacted, then thread 0 compacts serially
        if is_valid {
            shared_samples[local_idx] = color;
        }

        workgroupBarrier();

        if local_idx == 0u {
            var sc = 0u;
            for (var pi = 0u; pi < 16u; pi += 1u) {
                if shared_pixel_valid[pi] {
                    shared_samples[sc] = shared_samples[pi];
                    sc += 1u;
                }
            }
            shared_sample_count = sc;
        }
    }

    workgroupBarrier();

    // BC2 always uses 4-color mode; cluster-fit over all valid (in-bounds) pixels
    let total_iterations = binom_iter_total_iterations(shared_sample_count - 1u, 3u);
    let iter_per_thread = (total_iterations + 63u) / 64u;
    let iter_begin = local_idx * iter_per_thread;
    let iter_end = min((local_idx + 1u) * iter_per_thread, total_iterations);

    var error = 1e10;
    var endpoint0 = vec3<f32>(0.0);
    var endpoint1 = vec3<f32>(0.0);
    var indices: array<u32, 16>;

    let cluster_fit = cluster_fit_vec3(
        shared_samples,
        shared_sample_count,
        4u,
        vec3<u32>(5u, 6u, 5u),
        0u,
        iter_begin,
        iter_end,
    );
    error = cluster_fit.error;
    endpoint0 = cluster_fit.endpoint0;
    endpoint1 = cluster_fit.endpoint1;
    indices = cluster_fit.indices;

    let min_error = workgroupMin_f32(error, 64, sg_size, local_idx);
    var min_local_idx = local_idx;
    if error > min_error {
        min_local_idx = 0xFFFFFFFFu;
    }
    min_local_idx = workgroupMin_u32(min_local_idx, 64, sg_size, local_idx);

    if local_idx == min_local_idx {
        var color0 = rgb565(endpoint0);
        var color1 = rgb565(endpoint1);

        // Pack the explicit alpha nibbles.
        // Layout: pixel (px, py), byte index b = py*2 + px/2.
        // Bytes 0–3 → alpha_lo (little-endian), bytes 4–7 → alpha_hi.
        // Nibble shift within byte = (px % 2) * 4.
        var alpha_lo = 0u;
        var alpha_hi = 0u;
        for (var pi = 0u; pi < 16u; pi += 1u) {
            let apx = pi % 4u;
            let apy = pi / 4u;
            let b = apy * 2u + apx / 2u;
            let nibble = shared_alpha_nibbles[pi];
            let shift = (b % 4u) * 8u + (apx % 2u) * 4u;
            if b < 4u {
                alpha_lo |= nibble << shift;
            } else {
                alpha_hi |= nibble << shift;
            }
        }

        if color0 == color1 {
            blocks_output[block_idx] = make_bc2_block(alpha_lo, alpha_hi, color0, 0u, 0u);
            return;
        }

        // BC2 always uses 4-color mode: color0 must be > color1
        if color0 < color1 {
            let tmp = color1;
            color1 = color0;
            color0 = tmp;

            for (var i = 0u; i < 16u; i += 1u) {
                indices[i] = 3u - indices[i];
            }
        }

        var index_index = 0u;
        var packed_indices = 0u;

        for (var j = 0u; j < 4u; j += 1u) {
            for (var i = 0u; i < 4u; i += 1u) {
                let x = block_x * 4u + i;
                let y = block_y * 4u + j;

                var index = 0u;
                if x < params.in_width && y < params.in_height {
                    let remap = array<u32, 4>(0u, 2u, 3u, 1u);
                    index = remap[indices[index_index]];
                    index_index += 1u;
                }

                packed_indices |= index << (j * 8u + i * 2u);
            }
        }

        blocks_output[block_idx] = make_bc2_block(alpha_lo, alpha_hi, color0, color1, packed_indices);
    }
}
