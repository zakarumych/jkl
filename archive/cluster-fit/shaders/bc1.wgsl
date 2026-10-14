
struct Block {
    color01: u32,
    indices: u32,
}

fn rgb565(color: vec3<f32>) -> u32 {
    let r = u32(color.r * 31.0 + 0.5);
    let g = u32(color.g * 63.0 + 0.5);
    let b = u32(color.b * 31.0 + 0.5);

    return (r << 11) | (g << 5) | b;
}

fn make_block(color0: u32, color1: u32, indices: u32) -> Block {
    let color01 = color0 | (color1 << 16);
    return Block(color01, indices);
}

@group(0) @binding(0)
var<storage, read> image_input: array<u32>;

@group(0) @binding(1)
var<storage, read_write> blocks_output: array<Block>;

struct Params {
    in_width: u32,
    in_height: u32,
    layers: u32,
    in_row_stride: u32,
    in_plane_stride: u32,
    out_row_stride: u32,
    out_plane_stride: u32,
    alpha_threshold: f32,
    x_offset: u32,
    y_offset: u32,
    z_offset: u32,
}

var<immediate> params: Params;

fn div_ceil(a: u32, b: u32) -> u32 {
    let rem = a % b;
    let div = a / b;
    if rem > 0u {
        return div + 1u;
    } else {
        return div;
    }
}

// Workgroup shared memory for per-thread cluster-fit errors and best-fit tracking
var<workgroup> shared_sample_count: u32;
var<workgroup> shared_transparent_count: u32;
var<workgroup> shared_samples: array<vec3<f32>, 16>;
var<workgroup> shared_transparents: array<bool, 16>;

@compute @workgroup_size(1, 1, 64)
fn compress_bc1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
) {
    // Each workgroup processes one 4×4 block; workgroup size is 64×1×1
    // Workgroup coordinate directly maps to block coordinate
    let block_x = wg_id.x + params.x_offset;  // x/y/z offsets batch blocks across submissions to avoid GPU TDR
    let block_y = wg_id.y + params.y_offset;
    let block_z = wg_id.z + params.z_offset;

    // Boundary checks
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

    // Threads 0-15 each load one pixel; all 64 threads participate in the subgroup prefix sum
    // (requires sg_size >= 16; threads 16-63 contribute 0 to the sums)
    var color = vec3<f32>(0.0);
    var is_opaque = false;
    var is_transparent = false;

    if local_idx < 16u {
        let px = local_idx % 4u;
        let py = local_idx / 4u;
        let x = block_x * 4u + px;
        let y = block_y * 4u + py;

        if x < params.in_width && y < params.in_height {
            let c = unpack4x8unorm(image_input[block_z * params.in_plane_stride + y * params.in_row_stride + x]);
            if c.a >= params.alpha_threshold {
                color = c.rgb;
                is_opaque = true;
                shared_transparents[local_idx] = false;
            } else {
                is_transparent = true;
                shared_transparents[local_idx] = true;
            }
        }
    }

    let opaque_u = select(0u, 1u, is_opaque);

    if sg_size >= 16u {
        // Compiler constant-folds this branch; subgroup prefix sum assigns write slots in order
        let my_slot = subgroupExclusiveAdd(opaque_u);

        if is_opaque {
            shared_samples[my_slot] = color;
        }

        // Thread 15's exclusive sum + own value = total opaque count across threads 0-15
        if local_idx == 15u {
            shared_sample_count = my_slot + opaque_u;
            shared_transparent_count = subgroupAdd(select(0u, 1u, is_transparent));
        }
    } else {
        // Fallback for sg_size < 16: store uncompacted, then thread 0 compacts serially
        if is_opaque {
            shared_samples[local_idx] = color;
        }

        workgroupBarrier();

        if local_idx == 0u {
            var sc = 0u;
            var tc = 0u;
            for (var pi = 0u; pi < 16u; pi += 1u) {
                let px = pi % 4u;
                let py = pi / 4u;
                if block_x * 4u + px < params.in_width && block_y * 4u + py < params.in_height {
                    if shared_transparents[pi] {
                        tc += 1u;
                    } else {
                        shared_samples[sc] = shared_samples[pi];
                        sc += 1u;
                    }
                }
            }
            shared_sample_count = sc;
            shared_transparent_count = tc;
        }
    }

    workgroupBarrier();

    // Early exit for all-transparent blocks
    if shared_sample_count == 0u {
        if local_idx == 0u {
            blocks_output[block_idx] = make_block(0, 0, 0xFFFFFFFFu);
        }
        return;
    }

    // Each thread processes a sub-range of cluster-fit iterations
    let total_iterations = binom_iter_total_iterations(shared_sample_count - 1u, 3u);
    let iter_per_thread = (total_iterations + 63u) / 64u;  // Ceiling division
    let iter_begin = local_idx * iter_per_thread;
    let iter_end = min((local_idx + 1u) * iter_per_thread, total_iterations);

    var error = 1e10;  // Large initial error
    var endpoint0 = vec3<f32>(0.0);
    var endpoint1 = vec3<f32>(0.0);
    var indices: array<u32, 16>;

    // Each thread evaluates cluster-fit for its iteration range
    if shared_transparent_count == 0u {
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
    } else {
        let cluster_fit = cluster_fit_vec3(
            shared_samples,
            shared_sample_count,
            3u,
            vec3<u32>(5u, 6u, 5u),
            0u,
            iter_begin,
            iter_end,
        );
        error = cluster_fit.error;
        endpoint0 = cluster_fit.endpoint0;
        endpoint1 = cluster_fit.endpoint1;
        indices = cluster_fit.indices;
    }

    let min_error = workgroupMin_f32(error, 64, sg_size, local_idx);
    var min_local_idx = local_idx;
    if error > min_error {
        min_local_idx = 0xFFFFFFFFu;  // Invalidate this thread's result
    }
    min_local_idx = workgroupMin_u32(min_local_idx, 64, sg_size, local_idx);

    // Thread with the minimum error writes the final block output, applying color ordering rules and packing indices
    if local_idx == min_local_idx {
        var color0 = rgb565(endpoint0);
        var color1 = rgb565(endpoint1);

        // Pack indices for output
        var index_index = 0u;
        var packed_indices = 0u;

        // Handle color ordering for opaque case
        if shared_transparent_count == 0u {
            if color0 == color1 {
                blocks_output[block_idx] = make_block(color0, 0, 0u);
                return;
            }

            if color0 < color1 {
                // Swap colors so that color0 is the larger one
                let tmp = color1;
                color1 = color0;
                color0 = tmp;

                for (var i = 0u; i < 16u; i += 1u) {
                    indices[i] = 3u - indices[i];
                }
            }

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
        } else {
            // Handle color ordering for transparent case
            if color0 > color1 {
                // Swap colors so that color0 is the smaller one
                let tmp = color1;
                color1 = color0;
                color0 = tmp;

                for (var i = 0u; i < 16u; i += 1u) {
                    indices[i] = 2u - indices[i];
                }
            }

            for (var j = 0u; j < 4u; j += 1u) {
                for (var i = 0u; i < 4u; i += 1u) {
                    let x = block_x * 4u + i;
                    let y = block_y * 4u + j;

                    var index = 0u;
                    if x < params.in_width && y < params.in_height {
                        if shared_transparents[j * 4u + i] {
                            index = 3u;
                        } else {
                            let remap = array<u32, 3>(0u, 2u, 1u);
                            index = remap[indices[index_index]];
                            index_index += 1u;
                        }
                    }

                    packed_indices |= index << (j * 8u + i * 2u);
                }
            }
        }

        blocks_output[block_idx] = make_block(color0, color1, packed_indices);
    }
}
