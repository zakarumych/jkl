
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
    let color01 = (color0 << 16) | color1;
    return Block(color01, indices);
}

@group(0) @binding(0)
var<storage, read> image_input: array<u32>;

@group(0) @binding(1)
var<storage, read_write> blocks_output: array<Block>;

struct Params {
    width: u32,
    height: u32,
    layers: u32,
    in_row_stride: u32,
    in_plane_stride: u32,
    out_row_stride: u32,
    out_plane_stride: u32,
    alpha_threshold: f32,
}

@group(0) @binding(2)
var<uniform> params: Params;

fn div_ceil(a: u32, b: u32) -> u32 {
    let rem = a % b;
    let div = a / b;
    if rem > 0u {
        return div + 1u;
    } else {
        return div;
    }
}

@compute @workgroup_size(8, 8, 1)
fn compress_bc1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x * 4u >= params.width {
        return;
    }
    if gid.y * 4u >= params.height {
        return;
    }
    if gid.z >= params.layers {
        return;
    }

    let block_idx = gid.z * params.out_plane_stride + gid.y * params.out_row_stride + gid.x;

    var samples = array<vec3<f32>, 16>();
    var sample_count = 0u;
    var transparents = array<bool, 16>();
    var transparent_count = 0u;

    for (var j = 0u; j < 4u; j += 1u) {
        for (var i = 0u; i < 4u; i += 1u) {
            let x = gid.x * 4u + i;
            let y = gid.y * 4u + j;

            if x < params.width && y < params.height {
                let c = unpack4x8unorm(image_input[gid.z * params.in_plane_stride + y * params.in_row_stride + x]);
                if c.a <= params.alpha_threshold {
                    transparents[j * 4u + i] = true;
                    transparent_count += 1u;
                } else {
                    samples[sample_count] = c.rgb;
                    sample_count += 1u;
                }
            }
        }
    }

    if sample_count == 0u {
        // All texels are transparent
        blocks_output[block_idx] = make_block(0, 0, 0xFFFFFFFFu);
        return;
    }

    var color0 = 0u;
    var color1 = 0u;
    var indices = array<u32, 16>();

    if transparent_count == 0u {
        let cluster_fit = cluster_fit_vec3(samples, sample_count, 4u, vec3<u32>(5u, 6u, 5u), 0u);
        color0 = rgb565(cluster_fit.endpoint0);
        color1 = rgb565(cluster_fit.endpoint1);
        indices = cluster_fit.indices;

        if color0 == color1 {
            blocks_output[block_idx] = make_block(color0, color0, 0u);
            return;
        }

        if color0 < color1 {
            // Swap colors so that color0 is the larger one
            let tmp = color1;
            color1 = color0;
            color0 = tmp;

            for (var i = 0u; i < sample_count; i += 1u) {
                indices[i] = 3u - indices[i];
            }
        }
    } else {
        let cluster_fit = cluster_fit_vec3(samples, sample_count, 3u, vec3<u32>(5u, 6u, 5u), 0u);
        color0 = rgb565(cluster_fit.endpoint0);
        color1 = rgb565(cluster_fit.endpoint1);
        indices = cluster_fit.indices;

        if color0 > color1 {
            // Swap colors so that color0 is the smaller one
            let tmp = color1;
            color1 = color0;
            color0 = tmp;

            for (var i = 0u; i < sample_count; i += 1u) {
                indices[i] = 2u - indices[i];
            }
        }
    }

    var index_index = 0u;
    var packed_indices = 0u;

    for (var j = 0u; j < 4u; j += 1u) {
        for (var i = 0u; i < 4u; i += 1u) {
            let x = gid.x * 4u + i;
            let y = gid.y * 4u + j;

            var index = 0u;
            if x < params.width && y < params.height {
                if transparents[j * 4u + i] {
                    index = 3u;
                } else {
                    index = indices[index_index];
                    index_index += 1u;
                }
            }

            packed_indices |= index << ((j * 4u + i) * 2u);
        }
    }

    blocks_output[block_idx] = make_block(color0, color1, packed_indices);
}
