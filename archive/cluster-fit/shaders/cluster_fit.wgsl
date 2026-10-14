
// Least-squares cluster-fit quantization helpers for block encoders.
//
// This mirrors the Rust implementation for f32 / vec2<f32> / vec3<f32> and
// extends it to vec4<f32> with the same exhaustive partition search model.

const CLUSTER_FIT_MAX_SAMPLES: u32 = 16u;
const CLUSTER_FIT_MAX_PALETTE: u32 = 8u;

const CLUSTER_FIT_EPSILON: f32 = 1e-8;

const CLUSTER_ERROR_DISTANCE: u32 = 0u;
const CLUSTER_ERROR_PERCEPTUAL: u32 = 1u;

struct ClusterFitF32 {
    endpoints: vec2<f32>,
    indices: array<u32, 16>,
    error: f32,
};

struct ClusterFitVec2 {
    endpoint0: vec2<f32>,
    endpoint1: vec2<f32>,
    indices: array<u32, 16>,
    error: f32,
};

struct ClusterFitVec3 {
    endpoint0: vec3<f32>,
    endpoint1: vec3<f32>,
    indices: array<u32, 16>,
    error: f32,
};

struct ClusterFitVec4 {
    endpoint0: vec4<f32>,
    endpoint1: vec4<f32>,
    indices: array<u32, 16>,
    error: f32,
};

fn norm2(v: vec2<f32>) -> vec2<f32> {
    let len = length(v);
    if len > 1.0e-6 {
        return v / len;
    }
    return vec2<f32>(1.0, 0.0);
}

fn norm3(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len > 1.0e-6 {
        return v / len;
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

fn norm4(v: vec4<f32>) -> vec4<f32> {
    let len = length(v);
    if len > 1.0e-6 {
        return v / len;
    }
    return vec4<f32>(1.0, 0.0, 0.0, 0.0);
}

fn bit_levels(bits: u32) -> f32 {
    let b = min(bits, 31u);
    if b == 0u {
        return 1.0;
    }
    return f32((1u << b) - 1u);
}

fn snap_unorm_f32(v: f32, bits: u32) -> f32 {
    if bits == 0u {
        return clamp(v, 0.0, 1.0);
    }

    let levels = bit_levels(bits);
    let q = round(clamp(v, 0.0, 1.0) * levels);
    return q / levels;
}

fn snap_unorm_vec2(v: vec2<f32>, bits: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(
        snap_unorm_f32(v.x, bits.x),
        snap_unorm_f32(v.y, bits.y),
    );
}

fn snap_unorm_vec3(v: vec3<f32>, bits: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(
        snap_unorm_f32(v.x, bits.x),
        snap_unorm_f32(v.y, bits.y),
        snap_unorm_f32(v.z, bits.z),
    );
}

fn snap_unorm_vec4(v: vec4<f32>, bits: vec4<u32>) -> vec4<f32> {
    return vec4<f32>(
        snap_unorm_f32(v.x, bits.x),
        snap_unorm_f32(v.y, bits.y),
        snap_unorm_f32(v.z, bits.z),
        snap_unorm_f32(v.w, bits.w),
    );
}

fn yiq_from_rgb(c: vec3<f32>) -> vec3<f32> {
    let y = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    let i = 0.5959 * c.r - 0.2746 * c.g - 0.3213 * c.b;
    let q = 0.2115 * c.r - 0.5227 * c.g + 0.3112 * c.b;
    return vec3<f32>(y, i, q);
}

fn perceptual_distance_rgb(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let ya = yiq_from_rgb(a);
    let yb = yiq_from_rgb(b);

    let dy = ya.x - yb.x;
    let di = ya.y - yb.y;
    let dq = ya.z - yb.z;

    let luminance_diff = dy * dy;
    let chrominance_diff = 0.25 * (di * di + dq * dq);

    return sqrt(luminance_diff + chrominance_diff);
}

fn error_f32(mode: u32, a: f32, b: f32) -> f32 {
    _ = mode;
    return abs(a - b);
}

fn error_vec2(mode: u32, a: vec2<f32>, b: vec2<f32>) -> f32 {
    _ = mode;
    return distance(a, b);
}

fn error_vec3(mode: u32, a: vec3<f32>, b: vec3<f32>) -> f32 {
    if mode == CLUSTER_ERROR_PERCEPTUAL {
        return perceptual_distance_rgb(a, b);
    }
    return distance(a, b);
}

fn error_vec4(mode: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    if mode == CLUSTER_ERROR_PERCEPTUAL {
        let rgb = perceptual_distance_rgb(a.xyz, b.xyz);
        let da = a.w - b.w;
        return sqrt(rgb * rgb + da * da);
    }
    return distance(a, b);
}

fn fallback_endpoints_f32(samples: array<f32, 16>, sample_count: u32) -> vec2<f32> {
    var min_v = 3.402823e38;
    var max_v = -3.402823e38;

    for (var i = 0u; i < sample_count; i += 1u) {
        min_v = min(min_v, samples[i]);
        max_v = max(max_v, samples[i]);
    }

    return vec2<f32>(min_v, max_v);
}

fn fallback_endpoints_vec2(samples: array<vec2<f32>, 16>, sample_count: u32) -> array<vec2<f32>, 2> {
    var min_v = vec2<f32>(3.402823e38);
    var max_v = vec2<f32>(-3.402823e38);

    for (var i = 0u; i < sample_count; i += 1u) {
        min_v = min(min_v, samples[i]);
        max_v = max(max_v, samples[i]);
    }

    var out: array<vec2<f32>, 2>;
    out[0] = min_v;
    out[1] = max_v;
    return out;
}

fn fallback_endpoints_vec3(samples: array<vec3<f32>, 16>, sample_count: u32) -> array<vec3<f32>, 2> {
    var min_v = vec3<f32>(3.402823e38);
    var max_v = vec3<f32>(-3.402823e38);

    for (var i = 0u; i < sample_count; i += 1u) {
        min_v = min(min_v, samples[i]);
        max_v = max(max_v, samples[i]);
    }

    var out: array<vec3<f32>, 2>;
    out[0] = min_v;
    out[1] = max_v;
    return out;
}

fn fallback_endpoints_vec4(samples: array<vec4<f32>, 16>, sample_count: u32) -> array<vec4<f32>, 2> {
    var min_v = vec4<f32>(3.402823e38);
    var max_v = vec4<f32>(-3.402823e38);

    for (var i = 0u; i < sample_count; i += 1u) {
        min_v = min(min_v, samples[i]);
        max_v = max(max_v, samples[i]);
    }

    var out: array<vec4<f32>, 2>;
    out[0] = min_v;
    out[1] = max_v;
    return out;
}

fn max_variance_diagonal_axis2(samples: array<vec2<f32>, 16>, sample_count: u32) -> vec2<f32> {
    var min_v = vec2<f32>(3.402823e38);
    var max_v = vec2<f32>(-3.402823e38);

    for (var i = 0u; i < sample_count; i += 1u) {
        min_v = min(min_v, samples[i]);
        max_v = max(max_v, samples[i]);
    }

    let center = (min_v + max_v) * 0.5;

    var diagonals: array<vec2<f32>, 2>;
    diagonals[0] = norm2(vec2<f32>(max_v.x - min_v.x, max_v.y - min_v.y));
    diagonals[1] = norm2(vec2<f32>(max_v.x - min_v.x, min_v.y - max_v.y));

    var best_diag = vec2<f32>(0.0);
    var best_var = -1.0;

    for (var d = 0u; d < 2u; d += 1u) {
        let diagonal = diagonals[d];
        var var_acc = 0.0;

        for (var i = 0u; i < sample_count; i += 1u) {
            let t = dot(samples[i] - center, diagonal);
            var_acc = var_acc + t * t;
        }

        if var_acc > best_var {
            best_var = var_acc;
            best_diag = diagonal;
        }
    }

    return best_diag;
}

fn max_variance_diagonal_axis3(samples: array<vec3<f32>, 16>, sample_count: u32) -> vec3<f32> {
    var min_v = vec3<f32>(3.402823e38);
    var max_v = vec3<f32>(-3.402823e38);

    for (var i = 0u; i < sample_count; i += 1u) {
        min_v = min(min_v, samples[i]);
        max_v = max(max_v, samples[i]);
    }

    let center = (min_v + max_v) * 0.5;

    var diagonals: array<vec3<f32>, 4>;
    diagonals[0] = norm3(vec3<f32>(max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z));
    diagonals[1] = norm3(vec3<f32>(max_v.x - min_v.x, max_v.y - min_v.y, min_v.z - max_v.z));
    diagonals[2] = norm3(vec3<f32>(max_v.x - min_v.x, min_v.y - max_v.y, max_v.z - min_v.z));
    diagonals[3] = norm3(vec3<f32>(min_v.x - max_v.x, max_v.y - min_v.y, max_v.z - min_v.z));

    var best_diag = vec3<f32>(0.0);
    var best_var = -1.0;

    for (var d = 0u; d < 4u; d += 1u) {
        let diagonal = diagonals[d];
        var var_acc = 0.0;

        for (var i = 0u; i < sample_count; i += 1u) {
            let t = dot(samples[i] - center, diagonal);
            var_acc = var_acc + t * t;
        }

        if var_acc > best_var {
            best_var = var_acc;
            best_diag = diagonal;
        }
    }

    return best_diag;
}

fn max_variance_diagonal_axis4(samples: array<vec4<f32>, 16>, sample_count: u32) -> vec4<f32> {
    var min_v = vec4<f32>(3.402823e38);
    var max_v = vec4<f32>(-3.402823e38);

    for (var i = 0u; i < sample_count; i += 1u) {
        min_v = min(min_v, samples[i]);
        max_v = max(max_v, samples[i]);
    }

    let center = (min_v + max_v) * 0.5;

    let dx = max_v.x - min_v.x;
    let dy = max_v.y - min_v.y;
    let dz = max_v.z - min_v.z;
    let dw = max_v.w - min_v.w;

    var diagonals: array<vec4<f32>, 8>;
    diagonals[0] = norm4(vec4<f32>(dx, dy, dz, dw));
    diagonals[1] = norm4(vec4<f32>(dx, dy, dz, -dw));
    diagonals[2] = norm4(vec4<f32>(dx, dy, -dz, dw));
    diagonals[3] = norm4(vec4<f32>(dx, dy, -dz, -dw));
    diagonals[4] = norm4(vec4<f32>(dx, -dy, dz, dw));
    diagonals[5] = norm4(vec4<f32>(dx, -dy, dz, -dw));
    diagonals[6] = norm4(vec4<f32>(dx, -dy, -dz, dw));
    diagonals[7] = norm4(vec4<f32>(dx, -dy, -dz, -dw));

    var best_diag = vec4<f32>(0.0);
    var best_var = -1.0;

    for (var d = 0u; d < 8u; d += 1u) {
        let diagonal = diagonals[d];
        var var_acc = 0.0;

        for (var i = 0u; i < sample_count; i += 1u) {
            let t = dot(samples[i] - center, diagonal);
            var_acc = var_acc + t * t;
        }

        if var_acc > best_var {
            best_var = var_acc;
            best_diag = diagonal;
        }
    }

    return best_diag;
}

fn principal_axis_vec2(samples: array<vec2<f32>, 16>, sample_count: u32) -> vec2<f32> {
    return max_variance_diagonal_axis2(samples, sample_count);
}

fn principal_axis_vec3(samples: array<vec3<f32>, 16>, sample_count: u32) -> vec3<f32> {
    return max_variance_diagonal_axis3(samples, sample_count);
}

fn principal_axis_vec4(samples: array<vec4<f32>, 16>, sample_count: u32) -> vec4<f32> {
    return max_variance_diagonal_axis4(samples, sample_count);
}

fn project_f32(sample: f32) -> f32 {
    return sample;
}

fn project_vec2(sample: vec2<f32>, axis: vec2<f32>) -> f32 {
    return dot(sample, axis);
}

fn project_vec3(sample: vec3<f32>, axis: vec3<f32>) -> f32 {
    return dot(sample, axis);
}

fn project_vec4(sample: vec4<f32>, axis: vec4<f32>) -> f32 {
    return dot(sample, axis);
}

fn solve_endpoints_f32(
    weights: array<f32, 16>,
    samples: array<f32, 16>,
    sample_count: u32,
) -> vec3<f32> {
    var A = 0.0;
    var B = 0.0;
    var C = 0.0;

    var X = 0.0;
    var Y = 0.0;

    for (var i = 0u; i < sample_count; i += 1u) {
        let w = weights[i];
        let u = 1.0 - w;
        let s = samples[i];

        A = A + u * u;
        B = B + u * w;
        C = C + w * w;

        X = X + s * u;
        Y = Y + s * w;
    }

    let D = A * C - B * B;
    if abs(D) < CLUSTER_FIT_EPSILON {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let invD = 1.0 / D;
    let c0 = (X * C - Y * B) * invD;
    let c1 = (Y * A - X * B) * invD;

    return vec3<f32>(1.0, c0, c1);
}

fn solve_endpoints_vec2(
    weights: array<f32, 16>,
    samples: array<vec2<f32>, 16>,
    sample_count: u32,
) -> array<vec2<f32>, 3> {
    var A = 0.0;
    var B = 0.0;
    var C = 0.0;

    var X = vec2<f32>(0.0);
    var Y = vec2<f32>(0.0);

    for (var i = 0u; i < sample_count; i += 1u) {
        let w = weights[i];
        let u = 1.0 - w;
        let s = samples[i];

        A = A + u * u;
        B = B + u * w;
        C = C + w * w;

        X = X + s * u;
        Y = Y + s * w;
    }

    var out: array<vec2<f32>, 3>;

    let D = A * C - B * B;
    if abs(D) < CLUSTER_FIT_EPSILON {
        out[0] = vec2<f32>(0.0);
        out[1] = vec2<f32>(0.0);
        out[2] = vec2<f32>(0.0);
        return out;
    }

    let invD = 1.0 / D;
    out[0] = vec2<f32>(1.0, 1.0);
    out[1] = (X * C - Y * B) * invD;
    out[2] = (Y * A - X * B) * invD;
    return out;
}

fn solve_endpoints_vec3(
    weights: array<f32, 16>,
    samples: array<vec3<f32>, 16>,
    sample_count: u32,
) -> array<vec3<f32>, 3> {
    var A = 0.0;
    var B = 0.0;
    var C = 0.0;

    var X = vec3<f32>(0.0);
    var Y = vec3<f32>(0.0);

    for (var i = 0u; i < sample_count; i += 1u) {
        let w = weights[i];
        let u = 1.0 - w;
        let s = samples[i];

        A = A + u * u;
        B = B + u * w;
        C = C + w * w;

        X = X + s * u;
        Y = Y + s * w;
    }

    var out: array<vec3<f32>, 3>;

    let D = A * C - B * B;
    if abs(D) < CLUSTER_FIT_EPSILON {
        out[0] = vec3<f32>(0.0);
        out[1] = vec3<f32>(0.0);
        out[2] = vec3<f32>(0.0);
        return out;
    }

    let invD = 1.0 / D;
    out[0] = vec3<f32>(1.0);
    out[1] = (X * C - Y * B) * invD;
    out[2] = (Y * A - X * B) * invD;
    return out;
}

fn solve_endpoints_vec4(
    weights: array<f32, 16>,
    samples: array<vec4<f32>, 16>,
    sample_count: u32,
) -> array<vec4<f32>, 3> {
    var A = 0.0;
    var B = 0.0;
    var C = 0.0;

    var X = vec4<f32>(0.0);
    var Y = vec4<f32>(0.0);

    for (var i = 0u; i < sample_count; i += 1u) {
        let w = weights[i];
        let u = 1.0 - w;
        let s = samples[i];

        A = A + u * u;
        B = B + u * w;
        C = C + w * w;

        X = X + s * u;
        Y = Y + s * w;
    }

    var out: array<vec4<f32>, 3>;

    let D = A * C - B * B;
    if abs(D) < CLUSTER_FIT_EPSILON {
        out[0] = vec4<f32>(0.0);
        out[1] = vec4<f32>(0.0);
        out[2] = vec4<f32>(0.0);
        return out;
    }

    let invD = 1.0 / D;
    out[0] = vec4<f32>(1.0);
    out[1] = (X * C - Y * B) * invD;
    out[2] = (Y * A - X * B) * invD;
    return out;
}

fn build_palette_f32(c0: f32, c1: f32, bits: u32, palette_count: u32) -> array<f32, 8> {
    var palette: array<f32, 8>;
    palette[0] = snap_unorm_f32(c0, bits);

    for (var i = 1u; i + 1u < palette_count; i += 1u) {
        let t = f32(i) / f32(palette_count - 1u);
        palette[i] = snap_unorm_f32(c0 * (1.0 - t) + c1 * t, bits);
    }

    palette[palette_count - 1u] = snap_unorm_f32(c1, bits);
    return palette;
}

fn build_palette_vec2(
    c0: vec2<f32>,
    c1: vec2<f32>,
    bits: vec2<u32>,
    palette_count: u32,
) -> array<vec2<f32>, 8> {
    var palette: array<vec2<f32>, 8>;
    palette[0] = snap_unorm_vec2(c0, bits);

    for (var i = 1u; i + 1u < palette_count; i += 1u) {
        let t = f32(i) / f32(palette_count - 1u);
        palette[i] = snap_unorm_vec2(c0 * (1.0 - t) + c1 * t, bits);
    }

    palette[palette_count - 1u] = snap_unorm_vec2(c1, bits);
    return palette;
}

fn build_palette_vec3(
    c0: vec3<f32>,
    c1: vec3<f32>,
    bits: vec3<u32>,
    palette_count: u32,
) -> array<vec3<f32>, 8> {
    var palette: array<vec3<f32>, 8>;
    palette[0] = snap_unorm_vec3(c0, bits);

    for (var i = 1u; i + 1u < palette_count; i += 1u) {
        let t = f32(i) / f32(palette_count - 1u);
        palette[i] = snap_unorm_vec3(c0 * (1.0 - t) + c1 * t, bits);
    }

    palette[palette_count - 1u] = snap_unorm_vec3(c1, bits);
    return palette;
}

fn build_palette_vec4(
    c0: vec4<f32>,
    c1: vec4<f32>,
    bits: vec4<u32>,
    palette_count: u32,
) -> array<vec4<f32>, 8> {
    var palette: array<vec4<f32>, 8>;
    palette[0] = snap_unorm_vec4(c0, bits);

    for (var i = 1u; i + 1u < palette_count; i += 1u) {
        let t = f32(i) / f32(palette_count - 1u);
        palette[i] = snap_unorm_vec4(c0 * (1.0 - t) + c1 * t, bits);
    }

    palette[palette_count - 1u] = snap_unorm_vec4(c1, bits);
    return palette;
}

fn index_error_f32(sample: f32, palette: array<f32, 8>, palette_count: u32, mode: u32) -> vec2<f32> {
    var best_index = 0u;
    var best_error = 3.402823e38;

    for (var i = 0u; i < palette_count; i += 1u) {
        let e = error_f32(mode, sample, palette[i]);
        if e < best_error {
            best_error = e;
            best_index = i;
        }
    }

    return vec2<f32>(f32(best_index), best_error);
}

fn index_error_vec2(sample: vec2<f32>, palette: array<vec2<f32>, 8>, palette_count: u32, mode: u32) -> vec2<f32> {
    var best_index = 0u;
    var best_error = 3.402823e38;

    for (var i = 0u; i < palette_count; i += 1u) {
        let e = error_vec2(mode, sample, palette[i]);
        if e < best_error {
            best_error = e;
            best_index = i;
        }
    }

    return vec2<f32>(f32(best_index), best_error);
}

fn index_error_vec3(sample: vec3<f32>, palette: array<vec3<f32>, 8>, palette_count: u32, mode: u32) -> vec2<f32> {
    var best_index = 0u;
    var best_error = 3.402823e38;

    for (var i = 0u; i < palette_count; i += 1u) {
        let e = error_vec3(mode, sample, palette[i]);
        if e < best_error {
            best_error = e;
            best_index = i;
        }
    }

    return vec2<f32>(f32(best_index), best_error);
}

fn index_error_vec4(sample: vec4<f32>, palette: array<vec4<f32>, 8>, palette_count: u32, mode: u32) -> vec2<f32> {
    var best_index = 0u;
    var best_error = 3.402823e38;

    for (var i = 0u; i < palette_count; i += 1u) {
        let e = error_vec4(mode, sample, palette[i]);
        if e < best_error {
            best_error = e;
            best_index = i;
        }
    }

    return vec2<f32>(f32(best_index), best_error);
}

fn sort_by_projection(order_indices: ptr<function, array<u32, 16>>, projections: ptr<function, array<f32, 16>>, sample_count: u32) {
    for (var i = 0u; i < sample_count; i += 1u) {
        for (var j = i + 1u; j < sample_count; j += 1u) {
            if (*projections)[i] > (*projections)[j] {
                let pi = (*projections)[i];
                (*projections)[i] = (*projections)[j];
                (*projections)[j] = pi;

                let oi = (*order_indices)[i];
                (*order_indices)[i] = (*order_indices)[j];
                (*order_indices)[j] = oi;
            }
        }
    }
}

fn cluster_fit_vec3(
    samples: array<vec3<f32>, 16>,
    sample_count: u32,
    palette_count: u32,
    bits: vec3<u32>,
    error_mode: u32,
    iter_begin: u32,
    iter_end: u32,
) -> ClusterFitVec3 {
    var out: ClusterFitVec3;
    out.endpoint0 = vec3<f32>(0.0);
    out.endpoint1 = vec3<f32>(0.0);
    out.indices = array<u32, 16>();
    out.error = 0.0;

    if sample_count == 0u || palette_count < 2u || palette_count > CLUSTER_FIT_MAX_PALETTE || sample_count > CLUSTER_FIT_MAX_SAMPLES || sample_count < palette_count {
        return out;
    }

    let axis = principal_axis_vec3(samples, sample_count);

    var order_idx: array<u32, 16>;
    var projections: array<f32, 16>;

    for (var i = 0u; i < sample_count; i += 1u) {
        order_idx[i] = i;
        projections[i] = project_vec3(samples[i], axis);
    }

    sort_by_projection(&order_idx, &projections, sample_count);

    let endpoints0 = fallback_endpoints_vec3(samples, sample_count);

    var best0 = endpoints0[0];
    var best1 = endpoints0[1];
    var best_indices = array<u32, 16>();
    var best_error = 0.0;

    {
        let palette = build_palette_vec3(best0, best1, bits, palette_count);
        for (var i = 0u; i < sample_count; i += 1u) {
            let sample_index = order_idx[i];
            let ie = index_error_vec3(samples[sample_index], palette, palette_count, error_mode);
            best_indices[sample_index] = u32(ie.x);
            best_error += ie.y;
        }
    }

    var iter_state = binom_iter_init_at(sample_count - 1u, palette_count - 1u, iter_begin);

    for (var iter = iter_begin; iter < iter_end; iter += 1u) {
        var weights = array<f32, 16>();

        for (var i = 0u; i < sample_count; i += 1u) {
            var cluster = 0u;
            for (var c = 0u; c < palette_count - 1u; c += 1u) {
                if i > iter_state.indices[c] {
                    cluster += 1u;
                }
            }
            let t = f32(cluster) / f32(palette_count - 1u);
            weights[order_idx[i]] = t;
        }

        let solved = solve_endpoints_vec3(weights, samples, sample_count);
        if solved[0].x > 0.0 {
            let c0 = snap_unorm_vec3(solved[1], bits);
            let c1 = snap_unorm_vec3(solved[2], bits);

            let palette = build_palette_vec3(c0, c1, bits, palette_count);

            var total_error = 0.0;
            var indices = array<u32, 16>();

            for (var i = 0u; i < sample_count; i += 1u) {
                let sample_index = order_idx[i];
                let ie = index_error_vec3(samples[sample_index], palette, palette_count, error_mode);
                indices[sample_index] = u32(ie.x);
                total_error = total_error + ie.y;
            }

            if best_error > total_error {
                best_error = total_error;
                best0 = c0;
                best1 = c1;
                best_indices = indices;
            }
        }

        binom_iter_next(&iter_state);
    }

    out.endpoint0 = best0;
    out.endpoint1 = best1;
    out.indices = best_indices;
    out.error = best_error;

    return out;
}
