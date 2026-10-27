// One invocation owns one patch, including its attribute and scratch ranges.
// Metadata header: patch/vertex/triangle counts, UV count, flags, tangent UV set,
// four precisions, vertex word stride, patch word stride, stream count, padding.
// Patch record: six counts/bases, sixteen inverse-matrix words, four words per
// UV transform, then five words per stream. Packing lives in mesh::gpu::metadata.
@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> metadata: array<u32>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<u32>;
// Global triangle indices followed by one primitive ID per triangle.
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;
// Three words per vertex, reused between spatial decoding and vector accumulation.
@group(0) @binding(5) var<storage, read_write> scratch: array<u32>;
const SENTINEL: u32 = 0x7fc00001u;
const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;
struct Rans {
    state: vec2<u32>,
    cursor: u32,
    end: u32,
    start: u32,
    count: u32,
}

fn umuladd64(a: u32, b: u32, c: u32) -> vec2<u32> {
    let p00 = (a & 65535u) * (b & 65535u);
    let p01 = (a & 65535u) * (b >> 16u);
    let p10 = (a >> 16u) * (b & 65535u);
    let p11 = (a >> 16u) * (b >> 16u);
    let lo0 = p00 + c;
    var carry = select(0u, 1u, lo0 < p00);
    let mid = p01 + p10;
    carry += select(0u, 65536u, mid < p01);
    let low = lo0 + (mid << 16u);
    carry += select(0u, 1u, low < lo0);
    return vec2(low, p11 + (mid >> 16u) + carry);
}

// Convert an unsigned 41-bit determinant magnitude with one ties-to-even rounding.
fn magnitude_float(v: vec2<u32>) -> f32 {
    if (v.y == 0u) {
        return f32(v.x);
    }
    let shift = 32u + firstLeadingBit(v.y) - 23u;
    var top = (v.x >> shift) | (v.y << (32u - shift));
    let remainder = v.x & ((1u << shift) - 1u);
    let half = 1u << (shift - 1u);
    if (remainder > half || (remainder == half && (top & 1u) != 0u)) {
        top += 1u;
    }
    return f32(top) * bitcast<f32>((127u + shift) << 23u);
}

// UV differences are at most twenty bits. Both products and their signed
// difference are evaluated exactly before converting the determinant to f32.
fn grid_determinant(d1: vec2<i32>, d2: vec2<i32>) -> f32 {
    let a = umuladd64(u32(abs(d1.x)), u32(abs(d2.y)), 0u);
    let b = umuladd64(u32(abs(d1.y)), u32(abs(d2.x)), 0u);
    var negative = (d1.x < 0) != (d2.y < 0);
    let b_negative = (d1.y < 0) != (d2.x < 0);
    var magnitude: vec2<u32>;
    if (negative != b_negative) {
        let low = a.x + b.x;
        magnitude = vec2(low, a.y + b.y + select(0u, 1u, low < a.x));
    } else {
        let a_greater = a.y > b.y || (a.y == b.y && a.x >= b.x);
        let large = select(b, a, a_greater);
        let small = select(a, b, a_greater);
        magnitude = vec2(large.x - small.x, large.y - small.y - select(0u, 1u, large.x < small.x));
        negative = negative != !a_greater;
    }
    return select(magnitude_float(magnitude), -magnitude_float(magnitude), negative);
}

fn stream(p: u32, role: u32) -> Rans {
    let s = p + 22u + 4u * metadata[3] + role * 5u;
    let off = metadata[s];
    var r: Rans;
    r.cursor = off;
    r.end = off + metadata[s + 1u];
    r.start = metadata[s + 2u];
    r.count = metadata[s + 3u];
    r.state = vec2(0u);
    return r;
}

fn next(r: ptr<function, Rans>) -> u32 {
    if ((*r).state.y == 0u && (*r).cursor < (*r).end) {
        (*r).state = vec2(payload[(*r).cursor], (*r).state.x);
        (*r).cursor += 1u;
    }
    if ((*r).state.y == 0u && (*r).cursor < (*r).end) {
        (*r).state = vec2(payload[(*r).cursor], (*r).state.x);
        (*r).cursor += 1u;
    }
    let bucket = (*r).state.x;
    var lo = (*r).start;
    var hi = lo + (*r).count;
    loop {
        if (lo + 1u >= hi) {
            break;
        }
        let mid = (lo + hi) >> 1u;
        if (table[mid * 3u + 1u] <= bucket) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = lo * 3u;
    (*r).state = umuladd64((*r).state.y, table[t], bucket - table[t + 1u]);
    return table[t + 2u];
}

fn unzigzag(v: u32) -> i32 {
    return select(i32(v >> 1u), -i32(v >> 1u) - 1, (v & 1u) != 0u);
}

fn finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return (a.x * b.x + a.y * b.y) + a.z * b.z;
}

fn unit(a: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot3(a, a));
    if (len > 0.0 && finite(len)) {
        return a * (1.0 / len);
    }
    return vec3(0.0, 0.0, 1.0);
}

fn axis(n: vec3<f32>) -> vec3<f32> {
    var k = 0u;
    if (abs(n.y) < abs(n[k])) {
        k = 1u;
    }
    if (abs(n.z) < abs(n[k])) {
        k = 2u;
    }
    var a = vec3(0.0);
    a[k] = 1.0;
    return unit(cross(n, a));
}

fn load_q(id: u32) -> vec3<u32> {
    return vec3(scratch[id * 3u], scratch[id * 3u + 1u], scratch[id * 3u + 2u]);
}

fn store_q(id: u32, v: vec3<u32>) {
    scratch[id * 3u] = v.x;
    scratch[id * 3u + 1u] = v.y;
    scratch[id * 3u + 2u] = v.z;
}

fn load_sum(id: u32) -> vec3<f32> {
    return bitcast<vec3<f32>>(load_q(id));
}

fn store_sum(id: u32, v: vec3<f32>) {
    store_q(id, bitcast<vec3<u32>>(v));
}

fn value(id: u32, off: u32) -> vec3<f32> {
    let b = id * metadata[10] + off;
    return bitcast<vec3<f32>>(vec3(vertices[b], vertices[b + 1u], vertices[b + 2u]));
}

fn put_value(id: u32, off: u32, v: vec3<f32>) {
    let b = id * metadata[10] + off;
    let w = bitcast<vec3<u32>>(v);
    vertices[b] = w.x;
    vertices[b + 1u] = w.y;
    vertices[b + 2u] = w.z;
}

fn uv_offset(uv_set: u32) -> u32 {
    return 3u + select(0u, 3u, (metadata[4] & 1u) != 0u) +
           select(0u, 4u, (metadata[4] & 2u) != 0u) + uv_set * 2u;
}

fn uv_grid(id: u32, uv_set: u32) -> vec2<i32> {
    let b = id * metadata[10] + uv_offset(uv_set);
    return vec2<i32>(vec2(vertices[b], vertices[b + 1u]));
}

fn triangle(face: u32) -> vec3<u32> {
    let b = face * 3u;
    return vec3(indices[b], indices[b + 1u], indices[b + 2u]);
}

fn stored_triangle(p: u32, i: u32) -> vec3<u32> {
    var t = triangle(metadata[p + 1u] + i) - vec3(metadata[p]);
    if ((i & 1u) != 0u) {
        t = t.xzy;
    }
    return t;
}

fn contains(t: vec3<u32>, v: u32) -> bool {
    return any(t == vec3(v));
}

fn emit(id: u32, pred: vec3<u32>, dimensions: u32, mask: u32, r: ptr<function, Rans>) {
    if (!all(load_q(id) == vec3(SENTINEL))) {
        return;
    }
    var q = vec3(0u);
    for (var k = 0u; k < dimensions; k++) {
        q[k] = (pred[k] + next(r)) & mask;
    }
    store_q(id, q);
}

// Each spatial attribute uses its own entropy state and reuses three scratch words.
fn decode_spatial(p: u32, role: u32, uv_set: u32) {
    let base = metadata[p];
    let count = metadata[p + 5u];
    let dims = select(2u, 3u, role == 2u);
    let bits = select(metadata[7], metadata[6], role == 2u);
    let mask = (1u << bits) - 1u;
    for (var i = 0u; i < count; i++) {
        store_q(base + i, vec3(SENTINEL));
    }
    var r = stream(p, role);
    var prev = vec3(0u);
    var has_prev = false;
    var last = base;
    for (var i = 0u; i < metadata[p + 2u]; i++) {
        let t = stored_triangle(p, i);
        var overlap = vec3(0u);
        var n = 0u;
        if (has_prev) {
            for (var c = 0u; c < 3u; c++) {
                let v = t[c];
                var duplicate = false;
                for (var j = 0u; j < n; j++) {
                    if (overlap[j] == v) {
                        duplicate = true;
                    }
                }
                if (contains(prev, v) && !duplicate) {
                    overlap[n] = v;
                    n += 1u;
                }
            }
        }
        var pred = vec3(0u);
        if (n >= 2u) {
            pred = load_q(base + overlap.x) + load_q(base + overlap.y);
            for (var c = 0u; c < 3u; c++) {
                let v = prev[c];
                var is_shared = false;
                for (var j = 0u; j < n; j++) {
                    if (overlap[j] == v) {
                        is_shared = true;
                    }
                }
                if (!is_shared) {
                    pred -= load_q(base + v);
                    break;
                }
            }
        } else if (n == 1u) {
            pred = load_q(base + overlap.x);
        }
        for (var c = 0u; c < 3u; c++) {
            let id = base + t[c];
            var guess = pred;
            if (n == 0u && c != 0u) {
                guess = load_q(base + t.x);
            }
            if (all(load_q(id) == vec3(SENTINEL))) {
                emit(id, guess, dims, mask, &r);
                last = id;
            }
        }
        prev = t;
        has_prev = true;
    }
    for (var i = 0u; i < count; i++) {
        let id = base + i;
        if (all(load_q(id) == vec3(SENTINEL))) {
            var pred = vec3(0u);
            if (id != base) {
                pred = load_q(last);
            }
            emit(id, pred, dims, mask, &r);
            last = id;
        }
    }
    for (var i = 0u; i < count; i++) {
        let id = base + i;
        let q = vec3<f32>(load_q(id)) / f32(mask);
        if (role == 2u) {
            var v = vec3(0.0);
            for (var k = 0u; k < 3u; k++) {
                let m = p + 6u + k * 4u;
                v[k] = ((q.x * bitcast<f32>(metadata[m]) + q.y * bitcast<f32>(metadata[m + 1u])) +
                        q.z * bitcast<f32>(metadata[m + 2u])) +
                       bitcast<f32>(metadata[m + 3u]);
            }
            put_value(id, 0u, v);
        } else {
            let b = p + 22u + uv_set * 4u;
            let out = id * metadata[10] + uv_offset(uv_set);
            for (var k = 0u; k < 2u; k++) {
                if ((metadata[4] & 2u) != 0u && uv_set == metadata[5]) {
                    vertices[out + k] = load_q(id)[k];
                } else {
                    vertices[out + k] = bitcast<u32>(bitcast<f32>(metadata[b + k]) +
                                                     q[k] * bitcast<f32>(metadata[b + k + 2u]));
                }
            }
        }
    }
}

fn decode_normal(aq: u32, bq: u32, predicted: vec3<f32>) -> vec3<f32> {
    let a = f32(unzigzag(aq)) * (PI / (2.0 * f32((1u << (metadata[8] - 1u)) - 1u)));
    let b = f32(unzigzag(bq)) * (TAU / f32(1u << metadata[8]));
    let n = unit(predicted);
    let u = axis(n);
    let v = cross(u, n);
    return unit(n * (cos(a) * cos(b)) - u * (cos(a) * sin(b)) + v * sin(a));
}

// Area sums use internal triangles only, with original winding already restored.
fn decode_normals(p: u32, role: u32) {
    let base = metadata[p];
    let count = metadata[p + 5u];
    for (var i = 0u; i < count; i++) {
        store_sum(base + i, vec3(0.0));
    }
    for (var i = 0u; i < metadata[p + 2u]; i++) {
        let t = triangle(metadata[p + 1u] + i);
        let g = cross(value(t.y, 0u) - value(t.x, 0u), value(t.z, 0u) - value(t.x, 0u));
        for (var c = 0u; c < 3u; c++) {
            store_sum(t[c], load_sum(t[c]) + g);
        }
    }
    var r = stream(p, role);
    var previous = vec3(0.0, 0.0, 1.0);
    for (var i = 0u; i < count; i++) {
        let id = base + i;
        var predicted = previous;
        if (i < metadata[p + 4u]) {
            predicted = unit(load_sum(id));
        }
        let a = next(&r);
        let b = next(&r);
        previous = decode_normal(a, b, predicted);
        put_value(id, 3u, previous);
    }
}

// Tangent predictors use decoded positions, exact UV-grid differences, and
// decoded normals. The common positive patch UV scale cancels on normalization.
fn decode_tangents(p: u32, role: u32) {
    let base = metadata[p];
    let count = metadata[p + 5u];
    let uv_set = metadata[5];
    for (var i = 0u; i < count; i++) {
        store_sum(base + i, vec3(0.0));
    }
    for (var i = 0u; i < metadata[p + 2u]; i++) {
        let t = triangle(metadata[p + 1u] + i);
        let e1 = value(t.y, 0u) - value(t.x, 0u);
        let e2 = value(t.z, 0u) - value(t.x, 0u);
        let d1 = uv_grid(t.y, uv_set) - uv_grid(t.x, uv_set);
        let d2 = uv_grid(t.z, uv_set) - uv_grid(t.x, uv_set);
        let det = grid_determinant(d1, d2);
        if (det != 0.0 && finite(det)) {
            let tangent = (e1 * f32(d2.y) - e2 * f32(d1.y)) * (1.0 / det);
            if (finite(tangent.x) && finite(tangent.y) && finite(tangent.z)) {
                for (var c = 0u; c < 3u; c++) {
                    store_sum(t[c], load_sum(t[c]) + tangent);
                }
            }
        }
    }
    var r = stream(p, role);
    var signs = stream(p, role + 1u);
    for (var i = 0u; i < count; i++) {
        let id = base + i;
        let n = value(id, 3u);
        let sum = load_sum(id);
        let projected = sum - n * dot3(sum, n);
        var zero = axis(unit(n));
        if (dot3(projected, projected) > 0.0 && finite(projected.x) && finite(projected.y) &&
            finite(projected.z)) {
            zero = unit(projected);
        }
        let a = f32(unzigzag(next(&r))) * (TAU / f32(1u << metadata[9]));
        let tangent = unit(zero * cos(a) + cross(zero, n) * sin(a));
        put_value(id, 6u, tangent);
        vertices[id * metadata[10] + 9u] = bitcast<u32>(select(1.0, -1.0, next(&signs) != 0u));
    }
    // The selected UV output words retain exact grid coordinates until predictors
    // finish; convert them in place to their renderer-facing f32 representation.
    let bounds = p + 22u + uv_set * 4u;
    let scale = f32((1u << metadata[7]) - 1u);
    for (var i = 0u; i < count; i++) {
        let out = (base + i) * metadata[10] + uv_offset(uv_set);
        for (var k = 0u; k < 2u; k++) {
            vertices[out + k] = bitcast<u32>(bitcast<f32>(metadata[bounds + k]) +
                f32(vertices[out + k]) / scale * bitcast<f32>(metadata[bounds + k + 2u]));
        }
    }
}
@compute @workgroup_size(64) fn decode_mesh(@builtin(global_invocation_id) tid: vec3<u32>) {
    if (tid.x >= metadata[0]) {
        return;
    }
    let p = 16u + tid.x * metadata[11];
    let base = metadata[p];
    let facebase = metadata[p + 1u];
    let head = metadata[p + 2u];
    let faces = metadata[p + 3u];
    var r = stream(p, 0u);
    var primitive = stream(p, 1u);
    var prev = vec3(0u);
    for (var i = 0u; i < faces; i++) {
        if (i == head && head != 0u) {
            prev += vec3(base);
        }
        var t = vec3(0u);
        for (var c = 0u; c < 3u; c++) {
            t[c] = prev[c] + bitcast<u32>(unzigzag(next(&r)));
        }
        prev = t;
        if (i < head) {
            t += vec3(base);
        }
        if ((i & 1u) != 0u) {
            t = t.xzy;
        }
        let out = (facebase + i) * 3u;
        indices[out] = t.x;
        indices[out + 1u] = t.y;
        indices[out + 2u] = t.z;
        indices[metadata[2] * 3u + facebase + i] = next(&primitive);
    }
    decode_spatial(p, 2u, 0u);
    for (var uv_set = 0u; uv_set < metadata[3]; uv_set++) {
        decode_spatial(p, 3u + uv_set, uv_set);
    }
    var role = 3u + metadata[3];
    if ((metadata[4] & 1u) != 0u) {
        decode_normals(p, role);
        role += 1u;
    }
    if ((metadata[4] & 2u) != 0u) {
        decode_tangents(p, role);
        role += 2u;
    }
    if ((metadata[4] & 4u) != 0u) {
        var colors = stream(p, role);
        for (var i = 0u; i < metadata[p + 5u]; i++) {
            vertices[(base + i) * metadata[10] + metadata[10] - 1u] = next(&colors);
        }
    }
}


