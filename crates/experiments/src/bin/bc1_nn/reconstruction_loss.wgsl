fn bc1_loss_endpoints(prediction: Output, target_value: Target) -> array<vec3<f32>, 2> {
    let a = vec3<f32>(target_value[0], target_value[1], target_value[2]) + vec3<f32>(prediction[0], prediction[1], prediction[2]);
    let b = vec3<f32>(target_value[45], target_value[46], target_value[47]) + vec3<f32>(prediction[3], prediction[4], prediction[5]);
    let levels = vec3<f32>(31.0, 63.0, 31.0);
    return array<vec3<f32>, 2>(floor(clamp(a, vec3<f32>(0.0), vec3<f32>(1.0)) * levels + vec3<f32>(0.5)) / levels,
        floor(clamp(b, vec3<f32>(0.0), vec3<f32>(1.0)) * levels + vec3<f32>(0.5)) / levels);
}

fn bc1_loss_weight(a: vec3<f32>, b: vec3<f32>, color: vec3<f32>) -> f32 {
    if all(a == b) { return 0.0; }
    let weights = array<f32, 4>(0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0);
    var best = 1e30;
    var beta = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let delta = a * (1.0 - weights[k]) + b * weights[k] - color;
        let squared = dot(delta, delta);
        if squared < best { best = squared; beta = weights[k]; }
    }
    return beta;
}

fn nn_loss(prediction: Output, target_value: Target) -> f32 {
    let endpoints = bc1_loss_endpoints(prediction, target_value);
    var value = 0.0;
    for (var texel = 0u; texel < 16u; texel++) {
        let color = vec3<f32>(target_value[3u * texel], target_value[3u * texel + 1u], target_value[3u * texel + 2u]);
        let beta = bc1_loss_weight(endpoints[0], endpoints[1], color);
        let delta = endpoints[0] * (1.0 - beta) + endpoints[1] * beta - color;
        value += (sqrt(dot(delta, delta) + 1e-12) - 1e-6) / 16.0;
    }
    for (var i = 0u; i < 6u; i++) {
        let delta = prediction[i] - target_value[48u + i];
        value += 0.005 * delta * delta;
    }
    return value;
}

fn nn_loss_derivative(prediction: Output, target_value: Target) -> Output {
    let endpoints = bc1_loss_endpoints(prediction, target_value);
    var result: Output;
    var mask: Output;
    for (var i = 0u; i < 6u; i++) {
        result[i] = 0.01 * (prediction[i] - target_value[48u + i]);
        var base = i;
        if i >= 3u { base = 45u + i - 3u; }
        let z = target_value[base] + prediction[i];
        mask[i] = select(0.0, 1.0, z >= 0.0 && z <= 1.0);
    }
    for (var texel = 0u; texel < 16u; texel++) {
        let color = vec3<f32>(target_value[3u * texel], target_value[3u * texel + 1u], target_value[3u * texel + 2u]);
        let beta = bc1_loss_weight(endpoints[0], endpoints[1], color);
        let delta = endpoints[0] * (1.0 - beta) + endpoints[1] * beta - color;
        let grad = delta / (16.0 * sqrt(dot(delta, delta) + 1e-12));
        for (var c = 0u; c < 3u; c++) {
            result[c] += (1.0 - beta) * grad[c] * mask[c];
            result[c + 3u] += beta * grad[c] * mask[c + 3u];
        }
    }
    return result;
}
