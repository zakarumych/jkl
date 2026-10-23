//! Quantized BC1 reconstruction loss with a frozen-assignment straight-through gradient.

use mev::kernels::nn::Loss;

const EPSILON: f32 = 1e-6;
const ANCHOR: f32 = 0.005;
const WEIGHTS: [f32; 4] = [0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0];

pub fn loss() -> Loss {
    Loss::Custom {
        target_size: 54,
        source: include_str!("reconstruction_loss.wgsl").into(),
    }
}

/// Packs normalized PCA-sorted RGB inputs and teacher endpoint residuals.
pub fn target(input: &[f32], teacher_residual: &[f32]) -> [f32; 54] {
    assert_eq!(input.len(), 48);
    assert_eq!(teacher_residual.len(), 6);
    let mut result = [0.0; 54];
    result[..48].copy_from_slice(input);
    result[48..].copy_from_slice(teacher_residual);
    result
}

fn endpoints(prediction: &[f32; 6], target: &[f32; 54]) -> ([f32; 6], [f32; 6]) {
    let mut quantized = [0.0; 6];
    let mut mask = [0.0; 6];
    for i in 0..6 {
        let base = if i < 3 { i } else { 45 + i - 3 };
        let z = target[base] + prediction[i];
        let levels = if i % 3 == 1 { 63.0 } else { 31.0 };
        quantized[i] = (z.clamp(0.0, 1.0) * levels + 0.5).floor() / levels;
        mask[i] = if (0.0..=1.0).contains(&z) { 1.0 } else { 0.0 };
    }
    (quantized, mask)
}

fn assignments(quantized: &[f32; 6], target: &[f32; 54]) -> [f32; 16] {
    let equal = quantized[..3] == quantized[3..];
    std::array::from_fn(|texel| {
        let mut best = f32::INFINITY;
        let mut weight = 0.0;
        for &beta in &WEIGHTS[..if equal { 1 } else { 4 }] {
            let squared = (0..3)
                .map(|c| {
                    let d = quantized[c] * (1.0 - beta) + quantized[c + 3] * beta
                        - target[3 * texel + c];
                    d * d
                })
                .sum::<f32>();
            if squared < best {
                best = squared;
                weight = beta;
            }
        }
        weight
    })
}

#[allow(dead_code)]
pub fn loss_value(prediction: &[f32; 6], target: &[f32; 54]) -> f32 {
    let (quantized, _) = endpoints(prediction, target);
    let weights = assignments(&quantized, target);
    let mut value = 0.0;
    for (texel, &beta) in weights.iter().enumerate() {
        let squared = (0..3)
            .map(|c| {
                let d =
                    quantized[c] * (1.0 - beta) + quantized[c + 3] * beta - target[3 * texel + c];
                d * d
            })
            .sum::<f32>();
        value += ((squared + EPSILON * EPSILON).sqrt() - EPSILON) / 16.0;
    }
    for i in 0..6 {
        value += ANCHOR * (prediction[i] - target[48 + i]).powi(2);
    }
    value
}

/// Differentiates fixed nearest-palette assignments; RGB565 rounding uses unit STE.
#[allow(dead_code)]
pub fn gradient(prediction: &[f32; 6], target: &[f32; 54]) -> [f32; 6] {
    let (quantized, mask) = endpoints(prediction, target);
    let weights = assignments(&quantized, target);
    let mut result = std::array::from_fn(|i| 2.0 * ANCHOR * (prediction[i] - target[48 + i]));
    for (texel, &beta) in weights.iter().enumerate() {
        let delta: [f32; 3] = std::array::from_fn(|c| {
            quantized[c] * (1.0 - beta) + quantized[c + 3] * beta - target[3 * texel + c]
        });
        let scale =
            1.0 / (16.0 * (delta.iter().map(|d| d * d).sum::<f32>() + EPSILON * EPSILON).sqrt());
        for c in 0..3 {
            result[c] += (1.0 - beta) * delta[c] * scale * mask[c];
            result[c + 3] += beta * delta[c] * scale * mask[c + 3];
        }
    }
    result
}

/// Checks custom WGSL loss values and gradients against the CPU reference.
pub fn verify_gpu(
    device: &mev::Device,
    queue: &mut mev::Queue,
) -> Result<(), Box<dyn std::error::Error>> {
    use mev::{Arguments, BufferUsage, PipelineStages};

    #[derive(Arguments)]
    struct Bindings {
        #[mev(storage, compute)]
        input: mev::Buffer,
        #[mev(storage, compute)]
        output: mev::Buffer,
    }

    let colors: [f32; 48] =
        std::array::from_fn(|i| 0.1 + (i / 3) as f32 * 0.047 + (i % 3) as f32 * 0.021);
    let teacher = [0.03, -0.02, 0.01, -0.01, 0.02, -0.03];
    let mut black_white = [0.0; 48];
    black_white[24..].fill(1.0);
    let fixtures = [
        (
            [0.04, -0.03, 0.02, -0.06, 0.01, -0.04],
            target(&colors, &teacher),
        ),
        ([-0.4, 0.05, 0.0, 0.5, 0.0, -0.1], target(&colors, &teacher)),
        ([-1.0; 6], target(&[1.0; 48], &[0.0; 6])),
        ([0.0; 6], target(&[1.0; 48], &[0.0; 6])),
        ([0.0; 6], target(&black_white, &[0.0; 6])),
    ];
    let mut packed = Vec::with_capacity(fixtures.len() * 60);
    for (prediction, target) in &fixtures {
        packed.extend_from_slice(prediction);
        packed.extend_from_slice(target);
    }
    let shader = format!(
        "alias Output = array<f32, 6>;\nalias Target = array<f32, 54>;\n{}\n{}",
        include_str!("reconstruction_loss.wgsl"),
        r#"
@group(0) @binding(0) var<storage, read> diagnostic_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> diagnostic_output: array<f32>;
@compute @workgroup_size(1)
fn check_loss(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let sample = invocation.x;
    var prediction: Output;
    var target_value: Target;
    for (var i = 0u; i < 6u; i++) { prediction[i] = diagnostic_input[sample * 60u + i]; }
    for (var i = 0u; i < 54u; i++) { target_value[i] = diagnostic_input[sample * 60u + 6u + i]; }
    let gradient = nn_loss_derivative(prediction, target_value);
    for (var i = 0u; i < 6u; i++) { diagnostic_output[sample * 7u + i] = gradient[i]; }
    diagnostic_output[sample * 7u + 6u] = nn_loss(prediction, target_value);
}
"#
    );
    let library = device.new_shader_library(mev::LibraryDesc {
        name: "BC1 reconstruction loss verification",
        input: mev::LibraryInput::wgsl(shader),
    })?;
    let pipeline = device.new_compute_pipeline(mev::ComputePipelineDesc {
        name: "BC1 reconstruction loss verification",
        shader: library.entry("check_loss"),
        work_group_size: [1, 1, 1],
        constants: 0,
        arguments: &[Bindings::LAYOUT],
    })?;
    let make = |name, size, usage| device.new_buffer(mev::BufferDesc { name, size, usage });
    let mut upload = make(
        "BC1 reconstruction verification upload",
        packed.len() * 4,
        BufferUsage::HOST_WRITE | BufferUsage::TRANSFER_SRC,
    );
    upload.write(0, bytemuck::cast_slice(&packed))?;
    let output_size = fixtures.len() * 7 * 4;
    let bindings = Bindings {
        input: make(
            "BC1 reconstruction verification input",
            packed.len() * 4,
            BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
        ),
        output: make(
            "BC1 reconstruction verification output",
            output_size,
            BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
        ),
    };
    let mut readback = make(
        "BC1 reconstruction verification readback",
        output_size,
        BufferUsage::HOST_READ | BufferUsage::TRANSFER_DST,
    );
    let mut encoder = queue.new_command_encoder();
    encoder
        .copy()
        .copy_buffer_to_buffer(&upload, &bindings.input);
    encoder.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
    {
        let mut compute = encoder.compute();
        compute.with_pipeline(&pipeline);
        compute.with_arguments(0, &bindings);
        compute.dispatch(mev::Extent3::new(fixtures.len() as u32, 1, 1));
    }
    encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
    encoder
        .copy()
        .copy_buffer_to_buffer(&bindings.output, &readback);
    queue.submit_checkpoint([encoder.finish()])?;
    queue.wait_idle()?;
    let mut actual = vec![0.0_f32; fixtures.len() * 7];
    readback.read(0, bytemuck::cast_slice_mut(&mut actual))?;
    for (sample, (prediction, target)) in fixtures.iter().enumerate() {
        let derivative = gradient(prediction, target);
        let mut expected = [0.0; 7];
        expected[..6].copy_from_slice(&derivative);
        expected[6] = loss_value(prediction, target);
        for (component, &expected) in expected.iter().enumerate() {
            let actual = actual[sample * 7 + component];
            let tolerance = 2e-5 * (1.0 + expected.abs());
            if !actual.is_finite() || (actual - expected).abs() > tolerance {
                return Err(format!("BC1 reconstruction GPU/CPU mismatch sample {sample} component {component}: GPU {actual}, CPU {expected}, tolerance {tolerance}").into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> [f32; 54] {
        target(
            &std::array::from_fn::<_, 48, _>(|i| {
                let texel = i / 3;
                0.1 + texel as f32 * 0.047 + (i % 3) as f32 * 0.021
            }),
            &[0.03, -0.02, 0.01, -0.01, 0.02, -0.03],
        )
    }

    // Hold palette assignment and rounding offset fixed while perturbing endpoints.
    fn frozen_value(prediction: &[f64; 6], origin: &[f32; 6], target: &[f32; 54]) -> f64 {
        let (quantized, mask) = endpoints(origin, target);
        let weights = assignments(&quantized, target);
        let endpoint: [f64; 6] = std::array::from_fn(|i| {
            quantized[i] as f64 + (prediction[i] - origin[i] as f64) * mask[i] as f64
        });
        let mut value = 0.0;
        for (texel, &beta) in weights.iter().enumerate() {
            let beta = beta as f64;
            let squared = (0..3)
                .map(|c| {
                    let d = endpoint[c] * (1.0 - beta) + endpoint[c + 3] * beta
                        - target[texel * 3 + c] as f64;
                    d * d
                })
                .sum::<f64>();
            value += ((squared + (EPSILON as f64).powi(2)).sqrt() - EPSILON as f64) / 16.0;
        }
        for i in 0..6 {
            value += ANCHOR as f64 * (prediction[i] - target[48 + i] as f64).powi(2);
        }
        value
    }

    #[test]
    fn frozen_ste_gradient_matches_finite_differences() {
        let target = fixture();
        for prediction in [
            [0.04, -0.03, 0.02, -0.06, 0.01, -0.04],
            [-0.4, 0.05, 0.0, 0.5, 0.0, -0.1],
        ] {
            let analytic = gradient(&prediction, &target);
            for i in 0..6 {
                let mut plus = prediction.map(f64::from);
                let mut minus = plus;
                plus[i] += 1e-6;
                minus[i] -= 1e-6;
                let numeric = (frozen_value(&plus, &prediction, &target)
                    - frozen_value(&minus, &prediction, &target))
                    / 2e-6;
                assert!(
                    (analytic[i] as f64 - numeric).abs() < 1e-4,
                    "component {i}: analytic={} numeric={numeric}",
                    analytic[i]
                );
            }
        }
    }

    #[test]
    fn equal_endpoints_use_opaque_endpoint_zero() {
        let target = target(&[1.0; 48], &[0.0; 6]);
        let prediction = [-1.0; 6];
        assert!(
            (loss_value(&prediction, &target) - (3.0_f32.sqrt() - EPSILON + 0.03)).abs() < 1e-5
        );
        let grad = gradient(&prediction, &target);
        for c in 0..3 {
            assert!(grad[c] < -0.5);
            assert_eq!(grad[c + 3], -0.01);
        }
        let white = [0.0; 6];
        assert_eq!(loss_value(&white, &target), 0.0);
        assert_eq!(gradient(&white, &target), [0.0; 6]);
    }

    #[test]
    fn black_white_palette_and_clamped_outputs_are_finite() {
        let mut input = [0.0; 48];
        input[24..].fill(1.0);
        let target = target(&input, &[0.0; 6]);
        assert_eq!(loss_value(&[0.0; 6], &target), 0.0);
        for prediction in [[0.0; 6], [3.0, -3.0, 2.0, -2.0, 4.0, -4.0]] {
            assert!(loss_value(&prediction, &target).is_finite());
            assert!(gradient(&prediction, &target).iter().all(|v| v.is_finite()));
        }
    }
}
