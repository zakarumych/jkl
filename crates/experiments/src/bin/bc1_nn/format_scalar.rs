//! Scalar BC4/BC5 experiment kernels. Candidate neighbors execute sequentially per channel.
use super::{Result, Texels};
use jkl::{
    image::block::bc4::Block,
    math::{R8U, R32F},
};
use mev::kernels::nn::{Batch, Nn};
use mev::{Arguments, Buffer, BufferDesc, BufferUsage, Device, DeviceRepr, PipelineStages, Queue};
use std::time::Instant;
#[derive(Clone, Copy, Debug)]
pub enum Method {
    CurrentBuggy,
    CurrentCorrected,
    Neural,
    Extrema,
    Teacher,
}
impl Method {
    fn id(self) -> u32 {
        match self {
            Self::CurrentBuggy => 0,
            Self::CurrentCorrected => 1,
            Self::Neural => 2,
            Self::Extrema => 3,
            Self::Teacher => 4,
        }
    }
}
pub struct BenchResult {
    pub blocks: Vec<[u8; 16]>,
    pub resident_ms: f64,
    pub host_ms: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, mev::AutoDeviceRepr)]
struct Params {
    count: u32,
    channels: u32,
    method: u32,
    pad: u32,
}
#[derive(Arguments)]
struct Bindings {
    #[mev(storage, compute)]
    rgb: Buffer,
    #[mev(storage, compute)]
    input: Buffer,
    #[mev(storage, compute)]
    residual: Buffer,
    #[mev(storage, compute)]
    output: Buffer,
}
pub struct ScalarEncoder {
    bindings: Bindings,
    preprocess: mev::ComputePipeline,
    encode: mev::ComputePipeline,
    upload: Buffer,
    readback: Buffer,
    packed: Vec<u8>,
    bytes: Vec<u8>,
    count: u32,
    channels: u32,
}
fn pack(blocks: &[Texels], out: &mut [u8]) {
    for (b, t) in blocks.iter().enumerate() {
        for (p, c) in t.iter().enumerate() {
            out[(b * 16 + p) * 4..(b * 16 + p) * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
impl ScalarEncoder {
    pub fn new(device: &Device, blocks: &[Texels], channels: u32) -> Result<Self> {
        if blocks.is_empty() || !(1..=2).contains(&channels) {
            return Err("scalar encoder requires blocks and one or two channels".into());
        }
        let count = u32::try_from(blocks.len())?;
        if (count * channels).div_ceil(64) > 65535 {
            return Err("scalar batch exceeds dispatch limit".into());
        }
        let make = |name, size, usage| device.new_buffer(BufferDesc { name, size, usage });
        let bindings = Bindings {
            rgb: make(
                "scalar RGB",
                blocks.len() * 64,
                BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            ),
            input: make(
                "scalar sorted",
                blocks.len() * channels as usize * 16 * 4,
                BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
            ),
            residual: make(
                "scalar residual",
                blocks.len() * channels as usize * 2 * 4,
                BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
            ),
            output: make(
                "scalar blocks",
                blocks.len() * channels as usize * 8,
                BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
            ),
        };
        let library = device.new_shader_library(mev::LibraryDesc {
            name: "scalar formats",
            input: mev::LibraryInput::wgsl(include_str!("format_scalar.wgsl")),
        })?;
        let pipeline = |entry| {
            device.new_compute_pipeline(mev::ComputePipelineDesc {
                name: entry,
                shader: library.entry(entry),
                work_group_size: [64, 1, 1],
                constants: Params::SIZE,
                arguments: &[Bindings::LAYOUT],
            })
        };
        let mut packed = vec![0; blocks.len() * 64];
        pack(blocks, &mut packed);
        Ok(Self {
            preprocess: pipeline("preprocess")?,
            encode: pipeline("encode")?,
            bindings,
            upload: make(
                "scalar upload",
                packed.len(),
                BufferUsage::HOST_WRITE | BufferUsage::TRANSFER_SRC,
            ),
            readback: make(
                "scalar readback",
                blocks.len() * channels as usize * 8,
                BufferUsage::HOST_READ | BufferUsage::TRANSFER_DST,
            ),
            bytes: vec![0; blocks.len() * channels as usize * 8],
            packed,
            count,
            channels,
        })
    }
    pub fn residual_buffer(&self) -> &Buffer {
        &self.bindings.residual
    }
    pub fn canonical_input(&self) -> &Buffer {
        &self.bindings.input
    }
    pub fn upload(&mut self, q: &mut Queue, blocks: &[Texels]) -> Result<()> {
        if blocks.len() != self.count as usize {
            return Err("scalar block count mismatch".into());
        }
        pack(blocks, &mut self.packed);
        self.upload.write(0, &self.packed)?;
        let mut e = q.new_command_encoder();
        e.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        e.copy()
            .copy_buffer_to_buffer(&self.upload, &self.bindings.rgb);
        e.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
        q.submit_checkpoint([e.finish()])?;
        Ok(())
    }
    fn dispatch(&self, q: &mut Queue, p: &mev::ComputePipeline, method: Method) -> Result<()> {
        let mut e = q.new_command_encoder();
        e.barrier(
            PipelineStages::TRANSFER | PipelineStages::COMPUTE_SHADER,
            PipelineStages::COMPUTE_SHADER,
        );
        {
            let mut c = e.compute();
            c.with_pipeline(p);
            c.with_arguments(0, &self.bindings);
            c.with_constants(&Params {
                count: self.count,
                channels: self.channels,
                method: method.id(),
                pad: 0,
            });
            c.dispatch(mev::Extent3::new(
                (self.count * self.channels).div_ceil(64),
                1,
                1,
            ));
        }
        q.submit_checkpoint([e.finish()])?;
        Ok(())
    }
    pub fn preprocess(&self, q: &mut Queue) -> Result<()> {
        self.dispatch(q, &self.preprocess, Method::Extrema)
    }
    fn run(&self, q: &mut Queue, nn: Option<&Nn>, method: Method) -> Result<()> {
        self.preprocess(q)?;
        if matches!(method, Method::Neural) {
            nn.ok_or("scalar neural mode requires model")?.inference(
                q,
                self.bindings.input.slice(..),
                self.bindings.residual.slice(..),
                &[],
                Batch::new(self.count * self.channels),
            )?;
        }
        self.dispatch(q, &self.encode, method)
    }
    fn read(&mut self, q: &mut Queue) -> Result<()> {
        let mut e = q.new_command_encoder();
        e.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        e.copy()
            .copy_buffer_to_buffer(&self.bindings.output, &self.readback);
        q.submit_checkpoint([e.finish()])?;
        q.wait_idle()?;
        self.readback.read(0, &mut self.bytes)?;
        Ok(())
    }
    pub fn benchmark(
        &mut self,
        q: &mut Queue,
        nn: Option<&Nn>,
        blocks: &[Texels],
        repeats: usize,
        method: Method,
    ) -> Result<BenchResult> {
        if repeats == 0 {
            return Err("scalar repeats must be positive".into());
        }
        self.upload(q, blocks)?;
        self.run(q, nn, method)?;
        q.wait_idle()?;
        let mut resident = vec![];
        let mut host = vec![];
        for _ in 0..repeats {
            let t = Instant::now();
            self.run(q, nn, method)?;
            q.wait_idle()?;
            resident.push(t.elapsed().as_secs_f64() * 1000.);
        }
        for _ in 0..repeats {
            let t = Instant::now();
            self.upload(q, blocks)?;
            self.run(q, nn, method)?;
            self.read(q)?;
            host.push(t.elapsed().as_secs_f64() * 1000.);
        }
        let mut output = vec![[0; 16]; self.count as usize];
        let stride = self.channels as usize * 8;
        for (b, s) in output.iter_mut().zip(self.bytes.chunks_exact(stride)) {
            b[..stride].copy_from_slice(s);
        }
        Ok(BenchResult {
            blocks: output,
            resident_ms: median(&mut resident),
            host_ms: median(&mut host),
        })
    }
    pub fn teacher(&mut self, q: &mut Queue, blocks: &[Texels]) -> Result<Vec<Block>> {
        self.upload(q, blocks)?;
        self.run(q, None, Method::Teacher)?;
        self.read(q)?;
        Ok(self
            .bytes
            .chunks_exact(8)
            .map(|b| Block::from_bytes(b.try_into().unwrap()))
            .collect())
    }
}
/// CPU implementation with optional correction limited to valid-index packing.
pub fn current_cpu(texels: &Texels, channel: usize, corrected: bool) -> Block {
    let colors = std::array::from_fn(|y| {
        std::array::from_fn(|x| (R32F::new(texels[y * 4 + x][channel] as f32 / 255.), true))
    });
    if !corrected {
        return Block::encode(colors);
    }
    let samples = texels.map(|t| t[channel] as f32 / 255.);
    let mut cf = jkl::algos::cluster_fit::cluster_fit::<f32, 8, 16>(
        &samples,
        |v| R8U::from_f32(R32F::new(v)).into_f32().0,
        |a, b| (a - b).abs(),
    );
    let mut a = R8U::from_f32(R32F::new(cf.endpoints.0));
    let mut b = R8U::from_f32(R32F::new(cf.endpoints.1));
    if a == b {
        return Block {
            color0: a,
            color1: R8U::BLACK,
            indices: [0; 6],
        };
    }
    if a.bits() < b.bits() {
        std::mem::swap(&mut a, &mut b);
        for i in &mut cf.indices {
            *i = 7 - *i;
        }
    }
    let map = [0u64, 2, 3, 4, 5, 6, 7, 1];
    let mut bits = 0u64;
    for (i, idx) in cf.indices.into_iter().enumerate() {
        bits |= map[idx] << (i * 3);
    }
    let raw = bits.to_le_bytes();
    Block {
        color0: a,
        color1: b,
        indices: raw[..6].try_into().unwrap(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corrected_flat() {
        for c in [0, 1, 127, 254, 255] {
            let t = [[c, 0, 0]; 16];
            let b = current_cpu(&t, 0, true);
            assert_eq!(b.bytes()[0], c);
            assert_eq!(b.indices, [0; 6]);
        }
    }
    #[test]
    fn corrected_valid_indices() {
        let t = std::array::from_fn(|i| [(i * 17) as u8, 0, 0]);
        let buggy = current_cpu(&t, 0, false);
        let fixed = current_cpu(&t, 0, true);
        assert_eq!(buggy.indices, [0; 6]);
        assert_ne!(fixed.indices, [0; 6]);
        assert_eq!(buggy.color0, fixed.color0);
        assert_eq!(buggy.color1, fixed.color1);
        let error = |b: Block| {
            b.decode()
                .into_iter()
                .flatten()
                .zip(t)
                .map(|(a, b)| (a.0 - b[0] as f32 / 255.).abs())
                .sum::<f32>()
        };
        assert!(error(fixed) < error(buggy));
    }
    #[test]
    fn full_index_word_roundtrip() {
        let mut bits = 0u64;
        for i in 0..16 {
            bits |= ((i % 8) as u64) << (i * 3);
        }
        let raw = bits.to_le_bytes();
        let b = Block::from_bytes([255, 0, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]]);
        let decoded = b.decode();
        for i in 0..16 {
            let k = i % 8;
            let expected = if k == 0 {
                1.
            } else if k == 1 {
                0.
            } else {
                (8 - k) as f32 / 7.
            };
            assert!((decoded[i / 4][i % 4].0 - expected).abs() < 1e-6);
        }
    }
}

/// Compare GPU baseline bytes and decoded absolute-error sums against actual CPU semantics.
pub fn current_parity(
    blocks: &[Texels],
    channels: u32,
    encoded: &[[u8; 16]],
    corrected: bool,
) -> serde_json::Value {
    assert_eq!(blocks.len(), encoded.len());
    let mut mismatches = 0usize;
    let mut max_endpoint_delta = 0u8;
    let mut max_error_delta = 0f64;
    let mut cpu_total = 0f64;
    let mut gpu_total = 0f64;
    for (texels, bytes) in blocks.iter().zip(encoded) {
        for channel in 0..channels as usize {
            let cpu = current_cpu(texels, channel, corrected);
            let gpu = Block::from_bytes(bytes[channel * 8..channel * 8 + 8].try_into().unwrap());
            mismatches += usize::from(cpu.bytes() != gpu.bytes());
            for i in 0..2 {
                max_endpoint_delta =
                    max_endpoint_delta.max(cpu.bytes()[i].abs_diff(gpu.bytes()[i]));
            }
            let err = |b: Block| {
                b.decode()
                    .into_iter()
                    .flatten()
                    .zip(texels)
                    .map(|(a, t)| (a.0 as f64 - t[channel] as f64 / 255.).abs())
                    .sum::<f64>()
            };
            let ce = err(cpu);
            let ge = err(gpu);
            cpu_total += ce;
            gpu_total += ge;
            max_error_delta = max_error_delta.max((ce - ge).abs());
        }
    }
    serde_json::json!({"scalar_blocks":blocks.len()*channels as usize,"byte_mismatches":mismatches,"max_endpoint_code_delta":max_endpoint_delta,"max_block_absolute_error_sum_delta":max_error_delta,"cpu_absolute_error_sum":cpu_total,"gpu_absolute_error_sum":gpu_total})
}

/// Runtime packing and CPU-port checks on corpus blocks and adversarial scalar fixtures.
pub fn verify_gpu(
    device: &Device,
    queue: &mut Queue,
    sourceblocks: &[Texels],
    _nn: Option<&Nn>,
) -> Result<serde_json::Value> {
    let mut blocks: Vec<Texels> = sourceblocks.iter().take(65).copied().collect();
    for c in [0, 1, 2, 17, 63, 127, 128, 191, 254, 255] {
        blocks.push([[c, c, c]; 16]);
    }
    for (a, b) in [(0, 255), (0, 1), (1, 254), (127, 128), (17, 191), (63, 255)] {
        for pattern in 0..3 {
            blocks.push(std::array::from_fn(|i| {
                let c = if i % 3 == pattern { a } else { b };
                [c, 255 - c, c]
            }));
        }
    }
    blocks.push(std::array::from_fn(|i| {
        [(i * 17) as u8, (255 - i * 17) as u8, 0]
    }));
    let mut encoder = ScalarEncoder::new(device, &blocks, 2)?;
    let buggy = encoder.benchmark(queue, None, &blocks, 1, Method::CurrentBuggy)?;
    let corrected = encoder.benchmark(queue, None, &blocks, 1, Method::CurrentCorrected)?;
    let extrema = encoder.benchmark(queue, None, &blocks, 1, Method::Extrema)?;
    let teacher = encoder.teacher(queue, &blocks)?;
    let source_error = |block: Block, t: &Texels, ch: usize| {
        block
            .decode()
            .into_iter()
            .flatten()
            .zip(t)
            .map(|(p, t)| (p.0 as f64 - t[ch] as f64 / 255.).abs())
            .sum::<f64>()
    };
    let mut max_decoded_difference = 0f32;
    let mut extrema_violations = 0usize;
    let mut teacher_cpu_violations = 0usize;
    let mut max_teacher_excess = 0f64;
    for (i, t) in blocks.iter().enumerate() {
        for ch in 0..2 {
            let cp = current_cpu(t, ch, true);
            let gp = Block::from_bytes(corrected.blocks[i][ch * 8..ch * 8 + 8].try_into().unwrap());
            for (a, b) in cp
                .decode()
                .into_iter()
                .flatten()
                .zip(gp.decode().into_iter().flatten())
            {
                max_decoded_difference = max_decoded_difference.max((a.0 - b.0).abs());
            }
            let lo = t.iter().map(|t| t[ch]).min().unwrap();
            let hi = t.iter().map(|t| t[ch]).max().unwrap();
            let nearest = |a: u8, b: u8| {
                let palette = Block::from_bytes([a, b, 0, 0, 0, 0, 0, 0]);
                let mut raw = 0u64;
                for (p, t) in t.iter().enumerate() {
                    let mut k = 0usize;
                    let mut e = f64::INFINITY;
                    for j in 0..8 {
                        let bits = (j as u64) * 0x249249249249u64;
                        let bytes = bits.to_le_bytes();
                        let candidate = Block::from_bytes([
                            palette.color0.bits(),
                            palette.color1.bits(),
                            bytes[0],
                            bytes[1],
                            bytes[2],
                            bytes[3],
                            bytes[4],
                            bytes[5],
                        ]);
                        let v = candidate.decode()[0][0].0 as f64;
                        let d = (v - t[ch] as f64 / 255.).abs();
                        if d < e {
                            e = d;
                            k = j;
                        }
                    }
                    raw |= (k as u64) << (p * 3);
                }
                let bytes = raw.to_le_bytes();
                Block::from_bytes([
                    a, b, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5],
                ])
            };
            let baseline =
                source_error(nearest(lo, hi), t, ch).min(source_error(nearest(hi, lo), t, ch));
            let ep = Block::from_bytes(extrema.blocks[i][ch * 8..ch * 8 + 8].try_into().unwrap());
            if source_error(ep, t, ch) > baseline + 2e-5 {
                extrema_violations += 1;
            }
            let excess = source_error(teacher[i * 2 + ch], t, ch) - source_error(cp, t, ch);
            max_teacher_excess = max_teacher_excess.max(excess);
            if excess > 2e-5 {
                teacher_cpu_violations += 1;
            }
        }
    }
    if extrema_violations > 0 {
        return Err(format!(
            "scalar extrema regressed beyond minmax guard on {extrema_violations} fixtures"
        )
        .into());
    }
    let bug = current_parity(&blocks, 2, &buggy.blocks, false);
    let fixed = current_parity(&blocks, 2, &corrected.blocks, true);
    let exact = bug["byte_mismatches"] == 0 && fixed["byte_mismatches"] == 0;
    Ok(
        serde_json::json!({"fixtures":blocks.len(),"current_buggy":bug,"current_corrected":fixed,"max_decoded_channel_difference":max_decoded_difference,"port_parity":if exact{"byte_exact_on_fixtures"}else{"numerical_port_with_reported_discrepancies"},"extrema_minmax_guard_violations":extrema_violations,"teacher_actual_cpu_corrected_excess_count":teacher_cpu_violations,"teacher_max_absolute_error_sum_excess":max_teacher_excess,"teacher_retention":"corrected GPU port candidate retained by actual decoded absolute error; CPU candidate separately checked"}),
    )
}
