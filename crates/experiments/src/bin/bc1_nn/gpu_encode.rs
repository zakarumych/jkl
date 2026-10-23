//! Resident RGB8 preprocessing, endpoint prediction, and BC1 packing.
use jkl::image::block::bc1::Block;
use mev::kernels::nn::{Batch, Nn};
use mev::{Arguments, Buffer, BufferDesc, BufferUsage, Device, DeviceRepr, PipelineStages, Queue};
use std::{error::Error, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Texels = [[u8; 3]; 16];

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, mev::AutoDeviceRepr)]
struct Params {
    count: u32,
    mode: u32,
    gd_steps: u32,
    gd_rate: f32,
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

pub struct GpuEncoder {
    pub input: Buffer,
    pub output_residual: Buffer,
    preprocess_pipeline: mev::ComputePipeline,
    encode_pipeline: mev::ComputePipeline,
    relaxed_check_pipeline: mev::ComputePipeline,
    bindings: Bindings,
    upload: Buffer,
    readback: Buffer,
    packed: Vec<u8>,
    bytes: Vec<u8>,
    count: u32,
    uploaded: bool,
}
#[derive(Clone)]
pub struct EncodeResult {
    pub blocks: Vec<Block>,
    pub resident_ms: f64,
    pub host_ms: f64,
}

fn pack_rgb(blocks: &[Texels], packed: &mut [u8]) {
    for (block, texels) in blocks.iter().enumerate() {
        for (pixel, &[r, g, b]) in texels.iter().enumerate() {
            let offset = (block * 16 + pixel) * 4;
            packed[offset..offset + 4].copy_from_slice(&[r, g, b, 255]);
        }
    }
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}
impl GpuEncoder {
    pub fn output_buffer(&self) -> &Buffer {
        &self.bindings.output
    }
    pub fn encode_one_neighbor_round(
        &self,
        queue: &mut Queue,
        nn: Option<&Nn>,
        principal: bool,
    ) -> Result<()> {
        self.encode_config(queue, nn, if principal { 12 } else { 9 }, 1, 0.0)
    }
    pub fn new(device: &Device, blocks: &[Texels]) -> Result<Self> {
        if blocks.is_empty() {
            return Err("GPU BC1 encoder needs nonempty blocks".into());
        }
        let count = u32::try_from(blocks.len())?;
        if count.div_ceil(64) > 65535 {
            return Err("GPU BC1 encoder batch exceeds 65535 workgroups".into());
        }
        let make = |name, size, usage| device.new_buffer(BufferDesc { name, size, usage });
        let input = make(
            "BC1 GPU canonical input",
            blocks.len() * 48 * 4,
            BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
        );
        let output_residual = make(
            "BC1 GPU residual",
            blocks.len() * 6 * 4,
            BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
        );
        let bindings = Bindings {
            rgb: make(
                "BC1 GPU RGB8",
                blocks.len() * 64,
                BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            ),
            input: input.clone(),
            residual: output_residual.clone(),
            output: make(
                "BC1 GPU packed output",
                blocks.len() * 8,
                BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
            ),
        };
        let library = device.new_shader_library(mev::LibraryDesc {
            name: "BC1 GPU encoder",
            input: mev::LibraryInput::wgsl(include_str!("gpu_encode.wgsl")),
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
        pack_rgb(blocks, &mut packed);
        Ok(Self {
            input,
            output_residual,
            preprocess_pipeline: pipeline("preprocess")?,
            encode_pipeline: pipeline("encode")?,
            relaxed_check_pipeline: pipeline("check_relaxed")?,
            bindings,
            upload: make(
                "BC1 GPU upload",
                packed.len(),
                BufferUsage::HOST_WRITE | BufferUsage::TRANSFER_SRC,
            ),
            readback: make(
                "BC1 GPU readback",
                blocks.len() * 8,
                BufferUsage::HOST_READ | BufferUsage::TRANSFER_DST,
            ),
            packed,
            bytes: vec![0; blocks.len() * 8],
            count,
            uploaded: false,
        })
    }
    pub fn upload(&mut self, queue: &mut Queue, blocks: &[Texels]) -> Result<()> {
        if blocks.len() != self.count as usize {
            return Err("GPU encoder block count mismatch".into());
        }
        pack_rgb(blocks, &mut self.packed);
        self.upload.write(0, &self.packed)?;
        let mut encoder = queue.new_command_encoder();
        encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        encoder
            .copy()
            .copy_buffer_to_buffer(&self.upload, &self.bindings.rgb);
        encoder.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
        queue.submit_checkpoint([encoder.finish()])?;
        self.uploaded = true;
        Ok(())
    }
    fn dispatch(
        &self,
        queue: &mut Queue,
        pipeline: &mev::ComputePipeline,
        mode: u32,
        gd_steps: u32,
        gd_rate: f32,
    ) -> Result<()> {
        let mut encoder = queue.new_command_encoder();
        encoder.barrier(
            PipelineStages::TRANSFER | PipelineStages::COMPUTE_SHADER,
            PipelineStages::COMPUTE_SHADER,
        );
        {
            let mut compute = encoder.compute();
            compute.with_pipeline(pipeline);
            compute.with_arguments(0, &self.bindings);
            compute.with_constants(&Params {
                count: self.count,
                mode,
                gd_steps,
                gd_rate,
            });
            compute.dispatch(mev::Extent3::new(self.count.div_ceil(64), 1, 1));
        }
        queue.submit_checkpoint([encoder.finish()])?;
        Ok(())
    }
    pub fn preprocess(&self, queue: &mut Queue) -> Result<()> {
        if !self.uploaded {
            return Err("Upload RGB blocks before preprocessing".into());
        }
        self.dispatch(queue, &self.preprocess_pipeline, 0, 0, 0.0)
    }
    pub fn encode(&self, queue: &mut Queue, nn: Option<&Nn>, mode: u32) -> Result<()> {
        if mode > 3 {
            return Err("GPU BC1 default mode must be 0..3".into());
        }
        self.encode_config(queue, nn, mode, 0, 0.0)
    }
    fn encode_config(
        &self,
        queue: &mut Queue,
        nn: Option<&Nn>,
        mode: u32,
        gd_steps: u32,
        gd_rate: f32,
    ) -> Result<()> {
        if mode > 13 {
            return Err("GPU BC1 private mode must be 0..13".into());
        }
        if (mode < 3 || mode == 4 || (7..=10).contains(&mode)) && nn.is_none() {
            return Err("NN required for prediction modes".into());
        }
        if (11..=13).contains(&mode) {
            if !self.uploaded {
                return Err("Upload RGB blocks before preprocessing".into());
            }
            self.dispatch(queue, &self.preprocess_pipeline, mode, 0, 0.0)?;
        } else {
            self.preprocess(queue)?;
        }
        if mode < 3 || mode == 4 || (7..=10).contains(&mode) {
            nn.unwrap().inference(
                queue,
                self.input.slice(..),
                self.output_residual.slice(..),
                &[],
                Batch::new(self.count),
            )?;
        }
        self.dispatch(queue, &self.encode_pipeline, mode, gd_steps, gd_rate)
    }
    fn copy_output(&self, queue: &mut Queue) -> Result<()> {
        let mut encoder = queue.new_command_encoder();
        encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        encoder
            .copy()
            .copy_buffer_to_buffer(&self.bindings.output, &self.readback);
        queue.submit_checkpoint([encoder.finish()])?;
        Ok(())
    }
    pub fn benchmark(
        &mut self,
        queue: &mut Queue,
        nn: Option<&Nn>,
        blocks: &[Texels],
        repeats: usize,
        mode: u32,
    ) -> Result<EncodeResult> {
        self.benchmark_config(queue, nn, blocks, repeats, mode, 0, 0.0)
    }
    pub fn benchmark_principal_ls(
        &mut self,
        queue: &mut Queue,
        blocks: &[Texels],
        repeats: usize,
        passes: u32,
    ) -> Result<EncodeResult> {
        if passes == 0 || passes > 64 {
            return Err("principal-axis LS requires passes 1..64".into());
        }
        self.benchmark_config(queue, None, blocks, repeats, 11, passes, 0.0)
    }
    pub fn benchmark_gd(
        &mut self,
        queue: &mut Queue,
        nn: &Nn,
        blocks: &[Texels],
        repeats: usize,
        steps: u32,
        rate: f32,
    ) -> Result<EncodeResult> {
        if steps == 0 || steps > 64 || !rate.is_finite() || rate <= 0.0 {
            return Err("GD requires steps 1..64 and a finite positive rate".into());
        }
        self.benchmark_config(queue, Some(nn), blocks, repeats, 4, steps, rate)
    }
    pub fn benchmark_ls_gd(
        &mut self,
        queue: &mut Queue,
        nn: &Nn,
        blocks: &[Texels],
        repeats: usize,
        steps: u32,
        rate: f32,
        fixed_indices: bool,
    ) -> Result<EncodeResult> {
        if steps == 0 || steps > 64 || !rate.is_finite() || rate <= 0.0 {
            return Err("GD requires steps 1..64 and a finite positive rate".into());
        }
        self.benchmark_config(
            queue,
            Some(nn),
            blocks,
            repeats,
            if fixed_indices { 8 } else { 7 },
            steps,
            rate,
        )
    }
    pub fn benchmark_principal_neighbors(
        &mut self,
        queue: &mut Queue,
        blocks: &[Texels],
        repeats: usize,
        rounds: u32,
        reuse_indices: bool,
    ) -> Result<EncodeResult> {
        if rounds == 0 || rounds > 2 {
            return Err("principal neighbor comparison requires one or two rounds".into());
        }
        self.benchmark_config(
            queue,
            None,
            blocks,
            repeats,
            if reuse_indices { 13 } else { 12 },
            rounds,
            0.0,
        )
    }
    pub fn benchmark_neighbors(
        &mut self,
        queue: &mut Queue,
        nn: &Nn,
        blocks: &[Texels],
        repeats: usize,
        rounds: u32,
        reuse_indices: bool,
    ) -> Result<EncodeResult> {
        if rounds == 0 || rounds > 2 {
            return Err("neighbor comparison requires one or two rounds".into());
        }
        self.benchmark_config(
            queue,
            Some(nn),
            blocks,
            repeats,
            if reuse_indices { 10 } else { 9 },
            rounds,
            0.0,
        )
    }
    pub fn benchmark_range_gd(
        &mut self,
        queue: &mut Queue,
        blocks: &[Texels],
        repeats: usize,
        steps: u32,
        rate: f32,
    ) -> Result<EncodeResult> {
        if steps == 0 || steps > 64 || !rate.is_finite() || rate <= 0.0 {
            return Err("GD requires steps 1..64 and a finite positive rate".into());
        }
        self.benchmark_config(queue, None, blocks, repeats, 5, steps, rate)
    }
    pub fn benchmark_range_start(
        &mut self,
        queue: &mut Queue,
        blocks: &[Texels],
        repeats: usize,
    ) -> Result<EncodeResult> {
        self.benchmark_config(queue, None, blocks, repeats, 6, 0, 0.0)
    }
    pub fn check_relaxed(&self, queue: &mut Queue) -> Result<()> {
        self.dispatch(queue, &self.relaxed_check_pipeline, 0, 0, 0.0)
    }
    pub fn check_relaxed_fixed(&self, queue: &mut Queue) -> Result<()> {
        self.dispatch(queue, &self.relaxed_check_pipeline, 8, 0, 0.0)
    }
    fn benchmark_config(
        &mut self,
        queue: &mut Queue,
        nn: Option<&Nn>,
        blocks: &[Texels],
        repeats: usize,
        mode: u32,
        gd_steps: u32,
        gd_rate: f32,
    ) -> Result<EncodeResult> {
        if repeats == 0 {
            return Err("GPU BC1 benchmark needs at least one repeat".into());
        }
        queue.wait_idle()?;
        self.upload(queue, blocks)?;
        if mode <= 3 {
            self.encode(queue, nn, mode)?;
        } else {
            self.encode_config(queue, nn, mode, gd_steps, gd_rate)?;
        }
        queue.wait_idle()?;
        let mut resident = Vec::with_capacity(repeats);
        let mut host = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            self.encode_config(queue, nn, mode, gd_steps, gd_rate)?;
            queue.wait_idle()?;
            resident.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        for _ in 0..repeats {
            let start = Instant::now();
            self.upload(queue, blocks)?;
            self.encode_config(queue, nn, mode, gd_steps, gd_rate)?;
            self.copy_output(queue)?;
            queue.wait_idle()?;
            self.readback.read(0, &mut self.bytes)?;
            host.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(EncodeResult {
            blocks: self
                .bytes
                .chunks_exact(8)
                .map(|b| Block::from_bytes(b.try_into().unwrap()))
                .collect(),
            resident_ms: median(&mut resident),
            host_ms: median(&mut host),
        })
    }
}
