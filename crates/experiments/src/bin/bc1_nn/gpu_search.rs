//! Timing of the archived BC1 WGSL cluster search on opaque RGB blocks.

use std::{error::Error, time::Instant};

use jkl::image::block::bc1::Block;
use mev::{Arguments, BufferUsage, DeviceRepr, PipelineStages};

const BINOM: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../archive/cluster-fit/shaders/binom.wgsl"
));
const REDUCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../archive/cluster-fit/shaders/reduce.wgsl"
));
const CLUSTER_FIT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../archive/cluster-fit/shaders/cluster_fit.wgsl"
));
const BC1: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../archive/cluster-fit/shaders/bc1.wgsl"
));

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, mev::AutoDeviceRepr)]
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

#[derive(Arguments)]
struct SearchArguments {
    #[mev(storage, compute)]
    input: mev::Buffer,
    #[mev(storage, compute)]
    output: mev::Buffer,
}

pub struct SearchResult {
    pub blocks: Vec<Block>,
    /// Median command recording, submission, and queue completion time.
    pub resident_ms: f64,
    /// Median RGB packing, upload, search, readback, and queue completion time.
    pub end_to_end_ms: f64,
    pub count: usize,
    /// Workgroups executed; one per requested block.
    pub dispatched_count: usize,
    pub label: &'static str,
}

fn pack_rgb(blocks: &[[[u8; 3]; 16]], columns: usize, packed: &mut [u8]) {
    for (block_index, block) in blocks.iter().enumerate() {
        let block_x = block_index % columns;
        let block_y = block_index / columns;
        for (pixel, &[r, g, b]) in block.iter().enumerate() {
            let x = block_x * 4 + pixel % 4;
            let y = block_y * 4 + pixel / 4;
            let offset = (y * columns * 4 + x) * 4;
            packed[offset..offset + 4].copy_from_slice(&[r, g, b, 255]);
        }
    }
}

fn search(
    queue: &mut mev::Queue,
    pipeline: &mev::ComputePipeline,
    arguments: &SearchArguments,
    params: Params,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    // One row per submission keeps each command buffer at at most 1024 blocks.
    for row in 0..params.in_height / 4 {
        let mut encoder = queue.new_command_encoder();
        encoder.barrier(
            PipelineStages::TRANSFER | PipelineStages::COMPUTE_SHADER,
            PipelineStages::COMPUTE_SHADER,
        );
        {
            let mut compute = encoder.compute();
            compute.with_pipeline(pipeline);
            compute.with_arguments(0, arguments);
            compute.with_constants(&Params {
                y_offset: row,
                ..params
            });
            let columns = params.out_row_stride as usize;
            let row_count = columns.min(count - row as usize * columns);
            compute.dispatch(mev::Extent3::new(row_count as u32, 1, 1));
        }
        queue.submit_checkpoint([encoder.finish()])?;
    }
    Ok(())
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[middle - 1] + samples[middle]) * 0.5
    } else {
        samples[middle]
    }
}

pub struct PreparedSearch {
    pipeline: mev::ComputePipeline,
    arguments: SearchArguments,
    params: Params,
    upload: mev::Buffer,
    readback: mev::Buffer,
    packed: Vec<u8>,
    bytes: Vec<u8>,
    columns: usize,
    count: usize,
}
impl PreparedSearch {
    pub fn new(device: &mev::Device, blocks: &[[[u8; 3]; 16]]) -> Result<Self, Box<dyn Error>> {
        if blocks.is_empty() {
            return Err("GPU BC1 search needs nonempty blocks and at least one repeat".into());
        }
        let columns = blocks.len().min(1024);
        let rows = blocks.len().div_ceil(columns);
        let padded_count = columns
            .checked_mul(rows)
            .ok_or("BC1 block count overflow")?;
        let input_size = padded_count
            .checked_mul(64)
            .ok_or("BC1 input size overflow")?;
        let input_pixels = u32::try_from(input_size / 4)?;
        let output_size = padded_count
            .checked_mul(8)
            .ok_or("BC1 output size overflow")?;
        let params = Params {
            in_width: u32::try_from(columns * 4)?,
            in_height: u32::try_from(rows.checked_mul(4).ok_or("BC1 height overflow")?)?,
            layers: 1,
            in_row_stride: u32::try_from(columns * 4)?,
            in_plane_stride: input_pixels,
            out_row_stride: u32::try_from(columns)?,
            out_plane_stride: u32::try_from(padded_count)?,
            alpha_threshold: 0.5,
            x_offset: 0,
            y_offset: 0,
            z_offset: 0,
        };
        let library = device.new_shader_library(mev::LibraryDesc {
            name: "BC1 experiment production search",
            input: mev::LibraryInput::wgsl(format!("{BINOM}\n{REDUCE}\n{CLUSTER_FIT}\n{BC1}")),
        })?;
        let pipeline = device.new_compute_pipeline(mev::ComputePipelineDesc {
            name: "BC1 experiment production search",
            shader: library.entry("compress_bc1"),
            work_group_size: [1, 1, 64],
            constants: Params::SIZE,
            arguments: &[SearchArguments::LAYOUT],
        })?;
        let arguments = SearchArguments {
            input: device.new_buffer(mev::BufferDesc {
                name: "BC1 search RGBA input",
                size: input_size,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            }),
            output: device.new_buffer(mev::BufferDesc {
                name: "BC1 search block output",
                size: output_size,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
            }),
        };
        let upload = device.new_buffer(mev::BufferDesc {
            name: "BC1 search upload",
            size: input_size,
            usage: BufferUsage::HOST_WRITE | BufferUsage::TRANSFER_SRC,
        });
        let readback = device.new_buffer(mev::BufferDesc {
            name: "BC1 search readback",
            size: output_size,
            usage: BufferUsage::HOST_READ | BufferUsage::TRANSFER_DST,
        });
        let packed = vec![255; input_size];
        let bytes = vec![0; output_size];
        Ok(Self {
            pipeline,
            arguments,
            params,
            upload,
            readback,
            packed,
            bytes,
            columns,
            count: blocks.len(),
        })
    }
    pub fn benchmark(
        &mut self,
        queue: &mut mev::Queue,
        blocks: &[[[u8; 3]; 16]],
        repeats: usize,
    ) -> Result<SearchResult, Box<dyn Error>> {
        if blocks.len() != self.count || repeats == 0 {
            return Err("search count mismatch or zero repeats".into());
        }
        let Self {
            pipeline,
            arguments,
            params,
            upload,
            readback,
            packed,
            bytes,
            columns,
            count: _,
        } = self;
        let params = *params;
        let columns = *columns;
        let mut resident_samples = Vec::with_capacity(repeats);
        let mut total_samples = Vec::with_capacity(repeats);

        queue.wait_idle()?;
        pack_rgb(blocks, columns, packed);
        upload.write(0, packed)?;
        let mut encoder = queue.new_command_encoder();
        encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        encoder
            .copy()
            .copy_buffer_to_buffer(upload, &arguments.input);
        encoder.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
        queue.submit_checkpoint([encoder.finish()])?;
        search(queue, pipeline, arguments, params, blocks.len())?;
        queue.wait_idle()?;

        for _ in 0..repeats {
            let start = Instant::now();
            search(queue, pipeline, arguments, params, blocks.len())?;
            queue.wait_idle()?;
            resident_samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        for _ in 0..repeats {
            let start = Instant::now();
            pack_rgb(blocks, columns, packed);
            upload.write(0, packed)?;
            let mut encoder = queue.new_command_encoder();
            encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
            encoder
                .copy()
                .copy_buffer_to_buffer(upload, &arguments.input);
            encoder.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
            queue.submit_checkpoint([encoder.finish()])?;
            search(queue, pipeline, arguments, params, blocks.len())?;
            let mut encoder = queue.new_command_encoder();
            encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
            encoder
                .copy()
                .copy_buffer_to_buffer(&arguments.output, readback);
            queue.submit_checkpoint([encoder.finish()])?;
            queue.wait_idle()?;
            readback.read(0, bytes)?;
            total_samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        Ok(SearchResult {
            blocks: bytes
                .chunks_exact(8)
                .take(blocks.len())
                .map(|bytes| Block::from_bytes(bytes.try_into().unwrap()))
                .collect(),
            resident_ms: median(&mut resident_samples),
            end_to_end_ms: median(&mut total_samples),
            count: blocks.len(),
            dispatched_count: blocks.len(),
            label: "production WGSL RGB-distance cluster fit, 120 partition iterations / 64 lanes for opaque 16-texel blocks",
        })
    }
}
pub fn run(
    device: &mev::Device,
    queue: &mut mev::Queue,
    blocks: &[[[u8; 3]; 16]],
    repeats: usize,
) -> Result<SearchResult, Box<dyn Error>> {
    PreparedSearch::new(device, blocks)?.benchmark(queue, blocks, repeats)
}
