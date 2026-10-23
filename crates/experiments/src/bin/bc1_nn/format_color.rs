use std::{error::Error, time::Instant};

use super::{gpu_encode::GpuEncoder, Texels};
use mev::kernels::nn::Nn;
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
const BC2: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../archive/cluster-fit/shaders/bc2.wgsl"
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

pub struct NativeResult {
    pub blocks: Vec<[u8; 16]>,
    /// Median command recording, submission, and queue completion time.
    pub resident_ms: f64,
    /// Median RGB packing, upload, search, readback, and queue completion time.
    pub end_to_end_ms: f64,
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

struct NativeSearch {
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
impl NativeSearch {
    pub fn new(
        device: &mev::Device,
        blocks: &[[[u8; 3]; 16]],
        format: Format,
    ) -> Result<Self, Box<dyn Error>> {
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
            .checked_mul(16)
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
            input: mev::LibraryInput::wgsl(current_source(format)),
        })?;
        let pipeline = device.new_compute_pipeline(mev::ComputePipelineDesc {
            name: "BC1 experiment production search",
            shader: library.entry("compress_bc2"),
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
    ) -> Result<NativeResult, Box<dyn Error>> {
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

        Ok(NativeResult {
            blocks: bytes
                .chunks_exact(16)
                .take(blocks.len())
                .map(|bytes| bytes.try_into().unwrap())
                .collect(),
            resident_ms: median(&mut resident_samples),
            end_to_end_ms: median(&mut total_samples),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Format {
    Bc2,
    Bc3,
}
#[derive(Clone, Copy, Debug)]
pub enum Method {
    CurrentGpu,
    Neural,
    Extrema,
}
pub struct BenchResult {
    pub blocks: Vec<[u8; 16]>,
    pub resident_ms: f64,
    pub host_ms: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, mev::AutoDeviceRepr)]
struct PackParams {
    count: u32,
    format: u32,
}
#[derive(mev::Arguments)]
struct PackArguments {
    #[mev(storage, compute)]
    color: mev::Buffer,
    #[mev(storage, compute)]
    output: mev::Buffer,
}
pub struct ColorEncoder {
    color: GpuEncoder,
    current: NativeSearch,
    pack: mev::ComputePipeline,
    arguments: PackArguments,
    readback: mev::Buffer,
    bytes: Vec<u8>,
    params: PackParams,
}
impl ColorEncoder {
    pub fn new(
        device: &mev::Device,
        blocks: &[Texels],
        format: Format,
    ) -> Result<Self, Box<dyn Error>> {
        let color = GpuEncoder::new(device, blocks)?;
        let current = NativeSearch::new(device, blocks, format)?;
        let library = device.new_shader_library(mev::LibraryDesc {
            name: "BC2/BC3 opaque alpha packing",
            input: mev::LibraryInput::wgsl(include_str!("format_color.wgsl")),
        })?;
        let pack = device.new_compute_pipeline(mev::ComputePipelineDesc {
            name: "BC2/BC3 opaque alpha packing",
            shader: library.entry("append_alpha"),
            work_group_size: [64, 1, 1],
            constants: PackParams::SIZE,
            arguments: &[PackArguments::LAYOUT],
        })?;
        let arguments = PackArguments {
            color: color.output_buffer().clone(),
            output: device.new_buffer(mev::BufferDesc {
                name: "BC2/BC3 complete output",
                size: blocks.len() * 16,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
            }),
        };
        let readback = device.new_buffer(mev::BufferDesc {
            name: "BC2/BC3 complete readback",
            size: blocks.len() * 16,
            usage: BufferUsage::HOST_READ | BufferUsage::TRANSFER_DST,
        });
        Ok(Self {
            color,
            current,
            pack,
            arguments,
            readback,
            bytes: vec![0; blocks.len() * 16],
            params: PackParams {
                count: u32::try_from(blocks.len())?,
                format: match format {
                    Format::Bc2 => 2,
                    Format::Bc3 => 3,
                },
            },
        })
    }
    fn encode(
        &self,
        queue: &mut mev::Queue,
        nn: Option<&Nn>,
        method: Method,
    ) -> Result<(), Box<dyn Error>> {
        self.color
            .encode_one_neighbor_round(queue, nn, matches!(method, Method::Extrema))?;
        let mut encoder = queue.new_command_encoder();
        encoder.barrier(
            PipelineStages::COMPUTE_SHADER,
            PipelineStages::COMPUTE_SHADER,
        );
        {
            let mut compute = encoder.compute();
            compute.with_pipeline(&self.pack);
            compute.with_arguments(0, &self.arguments);
            compute.with_constants(&self.params);
            compute.dispatch(mev::Extent3::new(self.params.count.div_ceil(64), 1, 1));
        }
        queue.submit_checkpoint([encoder.finish()])?;
        Ok(())
    }
    pub fn benchmark(
        &mut self,
        queue: &mut mev::Queue,
        nn: Option<&Nn>,
        blocks: &[Texels],
        repeats: usize,
        method: Method,
    ) -> Result<BenchResult, Box<dyn Error>> {
        if matches!(method, Method::CurrentGpu) {
            let r = self.current.benchmark(queue, blocks, repeats)?;
            return Ok(BenchResult {
                blocks: r.blocks,
                resident_ms: r.resident_ms,
                host_ms: r.end_to_end_ms,
            });
        }
        if repeats == 0 || blocks.len() != self.params.count as usize {
            return Err("BC2/BC3 count mismatch or zero repeats".into());
        }
        queue.wait_idle()?;
        self.color.upload(queue, blocks)?;
        self.encode(queue, nn, method)?;
        queue.wait_idle()?;
        let mut resident = Vec::with_capacity(repeats);
        let mut host = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            self.encode(queue, nn, method)?;
            queue.wait_idle()?;
            resident.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        for _ in 0..repeats {
            let start = Instant::now();
            self.color.upload(queue, blocks)?;
            self.encode(queue, nn, method)?;
            let mut encoder = queue.new_command_encoder();
            encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
            encoder
                .copy()
                .copy_buffer_to_buffer(&self.arguments.output, &self.readback);
            queue.submit_checkpoint([encoder.finish()])?;
            queue.wait_idle()?;
            self.readback.read(0, &mut self.bytes)?;
            host.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(BenchResult {
            blocks: self
                .bytes
                .chunks_exact(16)
                .map(|b| b.try_into().unwrap())
                .collect(),
            resident_ms: median(&mut resident),
            host_ms: median(&mut host),
        })
    }
}
pub fn current_cpu(t: &Texels, format: Format) -> [u8; 16] {
    use jkl::{
        image::block::{bc2, bc3},
        math::Rgb32F,
    };
    let grid = std::array::from_fn(|y| {
        std::array::from_fn(|x| {
            let c = t[y * 4 + x];
            (
                Rgb32F::new(
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ),
                true,
            )
        })
    });
    match format {
        Format::Bc2 => bc2::Block::encode(grid).bytes(),
        Format::Bc3 => bc3::Block::encode(grid).bytes(),
    }
}
fn current_source(format: Format) -> String {
    if matches!(format, Format::Bc2) {
        return format!("{BINOM}\n{REDUCE}\n{CLUSTER_FIT}\n{BC2}");
    }
    let mut cf = CLUSTER_FIT.to_owned();
    cf = cf.replace(
        "let q = round(clamp(v, 0.0, 1.0) * levels);",
        "let q = floor(clamp(v, 0.0, 1.0) * levels);",
    );
    cf = cf.replace(
        "return q / levels;",
        "return cpu_code_to_float(u32(q), bits);",
    );

    cf = cf.replace(
        "var iter_state = binom_iter_init_at(sample_count - 1u, palette_count - 1u, iter_begin);",
        "var cuts = vec3<u32>(0u);",
    );
    cf = cf.replace("iter_state.indices[c]", "cuts[c]");
    cf = cf.replace(
        "binom_iter_next(&iter_state);",
        r#"
        for (var c=2i;c>=0i;c-=1i) {
            if cuts[u32(c)] < sample_count - 2u {
                cuts[u32(c)] += 1u;
                for(var j=u32(c)+1u;j<3u;j+=1u) {cuts[j]=cuts[u32(c)];}
                break;
            }
        }
    "#,
    );
    let mut bc = BC2
        .replace("\r\n", "\n")
        .replace("unpack4x8unorm(image_input[block_z * params.in_plane_stride + y * params.in_row_stride + x])", "cpu_rgba(image_input[block_z * params.in_plane_stride + y * params.in_row_stride + x])")
        .replace(
            "let r = u32(color.r * 31.0 + 0.5);",
            "let r = u32(clamp(color.r * 31.0,0.0,31.0));",
        )
        .replace(
            "let g = u32(color.g * 63.0 + 0.5);",
            "let g = u32(clamp(color.g * 63.0,0.0,63.0));",
        )
        .replace(
            "let b = u32(color.b * 31.0 + 0.5);",
            "let b = u32(clamp(color.b * 31.0,0.0,31.0));",
        )
        .replace(
            "let iter_begin = local_idx * iter_per_thread;",
            "let iter_begin = 0u;",
        )
        .replace(
            "let iter_end = min((local_idx + 1u) * iter_per_thread, total_iterations);",
            "let iter_end = select(0u,64u,local_idx==0u);",
        )
        .replace(
            "vec3<u32>(5u, 6u, 5u),\n        0u,",
            "vec3<u32>(5u, 6u, 5u),\n        1u,",
        )
        .replace(
            "make_bc2_block(alpha_lo, alpha_hi,",
            "make_bc2_block(255u, 0u,",
        )
        .replace("color0, 0u, 0u)", "color0, color0, 0u)");
    // Only lane zero participates: CPU search visits exactly 64 successive cuts.
    bc = bc.replace(
        "error = cluster_fit.error;",
        "error = select(1e10,cluster_fit.error,local_idx==0u);",
    );
    // Repeated truncation requires the CPU's exact normalized-code f32 bits.
    let table = |name: &str, levels: u32| {
        format!(
            "const {name}: array<u32, {}> = array<u32, {}>({});",
            levels + 1,
            levels + 1,
            (0..=levels)
                .map(|i| format!("{}u", (i as f32 / levels as f32).to_bits()))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    let lookup = format!(
        "{}\n{}\n{}\n{}",
        table("CPU5", 31),
        table("CPU6", 63),
        table("CPU8", 255),
        r#"
fn cpu_code_to_float(q:u32,bits:u32)->f32 {
    if bits==5u {return bitcast<f32>(CPU5[q]);}
    if bits==6u {return bitcast<f32>(CPU6[q]);}
    return bitcast<f32>(CPU8[q]);
}
fn cpu_rgba(word:u32)->vec4<f32> {return vec4<f32>(bitcast<f32>(CPU8[word&255u]),bitcast<f32>(CPU8[(word>>8u)&255u]),bitcast<f32>(CPU8[(word>>16u)&255u]),1.0);}
"#
    );
    format!("{BINOM}\n{REDUCE}\n{lookup}\n{cf}\n{bc}")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn current_cpu_opaque_alpha() {
        let t = [[80, 140, 210]; 16];
        let b = current_cpu(&t, Format::Bc2);
        assert_eq!(&b[..8], &[255; 8]);
        let b = current_cpu(&t, Format::Bc3);
        assert_eq!(&b[..8], &[255, 0, 0, 0, 0, 0, 0, 0]);
    }
    #[test]
    fn sources_preserve_native_and_port_scope() {
        let native = current_source(Format::Bc2);
        assert!(native.ends_with(BC2));
        assert!(native.contains(CLUSTER_FIT));
        let port = current_source(Format::Bc3);
        assert!(port.contains("select(0u,64u,local_idx==0u)"));
        assert!(port.contains("floor(clamp(v"));
        assert!(port.contains("vec3<u32>(5u, 6u, 5u),\n        1u,"));
        assert!(!port.contains("iter_state.indices[c]"));
    }
}

fn decoded_rgb(bytes: &[u8; 16]) -> [[f32; 3]; 16] {
    let decode = |bits: u16| {
        [
            ((bits >> 11) & 31) as f32 / 31.0,
            ((bits >> 5) & 63) as f32 / 63.0,
            (bits & 31) as f32 / 31.0,
        ]
    };
    let a = decode(u16::from_le_bytes([bytes[8], bytes[9]]));
    let b = decode(u16::from_le_bytes([bytes[10], bytes[11]]));
    let palette = [
        a,
        b,
        std::array::from_fn(|c| a[c] * (2.0 / 3.0) + b[c] * (1.0 / 3.0)),
        std::array::from_fn(|c| a[c] * (1.0 / 3.0) + b[c] * (2.0 / 3.0)),
    ];
    std::array::from_fn(|i| palette[((bytes[12 + i / 4] >> (2 * (i % 4))) & 3) as usize])
}
fn source_error(source: &Texels, bytes: &[u8; 16]) -> f64 {
    source
        .iter()
        .zip(decoded_rgb(bytes))
        .map(|(s, d)| {
            (0..3)
                .map(|c| (s[c] as f64 / 255.0 - d[c] as f64).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .sum()
}
fn compare_decoded(
    source: &[Texels],
    actual: &[[u8; 16]],
    expected: &[[u8; 16]],
) -> serde_json::Value {
    let mut mismatch_details = Vec::new();
    let mut exact = 0;
    let mut decoded_exact = 0;
    let mut max_channel = 0.0f64;
    let mut max_error = 0.0f64;
    for (case, ((s, a), b)) in source.iter().zip(actual).zip(expected).enumerate() {
        exact += usize::from(a == b);
        let da = decoded_rgb(a);
        let db = decoded_rgb(b);
        let mut delta = 0.0f64;
        for (pa, pb) in da.iter().zip(db) {
            for c in 0..3 {
                delta = delta.max((pa[c] as f64 - pb[c] as f64).abs());
            }
        }
        decoded_exact += usize::from(delta == 0.0);
        max_channel = max_channel.max(delta);
        max_error = max_error.max((source_error(s, a) - source_error(s, b)).abs());
        if a != b {
            mismatch_details.push(serde_json::json!({"case":case,"source":s,"gpu":a,"cpu_or_control":b,
                "gpu_rgb_error":source_error(s,a),"cpu_rgb_error":source_error(s,b),
                "gpu_yiq_decoded_error":yiq_error(s,a,false),"cpu_yiq_decoded_error":yiq_error(s,b,false),
                "gpu_yiq_snapped_palette_error":yiq_error(s,a,true),"cpu_yiq_snapped_palette_error":yiq_error(s,b,true)}));
        }
    }
    serde_json::json!({"blocks":source.len(),"byte_exact_blocks":exact,"mismatched_blocks":source.len()-exact,
        "decoded_exact_blocks":decoded_exact,"max_decoded_channel_error":max_channel,"max_source_rgb_norm_error_difference":max_error,
        "mismatch_details":mismatch_details,
        "faithfulness":if exact==source.len(){"bit exact on checked cases"}else if decoded_exact==source.len(){"decoded exact on checked cases; bytes differ"}else{"algorithm port with measured discrepancies; not bit exact"}})
}
fn yiq_error(source: &Texels, bytes: &[u8; 16], snap: bool) -> f64 {
    use jkl::math::{Rgb32F, Rgb565, Yiq32F};
    source
        .iter()
        .zip(decoded_rgb(bytes))
        .map(|(s, d)| {
            let actual = Rgb32F(d);
            let actual = if snap {
                Rgb565::from_f32(actual).into_f32()
            } else {
                actual
            };
            Yiq32F::perceptual_distance(
                Yiq32F::from_rgb(Rgb32F(s.map(|c| c as f32 / 255.0))),
                Yiq32F::from_rgb(actual),
            ) as f64
        })
        .sum()
}
fn assert_opaque(format: Format, blocks: &[[u8; 16]]) -> Result<(), Box<dyn Error>> {
    for b in blocks {
        let expected: &[u8] = match format {
            Format::Bc2 => &[255; 8],
            Format::Bc3 => &[255, 0, 0, 0, 0, 0, 0, 0],
        };
        if &b[..8] != expected {
            return Err("full-format GPU alpha is not opaque".into());
        }
        if decoded_rgb(b)
            .iter()
            .flatten()
            .any(|c| !c.is_finite() || *c < 0.0 || *c > 1.0)
        {
            return Err("invalid full-format GPU RGB decode".into());
        }
    }
    Ok(())
}
/// Checks native baseline scope, CPU-port differences, and complete-format packing.
pub fn verify_gpu(
    device: &mev::Device,
    queue: &mut mev::Queue,
    source: &[Texels],
    nn: Option<&Nn>,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut cases = source.iter().take(65).copied().collect::<Vec<_>>();
    for color in [
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 255],
        [80, 140, 210],
        [1, 254, 127],
    ] {
        cases.push([color; 16]);
    }
    cases.push(std::array::from_fn(|i| {
        if i % 2 == 0 {
            [255, 0, 0]
        } else {
            [0, 255, 255]
        }
    }));
    cases.push(std::array::from_fn(|i| {
        [i as u8 * 17, 255 - i as u8 * 17, ((i * 73) % 256) as u8]
    }));
    cases.push(std::array::from_fn(|i| {
        if i < 11 {
            [240, 50, 5]
        } else if i < 14 {
            [20, 130, 255]
        } else {
            [5, 220, 10]
        }
    }));
    let mut color = GpuEncoder::new(device, &cases)?;
    let extrema = color.benchmark_principal_neighbors(queue, &cases, 1, 1, false)?;
    let neural = match nn {
        Some(nn) => Some(color.benchmark_neighbors(queue, nn, &cases, 1, 1, false)?),
        None => None,
    };
    let control = super::gpu_search::run(device, queue, &cases, 1)?;
    let mut rows = Vec::new();
    for format in [Format::Bc2, Format::Bc3] {
        let mut encoder = ColorEncoder::new(device, &cases, format)?;
        let current = encoder.benchmark(queue, nn, &cases, 1, Method::CurrentGpu)?;
        assert_opaque(format, &current.blocks)?;
        let expected = if matches!(format, Format::Bc3) {
            cases
                .iter()
                .map(|t| current_cpu(t, format))
                .collect::<Vec<_>>()
        } else {
            control
                .blocks
                .iter()
                .map(|b| {
                    let mut full = [255u8; 16];
                    full[8..].copy_from_slice(&b.bytes());
                    full
                })
                .collect::<Vec<_>>()
        };
        let current_check = compare_decoded(&cases, &current.blocks, &expected);
        let mut new_checks = Vec::new();
        for (method, gold) in [
            (Method::Extrema, Some(&extrema)),
            (Method::Neural, neural.as_ref()),
        ] {
            let Some(gold) = gold else { continue };
            let actual = encoder.benchmark(queue, nn, &cases, 1, method)?;
            assert_opaque(format, &actual.blocks)?;
            for (full, base) in actual.blocks.iter().zip(&gold.blocks) {
                if full[8..] != base.bytes() {
                    return Err("full-format color differs from original RGB pipeline".into());
                }
                let c0 = u16::from_le_bytes([full[8], full[9]]);
                let c1 = u16::from_le_bytes([full[10], full[11]]);
                if c0 < c1 || (c0 == c1 && full[12..].iter().any(|v| *v != 0)) {
                    return Err("opaque color endpoint/index ordering invalid".into());
                }
            }
            new_checks.push(serde_json::json!({"method":format!("{method:?}"),"color_bytes_exact":true,"alpha_opaque":true,"equal_endpoint_indices_safe":true}));
        }
        rows.push(serde_json::json!({"format":format!("{format:?}"),"current_gpu":current_check,
            "current_comparison":if matches!(format,Format::Bc2){"unchanged native BC2 shader against unchanged native BC1 opaque RGB control; CPU BC2 intentionally not gold"}else{"private GPU port against actual CPU BC3"},
            "new_pipeline_pack_checks":new_checks,"native_bc2_source_unchanged":matches!(format,Format::Bc2)}));
    }
    Ok(
        serde_json::json!({"cases":cases.len(),"real_source_blocks":source.len().min(65),"synthetic_cases":8,"formats":rows,"neural_pack_checked":nn.is_some()}),
    )
}
#[cfg(test)]
mod gpu_port_diagnostic {
    use super::*;
    #[test]
    #[ignore]
    fn diagnose_bc3_port() {
        let (device, mut queue, _) = super::super::select_device().unwrap();
        let path = std::env::var("BC3_PARITY_IMAGE").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../bc1-nn-corpus-smoke-data/test.png"
            )
            .to_owned()
        });
        let image = image::open(path).unwrap().to_rgb8();
        let columns = image.width().div_ceil(4) as usize;
        let rows = image.height().div_ceil(4) as usize;
        let count = (columns * rows).min(65);
        let blocks = (0..count)
            .map(|n| {
                let b = n * (columns * rows) / count;
                std::array::from_fn(|i| {
                    let x = (b % columns) * 4 + i % 4;
                    let y = (b / columns) * 4 + i / 4;
                    image
                        .get_pixel(
                            (x as u32).min(image.width() - 1),
                            (y as u32).min(image.height() - 1),
                        )
                        .0
                })
            })
            .collect::<Vec<_>>();
        let mut report = verify_gpu(&device, &mut queue, &blocks, None).unwrap();
        let mut debug = NativeSearch::new(&device, &blocks, Format::Bc3).unwrap();
        let source = current_source(Format::Bc3).replace(
            "make_bc2_block(255u, 0u,",
            "make_bc2_block(255u, bitcast<u32>(error),",
        );
        let library = device
            .new_shader_library(mev::LibraryDesc {
                name: "BC3 fit diagnostic",
                input: mev::LibraryInput::wgsl(source),
            })
            .unwrap();
        debug.pipeline = device
            .new_compute_pipeline(mev::ComputePipelineDesc {
                name: "BC3 fit diagnostic",
                shader: library.entry("compress_bc2"),
                work_group_size: [1, 1, 64],
                constants: Params::SIZE,
                arguments: &[SearchArguments::LAYOUT],
            })
            .unwrap();
        let gpu = debug.benchmark(&mut queue, &blocks, 1).unwrap();
        use jkl::math::{Rgb32F, Rgb565, Vec3, Yiq32F};
        report["actual_fit_objective"]=serde_json::json!(blocks.iter().zip(&gpu.blocks).enumerate().filter_map(|(i,(b,output))| {
            let samples=b.map(|c|Vec3(c.map(|v|v as f32/255.0)));
            let cf=jkl::algos::cluster_fit::cluster_fit::<Vec3,4,16>(&samples,|a|Rgb565::from_f32(Rgb32F(a.0)).into_f32().into(),|a,b|Yiq32F::perceptual_distance(Yiq32F::from_rgb(Rgb32F(a.0)),Yiq32F::from_rgb(Rgb32F(b.0))));
            let gpu_error=f32::from_le_bytes(output[4..8].try_into().unwrap());
            let expected=current_cpu(b,Format::Bc3);
            if expected[8..]==output[8..] {None}else{Some(serde_json::json!({"case":i,"gpu_fit_error":gpu_error,"cpu_fit_error":cf.error,"delta":gpu_error-cf.error}))}
        }).collect::<Vec<_>>());
        std::fs::write(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../build/bc3-port-parity.json"
            ),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }
}
