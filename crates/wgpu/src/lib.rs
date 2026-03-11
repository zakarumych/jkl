use std::io;

use jkl::{
    image::format::Format,
    jackal::image::{Compression, JackalReader},
    math::Rgb8U,
};
use wgpu::util::DeviceExt;

const RGB8U_RANS_WGSL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/rgb8u_rans.wgsl"
));

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RawEntry32 {
    sym: u32,
    freq: u32,
    cumul: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RawEntry64 {
    sym: u64,
    freq: u32,
    cumul: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RawTile {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    symbol_count: u32,
    tile_count: u32,
    width: u32,
    height: u32,
    stride: u32,
}

pub struct Uploader {
    rgb8u_ans_compute_pipeline: ComputePipeline,
}

impl Uploader {
    /// Create an uploader bound to the given device/queue pair. The
    /// device is cloned internally so the caller may discard its
    /// handle if desired.
    pub fn new(device: &wgpu::Device) -> Self {
        Uploader {
            rgb8u_ans_compute_pipeline: ComputePipeline::new_rgb8_ans(device),
        }
    }

    /// Upload a JKLI image from an existing `JackalReader`. Any I/O
    /// error produced by the reader is propagated directly; no extra
    /// error types are added.
    pub fn upload_from_reader<R>(
        &self,
        reader: &mut JackalReader<R>,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> io::Result<wgpu::Texture>
    where
        R: io::Read + io::Seek,
    {
        if reader.format() != Format::RGB8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported format",
            ));
        }
        if reader.compression() != Compression::Ans {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported compression",
            ));
        }

        let [width_usize, height_usize, _] = reader.extent().raw_size();
        let width = u32::try_from(width_usize)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "dimension too large"))?;
        let height = u32::try_from(height_usize)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "dimension too large"))?;

        let payload_len = reader.payload_len();

        usize::try_from(payload_len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "payload too large"))?;

        u32::try_from(payload_len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "payload too large"))?;

        dbg!(payload_len);
        let payload_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-wgpu-payload-bytes"),
            size: payload_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        let mut offsets = Vec::new();
        let mut tiles = Vec::new();
        {
            let mut mapped = payload_buf.slice(..).get_mapped_range_mut();

            let mut last_offset = 0;
            for tile_index in 0..reader.tiles() {
                let tile = reader.tile(tile_index);
                let tile_len = reader.tile_payload_len(tile_index) as usize;
                reader
                    .copy_tile_payload_into(tile_index, &mut mapped[last_offset..][..tile_len])?;
                offsets.push(last_offset as u32);
                last_offset += tile_len;
                tiles.push(RawTile {
                    x: u32::try_from(tile.rect.x).unwrap_or(0),
                    y: u32::try_from(tile.rect.y).unwrap_or(0),
                    w: u32::try_from(tile.rect.w).unwrap_or(0),
                    h: u32::try_from(tile.rect.h).unwrap_or(0),
                })
            }
            offsets.push(last_offset as u32);
        };
        payload_buf.unmap();

        let offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-wgpu-tile-byte-offsets"),
            contents: bytemuck::cast_slice(&offsets[..]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let tiles_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-wgpu-tiles"),
            contents: bytemuck::cast_slice(&tiles[..]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        match (reader.format(), reader.compression()) {
            (Format::RGB8, Compression::Ans) => {
                let ans_rgb8u_context =
                    jkl::bits::read_bits_scope(reader.context_reader()?, |context_reader| {
                        jkl::ans::Context::<jkl::vle::Vle<u32>>::read(context_reader)
                    })?;

                let table = ans_rgb8u_context.table();

                let symbol_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jkl-wgpu-symbol-table"),
                    size: payload_len,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });

                {
                    let mut mapped = symbol_buf.slice(..).get_mapped_range_mut();
                    let mut offset = 0;

                    for entry in table {
                        let rgb = Rgb8U::from_bits_interleaved(entry.symbol.0);

                        let raw_entry = RawEntry32 {
                            sym: rgb.bits(),
                            cumul: entry.cumul,
                            freq: entry.freq.get(),
                            pad: 0,
                        };

                        let raw_bytes = bytemuck::bytes_of(&raw_entry);
                        mapped[offset..][..raw_bytes.len()].copy_from_slice(raw_bytes);
                        offset += raw_bytes.len();
                    }
                };

                symbol_buf.unmap();

                let byte_stride = (width.checked_mul(4).unwrap_or(0) + 255) & !255;
                let output_buffer_size = byte_stride as u64 * height as u64;
                let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jkl-wgpu-output"),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                let tile_count = u32::try_from(tiles.len()).unwrap_or(0);
                let symbol_count = u32::try_from(table.len()).unwrap_or(0);
                let params = Params {
                    tile_count,
                    width,
                    stride: byte_stride / 4,
                    height,
                    symbol_count,
                };

                let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jkl-wgpu-params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
                });

                let compute = &self.rgb8u_ans_compute_pipeline;
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("jkl-wgpu-decode-bg"),
                    layout: &compute.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: payload_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: offsets_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: tiles_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: symbol_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("jkl-wgpu-decode-pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&compute.pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);

                    let groups_x = tile_count.div_ceil(64);
                    if groups_x > 0 {
                        cpass.dispatch_workgroups(groups_x, 1, 1);
                    }
                }

                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("jkl-wgpu-output-texture"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });

                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &out_buf,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(byte_stride),
                            rows_per_image: Some(height),
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );

                let total_cpu_gpu_transfer = payload_len as usize
                    + offsets.len() * std::mem::size_of::<u32>()
                    + tiles.len() * std::mem::size_of::<RawTile>()
                    + table.len() * std::mem::size_of::<RawEntry32>()
                    + bytemuck::bytes_of(&params).len();

                dbg!(total_cpu_gpu_transfer);
                dbg!(width * height * 3);

                Ok(texture)
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Clone)]
struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePipeline {
    fn new_rgb8_ans(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-wgpu-rgb8-rans-decompress-shader"),
            source: wgpu::ShaderSource::Wgsl(RGB8U_RANS_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jkl-wgpu-rgb8-rans-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jkl-wgpu-rgb8-rans-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-wgpu-rgb8-rans-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("decompress_rgb8_rans"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}
