use std::io;

use jkl::{
    image::format::Format,
    jackal::image::{Compression, JackalReader},
    math::{Rgb8U, Rgb565},
    vle::Vle,
};
use wgpu::util::DeviceExt;

const RANS_WGSL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/rans.wgsl"));

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RawEntry32 {
    symbol: u32,
    freq: u32,
    cumul: u32,
    pad: u32,
}

impl RawEntry32 {
    fn from_entry<T>(entry: jkl::ans::Entry<T>, f: impl FnOnce(T) -> u32) -> Self {
        RawEntry32 {
            symbol: f(entry.symbol),
            freq: entry.freq.get(),
            cumul: entry.cumul,
            pad: 0,
        }
    }

    fn from_u8(entry: jkl::ans::Entry<u8>) -> Self {
        Self::from_entry(entry, |bits| u32::from(bits))
    }

    fn from_u32(entry: jkl::ans::Entry<u32>) -> Self {
        Self::from_entry(entry, |bits| bits)
    }

    fn from_rgb8_vle(entry: jkl::ans::Entry<Vle<u32>>) -> Self {
        Self::from_entry(entry, |vle| Rgb8U::from_bits_interleaved(vle.0).bits())
    }

    fn from_rgb565_vle(entry: jkl::ans::Entry<Vle<u16>>) -> Self {
        Self::from_entry(entry, |vle| {
            u32::from(Rgb565::from_bits_interleaved(vle.0).bits())
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RawEntry64 {
    symbol: u64,
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
    symbol2_count: u32,
    tile_count: u32,
    width: u32,
    height: u32,
    stride: u32,
}

pub struct Uploader {
    compute_pipeline: ComputePipeline,
}

impl Uploader {
    /// Create an uploader bound to the given device/queue pair. The
    /// device is cloned internally so the caller may discard its
    /// handle if desired.
    pub fn new(device: &wgpu::Device) -> Self {
        Uploader {
            compute_pipeline: ComputePipeline::new(device),
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
        match reader.format() {
            Format::RGB8 | Format::RGBA8 | Format::BC1 => {}
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unsupported format",
                ));
            }
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
        dbg!(payload_len);

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
        dbg!(offsets[..].len());

        let tiles_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-wgpu-tiles"),
            contents: bytemuck::cast_slice(&tiles[..]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        dbg!(tiles[..].len());

        match (reader.format(), reader.compression()) {
            (Format::RGB8, Compression::Ans) => {
                let ans_context =
                    jkl::bits::read_bits_scope(reader.context_reader()?, |context_reader| {
                        jkl::ans::Context::<jkl::vle::Vle<u32>>::read(context_reader)
                    })?;

                let table = ans_context.table();
                let symbol_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-symbol-table",
                    table.iter().copied().map(RawEntry32::from_rgb8_vle),
                );
                dbg!(table[..].len());

                let byte_stride = (width.checked_mul(4).unwrap_or(0) + 255) & !255;
                let output_buffer_size = byte_stride as u64 * height as u64;
                let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jkl-wgpu-output"),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

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

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    symbol_count: table.len(),
                    symbol2_count: 0,
                    tiles: &tiles,
                    pipeline: &self.compute_pipeline.rgb8,
                    bind_group_layout: &self.compute_pipeline.bind_group_layout,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(device, encoder);

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

                Ok(texture)
            }
            (Format::BC1, Compression::Ans) => {
                let (colors_context, indices_context) =
                    jkl::bits::read_bits_scope(reader.context_reader()?, |context_reader| {
                        let colors = jkl::ans::Context::<jkl::vle::Vle<u16>>::read(context_reader)?;
                        let indices = jkl::ans::Context::<u8>::read(context_reader)?;

                        Ok((colors, indices))
                    })?;

                let colors_table = colors_context.table();
                let indices_table = indices_context.table();
                let symbol_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-symbol-table",
                    colors_table
                        .iter()
                        .copied()
                        .map(RawEntry32::from_rgb565_vle)
                        .chain(indices_table.iter().copied().map(RawEntry32::from_u8)),
                );
                dbg!(colors_table[..].len());
                dbg!(indices_table[..].len());

                let byte_stride = (width.checked_mul(16).unwrap_or(0) + 255) & !255;
                let output_buffer_size = byte_stride as u64 * height as u64;
                let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jkl-wgpu-output"),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("jkl-wgpu-output-texture"),
                    size: wgpu::Extent3d {
                        width: width * 4,
                        height: height * 4,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Bc1RgbaUnorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    symbol_count: colors_table.len(),
                    symbol2_count: indices_table.len(),
                    tiles: &tiles,
                    pipeline: &self.compute_pipeline.bc1,
                    bind_group_layout: &self.compute_pipeline.bind_group_layout,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(device, encoder);

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
                        width: width * 4,
                        height: height * 4,
                        depth_or_array_layers: 1,
                    },
                );

                Ok(texture)
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Clone)]
struct ComputePipeline {
    rgb8: wgpu::ComputePipeline,
    rgba8: wgpu::ComputePipeline,
    bc1: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePipeline {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-wgpu-rans-decompress-shader"),
            source: wgpu::ShaderSource::Wgsl(RANS_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jkl-wgpu-rans-bgl"),
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
            label: Some("jkl-wgpu-rans-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let rgb8 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-wgpu-rgb8-rans-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("decompress_rgb8_rans"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let rgba8 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-wgpu-rgba8-rans-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("decompress_rgba8_rans"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bc1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-wgpu-bc1-rans-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("decompress_bc1_rans"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        ComputePipeline {
            rgb8,
            rgba8,
            bc1,
            bind_group_layout,
        }
    }
}

fn write_buffer_from_iter<T>(buffer: &wgpu::Buffer, iter: impl IntoIterator<Item = T>) -> usize
where
    T: bytemuck::NoUninit,
{
    let mut mapped = buffer.slice(..).get_mapped_range_mut();
    let mut offset = 0;

    for item in iter {
        let raw_bytes = bytemuck::bytes_of(&item);
        mapped[offset..][..raw_bytes.len()].copy_from_slice(raw_bytes);
        offset += raw_bytes.len();
    }

    offset
}

fn create_buffer_from_iter<T>(
    device: &wgpu::Device,
    label: &str,
    iter: impl Iterator<Item = T>,
) -> wgpu::Buffer
where
    T: bytemuck::NoUninit,
{
    let (lower, upper) = iter.size_hint();
    assert_eq!(Some(lower), upper);

    let size = u64::try_from(lower * std::mem::size_of::<T>()).expect("size exceedes u64");

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });

    write_buffer_from_iter(&buffer, iter);
    buffer.unmap();

    buffer
}

struct DecompressDispatch<'a> {
    width: u32,
    height: u32,
    byte_stride: u32,
    symbol_count: usize,
    symbol2_count: usize,
    tiles: &'a [RawTile],
    pipeline: &'a wgpu::ComputePipeline,
    bind_group_layout: &'a wgpu::BindGroupLayout,
    payload_buf: &'a wgpu::Buffer,
    offsets_buf: &'a wgpu::Buffer,
    tiles_buf: &'a wgpu::Buffer,
    symbol_buf: &'a wgpu::Buffer,
    out_buf: &'a wgpu::Buffer,
}

impl DecompressDispatch<'_> {
    fn run(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let tile_count = u32::try_from(self.tiles.len()).unwrap_or(0);
        let symbol_count = u32::try_from(self.symbol_count).unwrap_or(0);
        let symbol2_count = u32::try_from(self.symbol2_count).unwrap_or(0);
        let params = Params {
            symbol_count,
            symbol2_count,
            tile_count,
            width: self.width,
            stride: self.byte_stride / 4,
            height: self.height,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-wgpu-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jkl-wgpu-decode-bg"),
            layout: self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.payload_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.tiles_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.symbol_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.out_buf.as_entire_binding(),
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
            cpass.set_pipeline(self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let groups_x = tile_count.div_ceil(64);
            if groups_x > 0 {
                cpass.dispatch_workgroups(groups_x, 1, 1);
            }
        }
    }
}
