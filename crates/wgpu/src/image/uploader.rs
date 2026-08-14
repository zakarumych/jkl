use std::io;

use jkl::{
    algos::{ans, lz77, vle::Vle},
    image::{Image, format::Format},
    jackal::image::{Compression, JackalImageReader},
    math::{Rgb8U, Rgb565},
};
use wgpu::util::DeviceExt;

use crate::image::GpuPixels;

const RANS_WGSL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/rans.wgsl"));

#[repr(C)]
#[derive(Copy, Clone, bytemuck::NoUninit)]
struct RawFreqEntry {
    freq: u32,
    cumul: u32,
}

impl<T> From<ans::Entry<T>> for RawFreqEntry {
    fn from(entry: ans::Entry<T>) -> Self {
        RawFreqEntry {
            freq: entry.freq.get(),
            cumul: entry.cumul,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::NoUninit)]
struct RawTile {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::NoUninit)]
struct Params {
    table1_count: u32,
    table2_count: u32,
    tile_count: u32,
    width: u32,
    height: u32,
    stride: u32,
}

pub struct Uploader {
    rgb8: wgpu::ComputePipeline,
    rgb8_lz77: wgpu::ComputePipeline,
    rgba8: wgpu::ComputePipeline,
    bc1: wgpu::ComputePipeline,
    bc1_lz77: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Uploader {
    /// Create an uploader bound to the given device/queue pair. The
    /// device is cloned internally so the caller may discard its
    /// handle if desired.
    pub fn new(device: &wgpu::Device) -> Self {
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jkl-wgpu-rans-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: std::mem::size_of::<Params>() as u32,
        });

        let rgb8 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-wgpu-rgb8-rans-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("decompress_rgb8_rans"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let rgb8_lz77 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-wgpu-rgb8-lz77-rans-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("decompress_rgb8_lz77_rans"),
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

        let bc1_lz77 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-wgpu-bc1-lz77-rans-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("decompress_bc1_lz77_rans"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Uploader {
            rgb8,
            rgb8_lz77,
            rgba8,
            bc1,
            bc1_lz77,
            bind_group_layout,
        }
    }

    /// Upload a JKLI image from an existing `JackalReader`.
    pub fn upload_from_reader<R>(
        &self,
        reader: &mut JackalImageReader<R>,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> io::Result<Image<GpuPixels>>
    where
        R: io::Read + io::Seek,
    {
        let [width_usize, height_usize, _] = reader.extent().raw_size();
        let width = u32::try_from(width_usize)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "dimension too large"))?;
        let height = u32::try_from(height_usize)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "dimension too large"))?;

        let payload_len = reader.tiles_payload_len();
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

        let mut tiles_scratch = Vec::new();

        {
            let mut mapped = payload_buf.slice(..).get_mapped_range_mut().unwrap();

            let mut last_offset = 0;
            for tile_index in 0..reader.tiles() {
                let tile = reader.tile(tile_index);
                let tile_len = reader.tile_payload_len(tile_index) as usize;

                tiles_scratch.resize(tile_len, 0);

                debug_assert!(
                    tile_len % 4 == 0,
                    "tile rANS payload should be multiple of 4"
                );

                reader.copy_tile_payload_into(tile_index, &mut tiles_scratch[..tile_len])?;

                mapped
                    .slice(last_offset..last_offset + tile_len)
                    .copy_from_slice(&tiles_scratch[..tile_len]);

                offsets.push((last_offset / 4) as u32);
                last_offset += tile_len;

                tiles.push(RawTile {
                    x: u32::try_from(tile.rect.x).unwrap_or(0),
                    y: u32::try_from(tile.rect.y).unwrap_or(0),
                    w: u32::try_from(tile.rect.w).unwrap_or(0),
                    h: u32::try_from(tile.rect.h).unwrap_or(0),
                })
            }
            offsets.push((last_offset / 4) as u32);
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
                        ans::Context::<Vle<u32>>::read(context_reader)
                    })?;

                let table = ans_context.table();
                dbg!(table[..].len());

                let table_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-table",
                    table.iter().copied().map(RawFreqEntry::from),
                );

                let symbol_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-symbols",
                    table
                        .iter()
                        .copied()
                        .map(|e| Rgb8U::from_bits_interleaved(e.symbol.0).bits()),
                );

                let byte_stride = (width * 4).div_ceil(256) * 256;
                let output_buffer_size = byte_stride as u64 * height as u64;
                let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jkl-wgpu-output"),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    table1_count: table.len(),
                    table2_count: 0,
                    tiles: &tiles,
                    pipeline: &self.rgb8,
                    bind_group_layout: &self.bind_group_layout,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    table_buf: &table_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(device, encoder);

                Ok(Image::with_stride(
                    jkl::image::Dimensions::D2,
                    [width_usize, height_usize, 1],
                    [byte_stride as usize, byte_stride as usize * height_usize],
                    GpuPixels::new(wgpu::TextureFormat::Rgba8Unorm, out_buf),
                ))
            }
            (Format::RGB8, Compression::Lz77Ans) => {
                let lz77_ans_context =
                    jkl::bits::read_bits_scope(reader.context_reader()?, |context_reader| {
                        ans::Context::<lz77::Token<Vle<u32>>>::read(context_reader)
                    })?;

                let table = lz77_ans_context.table();
                dbg!(table[..].len());

                let table_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-table",
                    table.iter().copied().map(RawFreqEntry::from),
                );

                let symbol_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-symbols",
                    table.iter().copied().map(|e| match e.symbol {
                        lz77::Token::Literal { symbol } => {
                            [0u32, Rgb8U::from_bits_interleaved(symbol.0).bits()]
                        }
                        lz77::Token::Reference {
                            length_minus_2,
                            distance,
                        } => [u32::from(length_minus_2) + 2, u32::from(distance)],
                    }),
                );

                let byte_stride = (width * 4).div_ceil(256) * 256;
                let output_buffer_size = byte_stride as u64 * height as u64;
                let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jkl-wgpu-output"),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    table1_count: table.len(),
                    table2_count: 0,
                    tiles: &tiles,
                    pipeline: &self.rgb8_lz77,
                    bind_group_layout: &self.bind_group_layout,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    table_buf: &table_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(device, encoder);

                Ok(Image::with_stride(
                    jkl::image::Dimensions::D2,
                    [width_usize, height_usize, 1],
                    [byte_stride as usize, byte_stride as usize * height_usize],
                    GpuPixels::new(wgpu::TextureFormat::Rgba8Unorm, out_buf),
                ))
            }
            (Format::RGBA8, Compression::Ans) => {
                let ans_context =
                    jkl::bits::read_bits_scope(reader.context_reader()?, |context_reader| {
                        ans::Context::<Vle<u32>>::read(context_reader)
                    })?;

                let table = ans_context.table();
                dbg!(table[..].len());

                let table_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-table",
                    table.iter().copied().map(RawFreqEntry::from),
                );

                let symbol_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-symbols",
                    table
                        .iter()
                        .copied()
                        .map(|e| Rgb8U::from_bits_interleaved(e.symbol.0).bits()),
                );

                let byte_stride = (width * 4).div_ceil(256) * 256;
                let output_buffer_size = byte_stride as u64 * height as u64;
                let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jkl-wgpu-output"),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    table1_count: table.len(),
                    table2_count: 0,
                    tiles: &tiles,
                    pipeline: &self.rgba8,
                    bind_group_layout: &self.bind_group_layout,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    table_buf: &table_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(device, encoder);

                Ok(Image::with_stride(
                    jkl::image::Dimensions::D2,
                    [width_usize, height_usize, 1],
                    [byte_stride as usize, byte_stride as usize * height_usize],
                    GpuPixels::new(wgpu::TextureFormat::Rgba8Unorm, out_buf),
                ))
            }
            (Format::BC1, Compression::Ans) => {
                let (colors_context, indices_context) =
                    jkl::bits::read_bits_scope(reader.context_reader()?, |context_reader| {
                        let colors = ans::Context::<Vle<u16>>::read(context_reader)?;
                        let indices = ans::Context::<u8>::read(context_reader)?;

                        Ok((colors, indices))
                    })?;

                let colors_table = colors_context.table();
                let indices_table = indices_context.table();

                dbg!(colors_table[..].len());
                dbg!(indices_table[..].len());

                let table_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-table",
                    colors_table
                        .iter()
                        .copied()
                        .map(RawFreqEntry::from)
                        .chain(indices_table.iter().copied().map(RawFreqEntry::from)),
                );

                // Buffer should have size multiple of 4
                // As it is accessed as array of u32 in shader.
                // Constructed iterator is appended by three 0 bytes below and then cut off at padded_len,
                // so that is is padded to exactly multiple of 4.
                let padded_len = (colors_table.len() * 2 + indices_table.len() + 3) & !3;

                let symbol_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-symbols",
                    colors_table
                        .iter()
                        .copied()
                        .flat_map(|e| {
                            Rgb565::from_bits_interleaved(e.symbol.0)
                                .bits()
                                .to_le_bytes()
                        })
                        .chain(indices_table.iter().copied().map(|e| e.symbol))
                        .chain([0u8; 3])
                        .take(padded_len),
                );

                let byte_stride = (width * 16).div_ceil(256) * 256;
                let output_buffer_size = byte_stride as u64 * height as u64;
                let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jkl-wgpu-output"),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    table1_count: colors_table.len(),
                    table2_count: indices_table.len(),
                    tiles: &tiles,
                    pipeline: &self.bc1,
                    bind_group_layout: &self.bind_group_layout,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    table_buf: &table_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(device, encoder);

                Ok(Image::with_stride(
                    jkl::image::Dimensions::D2,
                    [width_usize, height_usize, 1],
                    [byte_stride as usize, byte_stride as usize * height_usize],
                    GpuPixels::new(wgpu::TextureFormat::Bc1RgbaUnorm, out_buf),
                ))
            }
            (Format::BC1, Compression::Lz77Ans) => {
                let (colors_context, indices_context) =
                    jkl::bits::read_bits_scope(reader.context_reader()?, |context_reader| {
                        let colors = ans::Context::<lz77::Token<Vle<u16>>>::read(context_reader)?;
                        let indices = ans::Context::<lz77::Token<u8>>::read(context_reader)?;

                        Ok((colors, indices))
                    })?;

                let colors_table = colors_context.table();
                let indices_table = indices_context.table();

                dbg!(colors_table[..].len());
                dbg!(indices_table[..].len());

                let table_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-table",
                    colors_table
                        .iter()
                        .copied()
                        .map(RawFreqEntry::from)
                        .chain(indices_table.iter().copied().map(RawFreqEntry::from)),
                );

                let symbol_buf = create_buffer_from_iter(
                    device,
                    "jkl-wgpu-symbols",
                    colors_table
                        .iter()
                        .copied()
                        .map(|e| match e.symbol {
                            lz77::Token::Literal { symbol } => {
                                [0u16, Rgb565::from_bits_interleaved(symbol.0).bits()]
                            }
                            lz77::Token::Reference {
                                length_minus_2,
                                distance,
                            } => [length_minus_2 + 2, distance],
                        })
                        .chain(indices_table.iter().copied().map(|e| match e.symbol {
                            lz77::Token::Literal { symbol } => [0u16, u16::from(symbol)],
                            lz77::Token::Reference {
                                length_minus_2,
                                distance,
                            } => [length_minus_2 + 2, distance],
                        })),
                );

                let byte_stride = (width * 16).div_ceil(256) * 256;
                let output_buffer_size = byte_stride as u64 * height as u64;
                let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jkl-wgpu-output"),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    table1_count: colors_table.len(),
                    table2_count: indices_table.len(),
                    tiles: &tiles,
                    pipeline: &self.bc1_lz77,
                    bind_group_layout: &self.bind_group_layout,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    table_buf: &table_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(device, encoder);

                Ok(Image::with_stride(
                    jkl::image::Dimensions::D2,
                    [width_usize, height_usize, 1],
                    [byte_stride as usize, byte_stride as usize * height_usize],
                    GpuPixels::new(wgpu::TextureFormat::Bc1RgbaUnorm, out_buf),
                ))
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unsupported format-compression combination",
                ));
            }
        }
    }
}

fn write_buffer_from_iter<T>(buffer: &wgpu::Buffer, iter: impl IntoIterator<Item = T>) -> usize
where
    T: bytemuck::NoUninit,
{
    let mut mapped = buffer.slice(..).get_mapped_range_mut().unwrap();
    let mut offset = 0;

    for item in iter {
        let raw_bytes = bytemuck::bytes_of(&item);

        mapped
            .slice(offset..offset + raw_bytes.len())
            .copy_from_slice(raw_bytes);

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
    table1_count: usize,
    table2_count: usize,
    tiles: &'a [RawTile],
    pipeline: &'a wgpu::ComputePipeline,
    bind_group_layout: &'a wgpu::BindGroupLayout,
    payload_buf: &'a wgpu::Buffer,
    offsets_buf: &'a wgpu::Buffer,
    tiles_buf: &'a wgpu::Buffer,
    table_buf: &'a wgpu::Buffer,
    symbol_buf: &'a wgpu::Buffer,
    out_buf: &'a wgpu::Buffer,
}

impl DecompressDispatch<'_> {
    fn run(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let tile_count = u32::try_from(self.tiles.len()).unwrap_or(0);
        let table1_count = u32::try_from(self.table1_count).unwrap_or(0);
        let table2_count = u32::try_from(self.table2_count).unwrap_or(0);
        let params = Params {
            table1_count,
            table2_count,
            tile_count,
            width: self.width,
            stride: self.byte_stride / 4,
            height: self.height,
        };

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
                    resource: self.table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.symbol_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.out_buf.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jkl-wgpu-decode-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_immediates(0, bytemuck::bytes_of(&params));

        let groups_x = tile_count.div_ceil(64);
        if groups_x > 0 {
            cpass.dispatch_workgroups(groups_x, 1, 1);
        }
    }
}
