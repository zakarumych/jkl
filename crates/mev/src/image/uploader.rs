use std::io;

use jkl::{
    algos::{ans, lz77, vle::Vle},
    image::{Image, format::Format},
    jackal::image::{Compression, JackalImageReader},
    math::{Rgb8U, Rgb565},
};
use mev::{Arguments, BufferUsage};

use crate::image::GpuPixels;

const RANS_WGSL: mev::ShaderSource = mev::include_shader_source!(Wgsl concat!(env!("CARGO_MANIFEST_DIR"), "/../wgpu/shaders/rans.wgsl"));

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
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, mev::AutoDeviceRepr)]
struct Params {
    table1_count: u32,
    table2_count: u32,
    tile_count: u32,
    width: u32,
    height: u32,
    stride: u32,
}

pub struct Uploader {
    rgb8: mev::ComputePipeline,
    rgb8_lz77: mev::ComputePipeline,
    rgba8: mev::ComputePipeline,
    bc1: mev::ComputePipeline,
    bc1_lz77: mev::ComputePipeline,
}

#[derive(mev::Arguments)]
struct UploaderArguments {
    #[mev(compute)]
    #[mev(storage)]
    payload: mev::Buffer,

    #[mev(compute)]
    #[mev(storage)]
    offsets: mev::Buffer,

    #[mev(compute)]
    #[mev(storage)]
    tiles: mev::Buffer,

    #[mev(compute)]
    #[mev(storage)]
    table: mev::Buffer,

    #[mev(compute)]
    #[mev(storage)]
    symbols: mev::Buffer,

    #[mev(compute)]
    #[mev(storage)]
    output: mev::Buffer,
}

impl Uploader {
    /// Create an uploader bound to the given device/queue pair. The
    /// device is cloned internally so the caller may discard its
    /// handle if desired.
    pub fn new(device: &mev::Device) -> Result<Self, mev::OutOfMemory> {
        let shader = device
            .new_shader_library(mev::LibraryDesc {
                name: "jkl-mev-rans-decompress-shader",
                input: RANS_WGSL.into(),
            })
            .expect("Failed to create rans decompress shader library");

        let rgb8 = device
            .new_compute_pipeline(mev::ComputePipelineDesc {
                name: "jkl-mev-rgb8-rans-pipeline",
                shader: mev::Shader {
                    library: shader.clone(),
                    entry: "decompress_rgb8_rans".into(),
                },
                work_group_size: [61, 1, 1],
                constants: size_of::<Params>(),
                arguments: &[UploaderArguments::LAYOUT],
            })
            .expect("Failed to create rgb8 rans pipeline");

        let rgb8_lz77 = device
            .new_compute_pipeline(mev::ComputePipelineDesc {
                name: "jkl-mev-rgb8-lz77-rans-pipeline",
                shader: mev::Shader {
                    library: shader.clone(),
                    entry: "decompress_rgb8_lz77_rans".into(),
                },
                work_group_size: [61, 1, 1],
                constants: size_of::<Params>(),
                arguments: &[UploaderArguments::LAYOUT],
            })
            .expect("Failed to create rgb8 lz77 rans pipeline");

        let rgba8 = device
            .new_compute_pipeline(mev::ComputePipelineDesc {
                name: "jkl-mev-rgba8-rans-pipeline",
                shader: mev::Shader {
                    library: shader.clone(),
                    entry: "decompress_rgba8_rans".into(),
                },
                work_group_size: [61, 1, 1],
                constants: size_of::<Params>(),
                arguments: &[UploaderArguments::LAYOUT],
            })
            .expect("Failed to create rgba8 rans pipeline");

        let bc1 = device
            .new_compute_pipeline(mev::ComputePipelineDesc {
                name: "jkl-mev-bc1-rans-pipeline",
                shader: mev::Shader {
                    library: shader.clone(),
                    entry: "decompress_bc1_rans".into(),
                },
                work_group_size: [61, 1, 1],
                constants: size_of::<Params>(),
                arguments: &[UploaderArguments::LAYOUT],
            })
            .expect("Failed to create bc1 rans pipeline");

        let bc1_lz77 = device
            .new_compute_pipeline(mev::ComputePipelineDesc {
                name: "jkl-mev-bc1-lz77-rans-pipeline",
                shader: mev::Shader {
                    library: shader.clone(),
                    entry: "decompress_bc1_lz77_rans".into(),
                },
                work_group_size: [61, 1, 1],
                constants: size_of::<Params>(),
                arguments: &[UploaderArguments::LAYOUT],
            })
            .expect("Failed to create bc1 lz77 rans pipeline");

        Ok(Uploader {
            rgb8,
            rgb8_lz77,
            rgba8,
            bc1,
            bc1_lz77,
        })
    }

    /// Upload a JKLI image from an existing `JackalReader`.
    pub fn upload_from_reader<R>(
        &self,
        reader: &mut JackalImageReader<R>,
        device: &mev::Device,
        encoder: &mut mev::CommandEncoder,
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
        let mut payload_buf = device.new_buffer(mev::BufferDesc {
            name: "jkl-mev-payload-bytes",
            size: payload_len as usize,
            usage: mev::BufferUsage::STORAGE
                | mev::BufferUsage::TRANSFER_SRC
                | mev::BufferUsage::HOST_WRITE
                | mev::BufferUsage::TRANSIENT,
        });

        let mut offsets = Vec::new();
        let mut tiles = Vec::new();

        if let Ok(mut mapped) = payload_buf.write_mapped_range(..) {
            let slice = mapped.as_mut();

            let mut last_offset = 0;
            for tile_index in 0..reader.tiles() {
                let tile = reader.tile(tile_index);
                let tile_len = reader.tile_payload_len(tile_index) as usize;

                debug_assert!(
                    tile_len % 4 == 0,
                    "tile rANS payload should be multiple of 4"
                );

                reader.copy_tile_payload_into(tile_index, &mut slice[last_offset..][..tile_len])?;

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

        let offsets_buf = device.new_buffer_init(mev::BufferInitDesc {
            name: "jkl-mev-tile-byte-offsets",
            data: bytemuck::cast_slice(&offsets[..]),
            usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
        });
        dbg!(offsets[..].len());

        let tiles_buf = device.new_buffer_init(mev::BufferInitDesc {
            name: "jkl-mev-tiles",
            data: bytemuck::cast_slice(&tiles[..]),
            usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
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
                    mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
                    "jkl-mev-table",
                    table.iter().copied().map(RawFreqEntry::from),
                );

                let symbol_buf = create_buffer_from_iter(
                    device,
                    mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
                    "jkl-mev-symbols",
                    table
                        .iter()
                        .copied()
                        .map(|e| Rgb8U::from_bits_interleaved(e.symbol.0).bits()),
                );

                let byte_stride = (width * 4).div_ceil(256) * 256;
                let output_buffer_size = byte_stride * height;
                let out_buf = device.new_buffer(mev::BufferDesc {
                    name: "jkl-mev-output",
                    size: output_buffer_size as usize,
                    usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_SRC,
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    table1_count: table.len(),
                    table2_count: 0,
                    tiles: &tiles,
                    pipeline: &self.rgb8,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    table_buf: &table_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(encoder);

                Ok(Image::with_stride(
                    jkl::image::Dimensions::D2,
                    [width_usize, height_usize, 1],
                    [byte_stride as usize, byte_stride as usize * height_usize],
                    GpuPixels::new(mev::PixelFormat::Rgba8Unorm, out_buf),
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
                    mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
                    "jkl-mev-table",
                    table.iter().copied().map(RawFreqEntry::from),
                );

                let symbol_buf = create_buffer_from_iter(
                    device,
                    mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
                    "jkl-mev-symbols",
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
                let out_buf = device.new_buffer(mev::BufferDesc {
                    name: "jkl-mev-output",
                    size: output_buffer_size as usize,
                    usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_SRC,
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    table1_count: table.len(),
                    table2_count: 0,
                    tiles: &tiles,
                    pipeline: &self.rgb8_lz77,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    table_buf: &table_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(encoder);

                Ok(Image::with_stride(
                    jkl::image::Dimensions::D2,
                    [width_usize, height_usize, 1],
                    [byte_stride as usize, byte_stride as usize * height_usize],
                    GpuPixels::new(mev::PixelFormat::Rgba8Unorm, out_buf),
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
                    mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
                    "jkl-mev-table",
                    table.iter().copied().map(RawFreqEntry::from),
                );

                let symbol_buf = create_buffer_from_iter(
                    device,
                    mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
                    "jkl-mev-symbols",
                    table
                        .iter()
                        .copied()
                        .map(|e| Rgb8U::from_bits_interleaved(e.symbol.0).bits()),
                );

                let byte_stride = (width * 4).div_ceil(256) * 256;
                let output_buffer_size = byte_stride as u64 * height as u64;
                let out_buf = device.new_buffer(mev::BufferDesc {
                    name: "jkl-mev-output",
                    size: output_buffer_size as usize,
                    usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_SRC,
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    table1_count: table.len(),
                    table2_count: 0,
                    tiles: &tiles,
                    pipeline: &self.rgba8,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    table_buf: &table_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(encoder);

                Ok(Image::with_stride(
                    jkl::image::Dimensions::D2,
                    [width_usize, height_usize, 1],
                    [byte_stride as usize, byte_stride as usize * height_usize],
                    GpuPixels::new(mev::PixelFormat::Rgba8Unorm, out_buf),
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
                    mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
                    "jkl-mev-table",
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
                    mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
                    "jkl-mev-symbols",
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
                let out_buf = device.new_buffer(mev::BufferDesc {
                    name: "jkl-mev-output",
                    size: output_buffer_size as usize,
                    usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_SRC,
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    table1_count: colors_table.len(),
                    table2_count: indices_table.len(),
                    tiles: &tiles,
                    pipeline: &self.bc1,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    table_buf: &table_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(encoder);

                Ok(Image::with_stride(
                    jkl::image::Dimensions::D2,
                    [width_usize, height_usize, 1],
                    [byte_stride as usize, byte_stride as usize * height_usize],
                    GpuPixels::new(mev::PixelFormat::Bc1RgbaUnorm, out_buf),
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
                    mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
                    "jkl-mev-table",
                    colors_table
                        .iter()
                        .copied()
                        .map(RawFreqEntry::from)
                        .chain(indices_table.iter().copied().map(RawFreqEntry::from)),
                );

                let symbol_buf = create_buffer_from_iter(
                    device,
                    mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSIENT,
                    "jkl-mev-symbols",
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
                let out_buf = device.new_buffer(mev::BufferDesc {
                    name: "jkl-mev-output",
                    size: output_buffer_size as usize,
                    usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_SRC,
                });

                DecompressDispatch {
                    width,
                    height,
                    byte_stride,
                    table1_count: colors_table.len(),
                    table2_count: indices_table.len(),
                    tiles: &tiles,
                    pipeline: &self.bc1_lz77,
                    payload_buf: &payload_buf,
                    offsets_buf: &offsets_buf,
                    tiles_buf: &tiles_buf,
                    table_buf: &table_buf,
                    symbol_buf: &symbol_buf,
                    out_buf: &out_buf,
                }
                .run(encoder);

                Ok(Image::with_stride(
                    jkl::image::Dimensions::D2,
                    [width_usize, height_usize, 1],
                    [byte_stride as usize, byte_stride as usize * height_usize],
                    GpuPixels::new(mev::PixelFormat::Bc1RgbaUnorm, out_buf),
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

fn write_buffer_from_iter<T>(buffer: &mut mev::Buffer, iter: impl IntoIterator<Item = T>) -> usize
where
    T: bytemuck::NoUninit,
{
    if buffer.map(..).is_err() {
        return 0;
    };

    let mut offset = 0;
    if let Ok(mut mapped) = buffer.write_mapped_range(..) {
        let slice = mapped.as_mut();

        for item in iter {
            let raw_bytes = bytemuck::bytes_of(&item);
            slice[offset..][..raw_bytes.len()].copy_from_slice(raw_bytes);
            offset += raw_bytes.len();
        }
    }

    offset
}

fn create_buffer_from_iter<T>(
    device: &mev::Device,
    usage: BufferUsage,
    label: &str,
    iter: impl Iterator<Item = T>,
) -> mev::Buffer
where
    T: bytemuck::NoUninit,
{
    let (lower, upper) = iter.size_hint();
    assert_eq!(Some(lower), upper);

    let size = u64::try_from(lower * std::mem::size_of::<T>()).expect("size exceedes u64");

    let mut buffer = device.new_buffer(mev::BufferDesc {
        name: label,
        size: size as usize,
        usage: usage | mev::BufferUsage::HOST_WRITE,
    });

    write_buffer_from_iter(&mut buffer, iter);

    buffer
}

#[derive(mev::Arguments)]
struct DecompressArguments {
    #[mev(compute)]
    #[mev(storage)]
    payload_words: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    offsets: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    tiles: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    table: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    symbols: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    output: mev::Buffer,
}

struct DecompressDispatch<'a> {
    width: u32,
    height: u32,
    byte_stride: u32,
    table1_count: usize,
    table2_count: usize,
    tiles: &'a [RawTile],
    pipeline: &'a mev::ComputePipeline,
    payload_buf: &'a mev::Buffer,
    offsets_buf: &'a mev::Buffer,
    tiles_buf: &'a mev::Buffer,
    table_buf: &'a mev::Buffer,
    symbol_buf: &'a mev::Buffer,
    out_buf: &'a mev::Buffer,
}

impl DecompressDispatch<'_> {
    fn run(&self, encoder: &mut mev::CommandEncoder) {
        let tile_count = u32::try_from(self.tiles.len()).unwrap_or(0);
        if tile_count == 0 {
            return;
        }

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

        let arguments = DecompressArguments {
            payload_words: self.payload_buf.clone(),
            offsets: self.offsets_buf.clone(),
            tiles: self.tiles_buf.clone(),
            table: self.table_buf.clone(),
            symbols: self.symbol_buf.clone(),
            output: self.out_buf.clone(),
        };

        let mut compute_encoder = encoder.compute();
        compute_encoder.with_pipeline(self.pipeline);
        compute_encoder.with_arguments(0, &arguments);
        compute_encoder.with_constants(&params);

        let groups_x = tile_count.div_ceil(64);
        compute_encoder.dispatch(mev::Extent3::new(groups_x, 1, 1));
    }
}
