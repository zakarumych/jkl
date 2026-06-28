use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use image::{ColorType, ImageFormat, RgbImage, RgbaImage};
use jkl::{
    image::{
        Extent, ImageRef, OwnedImage,
        block::{bc1, bc2},
        format::Format,
        quality,
    },
    jackal::image::{Compression, JackalImageReader, Options},
    math::{Rgb8U, Rgba8U},
};
use jkl_wgpu::image::{WgpuImage, blocks::BlockCompressor};

#[derive(Parser, Debug)]
#[command(name = "jkl-cli")]
#[command(about = "JKLI encoder/decoder")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Encode(EncodeArgs),
    Decode(DecodeArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FormatArg {
    #[value(name = "rgb8")]
    Rgb8,
    #[value(name = "bc1")]
    Bc1,
    #[value(name = "bc2")]
    Bc2,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CompressionArg {
    #[value(name = "lz77")]
    Lz77,
    #[value(name = "ans")]
    Ans,
    #[value(name = "lz77+ans")]
    Lz77Ans,
    #[value(name = "rle+ans")]
    RleAns,
}

#[derive(Parser, Debug)]
struct EncodeArgs {
    input: PathBuf,
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[arg(long)]
    format: FormatArg,
    #[arg(long)]
    compression: Option<CompressionArg>,
    /// Measure and print quality metrics (MSE, PSNR, max error, GMSD) after encoding.
    #[arg(long)]
    analyze_compression_quality: bool,
}

#[derive(Parser, Debug)]
struct DecodeArgs {
    input: PathBuf,
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:?}");
        std::process::exit(1);
    }
}

struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    block_compressor: BlockCompressor,
}

impl WgpuContext {
    fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        });

        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        let f = async || {
            for adapter in adapters {
                let fut = adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("jkl-cli-device"),
                    required_features: wgpu::Features::SUBGROUP,
                    ..Default::default()
                });

                if let Ok((device, queue)) = fut.await {
                    return Some((device, queue));
                }
            }

            None
        };

        let (device, queue) = futures::executor::block_on(f())?;
        let block_compressor = BlockCompressor::new(&device);

        Some(WgpuContext {
            device,
            queue,
            block_compressor,
        })
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    let mut cx = WgpuContext::new();
    // let mut cx = None;

    match cli.command {
        Command::Encode(args) => encode_command(args, cx.as_mut()),
        Command::Decode(args) => decode_command(args),
    }
}

fn encode_command(args: EncodeArgs, cx: Option<&mut WgpuContext>) -> Result<()> {
    let output = args
        .output
        .unwrap_or_else(|| default_with_extension(&args.input, "jkli"));

    // BC2 needs RGBA source; other formats use RGB.
    let source = if matches!(args.format, FormatArg::Bc2) {
        load_bc2_source(&args.input)?
    } else {
        load_source(&args.input)?
    };
    let compression = map_compression(args.compression);

    let mut output_file = File::create(&output)
        .with_context(|| format!("failed to create output file: {}", output.display()))?;

    match args.format {
        FormatArg::Rgb8 => {
            let rgb = match source {
                SourceImage::Rgb(image) => image,
                SourceImage::Bc1(image) => bc1_to_rgb8(image.as_ref()),
                SourceImage::Bc2(image) => bc2_to_rgb8(image.as_ref()),
                SourceImage::Rgba(_) => unreachable!(),
            };

            jkl::jackal::image::write_image(
                rgb.as_ref(),
                Options::new().with_compression(compression),
                &mut output_file,
            )
            .with_context(|| format!("failed to write JKLI file: {}", output.display()))?;

            if args.analyze_compression_quality {
                // RGB8 uses lossless compression; decoded is bit-identical to original.
                print_quality_metrics(rgb.as_ref(), rgb.as_ref());
            }
        }
        FormatArg::Bc1 => {
            let (blocks, original) = match source {
                SourceImage::Rgb(image) => {
                    let blocks = rgb8_to_bc1(image.as_ref(), cx);
                    (blocks, Some(image))
                }
                SourceImage::Bc1(image) => (image, None),
                SourceImage::Bc2(image) => {
                    let rgb = bc2_to_rgb8(image.as_ref());
                    let blocks = rgb8_to_bc1(rgb.as_ref(), cx);
                    (blocks, None)
                }
                SourceImage::Rgba(_) => unreachable!(),
            };

            jkl::jackal::image::write_image(
                blocks.as_ref(),
                Options::new().with_compression(compression),
                &mut output_file,
            )
            .with_context(|| format!("failed to write JKLI file: {}", output.display()))?;

            if args.analyze_compression_quality {
                let decoded = bc1_to_rgb8(blocks.as_ref());
                match original {
                    Some(ref orig) => print_quality_metrics(orig.as_ref(), decoded.as_ref()),
                    None => print_quality_metrics(decoded.as_ref(), decoded.as_ref()),
                }
            }
        }
        FormatArg::Bc2 => {
            let (blocks, original) = match source {
                SourceImage::Rgba(image) => {
                    let blocks = rgba8_to_bc2(image.as_ref(), cx);
                    (blocks, Some(image))
                }
                SourceImage::Bc2(image) => (image, None),
                SourceImage::Rgb(_) | SourceImage::Bc1(_) => unreachable!(),
            };

            jkl::jackal::image::write_image(
                blocks.as_ref(),
                Options::new().with_compression(compression),
                &mut output_file,
            )
            .with_context(|| format!("failed to write JKLI file: {}", output.display()))?;

            if args.analyze_compression_quality {
                let decoded = bc2_to_rgb8(blocks.as_ref());
                match original {
                    Some(ref orig) => {
                        let orig_rgb = rgba8_to_rgb8(orig.as_ref());
                        print_quality_metrics(orig_rgb.as_ref(), decoded.as_ref());
                    }
                    None => print_quality_metrics(decoded.as_ref(), decoded.as_ref()),
                }
            }
        }
    }

    Ok(())
}

fn decode_command(args: DecodeArgs) -> Result<()> {
    let output = args
        .output
        .unwrap_or_else(|| default_with_extension(&args.input, "png"));

    let format = output_format_from_path(&output)?;
    let source = load_jkli(&args.input)?;

    match source {
        SourceImage::Rgb(image) => {
            let rgb_ref = image.as_ref();
            let mut bytes = Vec::with_capacity(rgb_ref.width() * rgb_ref.height() * 3);
            for pixel in rgb_ref.iter_pixels() {
                bytes.extend_from_slice(&pixel.bytes());
            }
            let img = RgbImage::from_raw(
                u32::try_from(rgb_ref.width()).context("image width does not fit into u32")?,
                u32::try_from(rgb_ref.height()).context("image height does not fit into u32")?,
                bytes,
            )
            .ok_or_else(|| anyhow!("failed to construct RGB image buffer"))?;
            img.save_with_format(&output, format)
                .with_context(|| format!("failed to write output image: {}", output.display()))?;
        }
        SourceImage::Rgba(_) => unreachable!(),
        SourceImage::Bc1(image) => {
            let rgb = bc1_to_rgb8(image.as_ref());
            let rgb_ref = rgb.as_ref();
            let mut bytes = Vec::with_capacity(rgb_ref.width() * rgb_ref.height() * 3);
            for pixel in rgb_ref.iter_pixels() {
                bytes.extend_from_slice(&pixel.bytes());
            }
            let img = RgbImage::from_raw(
                u32::try_from(rgb_ref.width()).context("image width does not fit into u32")?,
                u32::try_from(rgb_ref.height()).context("image height does not fit into u32")?,
                bytes,
            )
            .ok_or_else(|| anyhow!("failed to construct RGB image buffer"))?;
            img.save_with_format(&output, format)
                .with_context(|| format!("failed to write output image: {}", output.display()))?;
        }
        SourceImage::Bc2(image) => {
            let rgba = bc2_to_rgba8(image.as_ref());
            let rgba_ref = rgba.as_ref();
            let mut bytes = Vec::with_capacity(rgba_ref.width() * rgba_ref.height() * 4);
            for pixel in rgba_ref.iter_pixels() {
                bytes.extend_from_slice(&pixel.bytes());
            }
            let img = RgbaImage::from_raw(
                u32::try_from(rgba_ref.width()).context("image width does not fit into u32")?,
                u32::try_from(rgba_ref.height()).context("image height does not fit into u32")?,
                bytes,
            )
            .ok_or_else(|| anyhow!("failed to construct RGBA image buffer"))?;
            img.save_with_format(&output, format)
                .with_context(|| format!("failed to write output image: {}", output.display()))?;
        }
    }

    Ok(())
}

fn load_source(path: &Path) -> Result<SourceImage> {
    if is_jkli_file(path)? {
        return load_jkli(path);
    }

    load_regular_image(path)
}

fn load_regular_image(path: &Path) -> Result<SourceImage> {
    let image = image::open(path)
        .with_context(|| format!("failed to open image file: {}", path.display()))?;

    match image.color() {
        ColorType::Rgb8 | ColorType::Rgba8 => {}
        color => {
            bail!(
                "unsupported source image color type: {color:?}; only Rgb8 and Rgba8 are supported"
            )
        }
    }

    let rgb = image.to_rgb8();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;

    let mut pixels = Vec::with_capacity(width * height);
    for chunk in rgb.as_raw().chunks_exact(3) {
        pixels.push(Rgb8U::from_bytes([chunk[0], chunk[1], chunk[2]]));
    }

    Ok(SourceImage::Rgb(OwnedImage::new_2d(
        width,
        height,
        pixels.into_boxed_slice(),
    )))
}

fn load_bc2_source(path: &Path) -> Result<SourceImage> {
    if is_jkli_file(path)? {
        return load_jkli(path);
    }
    load_regular_image_rgba(path)
}

fn load_regular_image_rgba(path: &Path) -> Result<SourceImage> {
    let image = image::open(path)
        .with_context(|| format!("failed to open image file: {}", path.display()))?;

    let rgba = image.to_rgba8();
    let width = rgba.width() as usize;
    let height = rgba.height() as usize;

    let mut pixels = Vec::with_capacity(width * height);
    for chunk in rgba.as_raw().chunks_exact(4) {
        pixels.push(Rgba8U::from_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    Ok(SourceImage::Rgba(OwnedImage::new_2d(
        width,
        height,
        pixels.into_boxed_slice(),
    )))
}

fn load_jkli(path: &Path) -> Result<SourceImage> {
    let file = File::open(path)
        .with_context(|| format!("failed to open JKLI file: {}", path.display()))?;
    let mut reader = JackalImageReader::open(file)
        .with_context(|| format!("failed to read JKLI header from: {}", path.display()))?;

    match reader.format() {
        Format::RGB8 => {
            let width = reader.extent().width();
            let height = reader.extent().height();

            let mut image = OwnedImage::new_2d(
                width,
                height,
                vec![Rgb8U::BLACK; width * height].into_boxed_slice(),
            );
            let mut tile_reader = reader
                .tile_reader::<Rgb8U>()
                .context("failed to open RGB8 tile reader")?;

            let tiles_iter = tile_reader
                .tile_size()
                .iter_tiles(Extent::D2 { width, height });
            assert_eq!(tiles_iter.len(), tile_reader.tiles());

            {
                let mut image = image.as_mut();
                let mut pixels = image.get_plane_mut(0);

                for (tile_index, tile) in tiles_iter.enumerate() {
                    if tile.plane != 0 {
                        bail!("only single-plane images are supported");
                    }

                    let tile_image = pixels.get_rect_mut(tile.rect);
                    tile_reader
                        .read_tile(tile_index, tile_image)
                        .with_context(|| format!("failed to read tile {tile_index}"))?;
                }
            }

            Ok(SourceImage::Rgb(image))
        }
        Format::BC1 => {
            let width_blocks = reader.extent().width();
            let height_blocks = reader.extent().height();

            let mut image = OwnedImage::new_2d(
                width_blocks,
                height_blocks,
                vec![bc1::Block::BLACK; width_blocks * height_blocks].into_boxed_slice(),
            );
            let mut tile_reader = reader
                .tile_reader::<bc1::Block>()
                .context("failed to open BC1 tile reader")?;

            let tiles_iter = tile_reader.tile_size().iter_tiles(Extent::D2 {
                width: width_blocks,
                height: height_blocks,
            });
            assert_eq!(tiles_iter.len(), tile_reader.tiles());

            {
                let mut image = image.as_mut();
                let mut blocks = image.get_plane_mut(0);

                for (tile_index, tile) in tiles_iter.enumerate() {
                    if tile.plane != 0 {
                        bail!("only single-plane images are supported");
                    }

                    let tile_image = blocks.get_rect_mut(tile.rect);
                    tile_reader
                        .read_tile(tile_index, tile_image)
                        .with_context(|| format!("failed to read tile {tile_index}"))?;
                }
            }

            Ok(SourceImage::Bc1(image))
        }
        Format::BC2 => {
            let width_blocks = reader.extent().width();
            let height_blocks = reader.extent().height();

            let mut image = OwnedImage::new_2d(
                width_blocks,
                height_blocks,
                vec![bc2::Block::BLACK; width_blocks * height_blocks].into_boxed_slice(),
            );
            let mut tile_reader = reader
                .tile_reader::<bc2::Block>()
                .context("failed to open BC2 tile reader")?;

            let tiles_iter = tile_reader.tile_size().iter_tiles(Extent::D2 {
                width: width_blocks,
                height: height_blocks,
            });
            assert_eq!(tiles_iter.len(), tile_reader.tiles());

            {
                let mut image = image.as_mut();
                let mut blocks = image.get_plane_mut(0);

                for (tile_index, tile) in tiles_iter.enumerate() {
                    if tile.plane != 0 {
                        bail!("only single-plane images are supported");
                    }

                    let tile_image = blocks.get_rect_mut(tile.rect);
                    tile_reader
                        .read_tile(tile_index, tile_image)
                        .with_context(|| format!("failed to read tile {tile_index}"))?;
                }
            }

            Ok(SourceImage::Bc2(image))
        }
        other => bail!("unsupported JKLI format: {other:?}; only RGB8, BC1, and BC2 are supported"),
    }
}

fn map_compression(compression: Option<CompressionArg>) -> Compression {
    match compression {
        None => Compression::None,
        Some(CompressionArg::Lz77) => Compression::Lz77,
        Some(CompressionArg::Ans) => Compression::Ans,
        Some(CompressionArg::Lz77Ans) => Compression::Lz77Ans,
        Some(CompressionArg::RleAns) => Compression::RleAns,
    }
}

fn output_format_from_path(path: &Path) -> Result<ImageFormat> {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .ok_or_else(|| anyhow!("output image extension is required (.bmp, .png, .jpg/.jpeg)"))?;

    match ext.as_str() {
        "bmp" => Ok(ImageFormat::Bmp),
        "png" => Ok(ImageFormat::Png),
        "jpg" | "jpeg" => Ok(ImageFormat::Jpeg),
        _ => bail!("unsupported output format '.{ext}'; use bmp, png, jpg or jpeg"),
    }
}

fn is_jkli_file(path: &Path) -> Result<bool> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open input file: {}", path.display()))?;

    let mut magic = [0u8; 4];
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(magic == *b"JKLI"),
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(error) => {
            Err(error).with_context(|| format!("failed to read input file: {}", path.display()))
        }
    }
}

fn default_with_extension(path: &Path, extension: &str) -> PathBuf {
    let mut output = path.to_path_buf();
    output.set_extension(extension);
    output
}

fn rgb8_to_bc1(input: ImageRef<'_, Rgb8U>, cx: Option<&mut WgpuContext>) -> OwnedImage<bc1::Block> {
    match cx {
        None => {
            let start = std::time::Instant::now();

            let extent = input.extent();
            let dimentions = extent.dimensions();
            let raw_size = extent.raw_size();

            let mut output = OwnedImage::new(
                dimentions,
                [
                    raw_size[0].div_ceil(4),
                    raw_size[1].div_ceil(4),
                    raw_size[2],
                ],
                vec![
                    bc1::Block::BLACK;
                    raw_size[0].div_ceil(4) * raw_size[1].div_ceil(4) * raw_size[2]
                ]
                .into_boxed_slice(),
            );

            bc1::encode_image(input, |c| c.into_f32(), output.as_mut());

            eprintln!(
                "BC1 compression CPU time: {:.3} seconds",
                start.elapsed().as_secs_f32()
            );

            output
        }
        Some(cx) => {
            let start = std::time::Instant::now();

            // Upload RGB image to GPU
            let input_buffer = WgpuImage::upload(
                &cx.device,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::BufferUsages::STORAGE,
                input,
                |c| c.into_opaque().0,
            );

            // Run BC1 compression kernel
            let output_buffer = cx.block_compressor.compress_rgba_to_bc1(
                input_buffer,
                0.0,
                &cx.device,
                &cx.queue,
                1 << 10, // Batch size of 1024 blocks to avoid GPU timeout.
            );

            let mut encoder = cx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jkl-rgba-to-bc1-download-encoder"),
                });

            let download_buffer = WgpuImage::new(
                &cx.device,
                wgpu::TextureFormat::Bc1RgbaUnorm,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                output_buffer.extent(),
            );

            // Copy compressed data to CPU-readable buffer
            output_buffer.copy_to(&download_buffer, &mut encoder);

            // And map it for reading on CPU
            download_buffer.map_on_submit(wgpu::MapMode::Read, &mut encoder);

            let idx = cx.queue.submit(Some(encoder.finish()));
            cx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(idx),
                    timeout: None,
                })
                .unwrap();

            let extent = input.extent();
            let dimentions = extent.dimensions();
            let raw_size = extent.raw_size();

            let mut output = OwnedImage::new(
                dimentions,
                [
                    raw_size[0].div_ceil(4),
                    raw_size[1].div_ceil(4),
                    raw_size[2],
                ],
                vec![
                    bc1::Block::BLACK;
                    raw_size[0].div_ceil(4) * raw_size[1].div_ceil(4) * raw_size[2]
                ]
                .into_boxed_slice(),
            );

            // Read compressed data from GPU.
            download_buffer.download(output.as_mut(), bc1::Block::from_bytes);

            eprintln!(
                "BC1 compression GPU time: {:.3} seconds",
                start.elapsed().as_secs_f32()
            );

            output
        }
    }
}

fn bc1_to_rgb8(input: ImageRef<'_, bc1::Block>) -> OwnedImage<Rgb8U> {
    let extent = input.extent();
    let dimentions = extent.dimensions();
    let raw_size = extent.raw_size();

    let mut output = OwnedImage::new(
        dimentions,
        [raw_size[0] * 4, raw_size[1] * 4, raw_size[2]],
        vec![Rgb8U::BLACK; raw_size[0] * 4 * raw_size[1] * 4 * raw_size[2]].into_boxed_slice(),
    );

    bc1::decode_image(input, |c| Rgb8U::from_f32(c.rgb()), output.as_mut());

    output
}

fn rgba8_to_bc2(
    input: ImageRef<'_, Rgba8U>,
    cx: Option<&mut WgpuContext>,
) -> OwnedImage<bc2::Block> {
    match cx {
        None => {
            let start = std::time::Instant::now();

            let extent = input.extent();
            let dimensions = extent.dimensions();
            let raw_size = extent.raw_size();

            let mut output = OwnedImage::new(
                dimensions,
                [
                    raw_size[0].div_ceil(4),
                    raw_size[1].div_ceil(4),
                    raw_size[2],
                ],
                vec![
                    bc2::Block::BLACK;
                    raw_size[0].div_ceil(4) * raw_size[1].div_ceil(4) * raw_size[2]
                ]
                .into_boxed_slice(),
            );

            bc2::encode_image_with_alpha(input, |c| c.into_f32(), output.as_mut());

            eprintln!(
                "BC2 compression CPU time: {:.3} seconds",
                start.elapsed().as_secs_f32()
            );

            output
        }
        Some(cx) => {
            let start = std::time::Instant::now();

            // Upload RGBA image to GPU
            let input_buffer = WgpuImage::upload(
                &cx.device,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::BufferUsages::STORAGE,
                input,
                |c: Rgba8U| c.0,
            );

            // Run BC2 compression kernel
            let output_buffer = cx.block_compressor.compress_rgba_to_bc2(
                input_buffer,
                &cx.device,
                &cx.queue,
                1 << 10,
            );

            let mut encoder = cx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jkl-rgba-to-bc2-download-encoder"),
                });

            let download_buffer = WgpuImage::new(
                &cx.device,
                wgpu::TextureFormat::Bc2RgbaUnorm,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                output_buffer.extent(),
            );

            output_buffer.copy_to(&download_buffer, &mut encoder);
            download_buffer.map_on_submit(wgpu::MapMode::Read, &mut encoder);

            let idx = cx.queue.submit(Some(encoder.finish()));
            cx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(idx),
                    timeout: None,
                })
                .unwrap();

            let extent = input.extent();
            let dimensions = extent.dimensions();
            let raw_size = extent.raw_size();

            let mut output = OwnedImage::new(
                dimensions,
                [
                    raw_size[0].div_ceil(4),
                    raw_size[1].div_ceil(4),
                    raw_size[2],
                ],
                vec![
                    bc2::Block::BLACK;
                    raw_size[0].div_ceil(4) * raw_size[1].div_ceil(4) * raw_size[2]
                ]
                .into_boxed_slice(),
            );

            download_buffer.download(output.as_mut(), bc2::Block::from_bytes);

            eprintln!(
                "BC2 compression GPU time: {:.3} seconds",
                start.elapsed().as_secs_f32()
            );

            output
        }
    }
}

fn bc2_to_rgb8(input: ImageRef<'_, bc2::Block>) -> OwnedImage<Rgb8U> {
    let extent = input.extent();
    let dimensions = extent.dimensions();
    let raw_size = extent.raw_size();

    let mut output = OwnedImage::new(
        dimensions,
        [raw_size[0] * 4, raw_size[1] * 4, raw_size[2]],
        vec![Rgb8U::BLACK; raw_size[0] * 4 * raw_size[1] * 4 * raw_size[2]].into_boxed_slice(),
    );

    bc2::decode_image(input, |c| Rgb8U::from_f32(c.rgb()), output.as_mut());

    output
}

fn bc2_to_rgba8(input: ImageRef<'_, bc2::Block>) -> OwnedImage<Rgba8U> {
    let extent = input.extent();
    let dimensions = extent.dimensions();
    let raw_size = extent.raw_size();

    let mut output = OwnedImage::new(
        dimensions,
        [raw_size[0] * 4, raw_size[1] * 4, raw_size[2]],
        vec![Rgba8U::BLACK; raw_size[0] * 4 * raw_size[1] * 4 * raw_size[2]].into_boxed_slice(),
    );

    bc2::decode_image(input, Rgba8U::from_f32, output.as_mut());

    output
}

fn rgba8_to_rgb8(input: ImageRef<'_, Rgba8U>) -> OwnedImage<Rgb8U> {
    let extent = input.extent();
    let dimensions = extent.dimensions();
    let raw_size = extent.raw_size();

    let mut output = OwnedImage::new(
        dimensions,
        raw_size,
        vec![Rgb8U::BLACK; raw_size[0] * raw_size[1] * raw_size[2]].into_boxed_slice(),
    );

    for (src, dst) in input.iter_pixels().zip(output.as_mut().iter_pixels_mut()) {
        *dst = src.rgb();
    }

    output
}

fn print_quality_metrics(original: ImageRef<'_, Rgb8U>, decoded: ImageRef<'_, Rgb8U>) {
    let mse_val = quality::mse(original, decoded);
    let psnr_val = quality::psnr_from_mse::<Rgb8U>(mse_val);
    let max_err = quality::max_error(original, decoded);
    let gmsd_val = quality::gmsd(original, decoded);
    println!("MSE:       {:.6}", mse_val);
    if psnr_val.is_finite() {
        println!("PSNR:      {:.2} dB", psnr_val);
    } else {
        println!("PSNR:      inf dB");
    }
    println!("max error: {:.6}", max_err);
    println!("GMSD:      {:.6}", gmsd_val);
}

enum SourceImage {
    Rgb(OwnedImage<Rgb8U>),
    Rgba(OwnedImage<Rgba8U>),
    Bc1(OwnedImage<bc1::Block>),
    Bc2(OwnedImage<bc2::Block>),
}
