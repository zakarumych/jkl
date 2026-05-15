use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use image::{ColorType, ImageFormat, RgbImage};
use jkl::{
    image::{
        Extent, ImageRef, OwnedImage,
        block::bc1::{self, Block},
        format::Format,
    },
    jackal::image::{Compression, JackalReader, Options},
    math::{Rgb8U, Rgba8U},
};
use jkl_wgpu::image::{WgpuImage, WgpuPixels, blocks::BlockCompressor};

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

    // let mut cx = WgpuContext::new();
    let mut cx = None;

    match cli.command {
        Command::Encode(args) => encode_command(args, cx.as_mut()),
        Command::Decode(args) => decode_command(args),
    }
}

fn encode_command(args: EncodeArgs, cx: Option<&mut WgpuContext>) -> Result<()> {
    let output = args
        .output
        .unwrap_or_else(|| default_with_extension(&args.input, "jkli"));

    let source = load_source(&args.input)?;
    let compression = map_compression(args.compression);

    let mut output_file = File::create(&output)
        .with_context(|| format!("failed to create output file: {}", output.display()))?;

    match args.format {
        FormatArg::Rgb8 => {
            let rgb = match source {
                SourceImage::Rgb(image) => image,
                SourceImage::Bc1(image) => bc1_to_rgb8(image.as_ref()),
            };

            jkl::jackal::image::write_image(
                rgb.as_ref(),
                Options::new().with_compression(compression),
                &mut output_file,
            )
            .with_context(|| format!("failed to write JKLI file: {}", output.display()))?;
        }
        FormatArg::Bc1 => {
            let blocks = match source {
                SourceImage::Rgb(image) => rgb8_to_bc1(image.as_ref(), cx),
                SourceImage::Bc1(image) => image,
            };

            jkl::jackal::image::write_image(
                blocks.as_ref(),
                Options::new().with_compression(compression),
                &mut output_file,
            )
            .with_context(|| format!("failed to write JKLI file: {}", output.display()))?;
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

    let rgb = match source {
        SourceImage::Rgb(image) => image,
        SourceImage::Bc1(image) => bc1_to_rgb8(image.as_ref()),
    };

    let rgb_ref = rgb.as_ref();
    let mut bytes = Vec::with_capacity(rgb_ref.width() * rgb_ref.height() * 3);
    for pixel in rgb_ref.iter_pixels() {
        bytes.extend_from_slice(&pixel.bytes());
    }

    let image = RgbImage::from_raw(
        u32::try_from(rgb_ref.width()).context("image width does not fit into u32")?,
        u32::try_from(rgb_ref.height()).context("image height does not fit into u32")?,
        bytes,
    )
    .ok_or_else(|| anyhow!("failed to construct RGB image buffer"))?;

    image
        .save_with_format(&output, format)
        .with_context(|| format!("failed to write output image: {}", output.display()))?;

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

fn load_jkli(path: &Path) -> Result<SourceImage> {
    let file = File::open(path)
        .with_context(|| format!("failed to open JKLI file: {}", path.display()))?;
    let mut reader = JackalReader::open(file)
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
                .pixel_reader::<Rgb8U>()
                .context("failed to open RGB8 tile reader")?;

            let tiles_iter = tile_reader
                .tile_size()
                .iter_tiles(Extent::D2 { width, height });
            assert_eq!(tiles_iter.len(), tile_reader.tiles());

            {
                let mut image = image.as_mut();
                let mut pixels = image.plane_mut(0);

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
                vec![Block::BLACK; width_blocks * height_blocks].into_boxed_slice(),
            );
            let mut tile_reader = reader
                .pixel_reader::<Block>()
                .context("failed to open BC1 tile reader")?;

            let tiles_iter = tile_reader.tile_size().iter_tiles(Extent::D2 {
                width: width_blocks,
                height: height_blocks,
            });
            assert_eq!(tiles_iter.len(), tile_reader.tiles());

            {
                let mut image = image.as_mut();
                let mut blocks = image.plane_mut(0);

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
        other => bail!("unsupported JKLI format: {other:?}; only RGB8 and BC1 are supported"),
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

fn rgb8_to_bc1(input: ImageRef<'_, Rgb8U>, cx: Option<&mut WgpuContext>) -> OwnedImage<Block> {
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

fn bc1_to_rgb8(input: ImageRef<'_, Block>) -> OwnedImage<Rgb8U> {
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

enum SourceImage {
    Rgb(OwnedImage<Rgb8U>),
    Bc1(OwnedImage<Block>),
}
