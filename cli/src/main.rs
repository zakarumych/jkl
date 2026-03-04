use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use image::{ColorType, ImageFormat, RgbImage};
use jkl::{
    image::{block::bc1::Block, format::Format, Extent, Image2DMut, Image2DRef, ImageRef},
    jackal::image::{Compression, JackalReader, Options},
    math::Rgb8U,
};

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

fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Encode(args) => encode_command(args),
        Command::Decode(args) => decode_command(args),
    }
}

fn encode_command(args: EncodeArgs) -> Result<()> {
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
                rgb.as_image_ref(),
                Options::new().with_compression(compression),
                &mut output_file,
            )
            .with_context(|| format!("failed to write JKLI file: {}", output.display()))?;
        }
        FormatArg::Bc1 => {
            let blocks = match source {
                SourceImage::Rgb(image) => rgb8_to_bc1(image.as_ref()),
                SourceImage::Bc1(image) => image,
            };

            jkl::jackal::image::write_image(
                blocks.as_image_ref(),
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

    Ok(SourceImage::Rgb(OwnedImage2D::new(width, height, pixels)))
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

            let mut image = OwnedImage2D::new(width, height, vec![Rgb8U::BLACK; width * height]);
            let mut tile_reader = reader
                .pixel_reader::<Rgb8U>()
                .context("failed to open RGB8 tile reader")?;

            let tiles_iter = tile_reader
                .tile_size()
                .iter_tiles(Extent::D2 { width, height });
            assert_eq!(tiles_iter.len(), tile_reader.tiles());

            let mut pixels = image.as_mut();

            for (tile_index, tile) in tiles_iter.enumerate() {
                if tile.plane != 0 {
                    bail!("only single-plane images are supported");
                }

                let tile_image = pixels.get_rect_mut(tile.rect);
                tile_reader
                    .read_tile(tile_index, tile_image)
                    .with_context(|| format!("failed to read tile {tile_index}"))?;
            }

            Ok(SourceImage::Rgb(image))
        }
        Format::BC1 => {
            let width_blocks = reader.extent().width();
            let height_blocks = reader.extent().height();

            let mut image = OwnedImage2D::new(
                width_blocks,
                height_blocks,
                vec![Block::BLACK; width_blocks * height_blocks],
            );
            let mut tile_reader = reader
                .pixel_reader::<Block>()
                .context("failed to open BC1 tile reader")?;

            let tiles_iter = tile_reader.tile_size().iter_tiles(Extent::D2 {
                width: width_blocks,
                height: height_blocks,
            });
            assert_eq!(tiles_iter.len(), tile_reader.tiles());

            let mut blocks = image.as_mut();

            for (tile_index, tile) in tiles_iter.enumerate() {
                if tile.plane != 0 {
                    bail!("only single-plane images are supported");
                }

                let tile_image = blocks.get_rect_mut(tile.rect);
                tile_reader
                    .read_tile(tile_index, tile_image)
                    .with_context(|| format!("failed to read tile {tile_index}"))?;
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

fn rgb8_to_bc1(input: Image2DRef<'_, Rgb8U>) -> OwnedImage2D<Block> {
    let block_width = input.width().div_ceil(4);
    let block_height = input.height().div_ceil(4);

    let mut blocks = Vec::with_capacity(block_width * block_height);

    for by in 0..block_height {
        for bx in 0..block_width {
            let x = bx * 4;
            let y = by * 4;

            let colors = if x + 4 <= input.width() && y + 4 <= input.height() {
                input
                    .get_range(x, y, 4, 4)
                    .into_matrix::<4, 4>()
                    .map(|row| row.map(|c| c.into_f32()))
            } else {
                let mut colors = [[Rgb8U::BLACK.into_f32(); 4]; 4];
                for (dy, row) in colors.iter_mut().enumerate() {
                    for (dx, color) in row.iter_mut().enumerate() {
                        let sx = (x + dx).min(input.width() - 1);
                        let sy = (y + dy).min(input.height() - 1);
                        *color = input.get(sx, sy).into_f32();
                    }
                }
                colors
            };

            blocks.push(Block::encode(colors));
        }
    }

    OwnedImage2D::new(block_width, block_height, blocks)
}

fn bc1_to_rgb8(input: Image2DRef<'_, Block>) -> OwnedImage2D<Rgb8U> {
    let width = input.width() * 4;
    let height = input.height() * 4;
    let mut output = OwnedImage2D::new(width, height, vec![Rgb8U::BLACK; width * height]);
    let mut pixels = output.as_mut();

    for by in 0..input.height() {
        for bx in 0..input.width() {
            let block = *input.get(bx, by);
            let decoded = block.decode().map(|row| row.map(Rgb8U::from_f32));
            let mut tile = pixels.get_range_mut(bx * 4, by * 4, 4, 4);
            tile.copy_from_matrix(&decoded);
        }
    }

    output
}

enum SourceImage {
    Rgb(OwnedImage2D<Rgb8U>),
    Bc1(OwnedImage2D<Block>),
}

struct OwnedImage2D<T> {
    width: usize,
    height: usize,
    pixels: Vec<T>,
}

impl<T> OwnedImage2D<T> {
    fn new(width: usize, height: usize, pixels: Vec<T>) -> Self {
        assert_eq!(pixels.len(), width * height);
        Self {
            width,
            height,
            pixels,
        }
    }

    fn as_ref(&self) -> Image2DRef<'_, T> {
        Image2DRef::new(self.width, self.height, &self.pixels)
    }

    fn as_mut(&mut self) -> Image2DMut<'_, T> {
        Image2DMut::new(self.width, self.height, &mut self.pixels)
    }

    fn as_image_ref(&self) -> ImageRef<'_, T> {
        ImageRef::new_2d(self.width, self.height, &self.pixels)
    }
}
