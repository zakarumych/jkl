use jkl::{
    image::{
        Extent, ImageRef, OwnedImage,
        block::{bc1, bc2, bc3, bc4, bc5, bc6, bc7},
        format::Format,
        tiles::TileSize,
    },
    jackal::image::{Compression, JackalImageReader, Options, Pixel, write_image},
    math::{R8U, R32F, Rg8U, Rg32F, Rgb8U, Rgb32F, Rgba8U, Rgba32F},
};
use std::{io::Cursor, time::Instant};

pub const FORMATS: &[Format] = &[
    Format::R8,
    Format::RG8,
    Format::RGB8,
    Format::RGBA8,
    Format::BC1,
    Format::BC2,
    Format::BC3,
    Format::BC4,
    Format::BC5,
    Format::BC6,
    Format::BC7,
];
pub const COMPRESSIONS: &[Compression] = &[
    Compression::None,
    Compression::Lz77,
    Compression::Ans,
    Compression::Lz77Ans,
    Compression::RleAns,
];

#[derive(Clone, Debug, PartialEq)]
pub struct TextureSettings {
    pub format: Format,
    pub compression: Compression,
    pub fixed: bool,
    pub tile_width: u16,
    pub tile_height: u16,
    pub flat_cost: f32,
    pub size_cost: f32,
    pub alpha_threshold: f32,
}
impl Default for TextureSettings {
    fn default() -> Self {
        Self {
            format: Format::BC1,
            compression: Compression::Ans,
            fixed: false,
            tile_width: 64,
            tile_height: 64,
            flat_cost: 64.0,
            size_cost: 1.0,
            alpha_threshold: 0.5,
        }
    }
}
impl TextureSettings {
    pub fn options(&self) -> Result<Options, String> {
        let options = Options::new().with_compression(self.compression);
        if self.fixed {
            if self.tile_width == 0
                || self.tile_height == 0
                || self.tile_width % self.format.tile_width_granularity() != 0
                || self.tile_height % self.format.tile_height_granularity() != 0
            {
                return Err(format!(
                    "Tile dimensions must be positive multiples of {} × {}.",
                    self.format.tile_width_granularity(),
                    self.format.tile_height_granularity()
                ));
            }
            Ok(options.with_tile_size(TileSize {
                width: self.tile_width,
                height: self.tile_height,
            }))
        } else {
            if !self.flat_cost.is_finite()
                || self.flat_cost < 0.0
                || !self.size_cost.is_finite()
                || self.size_cost <= 0.0
            {
                return Err("Tile costs must be finite; size cost must be positive.".into());
            }
            Ok(options.with_optimal_tile_size(self.flat_cost, self.size_cost))
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextureSource {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[f32; 4]>,
}
use TextureSettings as Settings;
use TextureSource as Source;
pub struct TextureEncoded {
    pub bytes: Vec<u8>,
    pub decoded: Vec<[f32; 4]>,
    pub psnr: f64,
    pub seconds: f64,
}
use TextureEncoded as Encoded;
fn byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}
fn check_size(width: usize, height: usize) -> Result<(), String> {
    if width == 0
        || height == 0
        || width
            .checked_mul(height)
            .is_none_or(|n| n > 64 * 1024 * 1024)
    {
        return Err("Texture must have 1–67,108,864 pixels.".into());
    }
    Ok(())
}
fn serialize<T: Pixel>(
    width: usize,
    height: usize,
    data: &[T],
    settings: &Settings,
) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor::new(Vec::new());
    write_image(
        ImageRef::new_2d(width, height, data),
        settings.options()?,
        &mut cursor,
    )
    .map_err(|e| e.to_string())?;
    Ok(cursor.into_inner())
}
fn blocks<T>(source: &Source, f: impl Fn([[(Rgba32F, bool); 4]; 4]) -> T) -> Vec<T> {
    let mut out = Vec::new();
    for by in 0..source.height.div_ceil(4) {
        for bx in 0..source.width.div_ceil(4) {
            let colors = std::array::from_fn(|y| {
                std::array::from_fn(|x| {
                    let px = bx * 4 + x;
                    let py = by * 4 + y;
                    (
                        Rgba32F(
                            source.pixels[py.min(source.height - 1) * source.width
                                + px.min(source.width - 1)],
                        ),
                        px < source.width && py < source.height,
                    )
                })
            });
            out.push(f(colors));
        }
    }
    out
}
fn expand<T: Copy>(
    width: usize,
    height: usize,
    data: &[T],
    f: impl Fn(T) -> [[[f32; 4]; 4]; 4],
) -> Vec<[f32; 4]> {
    let bw = width.div_ceil(4);
    let mut pixels = vec![[0.0; 4]; width * height];
    for (i, &block) in data.iter().enumerate() {
        let decoded = f(block);
        for y in 0..4 {
            for x in 0..4 {
                let px = i % bw * 4 + x;
                let py = i / bw * 4 + y;
                if px < width && py < height {
                    pixels[py * width + px] = decoded[y][x];
                }
            }
        }
    }
    pixels
}
pub fn encode(source: &Source, settings: Settings) -> Result<Encoded, String> {
    check_size(source.width, source.height)?;
    if source.pixels.len() != source.width * source.height {
        return Err("Pixel count does not match texture dimensions.".into());
    }
    if source.pixels.iter().flatten().any(|v| !v.is_finite()) {
        return Err("Texture pixels must be finite.".into());
    }
    if !settings.alpha_threshold.is_finite() || !(0.0..=1.0).contains(&settings.alpha_threshold) {
        return Err("Alpha cutoff must be between 0 and 1.".into());
    }
    settings.options()?;
    let start = Instant::now();
    let w = source.width;
    let h = source.height;
    let bw = w.div_ceil(4);
    let bh = h.div_ceil(4);
    let (bytes, decoded) = match settings.format {
        Format::R8 => {
            let data: Vec<_> = source.pixels.iter().map(|p| R8U::new(byte(p[0]))).collect();
            (
                serialize(w, h, &data, &settings)?,
                data.iter()
                    .map(|p| {
                        let v = p.bits() as f32 / 255.0;
                        [v, v, v, 1.0]
                    })
                    .collect(),
            )
        }
        Format::RG8 => {
            let data: Vec<_> = source
                .pixels
                .iter()
                .map(|p| Rg8U::new(byte(p[0]), byte(p[1])))
                .collect();
            (
                serialize(w, h, &data, &settings)?,
                data.iter()
                    .map(|p| [p.0[0] as f32 / 255.0, p.0[1] as f32 / 255.0, 0.0, 1.0])
                    .collect(),
            )
        }
        Format::RGB8 => {
            let data: Vec<_> = source
                .pixels
                .iter()
                .map(|p| Rgb8U::new(byte(p[0]), byte(p[1]), byte(p[2])))
                .collect();
            (
                serialize(w, h, &data, &settings)?,
                data.iter()
                    .map(|p| {
                        [
                            p.0[0] as f32 / 255.0,
                            p.0[1] as f32 / 255.0,
                            p.0[2] as f32 / 255.0,
                            1.0,
                        ]
                    })
                    .collect(),
            )
        }
        Format::RGBA8 => {
            let data: Vec<_> = source.pixels.iter().map(|p| Rgba8U(p.map(byte))).collect();
            (
                serialize(w, h, &data, &settings)?,
                data.iter().map(|p| p.0.map(|v| v as f32 / 255.0)).collect(),
            )
        }
        Format::BC1 => {
            let data = blocks(source, |c| {
                bc1::Block::encode_with_alpha(c, settings.alpha_threshold)
            });
            (
                serialize(bw, bh, &data, &settings)?,
                expand(w, h, &data, |b| {
                    b.decode_with_alpha().map(|r| r.map(|p| p.0))
                }),
            )
        }
        Format::BC2 => {
            let data = blocks(source, bc2::Block::encode_with_alpha);
            (
                serialize(bw, bh, &data, &settings)?,
                expand(w, h, &data, |b| {
                    b.decode_with_alpha().map(|r| r.map(|p| p.0))
                }),
            )
        }
        Format::BC3 => {
            let data = blocks(source, bc3::Block::encode_with_alpha);
            (
                serialize(bw, bh, &data, &settings)?,
                expand(w, h, &data, |b| {
                    b.decode_with_alpha().map(|r| r.map(|p| p.0))
                }),
            )
        }
        Format::BC4 => {
            let data = blocks(source, |c| {
                bc4::Block::encode(c.map(|r| r.map(|(p, m)| (R32F(p.0[0]), m))))
            });
            (
                serialize(bw, bh, &data, &settings)?,
                expand(w, h, &data, |b| {
                    b.decode().map(|r| r.map(|p| [p.0, p.0, p.0, 1.0]))
                }),
            )
        }
        Format::BC5 => {
            let data = blocks(source, |c| {
                bc5::Block::encode(c.map(|r| r.map(|(p, m)| (Rg32F([p.0[0], p.0[1]]), m))))
            });
            (
                serialize(bw, bh, &data, &settings)?,
                expand(w, h, &data, |b| {
                    b.decode().map(|r| r.map(|p| [p.0[0], p.0[1], 0.0, 1.0]))
                }),
            )
        }
        Format::BC6 => {
            let data = blocks(source, |c| {
                bc6::Block::encode(c.map(|r| r.map(|(p, m)| (Rgb32F([p.0[0], p.0[1], p.0[2]]), m))))
            });
            (
                serialize(bw, bh, &data, &settings)?,
                expand(w, h, &data, |b| {
                    b.decode().map(|r| r.map(|p| [p.0[0], p.0[1], p.0[2], 1.0]))
                }),
            )
        }
        Format::BC7 => {
            let data = blocks(source, bc7::Block::encode);
            (
                serialize(bw, bh, &data, &settings)?,
                expand(w, h, &data, |b| b.decode().map(|r| r.map(|p| p.0))),
            )
        }
    };
    let (decoded_width, decoded_height, reopened, _) = decode_jkli(&bytes)?;
    if decoded_width < w || decoded_height < h {
        return Err("Encoded texture has invalid reconstructed dimensions.".into());
    }
    let actual: Vec<_> = (0..h)
        .flat_map(|y| {
            reopened[y * decoded_width..y * decoded_width + w]
                .iter()
                .copied()
        })
        .collect();
    if actual != decoded {
        return Err("Encoded texture failed reconstruction verification.".into());
    }
    let decoded = actual;
    let squared: f64 = source
        .pixels
        .iter()
        .zip(&decoded)
        .flat_map(|(a, b)| (0..4).map(move |c| ((a[c] - b[c]) as f64).powi(2)))
        .sum();
    let mse = squared / (w * h * 4) as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        -10.0 * mse.log10()
    };
    Ok(Encoded {
        bytes,
        decoded,
        psnr,
        seconds: start.elapsed().as_secs_f64(),
    })
}

fn read_pixels<T: Pixel>(
    reader: &mut JackalImageReader<Cursor<&[u8]>>,
    zero: T,
) -> Result<OwnedImage<T>, String> {
    let extent = reader.extent();
    let w = extent.width();
    let h = extent.height();
    check_size(w, h)?;
    let mut image = OwnedImage::new_2d(w, h, vec![zero; w * h].into_boxed_slice());
    let mut tiles = reader.tile_reader::<T>().map_err(|e| e.to_string())?;
    let regions = tiles.tile_size().iter_tiles(extent);
    if regions.len() != tiles.tiles() {
        return Err("Invalid tile count.".into());
    }
    let mut destination = image.as_mut();
    let mut plane = destination.get_plane_mut(0);
    for (i, tile) in regions.enumerate() {
        tiles
            .read_tile(i, plane.get_rect_mut(tile.rect))
            .map_err(|e| e.to_string())?;
    }
    Ok(image)
}
pub fn decode_jkli(bytes: &[u8]) -> Result<(usize, usize, Vec<[f32; 4]>, String), String> {
    let mut reader = JackalImageReader::open(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let Extent::D2 {
        width: w,
        height: h,
    } = reader.extent()
    else {
        return Err("Only 2D single-plane textures are supported.".into());
    };
    check_size(w, h)?;
    let format = reader.format();
    let info = format!(
        "Decoded JKLI {format:?}, {:?}, {} tiles",
        reader.compression(),
        reader.tiles()
    );
    let (width, height, pixels) = match format {
        Format::R8 => (
            w,
            h,
            read_pixels(&mut reader, R8U::new(0))?
                .data()
                .iter()
                .map(|p| {
                    let v = p.bits() as f32 / 255.0;
                    [v, v, v, 1.0]
                })
                .collect(),
        ),
        Format::RG8 => (
            w,
            h,
            read_pixels(&mut reader, Rg8U::new(0, 0))?
                .data()
                .iter()
                .map(|p| [p.0[0] as f32 / 255.0, p.0[1] as f32 / 255.0, 0.0, 1.0])
                .collect(),
        ),
        Format::RGB8 => (
            w,
            h,
            read_pixels(&mut reader, Rgb8U::BLACK)?
                .data()
                .iter()
                .map(|p| {
                    [
                        p.0[0] as f32 / 255.0,
                        p.0[1] as f32 / 255.0,
                        p.0[2] as f32 / 255.0,
                        1.0,
                    ]
                })
                .collect(),
        ),
        Format::RGBA8 => (
            w,
            h,
            read_pixels(&mut reader, Rgba8U([0; 4]))?
                .data()
                .iter()
                .map(|p| p.0.map(|v| v as f32 / 255.0))
                .collect(),
        ),
        Format::BC1 => {
            check_size(w * 4, h * 4)?;
            let data = read_pixels(&mut reader, bc1::Block::BLACK)?;
            (
                w * 4,
                h * 4,
                expand(w * 4, h * 4, data.data(), |b| {
                    b.decode_with_alpha().map(|r| r.map(|p| p.0))
                }),
            )
        }
        Format::BC2 => {
            check_size(w * 4, h * 4)?;
            let data = read_pixels(&mut reader, bc2::Block::BLACK)?;
            (
                w * 4,
                h * 4,
                expand(w * 4, h * 4, data.data(), |b| {
                    b.decode_with_alpha().map(|r| r.map(|p| p.0))
                }),
            )
        }
        Format::BC3 => {
            check_size(w * 4, h * 4)?;
            let data = read_pixels(&mut reader, bc3::Block::BLACK)?;
            (
                w * 4,
                h * 4,
                expand(w * 4, h * 4, data.data(), |b| {
                    b.decode_with_alpha().map(|r| r.map(|p| p.0))
                }),
            )
        }
        Format::BC4 => {
            check_size(w * 4, h * 4)?;
            let data = read_pixels(&mut reader, bc4::Block::BLACK)?;
            (
                w * 4,
                h * 4,
                expand(w * 4, h * 4, data.data(), |b| {
                    b.decode().map(|r| r.map(|p| [p.0, p.0, p.0, 1.0]))
                }),
            )
        }
        Format::BC5 => {
            check_size(w * 4, h * 4)?;
            let data = read_pixels(&mut reader, bc5::Block::BLACK)?;
            (
                w * 4,
                h * 4,
                expand(w * 4, h * 4, data.data(), |b| {
                    b.decode().map(|r| r.map(|p| [p.0[0], p.0[1], 0.0, 1.0]))
                }),
            )
        }
        Format::BC6 => {
            check_size(w * 4, h * 4)?;
            let data = read_pixels(&mut reader, bc6::Block::BLACK)?;
            (
                w * 4,
                h * 4,
                expand(w * 4, h * 4, data.data(), |b| {
                    b.decode().map(|r| r.map(|p| [p.0[0], p.0[1], p.0[2], 1.0]))
                }),
            )
        }
        Format::BC7 => {
            check_size(w * 4, h * 4)?;
            let data = read_pixels(&mut reader, bc7::Block::BLACK)?;
            (
                w * 4,
                h * 4,
                expand(w * 4, h * 4, data.data(), |b| {
                    b.decode().map(|r| r.map(|p| p.0))
                }),
            )
        }
    };
    Ok((width, height, pixels, info))
}
