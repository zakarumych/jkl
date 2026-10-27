//! Texture workbench. Codec work and file I/O run on a single background job.
use eframe::egui;
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
use std::{
    io::{Cursor, Read},
    path::PathBuf,
    sync::{Arc, mpsc},
    time::Instant,
};

const FORMATS: &[Format] = &[
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
const COMPRESSIONS: &[Compression] = &[
    Compression::None,
    Compression::Lz77,
    Compression::Ans,
    Compression::Lz77Ans,
    Compression::RleAns,
];

#[derive(Clone, PartialEq)]
struct Settings {
    format: Format,
    compression: Compression,
    fixed: bool,
    tile_width: u16,
    tile_height: u16,
    flat_cost: f32,
    size_cost: f32,
    alpha_threshold: f32,
}
impl Default for Settings {
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
impl Settings {
    fn options(&self) -> Result<Options, String> {
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

struct Source {
    name: String,
    width: usize,
    height: usize,
    pixels: Vec<[f32; 4]>,
    preview: egui::ColorImage,
    info: String,
}
struct Encoded {
    bytes: Vec<u8>,
    decoded: Vec<[f32; 4]>,
    preview: egui::ColorImage,
    error: egui::ColorImage,
    psnr: f64,
    seconds: f64,
    settings: Settings,
}
enum Completion {
    Loaded(Source),
    Encoded(Encoded),
    Saved(String),
}

pub struct TextureEditor {
    source: Option<Arc<Source>>,
    settings: Settings,
    result: Option<Arc<Encoded>>,
    job: Option<mpsc::Receiver<Result<Completion, String>>>,
    status: String,
    textures: [Option<egui::TextureHandle>; 3],
    view: usize,
    zoom: f32,
    fit: bool,
}
impl Default for TextureEditor {
    fn default() -> Self {
        Self {
            source: None,
            settings: Settings::default(),
            result: None,
            job: None,
            status: String::new(),
            textures: [None, None, None],
            view: 0,
            zoom: 1.0,
            fit: true,
        }
    }
}
impl TextureEditor {
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }
    fn start(
        &mut self,
        label: &str,
        f: impl FnOnce() -> Result<Completion, String> + Send + 'static,
    ) {
        if self.busy() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.job = Some(rx);
        self.status = label.into();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
                .unwrap_or_else(|_| Err("Texture operation failed inside the codec.".into()));
            let _ = tx.send(result);
        });
    }
    pub fn open(&mut self, path: PathBuf) {
        if self.busy() {
            self.status =
                "Wait for the current texture operation before opening another file.".into();
            return;
        }
        self.start("Loading texture…", move || {
            load(path).map(Completion::Loaded)
        });
    }
    pub(crate) fn poll(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.job else {
            return;
        };
        let result = match job.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(60));
                return;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("Texture worker stopped without a result.".into())
            }
        };
        self.job = None;
        match result {
            Ok(Completion::Loaded(source)) => {
                self.status = source.info.clone();
                self.source = Some(Arc::new(source));
                self.result = None;
                self.textures = [None, None, None];
                self.view = 0;
            }
            Ok(Completion::Encoded(result)) => {
                if result.settings == self.settings {
                    self.status = format!("Encoded in {:.3} s", result.seconds);
                    self.result = Some(Arc::new(result));
                    self.textures[1] = None;
                    self.textures[2] = None;
                    self.view = 1;
                }
            }
            Ok(Completion::Saved(message)) => self.status = message,
            Err(message) => self.status = format!("Error: {message}"),
        }
    }
    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.poll(ctx);
        let busy = self.busy();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("Open texture…"))
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(
                        "Textures",
                        &[
                            "png", "jpg", "jpeg", "bmp", "tga", "tif", "tiff", "webp", "hdr",
                            "exr", "jkli",
                        ],
                    )
                    .pick_file()
                {
                    self.open(path);
                }
            }
            if busy {
                ui.spinner();
            }
            ui.label(&self.status);
        });
        ui.separator();
        let Some(source) = self.source.clone() else {
            ui.heading("Texture workbench");
            ui.label("Open or drop an image or JKLI texture to encode, inspect, and decode it.");
            return;
        };
        let before = self.settings.clone();
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(270.0);
                egui::ScrollArea::vertical().id_salt("texture-settings").show(ui, |ui| {
                    ui.heading("Encoding");
                    ui.label(&source.name);
                    ui.label(format!("{} × {} pixels", source.width, source.height));
                    ui.add_enabled_ui(!busy, |ui| {
                        egui::ComboBox::from_label("Pixel format").selected_text(format!("{:?}", self.settings.format)).show_ui(ui, |ui| {
                            for &f in FORMATS { ui.selectable_value(&mut self.settings.format, f, format!("{f:?}")); }
                        });
                        {
                            egui::ComboBox::from_label("Compression").selected_text(format!("{:?}", self.settings.compression)).show_ui(ui, |ui| { for &c in COMPRESSIONS { ui.selectable_value(&mut self.settings.compression, c, format!("{c:?}")); } });
                            ui.checkbox(&mut self.settings.fixed, "Fixed tile size");
                            if self.settings.fixed {
                                ui.horizontal(|ui| { ui.label("Width"); ui.add(egui::DragValue::new(&mut self.settings.tile_width).range(1..=16384)); ui.label("Height"); ui.add(egui::DragValue::new(&mut self.settings.tile_height).range(1..=16384)); });
                                ui.small("Tile dimensions use stored pixels or blocks.");
                            } else {
                                ui.horizontal(|ui| { ui.label("Flat tile cost"); ui.add(egui::DragValue::new(&mut self.settings.flat_cost).range(0.0..=65536.0)); });
                                ui.horizontal(|ui| { ui.label("Size cost"); ui.add(egui::DragValue::new(&mut self.settings.size_cost).range(0.001..=1024.0).speed(0.1)); });
                            }
                        }
                        if self.settings.format == Format::BC1 { ui.add(egui::Slider::new(&mut self.settings.alpha_threshold, 0.0..=1.0).text("Alpha cutoff")); }
                        if self.settings.format == Format::BC6 { ui.label("Unsigned BC6H (linear HDR). All-mode CPU search can be slow. Preview and PNG export clamp HDR to 0–1."); }
                        if matches!(self.settings.format, Format::R8 | Format::BC4) { ui.small("Uses red channel; preview displays grayscale."); }
                        if matches!(self.settings.format, Format::RG8 | Format::BC5) { ui.small("Uses red and green channels; blue is zero."); }
                        if self.settings.format.block_width() == 4 && (source.width % 4 != 0 || source.height % 4 != 0) { ui.colored_label(egui::Color32::YELLOW, format!("Saved blocks cover {} × {} pixels. Preview/export crop to the {} × {} source. JKLI stores block extents only.", source.width.div_ceil(4)*4, source.height.div_ceil(4)*4, source.width, source.height)); }
                        if ui.button("Encode texture").clicked() {
                            let settings = self.settings.clone(); let source = source.clone();
                            self.start("Encoding texture…", move || encode(&source, settings).map(Completion::Encoded));
                        }
                    });
                    if let Some(result) = self.result.clone().filter(|r| r.settings == self.settings) {
                        ui.separator();
                        ui.label(format!("Output: {} bytes", result.bytes.len()));
                        ui.label(format!("RGBA PSNR: {:.2} dB", result.psnr));
                        ui.label(format!("Time: {:.3} s", result.seconds));
                        ui.add_enabled_ui(!busy, |ui| {
                            if ui.button("Save encoded…").clicked() {
                                if let Some(path) = rfd::FileDialog::new().add_filter("Jackal texture", &["jkli"]).set_file_name("texture.jkli").save_file() {
                                    let result = result.clone(); self.start("Saving texture…", move || { std::fs::write(&path, &result.bytes).map_err(|e| e.to_string())?; Ok(Completion::Saved(format!("Saved {}", path.display()))) });
                                }
                            }
                        });
                    }
                    if ui.add_enabled(!busy, egui::Button::new("Export decoded PNG…")).clicked() {
                        if let Some(path) = rfd::FileDialog::new().add_filter("PNG", &["png"]).set_file_name("decoded.png").save_file() {
                            let result = self.result.clone().filter(|r| r.settings == self.settings); let source = source.clone();
                            self.start("Exporting decoded pixels…", move || {
                                let pixels = result.as_ref().map_or(source.pixels.as_slice(), |r| r.decoded.as_slice());
                                let bytes: Vec<u8> = pixels.iter().flat_map(|p| p.map(byte)).collect();
                                image::save_buffer_with_format(&path, &bytes, source.width as u32, source.height as u32, image::ColorType::Rgba8, image::ImageFormat::Png).map_err(|e| e.to_string())?;
                                Ok(Completion::Saved(format!("Exported {}", path.display())))
                            });
                        }
                    }
                });
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.view, 0, "Source");
                    ui.add_enabled_ui(self.result.as_ref().is_some_and(|r| r.settings == self.settings), |ui| { ui.selectable_value(&mut self.view, 1, "Reconstruction"); ui.selectable_value(&mut self.view, 2, "Error ×4"); });
                    ui.checkbox(&mut self.fit, "Fit");
                    ui.add(egui::Slider::new(&mut self.zoom, 0.1..=8.0).logarithmic(true).text("Zoom"));
                });
                let view = if self.view > 0 && self.result.as_ref().is_none_or(|r| r.settings != self.settings) { 0 } else { self.view };
                if self.textures[view].is_none() {
                    let image = match view { 1 => self.result.as_ref().unwrap().preview.clone(), 2 => self.result.as_ref().unwrap().error.clone(), _ => source.preview.clone() };
                    self.textures[view] = Some(ctx.load_texture(format!("texture-view-{view}"), image, egui::TextureOptions::NEAREST));
                }
                let texture = self.textures[view].as_ref().unwrap();
                let natural = egui::vec2(source.width as f32, source.height as f32);
                let available = ui.available_size();
                let scale = if self.fit { (available.x/natural.x).min(available.y/natural.y).max(0.01) } else { self.zoom };
                egui::ScrollArea::both().id_salt("texture-preview").show(ui, |ui| {
                    let (rect, _) = ui.allocate_exact_size(natural * scale, egui::Sense::hover());
                    let clip = rect.intersect(ui.clip_rect());
                    let cell = 16.0;
                    for y in 0..(clip.height()/cell).ceil() as i32 { for x in 0..(clip.width()/cell).ceil() as i32 {
                        let r = egui::Rect::from_min_size(clip.min + egui::vec2(x as f32*cell,y as f32*cell), egui::vec2(cell,cell)).intersect(clip);
                        ui.painter().rect_filled(r, 0.0, egui::Color32::from_gray(if (x+y)%2 == 0 {45} else {65}));
                    }}
                    ui.painter().image(texture.id(), rect, egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0,1.0)), egui::Color32::WHITE);
                });
            });
        });
        if before != self.settings {
            self.result = None;
            self.textures[1] = None;
            self.textures[2] = None;
            self.view = 0;
        }
    }
}

fn byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}
fn preview(width: usize, height: usize, pixels: &[[f32; 4]]) -> egui::ColorImage {
    let bytes: Vec<u8> = pixels.iter().flat_map(|p| p.map(byte)).collect();
    egui::ColorImage::from_rgba_unmultiplied([width, height], &bytes)
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
fn load(path: PathBuf) -> Result<Source, String> {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let mut magic = [0; 4];
    let mut file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
    let is_jkli = file.read_exact(&mut magic).is_ok() && &magic == b"JKLI";
    let (width, height, pixels, info) = if is_jkli || path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("jkli"))
    {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        decode_jkli(&bytes)?
    } else {
        let reader = image::ImageReader::open(path)
            .map_err(|e| e.to_string())?
            .with_guessed_format()
            .map_err(|e| e.to_string())?;
        let image = reader.decode().map_err(|e| e.to_string())?;
        let (width, height) = (image.width() as usize, image.height() as usize);
        check_size(width, height)?;
        let hdr = matches!(
            image,
            image::DynamicImage::ImageRgb32F(_) | image::DynamicImage::ImageRgba32F(_)
        );
        let rgba = image.to_rgba32f();
        let pixels = rgba
            .into_raw()
            .chunks_exact(4)
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect::<Vec<_>>();
        (
            width,
            height,
            pixels,
            if hdr {
                "Loaded HDR source; preview clamps to 0–1.".into()
            } else {
                "Loaded image.".into()
            },
        )
    };
    Ok(Source {
        name,
        width,
        height,
        preview: preview(width, height, &pixels),
        pixels,
        info,
    })
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
fn encode(source: &Source, settings: Settings) -> Result<Encoded, String> {
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
    let mut squared = 0.0;
    let error: Vec<_> = source
        .pixels
        .iter()
        .zip(&decoded)
        .map(|(a, b)| {
            let mut p = [0.0; 4];
            for c in 0..4 {
                let d = (a[c] - b[c]).abs();
                squared += (d as f64).powi(2);
                p[c] = d * 4.0;
            }
            p[3] = 1.0;
            p
        })
        .collect();
    let mse = squared / (w * h * 4) as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        -10.0 * mse.log10()
    };
    Ok(Encoded {
        bytes,
        preview: preview(w, h, &decoded),
        error: preview(w, h, &error),
        decoded,
        psnr,
        seconds: start.elapsed().as_secs_f64(),
        settings,
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
fn decode_jkli(bytes: &[u8]) -> Result<(usize, usize, Vec<[f32; 4]>, String), String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    fn source(w: usize, h: usize) -> Source {
        let pixels = vec![[0.2, 0.4, 0.7, 0.8]; w * h];
        Source {
            name: "test".into(),
            width: w,
            height: h,
            preview: preview(w, h, &pixels),
            pixels,
            info: String::new(),
        }
    }
    #[test]
    fn rgba_container_roundtrip() {
        let s = source(5, 3);
        for &compression in COMPRESSIONS {
            let result = encode(
                &s,
                Settings {
                    format: Format::RGBA8,
                    compression,
                    ..Settings::default()
                },
            )
            .unwrap();
            let (w, h, pixels, _) = decode_jkli(&result.bytes).unwrap();
            assert_eq!((w, h), (5, 3));
            assert_eq!(pixels, result.decoded);
        }
    }
    #[test]
    fn odd_block_preview_and_container_extents() {
        let s = source(5, 3);
        let result = encode(&s, Settings::default()).unwrap();
        assert_eq!(result.decoded.len(), 15);
        let (w, h, reopened, _) = decode_jkli(&result.bytes).unwrap();
        assert_eq!((w, h), (8, 4));
        for y in 0..3 {
            assert_eq!(
                &reopened[y * w..y * w + 5],
                &result.decoded[y * 5..y * 5 + 5]
            );
        }
    }
    #[test]
    fn invalid_tile_alignment_rejected() {
        assert!(
            Settings {
                fixed: true,
                tile_width: 3,
                ..Settings::default()
            }
            .options()
            .is_err()
        );
    }
    #[test]
    fn every_format_reopens_to_exact_preview_pixels() {
        let s = source(4, 4);
        for &format in FORMATS {
            let result = encode(
                &s,
                Settings {
                    format,
                    ..Settings::default()
                },
            )
            .unwrap();
            let (w, h, pixels, _) = decode_jkli(&result.bytes).unwrap();
            assert_eq!((w, h), (4, 4), "{format:?}");
            assert_eq!(pixels, result.decoded, "{format:?}");
        }
    }
}
