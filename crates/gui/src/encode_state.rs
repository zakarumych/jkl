use std::{
    collections::HashMap,
    io::Cursor,
    sync::{Arc, mpsc},
    thread,
};

use jkl::{
    image::{
        Dimensions, ImageMut, ImageRef, OwnedImage,
        block::{bc1, bc2},
        quality,
    },
    jackal::image::{Compression, Options, write_image},
    math::{Rgb8U, Rgba8U, Rgba32F},
};

// ── Input image ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub enum Image {
    Rgb8(OwnedImage<Rgb8U>),
    Rgba8(OwnedImage<Rgba8U>),
    Bc1(OwnedImage<bc1::Block>),
    Bc2(OwnedImage<bc2::Block>),
}

impl Image {
    pub fn dimensions(&self) -> Dimensions {
        match self {
            Self::Rgb8(img) => img.dimensions(),
            Self::Rgba8(img) => img.dimensions(),
            Self::Bc1(img) => img.dimensions(),
            Self::Bc2(img) => img.dimensions(),
        }
    }

    pub fn width(&self) -> usize {
        match self {
            Self::Rgb8(img) => img.width(),
            Self::Rgba8(img) => img.width(),
            Self::Bc1(img) => img.width(),
            Self::Bc2(img) => img.width(),
        }
    }

    pub fn height(&self) -> usize {
        match self {
            Self::Rgb8(img) => img.height(),
            Self::Rgba8(img) => img.height(),
            Self::Bc1(img) => img.height(),
            Self::Bc2(img) => img.height(),
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Self::Rgb8(img) => img.depth(),
            Self::Rgba8(img) => img.depth(),
            Self::Bc1(img) => img.depth(),
            Self::Bc2(img) => img.depth(),
        }
    }

    pub fn layers(&self) -> usize {
        match self {
            Self::Rgb8(img) => img.layers(),
            Self::Rgba8(img) => img.layers(),
            Self::Bc1(img) => img.layers(),
            Self::Bc2(img) => img.layers(),
        }
    }

    pub fn format(&self) -> Format {
        match self {
            Self::Rgb8(_) => Format::RGB8,
            Self::Rgba8(_) => Format::RGBA8,
            Self::Bc1(_) => Format::BC1,
            Self::Bc2(_) => Format::BC2,
        }
    }

    fn into_format(&self, format: Format) -> Image {
        match format {
            Format::RGB8 => match self {
                Image::Rgb8(_) => self.clone(),
                Image::Rgba8(img) => Image::Rgb8(img.into_format()),
                Image::Bc1(img) => Image::Rgb8(img.into_format()),
                Image::Bc2(_) => unimplemented!(),
            },
            Format::RGBA8 => match self {
                Image::Rgb8(img) => Image::Rgba8(img.into_format()),
                Image::Rgba8(_) => self.clone(),
                Image::Bc1(img) => Image::Rgba8(img.into_format()),
                Image::Bc2(img) => Image::Rgba8(img.into_format()),
            },
            _ => unimplemented!(),
        }
    }
}

// ── Output format ──────────────────────────────────────────────────────────

pub use jkl::image::format::Format;

pub const FORMAT_ALL: &[Format] = &[Format::RGB8, Format::BC1, Format::BC2];

pub fn format_label(f: Format) -> &'static str {
    match f {
        Format::RGB8 => "RGB8",
        Format::BC1 => "BC1",
        Format::BC2 => "BC2",
        _ => unimplemented!(),
    }
}

// ── View mode ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewMode {
    Input,
    Encoded,
    Error,
    ErrorHeatmap,
}

impl ViewMode {
    pub const ALL: &'static [Self] = &[Self::Input, Self::Encoded, Self::Error, Self::ErrorHeatmap];

    pub fn label(self) -> &'static str {
        match self {
            Self::Input => "Input",
            Self::Encoded => "Encoded",
            Self::Error => "Error",
            Self::ErrorHeatmap => "Error Heatmap",
        }
    }
}

// ── Cache entries ────────────────────────────────────────────────────────

enum EncodedDataCacheEntry {
    InProgress(mpsc::Receiver<Result<Image, String>>),
    Done { data: Arc<Image>, psnr: Option<f64> },
    Error(String),
}

enum JkliCacheEntry {
    InProgress(mpsc::Receiver<Vec<u8>>),
    Done(Arc<Vec<u8>>),
}

// ── Encoder closure types ──────────────────────────────────────────────────

/// Called to start block/pixel encoding. Responsible for sending the result
/// (or error) through `tx`, typically by spawning a background thread.
pub type BlockEncoderFn =
    dyn Fn(Arc<Image>, Format, mpsc::Sender<Result<Image, String>>) + Send + Sync;

// ── EncoderState ───────────────────────────────────────────────────────────

pub struct EncoderState {
    input: Arc<Image>,
    texture_handle: Option<egui::TextureHandle>,
    encoded_cache: HashMap<Format, EncodedDataCacheEntry>,
    jkli_cache: HashMap<(Format, Compression), JkliCacheEntry>,
    encode_blocks: Arc<BlockEncoderFn>,
    selected_format: Format,
    selected_compression: Compression,
    pub(super) view_mode: ViewMode,
    pub(super) heatmap_threshold: f32,
    pub(super) error_gamma: f32,
    decoded_pixels: HashMap<Format, Image>,
    decoded_textures: HashMap<Format, egui::TextureHandle>,
    error_textures: HashMap<Format, (u32, egui::TextureHandle)>,
    heatmap_cache: Option<(Format, u32, u32, egui::TextureHandle)>,
}

impl EncoderState {
    /// Create with the default CPU-based encoders (spawns threads).
    pub fn new(input: Image) -> Self {
        Self::with_encoders(input, cpu_block_encoder)
    }

    /// Create with custom encoder closures.
    ///
    /// Each closure receives a [`mpsc::Sender`] and is responsible for
    /// eventually sending one result through it (success or error).
    pub fn with_encoders<BF>(input: Image, block_fn: BF) -> Self
    where
        BF: Fn(Arc<Image>, Format, mpsc::Sender<Result<Image, String>>) + Send + Sync + 'static,
    {
        Self {
            input: Arc::new(input),
            texture_handle: None,
            encoded_cache: HashMap::new(),
            jkli_cache: HashMap::new(),
            encode_blocks: Arc::new(block_fn),
            selected_format: Format::BC1,
            selected_compression: Compression::Ans,
            view_mode: ViewMode::Input,
            heatmap_threshold: 0.1,
            error_gamma: 1.0,
            decoded_pixels: HashMap::new(),
            decoded_textures: HashMap::new(),
            error_textures: HashMap::new(),
            heatmap_cache: None,
        }
    }

    /// Create with a custom block encoder and the default CPU JKLI encoder.
    pub fn with_block_encoder<BF>(input: Image, block_fn: BF) -> Self
    where
        BF: Fn(Arc<Image>, Format, mpsc::Sender<Result<Image, String>>) + Send + Sync + 'static,
    {
        Self::with_encoders(input, block_fn)
    }

    pub fn input(&self) -> &Image {
        &self.input
    }

    pub fn selected_format(&self) -> Format {
        self.selected_format
    }

    pub fn selected_compression(&self) -> Compression {
        self.selected_compression
    }

    pub fn set_selection(&mut self, format: Format, compression: Compression) {
        self.selected_format = format;
        self.selected_compression = compression;
    }

    pub fn current_jkli_data(&self) -> Option<Arc<Vec<u8>>> {
        self.serialized_data(self.selected_format, self.selected_compression)
    }

    // ── Texture ────────────────────────────────────────────────────────────

    pub fn ensure_texture(&mut self, ctx: &egui::Context) {
        if self.texture_handle.is_some() {
            return;
        }
        let color_image = match &*self.input {
            Image::Rgb8(img) => {
                let rgba: Vec<u8> = img
                    .data()
                    .iter()
                    .flat_map(|p| [p.0[0], p.0[1], p.0[2], 255u8])
                    .collect();
                egui::ColorImage::from_rgba_unmultiplied([img.width(), img.height()], &rgba)
            }
            Image::Rgba8(img) => {
                let rgba: Vec<u8> = img.data().iter().flat_map(|p| p.0).collect();
                egui::ColorImage::from_rgba_unmultiplied([img.width(), img.height()], &rgba)
            }
            _ => unimplemented!("preview not implemented for block-compressed input"),
        };
        self.texture_handle =
            Some(ctx.load_texture("encode-preview", color_image, egui::TextureOptions::LINEAR));
    }

    /// Generate / refresh preview textures for the comparison views.
    ///
    /// Textures for `Encoded` and `Error` are computed once per format.
    /// The `Heatmap` texture is regenerated whenever the format or
    /// [`heatmap_threshold`](Self::heatmap_threshold) changes.
    /// Does nothing if encoded data for `format` is not yet ready.
    pub fn ensure_view_textures(&mut self, format: Format, ctx: &egui::Context) {
        let encoded_arc = match self.encoded_cache.get(&format) {
            Some(EncodedDataCacheEntry::Done { data, .. }) => Arc::clone(data),
            _ => return,
        };

        let w = self.input.width();
        let h = self.input.height();

        // ── Decoded pixels (cached) ────────────────────────────────────────
        if !self.decoded_pixels.contains_key(&format) {
            self.decoded_pixels
                .insert(format, encoded_arc.into_format(self.input.format()));
        }

        // ── Encoded view texture ───────────────────────────────────────────
        if !self.decoded_textures.contains_key(&format) {
            let rgba: Vec<u8> = match &self.decoded_pixels[&format] {
                Image::Rgb8(img) => img
                    .data()
                    .iter()
                    .flat_map(|p| [p.0[0], p.0[1], p.0[2], 255u8])
                    .collect(),
                Image::Rgba8(img) => img.data().iter().flat_map(|p| p.0).collect(),
                _ => unreachable!(),
            };
            let ci = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
            let tex = ctx.load_texture("enc-decoded", ci, egui::TextureOptions::LINEAR);
            self.decoded_textures.insert(format, tex);
        }

        // ── Abs-error texture ──────────────────────────────────────────────
        let gamma_bits = self.error_gamma.to_bits();
        let error_stale = self
            .error_textures
            .get(&format)
            .map_or(true, |(g, _)| *g != gamma_bits);

        // ── Heatmap texture (regenerate on format / threshold / gamma / palette change) ──
        let threshold_bits = self.heatmap_threshold.to_bits();
        let heatmap_stale = self.heatmap_cache.as_ref().map_or(true, |(f, t, g, _)| {
            *f != format || *t != threshold_bits || *g != gamma_bits
        });

        if error_stale {
            let gamma = self.error_gamma;
            let rgba: Vec<u8> = match (&*self.input, &self.decoded_pixels[&format]) {
                (Image::Rgb8(orig), Image::Rgb8(decoded)) => orig
                    .data()
                    .iter()
                    .zip(decoded.data().iter())
                    .flat_map(|(a, b)| {
                        let dr = (a.0[0] as i32 - b.0[0] as i32).unsigned_abs() as f32 / 255.0;
                        let dg = (a.0[1] as i32 - b.0[1] as i32).unsigned_abs() as f32 / 255.0;
                        let db = (a.0[2] as i32 - b.0[2] as i32).unsigned_abs() as f32 / 255.0;
                        [
                            (dr.powf(gamma) * 255.0) as u8,
                            (dg.powf(gamma) * 255.0) as u8,
                            (db.powf(gamma) * 255.0) as u8,
                            255u8,
                        ]
                    })
                    .collect(),
                (Image::Rgba8(orig), Image::Rgba8(decoded)) => orig
                    .data()
                    .iter()
                    .zip(decoded.data().iter())
                    .flat_map(|(a, b)| {
                        let dr = (a.0[0] as i32 - b.0[0] as i32).unsigned_abs() as f32 / 255.0;
                        let dg = (a.0[1] as i32 - b.0[1] as i32).unsigned_abs() as f32 / 255.0;
                        let db = (a.0[2] as i32 - b.0[2] as i32).unsigned_abs() as f32 / 255.0;
                        let da = (a.0[3] as i32 - b.0[3] as i32).unsigned_abs() as f32 / 255.0;
                        [
                            (dr.powf(gamma) * 255.0) as u8,
                            (dg.powf(gamma) * 255.0) as u8,
                            (db.powf(gamma) * 255.0) as u8,
                            (da.powf(gamma) * 255.0) as u8,
                        ]
                    })
                    .collect(),
                _ => unreachable!(),
            };
            let ci = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
            let tex = ctx.load_texture("enc-abs-error", ci, egui::TextureOptions::LINEAR);
            self.error_textures.insert(format, (gamma_bits, tex));
        }

        if heatmap_stale {
            let thresh = self.heatmap_threshold;
            let gamma = self.error_gamma;
            let rgba: Vec<u8> = match (&*self.input, &self.decoded_pixels[&format]) {
                (Image::Rgb8(orig), Image::Rgb8(decoded)) => {
                    let error_map = quality::error_heatmap(orig.as_ref(), decoded.as_ref());
                    let error_range = <Rgb8U as quality::ErrorPixel>::error_range();
                    error_map
                        .data()
                        .iter()
                        .zip(orig.data().iter())
                        .flat_map(|(&e, a)| {
                            let error = e / error_range;
                            if error <= thresh {
                                [a.0[0], a.0[1], a.0[2], 255u8]
                            } else {
                                let t = ((error - thresh) / (1.0 - thresh + 1e-6)).clamp(0.0, 1.0);
                                let t_mapped = t.powf(gamma);
                                let [r, g, b] = heatvision_color(t_mapped);
                                [r, g, b, 255u8]
                            }
                        })
                        .collect()
                }
                (Image::Rgba8(orig), Image::Rgba8(decoded)) => {
                    let error_map = quality::error_heatmap(orig.as_ref(), decoded.as_ref());
                    let error_range = <Rgba8U as quality::ErrorPixel>::error_range();
                    error_map
                        .data()
                        .iter()
                        .zip(orig.data().iter())
                        .flat_map(|(&e, a)| {
                            let error = e / error_range;
                            if error <= thresh {
                                a.0
                            } else {
                                let t = ((error - thresh) / (1.0 - thresh + 1e-6)).clamp(0.0, 1.0);
                                let t_mapped = t.powf(gamma);
                                let [r, g, b] = heatvision_color(t_mapped);
                                [r, g, b, 255u8]
                            }
                        })
                        .collect()
                }
                _ => unreachable!(),
            };
            let ci = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
            let tex = ctx.load_texture("enc-heatmap", ci, egui::TextureOptions::LINEAR);
            self.heatmap_cache = Some((format, threshold_bits, gamma_bits, tex));
        }
    }

    /// Returns the preview texture appropriate for the current
    /// [`view_mode`](Self::view_mode), or `None` if not yet available.
    pub fn preview_texture(&self, format: Format) -> Option<(egui::TextureId, f32)> {
        let aspect = self.input.width() as f32 / self.input.height() as f32;
        let id = match self.view_mode {
            ViewMode::Input => self.texture_handle.as_ref()?.id(),
            ViewMode::Encoded => self.decoded_textures.get(&format)?.id(),
            ViewMode::Error => self.error_textures.get(&format)?.1.id(),
            ViewMode::ErrorHeatmap => self
                .heatmap_cache
                .as_ref()
                .filter(|(f, _, _, _)| *f == format)?
                .3
                .id(),
        };
        Some((id, aspect))
    }

    // ── Background task management ─────────────────────────────────────────

    /// Advance all in-progress background tasks. Call every frame.
    pub fn poll(&mut self) {
        let input = Arc::clone(&self.input);
        for entry in self.encoded_cache.values_mut() {
            let EncodedDataCacheEntry::InProgress(rx) = entry else {
                continue;
            };
            match rx.try_recv() {
                Ok(Ok(data)) => {
                    let psnr = compute_psnr(&data, &input);
                    *entry = EncodedDataCacheEntry::Done {
                        data: Arc::new(data),
                        psnr,
                    }
                }
                Ok(Err(e)) => *entry = EncodedDataCacheEntry::Error(e),
                Err(mpsc::TryRecvError::Disconnected) => {
                    *entry = EncodedDataCacheEntry::Error("encode thread disconnected".into())
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        for entry in self.jkli_cache.values_mut() {
            let JkliCacheEntry::InProgress(rx) = entry else {
                continue;
            };
            match rx.try_recv() {
                Ok(d) => *entry = JkliCacheEntry::Done(Arc::new(d)),
                Err(mpsc::TryRecvError::Disconnected) => panic!("jkli thread panicked"),
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    /// Start encoding for `format` if not already cached or in-progress.
    pub fn ensure_blocks(&mut self, format: Format) {
        if self.encoded_cache.contains_key(&format) {
            return;
        }
        let input = Arc::clone(&self.input);
        let (tx, rx) = mpsc::channel();
        self.encoded_cache
            .insert(format, EncodedDataCacheEntry::InProgress(rx));
        (self.encode_blocks)(input, format, tx);
    }

    /// Start JKLI serialization if blocks are ready and result not cached.
    pub fn ensure_jkli(&mut self, format: Format, compression: Compression) {
        if self.jkli_cache.contains_key(&(format, compression)) {
            return;
        }
        let data = match self.encoded_cache.get(&format) {
            Some(EncodedDataCacheEntry::Done { data, .. }) => Arc::clone(data),
            _ => return,
        };
        let (tx, rx) = mpsc::channel();
        self.jkli_cache
            .insert((format, compression), JkliCacheEntry::InProgress(rx));

        thread::spawn(move || {
            let _ = tx.send(serialize_jkli(&data, compression));
        });
    }

    // ── Accessors for UI ──────────────────────────────────────────────────

    pub fn compression_done(&self, format: Format) -> bool {
        matches!(
            self.encoded_cache.get(&format),
            Some(EncodedDataCacheEntry::Done { .. })
        )
    }

    pub fn compression_in_progress(&self, format: Format) -> bool {
        matches!(
            self.encoded_cache.get(&format),
            Some(EncodedDataCacheEntry::InProgress(_))
        )
    }

    pub fn compression_psnr(&self, format: Format) -> Option<f64> {
        match self.encoded_cache.get(&format)? {
            EncodedDataCacheEntry::Done { psnr, .. } => *psnr,
            _ => None,
        }
    }

    pub fn compression_error(&self, format: Format) -> Option<&str> {
        match self.encoded_cache.get(&format)? {
            EncodedDataCacheEntry::Error(e) => Some(e),
            _ => None,
        }
    }

    pub fn serialized_data(
        &self,
        format: Format,
        compression: Compression,
    ) -> Option<Arc<Vec<u8>>> {
        match self.jkli_cache.get(&(format, compression))? {
            JkliCacheEntry::Done(d) => Some(Arc::clone(d)),
            _ => None,
        }
    }

    pub fn serialization_in_progress(&self, format: Format, compression: Compression) -> bool {
        matches!(
            self.jkli_cache.get(&(format, compression)),
            Some(JkliCacheEntry::InProgress(_))
        )
    }

    pub fn has_any_in_progress(&self) -> bool {
        self.encoded_cache
            .values()
            .any(|e| matches!(e, EncodedDataCacheEntry::InProgress(_)))
            || self
                .jkli_cache
                .values()
                .any(|e| matches!(e, JkliCacheEntry::InProgress(_)))
    }
}

// ── CPU encoder closures ───────────────────────────────────────────────────

fn cpu_block_encoder(input: Arc<Image>, format: Format, tx: mpsc::Sender<Result<Image, String>>) {
    thread::spawn(move || match format {
        Format::BC1 => {
            let encoded = match &*input {
                Image::Rgb8(img) => {
                    let [w, h, d] = img.raw_extent();
                    let mut output = OwnedImage::new(
                        input.dimensions(),
                        [w.div_ceil(4), h.div_ceil(4), d],
                        vec![bc1::Block::BLACK; (w.div_ceil(4)) * (h.div_ceil(4)) * d]
                            .into_boxed_slice(),
                    );
                    bc1::encode_image(img.as_ref(), |p| p.into_f32(), output.as_mut());
                    output
                }
                Image::Rgba8(img) => {
                    let [w, h, d] = img.raw_extent();
                    let mut output = OwnedImage::new(
                        input.dimensions(),
                        [w.div_ceil(4), h.div_ceil(4), d],
                        vec![bc1::Block::BLACK; (w.div_ceil(4)) * (h.div_ceil(4)) * d]
                            .into_boxed_slice(),
                    );
                    bc1::encode_image_with_alpha(
                        img.as_ref(),
                        |p| p.into_f32(),
                        0.5,
                        output.as_mut(),
                    );
                    output
                }
                _ => unimplemented!("CPU encoder only supports RGB8/RGBA8 input"),
            };

            let _ = tx.send(Ok(Image::Bc1(encoded)));
        }
        Format::BC2 => {
            let encoded = match &*input {
                Image::Rgb8(img) => {
                    let [w, h, d] = img.raw_extent();
                    let mut output = OwnedImage::new(
                        input.dimensions(),
                        [w.div_ceil(4), h.div_ceil(4), d],
                        vec![bc2::Block::BLACK; (w.div_ceil(4)) * (h.div_ceil(4)) * d]
                            .into_boxed_slice(),
                    );
                    bc2::encode_image(img.as_ref(), |p| p.into_f32(), output.as_mut());
                    output
                }
                Image::Rgba8(img) => {
                    let [w, h, d] = img.raw_extent();
                    let mut output = OwnedImage::new(
                        input.dimensions(),
                        [w.div_ceil(4), h.div_ceil(4), d],
                        vec![bc2::Block::BLACK; (w.div_ceil(4)) * (h.div_ceil(4)) * d]
                            .into_boxed_slice(),
                    );
                    bc2::encode_image_with_alpha(img.as_ref(), |p| p.into_f32(), output.as_mut());
                    output
                }
                _ => unimplemented!("CPU encoder only supports RGB8/RGBA8 input"),
            };

            let _ = tx.send(Ok(Image::Bc2(encoded)));
        }
        _ => unimplemented!(),
    });
}

// ── PSNR helpers ───────────────────────────────────────────────────────────

fn compute_psnr(data: &Image, input: &Image) -> Option<f64> {
    match data {
        Image::Rgb8(_) => Some(f64::INFINITY),
        Image::Rgba8(_) => Some(f64::INFINITY),
        Image::Bc1(img) => Some(psnr_bc1(input, img)),
        Image::Bc2(img) => Some(psnr_bc2(input, img)),
    }
}

fn psnr_bc1(input: &Image, img: &OwnedImage<bc1::Block>) -> f64 {
    let w = input.width();
    let h = input.height();
    match input {
        Image::Rgb8(orig) => {
            let mut decoded = vec![Rgb8U::BLACK; w * h];
            bc1::decode_image(
                ImageRef::new_2d(img.width(), img.height(), img.data()),
                |c: Rgba32F| {
                    Rgb8U::new(
                        (c.r() * 255.0).round() as u8,
                        (c.g() * 255.0).round() as u8,
                        (c.b() * 255.0).round() as u8,
                    )
                },
                ImageMut::new_2d(w, h, decoded.as_mut_slice()),
            );
            quality::psnr::<Rgb8U>(orig.as_ref(), ImageRef::new_2d(w, h, decoded.as_slice()))
        }
        Image::Rgba8(orig) => {
            let mut decoded = vec![Rgba8U([0, 0, 0, 255]); w * h];
            bc1::decode_image(
                ImageRef::new_2d(img.width(), img.height(), img.data()),
                |c: Rgba32F| {
                    Rgba8U([
                        (c.r() * 255.0).round() as u8,
                        (c.g() * 255.0).round() as u8,
                        (c.b() * 255.0).round() as u8,
                        (c.a() * 255.0).round() as u8,
                    ])
                },
                ImageMut::new_2d(w, h, decoded.as_mut_slice()),
            );
            quality::psnr::<Rgba8U>(orig.as_ref(), ImageRef::new_2d(w, h, decoded.as_slice()))
        }
        _ => unreachable!(),
    }
}

fn psnr_bc2(input: &Image, img: &OwnedImage<bc2::Block>) -> f64 {
    let w = input.width();
    let h = input.height();
    match input {
        Image::Rgb8(orig) => {
            let mut decoded = vec![Rgb8U::BLACK; w * h];
            bc2::decode_image(
                ImageRef::new_2d(img.width(), img.height(), img.data()),
                |c: Rgba32F| {
                    Rgb8U::new(
                        (c.r() * 255.0).round() as u8,
                        (c.g() * 255.0).round() as u8,
                        (c.b() * 255.0).round() as u8,
                    )
                },
                ImageMut::new_2d(w, h, decoded.as_mut_slice()),
            );
            quality::psnr::<Rgb8U>(orig.as_ref(), ImageRef::new_2d(w, h, decoded.as_slice()))
        }
        Image::Rgba8(orig) => {
            let mut decoded = vec![Rgba8U([0, 0, 0, 255]); w * h];
            bc2::decode_image(
                ImageRef::new_2d(img.width(), img.height(), img.data()),
                |c: Rgba32F| {
                    Rgba8U([
                        (c.r() * 255.0).round() as u8,
                        (c.g() * 255.0).round() as u8,
                        (c.b() * 255.0).round() as u8,
                        (c.a() * 255.0).round() as u8,
                    ])
                },
                ImageMut::new_2d(w, h, decoded.as_mut_slice()),
            );
            quality::psnr::<Rgba8U>(orig.as_ref(), ImageRef::new_2d(w, h, decoded.as_slice()))
        }
        _ => unreachable!(),
    }
}

// ── Colour mapping ────────────────────────────────────────────────────────

/// Maps `t` ∈ [0, 1] to a false-colour heat-vision palette:
/// black → blue → cyan → green → yellow → red.
fn heatvision_color(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b): (f32, f32, f32) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, 0.0, s)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 1.0 - s)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

// ── JKLI serialization ────────────────────────────────────────────────────

fn serialize_jkli(data: &Image, compression: Compression) -> Vec<u8> {
    let opts = Options::new().with_compression(compression);
    let mut buf = Cursor::new(Vec::<u8>::new());
    match data {
        Image::Rgb8(img) => write_image(img.as_ref(), opts, &mut buf),
        Image::Rgba8(img) => write_image(img.as_ref(), opts, &mut buf),
        Image::Bc1(img) => write_image(img.as_ref(), opts, &mut buf),
        Image::Bc2(img) => write_image(img.as_ref(), opts, &mut buf),
    }
    .expect("write_image to Cursor<Vec<u8>> is infallible");
    buf.into_inner()
}
