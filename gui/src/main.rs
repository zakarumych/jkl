use eframe::egui::{
    self, Color32, ColorImage, ImageData, TextureHandle, TextureId, TextureOptions,
};
use texpak::math::{Rgb32F, Rgb8U, Rgba32F};

fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "TexPak App",
        native_options,
        Box::new(|cc| Ok(Box::new(TexPakApp::new(cc)))),
    )
    .unwrap();
}

#[derive(Default)]
struct TexPakApp {
    opt: usize,
    image: Option<image::RgbImage>,
    original_image: Option<TextureHandle>,
    compressed_image: Vec<texpak::bc1::Block>,
    decompressed_image: Option<TextureHandle>,
}

impl TexPakApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }
}

impl eframe::App for TexPakApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.original_image.is_none() {
            if let Some(image) = &self.image {
                self.original_image = Some(ctx.load_texture(
                    "Original",
                    ColorImage {
                        size: [image.width() as usize, image.height() as usize],
                        pixels: image.pixels().map(|&p| rgb_image_to_egui(p)).collect(),
                    },
                    TextureOptions::default(),
                ));
            }
        }

        if self.compressed_image.is_empty() {
            if let Some(image) = &self.image {
                let mut blocks = Vec::new();
                for y in (0..image.height() & !3).step_by(4) {
                    for x in (0..image.width() & !3).step_by(4) {
                        let block = [
                            [
                                rgb_image_to_texpak(*image.get_pixel(x + 0, y + 0)),
                                rgb_image_to_texpak(*image.get_pixel(x + 1, y + 0)),
                                rgb_image_to_texpak(*image.get_pixel(x + 2, y + 0)),
                                rgb_image_to_texpak(*image.get_pixel(x + 3, y + 0)),
                            ],
                            [
                                rgb_image_to_texpak(*image.get_pixel(x + 0, y + 1)),
                                rgb_image_to_texpak(*image.get_pixel(x + 1, y + 1)),
                                rgb_image_to_texpak(*image.get_pixel(x + 2, y + 1)),
                                rgb_image_to_texpak(*image.get_pixel(x + 3, y + 1)),
                            ],
                            [
                                rgb_image_to_texpak(*image.get_pixel(x + 0, y + 2)),
                                rgb_image_to_texpak(*image.get_pixel(x + 1, y + 2)),
                                rgb_image_to_texpak(*image.get_pixel(x + 2, y + 2)),
                                rgb_image_to_texpak(*image.get_pixel(x + 3, y + 2)),
                            ],
                            [
                                rgb_image_to_texpak(*image.get_pixel(x + 0, y + 3)),
                                rgb_image_to_texpak(*image.get_pixel(x + 1, y + 3)),
                                rgb_image_to_texpak(*image.get_pixel(x + 2, y + 3)),
                                rgb_image_to_texpak(*image.get_pixel(x + 3, y + 3)),
                            ],
                        ];

                        let block = texpak::bc1::Block::encode(block, self.opt);
                        blocks.push(block);
                    }
                }
                self.compressed_image = blocks;
            }
        }

        if self.decompressed_image.is_none() {
            if let Some(image) = &self.image {
                assert!(!self.compressed_image.is_empty());
                let mut blocks = self.compressed_image.iter();

                let mut pixels = Vec::new();

                for _ in (0..image.height() & !3).step_by(4) {
                    let mut blocks_row = Vec::new();

                    for _ in (0..image.width() & !3).step_by(4) {
                        let block = *blocks.next().unwrap();
                        let block = texpak::bc1::Block::decode(block);
                        blocks_row.push(block);
                    }

                    for i in 0..4 {
                        for block in blocks_row.iter() {
                            for j in 0..4 {
                                pixels.push(rgba_texpak_to_egui(block[i][j]));
                            }
                        }
                    }
                }

                self.decompressed_image = Some(ctx.load_texture(
                    "Decompressed",
                    ColorImage {
                        size: [
                            (image.width() as usize) & !3,
                            (image.height() as usize) & !3,
                        ],
                        pixels,
                    },
                    TextureOptions::default(),
                ));
            }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(dropped_file) = ctx.input(|i| i.raw.dropped_files.last().cloned()) {
                    if let Some(path) = &dropped_file.path {
                        if let Ok(image) = image::open(path) {
                            self.image = Some(image.into_rgb8());
                            self.compressed_image.clear();
                        }
                    }
                }

                if ui.button("Clear").clicked() {
                    self.image = None;
                    self.original_image = None;
                    self.compressed_image.clear();
                    self.decompressed_image = None;
                }

                ui.label("Opt:");
                let r = ui.add(
                    egui::DragValue::new(&mut self.opt)
                        .range(0..=100)
                        .clamp_existing_to_range(true),
                );

                if r.changed() {
                    self.compressed_image.clear();
                    self.decompressed_image = None;
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let size = ui.available_size();

            ui.horizontal(|ui| {
                if let Some(image) = &self.original_image {
                    ui.add(
                        egui::Image::new(image).fit_to_exact_size(egui::vec2(size.x * 0.5, size.y)),
                    );
                }

                if let Some(image) = &self.decompressed_image {
                    ui.add(
                        egui::Image::new(image).fit_to_exact_size(egui::vec2(size.x * 0.5, size.y)),
                    );
                }
            });
        });
    }
}

fn rgb_image_to_egui(rgb: image::Rgb<u8>) -> egui::Color32 {
    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

fn rgb_image_to_texpak(rgb: image::Rgb<u8>) -> Rgb32F {
    Rgb32F::new(
        rgb[0] as f32 / 255.0,
        rgb[1] as f32 / 255.0,
        rgb[2] as f32 / 255.0,
    )
}

fn rgb_texpak_to_egui(rgb: Rgb32F) -> Color32 {
    Color32::from_rgb(
        (rgb.r() * 255.0) as u8,
        (rgb.g() * 255.0) as u8,
        (rgb.b() * 255.0) as u8,
    )
}

fn rgba_texpak_to_egui(rgb: Rgba32F) -> Color32 {
    Color32::from_rgba_premultiplied(
        (rgb.r() * 255.0) as u8,
        (rgb.g() * 255.0) as u8,
        (rgb.b() * 255.0) as u8,
        (rgb.a() * 255.0) as u8,
    )
}
