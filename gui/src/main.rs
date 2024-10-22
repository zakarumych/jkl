use eframe::egui::{
    self, Color32, ColorImage, ImageData, TextureHandle, TextureId, TextureOptions,
};
use texpak::{
    bc1::Block,
    math::{Rgb32F, Rgb8U, Rgba32F},
};

fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "TexPak App",
        native_options,
        Box::new(|cc| Ok(Box::new(TexPakApp::new(cc)))),
    )
    .unwrap();
}

struct TexPakApp {
    opt: usize,
    texpresso: bool,
    image: Option<image::RgbImage>,
    original_image: Option<TextureHandle>,
    compressed_image: Vec<texpak::bc1::Block>,
    decompressed_image: Option<TextureHandle>,
    total_error: f32,
}

impl TexPakApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        TexPakApp {
            opt: 0,
            texpresso: false,
            image: None,
            original_image: None,
            compressed_image: Vec::new(),
            decompressed_image: None,
            total_error: 0.0,
        }
    }
}

impl eframe::App for TexPakApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut show_original_over_compressed = false;

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Clear").clicked() {
                    self.image = None;
                    self.original_image = None;
                    self.compressed_image.clear();
                    self.decompressed_image = None;
                }

                ui.label("Opt:");
                let r = ui.add(
                    egui::DragValue::new(&mut self.opt)
                        .range(0..=1000)
                        .clamp_existing_to_range(true),
                );

                if r.changed() {
                    self.compressed_image.clear();
                    self.decompressed_image = None;
                }

                let r = ui.button("Press me");

                show_original_over_compressed = r.is_pointer_button_down_on();

                ui.label("Total error:");
                ui.strong(format!("{:.4}", self.total_error));

                if ui.checkbox(&mut self.texpresso, "Use Texpresso").changed() {
                    self.compressed_image.clear();
                    self.decompressed_image = None;
                }
            });
        });

        if let Some(dropped_file) = ctx.input(|i| i.raw.dropped_files.last().cloned()) {
            if let Some(path) = &dropped_file.path {
                if let Ok(image) = image::open(path) {
                    self.image = Some(image.into_rgb8());
                    self.compressed_image.clear();
                    self.original_image = None;
                    self.decompressed_image = None;
                }
            }
        }

        if self.original_image.is_none() {
            if let Some(image) = &self.image {
                self.original_image = Some(ctx.load_texture(
                    "Original",
                    ColorImage {
                        size: [image.width() as usize, image.height() as usize],
                        pixels: image.pixels().map(|&p| rgb_image_to_egui(p)).collect(),
                    },
                    TextureOptions::NEAREST,
                ));
            }
        }

        if self.compressed_image.is_empty() {
            self.total_error = 0.0;
            if let Some(image) = &self.image {
                let mut blocks = Vec::new();
                for y in (0..image.height()).step_by(4) {
                    for x in (0..image.width()).step_by(4) {
                        if self.texpresso {
                            let block = [
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 0).min(image.width() - 1),
                                    (y + 0).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 1).min(image.width() - 1),
                                    (y + 0).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 2).min(image.width() - 1),
                                    (y + 0).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 3).min(image.width() - 1),
                                    (y + 0).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 0).min(image.width() - 1),
                                    (y + 1).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 1).min(image.width() - 1),
                                    (y + 1).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 2).min(image.width() - 1),
                                    (y + 1).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 3).min(image.width() - 1),
                                    (y + 1).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 0).min(image.width() - 1),
                                    (y + 2).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 1).min(image.width() - 1),
                                    (y + 2).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 2).min(image.width() - 1),
                                    (y + 2).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 3).min(image.width() - 1),
                                    (y + 2).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 0).min(image.width() - 1),
                                    (y + 3).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 1).min(image.width() - 1),
                                    (y + 3).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 2).min(image.width() - 1),
                                    (y + 3).min(image.height() - 1),
                                )),
                                rgb_image_to_raw(*image.get_pixel(
                                    (x + 3).min(image.width() - 1),
                                    (y + 3).min(image.height() - 1),
                                )),
                            ];

                            let mut encoded = [0; 8];
                            texpresso::Format::Bc1.compress_block_masked(
                                block,
                                !0,
                                texpresso::Params::default(),
                                &mut encoded,
                            );

                            blocks.push(Block::from_bytes(encoded));
                        } else {
                            let block = [
                                [
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 0).min(image.width() - 1),
                                        (y + 0).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 1).min(image.width() - 1),
                                        (y + 0).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 2).min(image.width() - 1),
                                        (y + 0).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 3).min(image.width() - 1),
                                        (y + 0).min(image.height() - 1),
                                    )),
                                ],
                                [
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 0).min(image.width() - 1),
                                        (y + 1).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 1).min(image.width() - 1),
                                        (y + 1).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 2).min(image.width() - 1),
                                        (y + 1).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 3).min(image.width() - 1),
                                        (y + 1).min(image.height() - 1),
                                    )),
                                ],
                                [
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 0).min(image.width() - 1),
                                        (y + 2).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 1).min(image.width() - 1),
                                        (y + 2).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 2).min(image.width() - 1),
                                        (y + 2).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 3).min(image.width() - 1),
                                        (y + 2).min(image.height() - 1),
                                    )),
                                ],
                                [
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 0).min(image.width() - 1),
                                        (y + 3).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 1).min(image.width() - 1),
                                        (y + 3).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 2).min(image.width() - 1),
                                        (y + 3).min(image.height() - 1),
                                    )),
                                    rgb_image_to_texpak(*image.get_pixel(
                                        (x + 3).min(image.width() - 1),
                                        (y + 3).min(image.height() - 1),
                                    )),
                                ],
                            ];

                            let block = texpak::bc1::Block::encode(block, self.opt);
                            blocks.push(block);
                        }
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

                for y in (0..image.height()).step_by(4) {
                    let mut blocks_row = Vec::new();

                    for _ in (0..image.width()).step_by(4) {
                        let block = *blocks.next().unwrap();
                        let block = texpak::bc1::Block::decode(block);
                        blocks_row.push(block);
                    }

                    for i in 0..4.min(image.height() - y) {
                        for (x, block) in blocks_row.iter().enumerate() {
                            let x = x as u32 * 4;
                            for j in 0..4.min(image.width() - x) {
                                pixels.push(rgba_texpak_to_egui(block[i as usize][j as usize]));
                            }
                        }
                    }
                }

                self.total_error = 0.0;
                for y in 0..image.height() {
                    for x in 0..image.width() {
                        let o = rgb_image_to_texpak(*image.get_pixel(x, y));
                        let d = rgb_egui_to_texpak(pixels[(y * image.width() + x) as usize]);
                        self.total_error += Rgb32F::distance(o, d);
                    }
                }

                self.decompressed_image = Some(ctx.load_texture(
                    "Decompressed",
                    ColorImage {
                        size: [image.width() as usize, image.height() as usize],
                        pixels,
                    },
                    TextureOptions::NEAREST,
                ));
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let size = ui.available_size();

            ui.horizontal(|ui| {
                if let Some(image) = &self.original_image {
                    ui.add(
                        egui::Image::new(image).fit_to_exact_size(egui::vec2(size.x * 0.5, size.y)),
                    );
                }

                if show_original_over_compressed {
                    if let Some(image) = &self.original_image {
                        ui.add(
                            egui::Image::new(image)
                                .fit_to_exact_size(egui::vec2(size.x * 0.5, size.y)),
                        );
                    }
                } else {
                    if let Some(image) = &self.decompressed_image {
                        ui.add(
                            egui::Image::new(image)
                                .fit_to_exact_size(egui::vec2(size.x * 0.5, size.y)),
                        );
                    }
                }
            });
        });
    }
}

fn rgb_image_to_egui(rgb: image::Rgb<u8>) -> egui::Color32 {
    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

fn rgb_image_to_raw(rgb: image::Rgb<u8>) -> [u8; 4] {
    [rgb[0], rgb[1], rgb[2], 255]
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

fn rgb_egui_to_texpak(rgb: Color32) -> Rgb32F {
    Rgb32F::new(
        rgb.r() as f32 / 255.0,
        rgb.g() as f32 / 255.0,
        rgb.b() as f32 / 255.0,
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
