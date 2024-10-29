use std::time::Duration;

use eframe::egui::{
    self, Color32, ColorImage, ImageData, TextureHandle, TextureId, TextureOptions,
};
use jkl::{
    bc1::{self, Block},
    math::{Rgb32F, Rgb565, Rgb8U, Rgba32F},
    nn::{self, Diffs},
    z_curve::BoundZCurve,
};
use rand::Rng;

fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Jackal",
        native_options,
        Box::new(|cc| Ok(Box::new(Jackal::new(cc)))),
    )
    .unwrap();
}

struct Jackal {
    opt: usize,
    image: Option<image::RgbImage>,
    original_image: Option<TextureHandle>,

    compressed_image: Vec<bc1::Block>,
    decompressed_image: Option<TextureHandle>,
    total_error: f32,

    nn_image: Vec<Rgb565>,
    decompressed_nn_image: Option<TextureHandle>,
    nn_model: nn::Model,
    nn_adam: nn::Adam,

    nn_diff_image: Vec<Rgb565>,
    decompressed_nn_diff_image: Option<TextureHandle>,

    jkl_image: Vec<u8>,
    jkl_image_blocks: Vec<bc1::Block>,
    decompressed_jkl_image: Option<TextureHandle>,
    total_jkl_error: f32,
}

impl Jackal {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Jackal {
            opt: 0,
            image: None,
            original_image: None,

            compressed_image: Vec::new(),
            decompressed_image: None,
            total_error: 0.0,

            nn_image: Vec::new(),
            decompressed_nn_image: None,
            nn_model: nn::Model::new(),
            nn_adam: nn::Adam::new(),

            nn_diff_image: Vec::new(),
            decompressed_nn_diff_image: None,

            jkl_image: Vec::new(),
            jkl_image_blocks: Vec::new(),
            decompressed_jkl_image: None,
            total_jkl_error: 0.0,
        }
    }
}

impl eframe::App for Jackal {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut show_original_over_compressed = false;
        let mut compress_into_jackal = false;

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Clear").clicked() {
                    self.image = None;
                    self.original_image = None;
                    self.compressed_image.clear();
                    self.decompressed_image = None;
                    self.nn_image.clear();
                    self.decompressed_nn_image = None;
                    self.nn_diff_image.clear();
                    self.decompressed_nn_diff_image = None;
                    self.jkl_image.clear();
                    self.jkl_image_blocks.clear();
                    self.decompressed_jkl_image = None;
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
                    self.nn_image.clear();
                    self.decompressed_nn_image = None;
                    self.nn_diff_image.clear();
                    self.decompressed_nn_diff_image = None;
                    self.jkl_image.clear();
                    self.jkl_image_blocks.clear();
                    self.decompressed_jkl_image = None;
                }

                let r = ui.button("Press me");

                show_original_over_compressed = r.is_pointer_button_down_on();
                ui.separator();

                ui.label("Total error:");
                ui.strong(format!("{:.4}", self.total_error));

                ui.separator();

                ui.label("BC1 size:");
                ui.strong(format!("{:.4}", size_of_val(&self.compressed_image[..])));

                ui.separator();

                let r = ui.button("Compress into Jackal");
                compress_into_jackal = r.clicked();

                if !self.jkl_image.is_empty() {
                    ui.label("JKL size:");
                    ui.strong(format!("{:.4}", size_of_val(&self.jkl_image[..])));
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
                    self.nn_image.clear();
                    self.decompressed_nn_image = None;
                    self.nn_diff_image.clear();
                    self.decompressed_nn_diff_image = None;
                    self.jkl_image.clear();
                    self.jkl_image_blocks.clear();
                    self.decompressed_jkl_image = None;
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
                for y in (0..image.height()).step_by(4) {
                    for x in (0..image.width()).step_by(4) {
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

                        let block = bc1::Block::encode(block, self.opt);
                        self.compressed_image.push(block);
                    }
                }
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
                        let block = bc1::Block::decode(block);
                        blocks_row.push(block);
                    }

                    for i in 0..4.min(image.height() - y) {
                        for (x, block) in blocks_row.iter().enumerate() {
                            let x = x as u32 * 4;
                            for j in 0..4.min(image.width() - x) {
                                pixels.push(rgb_texpak_to_egui(block[i as usize][j as usize]));
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

        if self.nn_image.is_empty() {
            if let Some(_) = &self.image {
                let blocks = &self.compressed_image[..];
                self.nn_image = blocks.iter().map(|b| b.color0).collect();
            }
        }

        if let Some(image) = &self.image {
            ctx.request_repaint();

            let width = ((image.width() + 3) / 4) as usize;
            let height = ((image.height() + 3) / 4) as usize;
            let blocks = &self.compressed_image[..];

            let get_kernel = |index, blocks: &[bc1::Block]| {
                let x: usize = index % width;
                let y: usize = index / width;

                let mut kernel: [Option<&Block>; 15] = [None; 15];

                let mut ki = 0;
                for dx in 0..4 {
                    for dy in 0..4 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }

                        let kx = x.saturating_sub(dx);
                        let ky = y.saturating_sub(dy);

                        if kx != x || ky != y {
                            kernel[ki] = Some(&blocks[ky * width + kx]);
                        }

                        ki += 1;
                    }
                }

                let red_kernel0 = kernel.map(|b| b.map_or(0.0, |b| b.color0.r() as f32 / 31.0));
                let green_kernel0 = kernel.map(|b| b.map_or(0.0, |b| b.color0.g() as f32 / 63.0));
                let blue_kernel0 = kernel.map(|b| b.map_or(0.0, |b| b.color0.b() as f32 / 31.0));

                let mut kernel0 = [0.0; 45];
                kernel0[0..15].copy_from_slice(&red_kernel0);
                kernel0[15..30].copy_from_slice(&green_kernel0);
                kernel0[30..45].copy_from_slice(&blue_kernel0);

                kernel0
            };

            let mut diffs = Diffs::new();
            let mut rng = rand::thread_rng();
            let size = 100;
            for _ in 0..size {
                let x = rng.gen_range(0..width);
                let y = rng.gen_range(0..height);

                let index = y as usize * width as usize + x as usize;

                let kernel0 = get_kernel(index, blocks);

                let block = &blocks[index];

                let expected0 = [
                    block.color0.r() as f32 / 31.0,
                    block.color0.g() as f32 / 63.0,
                    block.color0.b() as f32 / 31.0,
                ];

                let signals = self.nn_model.forward(kernel0);
                self.nn_model.backward(signals, expected0, &mut diffs);
            }
            self.nn_adam.optimize(&mut self.nn_model, &diffs, size);

            let nn_pixels = &mut self.nn_image[..];
            for index in 0..width * height {
                let kernel0 = get_kernel(index, blocks);

                let signals = self.nn_model.forward(kernel0);
                let output0 = signals.output();

                nn_pixels[index] = Rgb565::new(
                    (output0[0] * 31.0).clamp(0.0, 31.0) as u8,
                    (output0[1] * 63.0).clamp(0.0, 63.0) as u8,
                    (output0[2] * 31.0).clamp(0.0, 31.0) as u8,
                );
            }
        }

        match self.decompressed_nn_image {
            None => {
                if let Some(image) = &self.image {
                    assert!(!self.nn_image.is_empty());
                    let pixels = self.nn_image.iter().map(|p| rgb565_to_egui(*p)).collect();

                    let width = ((image.width() + 3) / 4) as usize;
                    let height = ((image.height() + 3) / 4) as usize;

                    self.decompressed_nn_image = Some(ctx.load_texture(
                        "NN Decompressed",
                        ColorImage {
                            size: [width, height],
                            pixels,
                        },
                        TextureOptions::NEAREST,
                    ));
                }
            }
            Some(ref mut texture) => {
                if let Some(image) = &self.image {
                    assert!(!self.nn_image.is_empty());
                    let pixels = self.nn_image.iter().map(|p| rgb565_to_egui(*p)).collect();

                    let width = ((image.width() + 3) / 4) as usize;
                    let height = ((image.height() + 3) / 4) as usize;

                    texture.set(
                        ColorImage {
                            size: [width, height],
                            pixels,
                        },
                        TextureOptions::NEAREST,
                    );
                }
            }
        }

        if !self.compressed_image.is_empty() && !self.nn_image.is_empty() {
            let len = self.compressed_image.len();
            assert_eq!(len, self.nn_image.len());

            if self.nn_diff_image.len() != len {
                self.nn_diff_image = self.nn_image.clone();
            }

            for index in 0..len {
                let block = &self.compressed_image[index];
                let nn_pixel = self.nn_image[index];
                let diff_pixel = &mut self.nn_diff_image[index];

                *diff_pixel = Rgb565::new(
                    (block.color0.r() as i16 - nn_pixel.r() as i16).abs() as u8 & 31,
                    (block.color0.g() as i16 - nn_pixel.g() as i16).abs() as u8 & 63,
                    (block.color0.b() as i16 - nn_pixel.b() as i16).abs() as u8 & 31,
                );
            }
        }

        match self.decompressed_nn_diff_image {
            None => {
                if let Some(image) = &self.image {
                    let pixels = self
                        .nn_diff_image
                        .iter()
                        .map(|p| rgb565_to_egui(*p))
                        .collect();

                    let width = ((image.width() + 3) / 4) as usize;
                    let height = ((image.height() + 3) / 4) as usize;

                    self.decompressed_nn_diff_image = Some(ctx.load_texture(
                        "NN Decompressed",
                        ColorImage {
                            size: [width, height],
                            pixels,
                        },
                        TextureOptions::NEAREST,
                    ));
                }
            }
            Some(ref mut texture) => {
                if let Some(image) = &self.image {
                    assert!(!self.nn_diff_image.is_empty());
                    let pixels = self
                        .nn_diff_image
                        .iter()
                        .map(|p| rgb565_to_egui(*p))
                        .collect();

                    let width = ((image.width() + 3) / 4) as usize;
                    let height = ((image.height() + 3) / 4) as usize;

                    texture.set(
                        ColorImage {
                            size: [width, height],
                            pixels,
                        },
                        TextureOptions::NEAREST,
                    );
                }
            }
        }

        if compress_into_jackal {
            self.jkl_image.clear();
            self.jkl_image_blocks.clear();
            self.decompressed_jkl_image = None;

            if let Some(image) = &self.image {
                let blocks = &self.compressed_image[..];
                let mut output = Vec::new();
                jkl::jackal::compress_bc1_texture(
                    &self.nn_model,
                    jkl::jackal::Extent::D2 {
                        width: (image.width() + 3) / 4,
                        height: (image.height() + 3) / 4,
                    },
                    blocks,
                    std::io::Cursor::new(&mut output),
                )
                .unwrap();

                self.jkl_image = output;

                let (_extent, blocks) =
                    jkl::jackal::decompress_bc1_texture(std::io::Cursor::new(&self.jkl_image[..]))
                        .unwrap();

                self.jkl_image_blocks = blocks;

                let mut blocks = self.jkl_image_blocks.iter();
                let mut pixels = Vec::new();

                for y in (0..image.height()).step_by(4) {
                    let mut blocks_row = Vec::new();

                    for _ in (0..image.width()).step_by(4) {
                        let block = *blocks.next().unwrap();
                        let block = bc1::Block::decode(block);
                        blocks_row.push(block);
                    }

                    for i in 0..4.min(image.height() - y) {
                        for (x, block) in blocks_row.iter().enumerate() {
                            let x = x as u32 * 4;
                            for j in 0..4.min(image.width() - x) {
                                pixels.push(rgb_texpak_to_egui(block[i as usize][j as usize]));
                            }
                        }
                    }
                }

                self.total_jkl_error = 0.0;
                for y in 0..image.height() {
                    for x in 0..image.width() {
                        let o = rgb_image_to_texpak(*image.get_pixel(x, y));
                        let d = rgb_egui_to_texpak(pixels[(y * image.width() + x) as usize]);
                        self.total_jkl_error += Rgb32F::distance(o, d);
                    }
                }

                self.decompressed_jkl_image = Some(ctx.load_texture(
                    "Decompressed JKL",
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
                        egui::Image::new(image)
                            .fit_to_exact_size(egui::vec2(size.x * 0.25, size.y)),
                    );
                }

                if show_original_over_compressed {
                    if let Some(image) = &self.original_image {
                        ui.add(
                            egui::Image::new(image)
                                .fit_to_exact_size(egui::vec2(size.x * 0.25, size.y)),
                        );
                    }
                } else {
                    if let Some(image) = &self.decompressed_image {
                        ui.add(
                            egui::Image::new(image)
                                .fit_to_exact_size(egui::vec2(size.x * 0.25, size.y)),
                        );
                    }
                }

                if let Some(image) = &self.decompressed_nn_image {
                    ui.add(
                        egui::Image::new(image)
                            .fit_to_exact_size(egui::vec2(size.x * 0.25, size.y)),
                    );
                }

                if let Some(image) = &self.decompressed_nn_diff_image {
                    ui.add(
                        egui::Image::new(image)
                            .fit_to_exact_size(egui::vec2(size.x * 0.25, size.y)),
                    );
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

fn rgb565_to_egui(rgb: Rgb565) -> Color32 {
    Color32::from_rgb(
        (rgb.r() as f32 * (255.0 / 31.0)) as u8,
        (rgb.g() as f32 * (255.0 / 63.0)) as u8,
        (rgb.b() as f32 * (255.0 / 31.0)) as u8,
    )
}
