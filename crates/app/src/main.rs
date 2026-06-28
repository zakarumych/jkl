use std::sync::{Arc, mpsc};
use std::thread;

use eframe::{
    egui::{self, Align2, Color32, FontId, Sense},
    egui_wgpu::{WgpuSetup, WgpuSetupCreateNew, wgpu},
};

use jkl::image::{ImageMut, ImageRef, OwnedImage};
use jkl::{
    image::block::{bc1, bc2},
    math::{Rgb8U, Rgb565, Rgba8U},
};
use jkl_gui::{EncoderState, EncodingWidget, Format, Image};

fn input_from_dynamic(img: image::DynamicImage) -> Result<Image, String> {
    use image::DynamicImage;
    match img {
        DynamicImage::ImageRgb8(buf) => {
            let (w, h) = buf.dimensions();
            let pixels: Box<[Rgb8U]> = buf
                .into_raw()
                .chunks_exact(3)
                .map(|c| Rgb8U::new(c[0], c[1], c[2]))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            Ok(Image::Rgb8(OwnedImage::new_2d(
                w as usize, h as usize, pixels,
            )))
        }
        DynamicImage::ImageRgba8(buf) => {
            let (w, h) = buf.dimensions();
            let pixels: Box<[Rgba8U]> = buf
                .into_raw()
                .chunks_exact(4)
                .map(|c| Rgba8U::new(c[0], c[1], c[2], c[3]))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            Ok(Image::Rgba8(OwnedImage::new_2d(
                w as usize, h as usize, pixels,
            )))
        }
        other => Err(format!(
            "unsupported pixel format {:?}; image must be RGB8 or RGBA8",
            other.color()
        )),
    }
}

fn main() {
    let mut native_options = eframe::NativeOptions::default();
    native_options.wgpu_options.wgpu_setup = WgpuSetup::CreateNew(WgpuSetupCreateNew {
        device_descriptor: Arc::new(|adapter| {
            let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            };
            wgpu::DeviceDescriptor {
                label: Some("egui wgpu device"),
                required_features: wgpu::Features::SHADER_INT64
                    | wgpu::Features::TEXTURE_COMPRESSION_BC
                    | wgpu::Features::SUBGROUP,
                required_limits: wgpu::Limits {
                    max_texture_dimension_2d: 8192,
                    ..base_limits
                },
                ..Default::default()
            }
        }),
        ..Default::default()
    });

    eframe::run_native(
        "Jackal",
        native_options,
        Box::new(|cc| {
            let wgpu_state = cc.wgpu_render_state.clone();
            Ok(Box::new(App {
                wgpu_state,
                encoder: None,
                status: None,
            }))
        }),
    )
    .unwrap();
}

// ── App state ─────────────────────────────────────────────────────────────

enum Status {
    Info(String),
    Error(String),
}

struct App {
    wgpu_state: Option<eframe::egui_wgpu::RenderState>,
    encoder: Option<EncoderState>,
    status: Option<Status>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle dropped files.
        if let Some(dropped) = ctx.input(|i| i.raw.dropped_files.last().cloned()) {
            if let Some(path) = &dropped.path {
                match image::open(path) {
                    Ok(img) => match input_from_dynamic(img) {
                        Ok(input) => {
                            let w = input.width();
                            let h = input.height();
                            let fmt = match &input {
                                Image::Rgb8(_) => "RGB8",
                                Image::Rgba8(_) => "RGBA8",
                                _ => unreachable!(),
                            };
                            let encoder = if let Some(rs) = &self.wgpu_state {
                                EncoderState::with_block_encoder(
                                    input,
                                    gpu_block_encoder(rs.device.clone(), rs.queue.clone()),
                                )
                            } else {
                                EncoderState::new(input)
                            };
                            self.encoder = Some(encoder);
                            self.status = Some(Status::Info(format!("Loaded {w}\u{d7}{h} {fmt}")));
                        }
                        Err(e) => {
                            self.status = Some(Status::Error(e));
                        }
                    },
                    Err(e) => {
                        self.status = Some(Status::Error(format!("Cannot open image: {e}")));
                    }
                }
            }
        }

        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(20.0)
            .show(ctx, |ui| {
                ui.with_layout(
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| match &self.status {
                        Some(Status::Error(text)) => {
                            ui.colored_label(Color32::LIGHT_RED, text);
                        }
                        Some(Status::Info(text)) => {
                            ui.label(text);
                        }
                        None => {}
                    },
                );
            });

        let mut save_data: Option<Arc<Vec<u8>>> = None;

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(encoder) = &mut self.encoder {
                EncodingWidget::new("encoding", encoder).show(ctx, ui);
                let ready = encoder.current_jkli_data();
                ui.add_space(8.0);
                if ui
                    .add_enabled(ready.is_some(), egui::Button::new("Save\u{2026}"))
                    .clicked()
                {
                    save_data = ready;
                }
            } else {
                // Drop zone
                let rect = ui.available_rect_before_wrap();
                ui.allocate_rect(rect, Sense::hover());

                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "Drop an image here",
                    FontId::proportional(22.0),
                    Color32::GRAY,
                );
            }
        });

        if let Some(data) = save_data {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Jackal Image", &["jkli"])
                .save_file()
            {
                self.status = Some(match std::fs::write(&path, data.as_slice()) {
                    Ok(()) => Status::Info(format!("Saved: {}", path.display())),
                    Err(e) => Status::Error(format!("Save failed: {e}")),
                });
            }
        }
    }
}

// ── GPU encoder ──────────────────────────────────────────────────────────────────

fn gpu_block_encoder(
    device: wgpu::Device,
    queue: wgpu::Queue,
) -> impl Fn(Arc<Image>, Format, mpsc::Sender<Result<Image, String>>) + Send + Sync + 'static {
    use jkl_wgpu::image::blocks::BlockCompressor;
    let compressor = Arc::new(BlockCompressor::new(&device));

    move |input: Arc<Image>, format: Format, tx: mpsc::Sender<Result<Image, String>>| {
        let device = device.clone();
        let queue = queue.clone();
        let compressor = Arc::clone(&compressor);
        thread::spawn(move || {
            let result = match format {
                Format::RGB8 => {
                    let w = input.width();
                    let h = input.height();
                    let pixels: Vec<Rgb8U> = match &*input {
                        Image::Rgb8(img) => img.data().iter().copied().collect(),
                        Image::Rgba8(img) => img
                            .data()
                            .iter()
                            .map(|p| Rgb8U::new(p.0[0], p.0[1], p.0[2]))
                            .collect(),
                        _ => unreachable!(),
                    };
                    Ok(Image::Rgb8(OwnedImage::new_2d(
                        w,
                        h,
                        pixels.into_boxed_slice(),
                    )))
                }
                Format::BC1 => gpu_encode_bc1(&input, &compressor, &device, &queue),
                Format::BC2 => gpu_encode_bc2(&input, &compressor, &device, &queue),
                _ => unimplemented!(),
            };
            let _ = tx.send(result);
        });
    }
}

fn rgba_to_gpu(
    input: &Image,
    device: &wgpu::Device,
) -> jkl::image::Image<jkl_wgpu::image::GpuPixels> {
    use jkl_wgpu::image::WgpuImage;

    let w = input.width();
    let h = input.height();

    let rgba: Vec<Rgba8U> = match input {
        Image::Rgb8(img) => img
            .data()
            .iter()
            .map(|p| Rgba8U::new(p.0[0], p.0[1], p.0[2], 255))
            .collect(),
        Image::Rgba8(img) => img.data().iter().copied().collect(),
        _ => unreachable!(),
    };

    <jkl::image::Image<jkl_wgpu::image::GpuPixels> as WgpuImage>::upload(
        device,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::BufferUsages::STORAGE,
        ImageRef::new_2d(w, h, rgba.as_slice()),
        |p: Rgba8U| u32::from_le_bytes(p.0),
    )
}

fn gpu_encode_bc1(
    input: &Image,
    compressor: &jkl_wgpu::image::blocks::BlockCompressor,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<Image, String> {
    use jkl_wgpu::image::WgpuImage;

    let gpu_input = rgba_to_gpu(input, device);
    let gpu_output = compressor.compress_rgba_to_bc1(gpu_input, 0.5, device, queue, 1 << 20);

    let [bw, bh, _] = gpu_output.extent().raw_size();

    let staging: jkl::image::Image<jkl_wgpu::image::GpuPixels> =
        <jkl::image::Image<jkl_wgpu::image::GpuPixels> as WgpuImage>::new(
            device,
            wgpu::TextureFormat::Bc1RgbaUnorm,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            gpu_output.extent(),
        );

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jkl-bc1-readback"),
    });
    gpu_output.copy_to(&staging, &mut enc);
    staging.map_on_submit(wgpu::MapMode::Read, &mut enc);
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mut blocks = vec![
        bc1::Block {
            color0: Rgb565::BLACK,
            color1: Rgb565::BLACK,
            indices: [0; 4],
        };
        bw * bh
    ];
    staging.download(
        ImageMut::new_2d(bw, bh, blocks.as_mut_slice()),
        |raw: [u8; 8]| bc1::Block {
            color0: Rgb565::from_bits(u16::from_le_bytes([raw[0], raw[1]])),
            color1: Rgb565::from_bits(u16::from_le_bytes([raw[2], raw[3]])),
            indices: [raw[4], raw[5], raw[6], raw[7]],
        },
    );

    Ok(Image::Bc1(OwnedImage::new_2d(
        bw,
        bh,
        blocks.into_boxed_slice(),
    )))
}

fn gpu_encode_bc2(
    input: &Image,
    compressor: &jkl_wgpu::image::blocks::BlockCompressor,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<Image, String> {
    use jkl_wgpu::image::WgpuImage;

    let gpu_input = rgba_to_gpu(input, device);
    let gpu_output = compressor.compress_rgba_to_bc2(gpu_input, device, queue, 1 << 20);

    let [bw, bh, _] = gpu_output.extent().raw_size();

    let staging: jkl::image::Image<jkl_wgpu::image::GpuPixels> =
        <jkl::image::Image<jkl_wgpu::image::GpuPixels> as WgpuImage>::new(
            device,
            wgpu::TextureFormat::Bc2RgbaUnorm,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            gpu_output.extent(),
        );

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jkl-bc2-readback"),
    });
    gpu_output.copy_to(&staging, &mut enc);
    staging.map_on_submit(wgpu::MapMode::Read, &mut enc);
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mut blocks = vec![
        bc2::Block {
            alpha: [0; 8],
            color0: Rgb565::BLACK,
            color1: Rgb565::BLACK,
            indices: [0; 4],
        };
        bw * bh
    ];
    staging.download(
        ImageMut::new_2d(bw, bh, blocks.as_mut_slice()),
        |raw: [u8; 16]| bc2::Block {
            alpha: [
                raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
            ],
            color0: Rgb565::from_bits(u16::from_le_bytes([raw[8], raw[9]])),
            color1: Rgb565::from_bits(u16::from_le_bytes([raw[10], raw[11]])),
            indices: [raw[12], raw[13], raw[14], raw[15]],
        },
    );

    Ok(Image::Bc2(OwnedImage::new_2d(
        bw,
        bh,
        blocks.into_boxed_slice(),
    )))
}
