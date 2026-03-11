use std::{
    fs::File,
    sync::{Arc, Mutex},
};

use eframe::{
    egui::{self, Align2, Color32, FontId, Pos2, Sense, Vec2},
    egui_wgpu::{
        Callback, CallbackResources, CallbackTrait, ScreenDescriptor, WgpuSetup,
        WgpuSetupCreateNew, wgpu,
    },
};
use jkl::jackal::image::JackalReader;
use jkl_wgpu::Uploader;

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
                required_features: wgpu::Features::SHADER_INT64 | wgpu::Features::TEXTURE_COMPRESSION_BC,
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
        Box::new(|cc| Ok(Box::new(Jackal::new(cc)))),
    )
    .unwrap();
}

struct PendingDecode {
    reader: JackalReader<File>,
}

struct PreviewPipeline {
    render_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

struct GpuPreview {
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

struct SharedPreview {
    pending: Option<PendingDecode>,
    gpu: Option<GpuPreview>,
    render_pipeline: Option<PreviewPipeline>,
    uploader: Option<Uploader>,
    target_format: wgpu::TextureFormat,
    last_error: Option<String>,
}

#[derive(Clone)]
struct PreviewCallback {
    shared: Arc<Mutex<SharedPreview>>,
}

struct Jackal {
    shared: Arc<Mutex<SharedPreview>>,
}

impl Jackal {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Jackal {
            shared: Arc::new(Mutex::new(SharedPreview {
                pending: None,
                gpu: None,
                render_pipeline: None,
                uploader: None,
                target_format: cc
                    .wgpu_render_state
                    .as_ref()
                    .map(|s| s.target_format)
                    .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb),
                last_error: None,
            })),
        }
    }

    fn load_jkli_preview(&self, path: &std::path::Path) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("failed to open file: {e}"))?;
        let reader = JackalReader::open(file).map_err(|e| format!("failed to open .jkli: {e}"))?;

        let mut shared = self
            .shared
            .lock()
            .map_err(|_| "preview state mutex poisoned".to_owned())?;

        shared.pending = Some(PendingDecode { reader });
        shared.last_error = None;

        Ok(())
    }

    fn preview_size(&self) -> Option<[u32; 2]> {
        let shared = self.shared.lock().ok()?;

        if let Some(gpu) = &shared.gpu {
            return Some([gpu.width, gpu.height]);
        }

        if let Some(pending) = &shared.pending {
            let extent = pending.reader.extent();
            return Some([extent.width() as u32, extent.depth() as u32]);
        }

        None
    }

    fn paint_preview(&self, ui: &mut egui::Ui, image_size: [u32; 2]) {
        let available = ui.available_size();
        let image_aspect = image_size[0] as f32 / image_size[1] as f32;
        let panel_aspect = if available.y > 0.0 {
            available.x / available.y
        } else {
            image_aspect
        };

        let draw_size = if panel_aspect > image_aspect {
            Vec2::new(available.y * image_aspect, available.y)
        } else {
            Vec2::new(available.x, available.x / image_aspect)
        };

        let (rect, _) = ui.allocate_exact_size(draw_size, Sense::hover());
        let callback = Callback::new_paint_callback(
            rect,
            PreviewCallback {
                shared: Arc::clone(&self.shared),
            },
        );
        ui.painter().add(callback);

        let text = format!("{} x {}", image_size[0], image_size[1]);
        ui.painter().text(
            Pos2::new(rect.left() + 8.0, rect.bottom() - 8.0),
            Align2::LEFT_BOTTOM,
            text,
            FontId::proportional(12.0),
            Color32::WHITE,
        );
    }
}

impl PreviewPipeline {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-gui-preview-shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0)
var image_tex: texture_2d<f32>;
@group(0) @binding(1)
var image_sampler: sampler;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;

    let pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0),
    );

    out.position = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv = uv[vi];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(image_tex, image_sampler, in.uv);
}
"#
                .into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jkl-gui-preview-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jkl-gui-preview-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jkl-gui-preview-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jkl-gui-preview-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            render_pipeline,
            bind_group_layout,
            sampler,
        }
    }
}

impl CallbackTrait for PreviewCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        _resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut shared = match self.shared.lock() {
            Ok(guard) => guard,
            Err(_) => return Vec::new(),
        };
        let shared = &mut *shared;

        if shared.render_pipeline.is_none() {
            shared.render_pipeline = Some(PreviewPipeline::new(device, shared.target_format));
        }

        let pending = match &mut shared.pending {
            Some(p) => p,
            None => return Vec::new(),
        };

        // ensure uploader exists
        if shared.uploader.is_none() {
            shared.uploader = Some(Uploader::new(device));
        }
        let uploader = shared.uploader.as_ref().unwrap();

        let texture = match uploader.upload_from_reader(&mut pending.reader, device, encoder) {
            Ok(tex) => tex,
            Err(e) => {
                shared.last_error = Some(format!("upload failed: {e}"));
                return Vec::new();
            }
        };

        shared.pending = None;

        let uploaded = texture; // we already have it

        let render = shared
            .render_pipeline
            .as_ref()
            .expect("render pipeline should exist");

        let sampled_view = uploaded.create_view(&wgpu::TextureViewDescriptor::default());
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jkl-gui-preview-bg"),
            layout: &render.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&sampled_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&render.sampler),
                },
            ],
        });

        shared.gpu = Some(GpuPreview {
            bind_group: render_bind_group,
            width: uploaded.width(),
            height: uploaded.height(),
        });

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _resources: &CallbackResources,
    ) {
        let shared = match self.shared.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };

        let pipeline = match &shared.render_pipeline {
            Some(p) => p,
            None => return,
        };
        let gpu = match &shared.gpu {
            Some(g) => g,
            None => return,
        };

        render_pass.set_pipeline(&pipeline.render_pipeline);
        render_pass.set_bind_group(0, &gpu.bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }
}

impl eframe::App for Jackal {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.label("Drop a .jkli file to preview (RGB8 + ANS via GPU shader)");

            if let Ok(shared) = self.shared.lock() {
                if let Some(err) = &shared.last_error {
                    ui.colored_label(Color32::LIGHT_RED, err);
                }
            }
        });

        if let Some(dropped_file) = ctx.input(|i| i.raw.dropped_files.last().cloned()) {
            if let Some(path) = &dropped_file.path {
                let is_jkli = path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("jkli"))
                    .unwrap_or(false);

                let mut shared = match self.shared.lock() {
                    Ok(s) => s,
                    Err(_) => return,
                };

                if !is_jkli {
                    shared.last_error = Some("Only .jkli files are supported".to_owned());
                } else {
                    drop(shared);
                    if let Err(err) = self.load_jkli_preview(path) {
                        if let Ok(mut s) = self.shared.lock() {
                            s.last_error = Some(err);
                        }
                    }
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(size) = self.preview_size() {
                self.paint_preview(ui, size);
            } else {
                let rect = ui.available_rect_before_wrap();
                ui.allocate_rect(rect, Sense::hover());
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "Drop a .jkli file to start preview",
                    FontId::proportional(18.0),
                    Color32::GRAY,
                );
            }
        });
    }
}
