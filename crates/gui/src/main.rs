use std::{
    fs::File,
    sync::{Arc, Mutex},
};

use eframe::egui_wgpu::wgpu::util::DeviceExt;
use eframe::{
    egui::{self, Align2, Color32, FontId, Pos2, Sense, Vec2},
    egui_wgpu::{
        Callback, CallbackResources, CallbackTrait, ScreenDescriptor, WgpuSetup,
        WgpuSetupCreateNew, wgpu,
    },
};
use jkl::{
    image::format::Format,
    jackal::image::{Compression, JackalReader, decompress_wgsl_kernel},
};

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
                required_features: wgpu::Features::SHADER_INT64,
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

#[derive(Clone)]
struct PendingDecode {
    width: u32,
    height: u32,
    generation: u64,
    payload_words: Vec<u32>,
    tile_word_offsets: Vec<u32>,
    symbol_cumul: Vec<u32>,
    symbol_freq: Vec<u32>,
    symbol_rgb8: Vec<u32>,
    tile_meta: Vec<u32>,
    ans_total: u32,
}

struct PreviewPipeline {
    render_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

struct GpuPreview {
    bind_group: wgpu::BindGroup,
    texture: wgpu::Texture,
    generation: u64,
    width: u32,
    height: u32,
}

struct SharedPreview {
    pending: Option<PendingDecode>,
    gpu: Option<GpuPreview>,
    render_pipeline: Option<PreviewPipeline>,
    compute_pipeline: Option<ComputePipeline>,
    target_format: wgpu::TextureFormat,
    generation: u64,
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
                compute_pipeline: None,
                target_format: cc
                    .wgpu_render_state
                    .as_ref()
                    .map(|s| s.target_format)
                    .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb),
                generation: 0,
                last_error: None,
            })),
        }
    }

    fn load_jkli_preview(&self, path: &std::path::Path) -> Result<(), String> {
        let file =
            File::open(path).map_err(|e| format!("failed to open '{}': {e}", path.display()))?;

        let mut reader = JackalReader::open(file)
            .map_err(|e| format!("failed to parse JKLI '{}': {e}", path.display()))?;

        if reader.format() != Format::RGB8 {
            return Err(format!(
                "unsupported format {:?}, expected RGB8",
                reader.format()
            ));
        }

        if reader.compression() != Compression::Ans {
            return Err(format!(
                "unsupported compression {:?}, expected Ans",
                reader.compression()
            ));
        }

        let [width_usize, height_usize, _] = reader.extent().raw_size();
        let width =
            u32::try_from(width_usize).map_err(|_| "image width does not fit u32".to_owned())?;
        let height =
            u32::try_from(height_usize).map_err(|_| "image height does not fit u32".to_owned())?;

        let payload_len = reader
            .tile_payload_len_bytes()
            .map_err(|e| format!("failed to query payload length: {e}"))?;

        if payload_len % 4 != 0 {
            return Err("tile payload byte length is not 4-byte aligned".to_owned());
        }

        let mut payload_bytes = vec![0u8; payload_len];
        let tile_offsets_bytes = reader
            .read_all_tile_payloads_into(&mut payload_bytes)
            .map_err(|e| format!("failed to read tile payload: {e}"))?;

        let mut tile_word_offsets = Vec::with_capacity(tile_offsets_bytes.len());
        for &offset in &tile_offsets_bytes {
            if offset % 4 != 0 {
                return Err("tile payload offsets are not 4-byte aligned".to_owned());
            }

            let word = u32::try_from(offset / 4)
                .map_err(|_| "tile payload offset does not fit u32".to_owned())?;
            tile_word_offsets.push(word);
        }

        let payload_words = payload_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();

        let ctx = reader
            .read_rgb8_ans_gpu_context()
            .map_err(|e| format!("failed to read ANS context: {e}"))?;

        let mut tile_meta = Vec::with_capacity(reader.tiles() * 2);
        for tile_index in 0..reader.tiles() {
            let tile = reader.tile(tile_index);

            let x = u32::try_from(tile.rect.x).map_err(|_| "tile x does not fit u32".to_owned())?;
            let y = u32::try_from(tile.rect.y).map_err(|_| "tile y does not fit u32".to_owned())?;
            let w =
                u32::try_from(tile.rect.w).map_err(|_| "tile width does not fit u32".to_owned())?;
            let h = u32::try_from(tile.rect.h)
                .map_err(|_| "tile height does not fit u32".to_owned())?;

            if x > 0xFFFF || y > 0xFFFF || w > 0xFFFF || h > 0xFFFF {
                return Err("tile metadata exceeds 16-bit packing range".to_owned());
            }

            tile_meta.push((y << 16) | x);
            tile_meta.push((h << 16) | w);
        }

        let mut shared = self
            .shared
            .lock()
            .map_err(|_| "preview state mutex poisoned".to_owned())?;

        shared.generation = shared.generation.wrapping_add(1);
        shared.pending = Some(PendingDecode {
            width,
            height,
            generation: shared.generation,
            payload_words,
            tile_word_offsets,
            symbol_cumul: ctx.symbol_cumul,
            symbol_freq: ctx.symbol_freq,
            symbol_rgb8: ctx.symbol_rgb8,
            tile_meta,
            ans_total: ctx.ans_total,
        });
        shared.last_error = None;

        Ok(())
    }

    fn preview_size(&self) -> Option<[u32; 2]> {
        let shared = self.shared.lock().ok()?;

        if let Some(gpu) = &shared.gpu {
            return Some([gpu.width, gpu.height]);
        }

        shared.pending.as_ref().map(|p| [p.width, p.height])
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

impl ComputePipeline {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-gui-rgb8-rans-decompress-shader"),
            source: wgpu::ShaderSource::Wgsl(
                decompress_wgsl_kernel(Format::RGB8, Compression::Ans).into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jkl-gui-rgb8-rans-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jkl-gui-rgb8-rans-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-gui-rgb8-rans-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("decompress_rgb8_rans"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

fn encode_params(p: &PendingDecode) -> [u8; 32] {
    let mut out = [0u8; 32];
    let tile_count = u32::try_from(p.tile_word_offsets.len().saturating_sub(1)).unwrap_or(0);

    out[0..4].copy_from_slice(&tile_count.to_le_bytes());
    out[4..8].copy_from_slice(&p.width.to_le_bytes());
    out[8..12].copy_from_slice(&p.height.to_le_bytes());
    out[12..16].copy_from_slice(&p.ans_total.to_le_bytes());
    out[16..20].copy_from_slice(&(u32::try_from(p.symbol_cumul.len()).unwrap_or(0)).to_le_bytes());

    out
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

        if shared.render_pipeline.is_none() {
            shared.render_pipeline = Some(PreviewPipeline::new(device, shared.target_format));
        }
        if shared.compute_pipeline.is_none() {
            shared.compute_pipeline = Some(ComputePipeline::new(device));
        }

        let pending = match &shared.pending {
            Some(p) => p.clone(),
            None => return Vec::new(),
        };

        let uploaded_generation = shared.gpu.as_ref().map(|g| g.generation).unwrap_or(0);
        if uploaded_generation == pending.generation {
            return Vec::new();
        }

        let payload_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-gui-payload-words"),
            contents: bytemuck::cast_slice(&pending.payload_words),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-gui-tile-word-offsets"),
            contents: bytemuck::cast_slice(&pending.tile_word_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let symbol_cumul_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-gui-symbol-cumul"),
            contents: bytemuck::cast_slice(&pending.symbol_cumul),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let symbol_freq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-gui-symbol-freq"),
            contents: bytemuck::cast_slice(&pending.symbol_freq),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let symbol_rgb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-gui-symbol-rgb"),
            contents: bytemuck::cast_slice(&pending.symbol_rgb8),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let tile_meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-gui-tile-meta"),
            contents: bytemuck::cast_slice(&pending.tile_meta),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let params_bytes = encode_params(&pending);
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-gui-decode-params"),
            contents: &params_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let out_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jkl-gui-output-texture"),
            size: wgpu::Extent3d {
                width: pending.width,
                height: pending.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let compute = shared
            .compute_pipeline
            .as_ref()
            .expect("compute pipeline should exist");

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jkl-gui-decode-bg"),
            layout: &compute.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: payload_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: symbol_cumul_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: symbol_freq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: symbol_rgb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: tile_meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jkl-gui-rgb8-rans-decode-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute.pipeline);
            cpass.set_bind_group(0, &compute_bind_group, &[]);

            let tile_count =
                u32::try_from(pending.tile_word_offsets.len().saturating_sub(1)).unwrap_or(0);
            let groups_x = tile_count.div_ceil(64);
            if groups_x > 0 {
                cpass.dispatch_workgroups(groups_x, 1, 1);
            }
        }

        let render = shared
            .render_pipeline
            .as_ref()
            .expect("render pipeline should exist");

        let sampled_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());
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
            texture: out_texture,
            generation: pending.generation,
            width: pending.width,
            height: pending.height,
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

        let _keep_alive = &gpu.texture;

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
