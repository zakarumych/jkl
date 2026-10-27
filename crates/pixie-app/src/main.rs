mod backend;
mod ui;

use pixie_ui::{
    Pixie,
    cursor::{CursorIcon, cursor_system},
    event::PixieEvent,
};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use ui::State;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    keyboard::ModifiersState,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct App {
    state: State,
    target: Option<mev::Image2D>,
    renderer: pixie_ui_mev::Renderer,
    cvt: mev::kernels::Cvt,
    // Queue teardown releases submitted frame references before the surface.
    queue: mev::Queue,
    surface: Option<mev::Surface>,
    window: Option<winit::window::Window>,
    // GPU allocations hold weak device references and must be dropped first.
    device: mev::Device,
    target_size: (u32, u32),
    scale: f64,
    modifiers: ModifiersState,
    cursor_cache: pixie_ui_winit::CursorCache,
    smoke_test: bool,
    frames: u32,
    error: Option<String>,
}

fn upload_previews(
    state: &mut State,
    device: &mev::Device,
    queue: &mut mev::Queue,
    renderer: &mut pixie_ui_mev::Renderer,
) -> Result<()> {
    while let Some((id, width, height, pixels)) = state.preview_upload() {
        if width == 0 || height == 0 || pixels.len() != width as usize * height as usize * 4 {
            return Err("Preview dimensions do not match its pixels".into());
        }
        let image = device.new_image(mev::ImageDesc::new_d2_texture(
            width,
            height,
            mev::PixelFormat::Rgba8Unorm,
        ));
        let staging = device.new_buffer_init(mev::BufferInitDesc {
            data: &pixels,
            usage: mev::BufferUsage::TRANSFER_SRC,
            name: "JKL preview upload",
        });
        let mut encoder = queue.new_command_encoder();
        encoder.init_image(
            mev::PipelineStages::empty(),
            mev::PipelineStages::TRANSFER,
            &image,
        );
        {
            let mut copy = encoder.copy();
            copy.copy_buffer_to_image(
                &staging,
                mev::TexelShape::new(mev::PixelFormat::Rgba8Unorm, image.extent()),
                &image,
            );
            copy.barrier(
                mev::PipelineStages::TRANSFER,
                mev::PipelineStages::FRAGMENT_SHADER,
            );
        }
        queue.submit_checkpoint([encoder.finish()])?;
        // The state reuses its preview IDs, so replacement releases the previous image.
        renderer.insert_texture(id, image);
    }
    Ok(())
}

fn draw_state(
    state: &mut State,
    device: &mev::Device,
    queue: &mut mev::Queue,
    renderer: &mut pixie_ui_mev::Renderer,
    target: &mev::Image2D,
) -> Result<()> {
    state.tick();
    state.prepare();
    upload_previews(state, device, queue, renderer)?;
    let mut pixie = Pixie::from(&mut state.world);
    cursor_system(pixie.world_mut());
    let mut draws = Vec::new();
    pixie.draw_ui(&mut draws);
    let mut encoder = queue.new_command_encoder();
    encoder.init_image(
        mev::PipelineStages::FRAGMENT_SHADER | mev::PipelineStages::COLOR_OUTPUT,
        mev::PipelineStages::COLOR_OUTPUT,
        target,
    );
    drop(encoder.render(mev::RenderPassDesc {
        name: "JKL background",
        color_attachments: &[
            mev::AttachmentDesc::new(target.as_image()).clear(mev::ClearColor::DARK_GRAY),
        ],
        depth_stencil_attachment: None,
    }));
    encoder.barrier(
        mev::PipelineStages::COLOR_OUTPUT,
        mev::PipelineStages::COLOR_OUTPUT,
    );
    queue.submit_checkpoint([encoder.finish()])?;
    renderer.render(queue, target, &draws, &pixie.ui())?;
    Ok(())
}

impl App {
    fn render(&mut self) -> Result<()> {
        let window = self.window.as_ref().ok_or("Window is unavailable")?;
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let width = (size.width as f64 / self.scale).ceil() as u32;
        let height = (size.height as f64 / self.scale).ceil() as u32;
        if self.target_size != (width, height) {
            self.target = Some(self.device.new_image(mev::ImageDesc::new_d2_rt(
                width,
                height,
                mev::PixelFormat::Rgba8Unorm,
            )));
            self.target_size = (width, height);
            self.state.resize(width as i32, height as i32);
        }
        let target = self.target.as_ref().ok_or("Render target is unavailable")?;
        draw_state(
            &mut self.state,
            &self.device,
            &mut self.queue,
            &mut self.renderer,
            target,
        )?;
        let mut frame = self
            .surface
            .as_mut()
            .ok_or("Window surface is unavailable")?
            .next_frame()?;
        self.queue
            .sync_frame(&mut frame, mev::PipelineStages::COLOR_OUTPUT);
        let mut encoder = self.queue.new_command_encoder();
        encoder.init_image(
            mev::PipelineStages::empty(),
            mev::PipelineStages::COLOR_OUTPUT,
            frame.image(),
        );
        encoder.barrier(
            mev::PipelineStages::COLOR_OUTPUT,
            mev::PipelineStages::FRAGMENT_SHADER,
        );
        self.cvt.image_to_image_with(
            &mut encoder,
            target,
            frame.image(),
            mev::kernels::CvtOptions::default().filter(mev::Filter::Nearest),
        );
        window.pre_present_notify();
        encoder.present(frame, mev::PipelineStages::COLOR_OUTPUT);
        self.queue.submit_checkpoint([encoder.finish()])?;
        self.frames += 1;
        Ok(())
    }

    fn fail(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        error: impl std::fmt::Display,
    ) {
        self.error = Some(error.to_string());
        event_loop.exit();
    }
}

impl ApplicationHandler for App {
    fn new_events(
        &mut self,
        _: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        if matches!(cause, winit::event::StartCause::ResumeTimeReached { .. }) {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = winit::window::WindowAttributes::default()
            .with_title("JKL / Asset Workbench")
            .with_inner_size(winit::dpi::LogicalSize::new(1360, 900))
            .with_min_inner_size(winit::dpi::LogicalSize::new(1040, 720));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => window,
            Err(error) => {
                self.fail(event_loop, error);
                return;
            }
        };
        let mut surface = match self.device.new_surface(&window, &window) {
            Ok(surface) => surface,
            Err(error) => {
                self.fail(event_loop, error);
                return;
            }
        };
        let size = window.inner_size();
        self.scale = window.scale_factor();
        surface.preferred_extent(mev::Extent2::new(size.width, size.height));
        surface.preferred_usage(mev::ImageUsage::TARGET);
        surface.preferred_present_mode(mev::PresentMode::Fifo);
        surface.preferred_image_count(3);
        self.state.resize(
            (size.width as f64 / self.scale).ceil() as i32,
            (size.height as f64 / self.scale).ceil() as i32,
        );
        self.state.prepare();
        window.set_ime_allowed(true);
        window.request_redraw();
        self.surface = Some(surface);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let redraw = matches!(event, WindowEvent::RedrawRequested);
        self.state.tick();
        self.state.prepare();
        let mut consumed = false;
        match &event {
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && self.modifiers.control_key() =>
            {
                if let winit::keyboard::Key::Character(text) = &event.logical_key {
                    if let Some(key) = text.chars().next().map(|c| c.to_ascii_lowercase()) {
                        if matches!(key, 'o' | 'e' | 's' | 'd') && !event.repeat {
                            self.state.shortcut(key);
                        }
                        if key == 'a' {
                            let focus = Pixie::from(&mut self.state.world).ui().focused();
                            if let Some(focus) = focus {
                                let length = self
                                    .state
                                    .world
                                    .get::<&pixie_ui::text::Text>(focus)
                                    .map(|text| text.string.len())
                                    .ok();
                                if let (Some(length), Ok(input)) = (
                                    length,
                                    self.state
                                        .world
                                        .get::<&mut pixie_ui::text::TextInput>(focus),
                                ) {
                                    input.selection = 0..length;
                                }
                            }
                        }
                    }
                }
                consumed = true;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(position) => {
                        position.y as f32 / 40.0
                    }
                };
                self.state.wheel(amount);
            }
            WindowEvent::DroppedFile(path) => self.state.open(path.clone(), false),
            WindowEvent::Ime(winit::event::Ime::Commit(text)) => {
                self.state.event(PixieEvent::Paste(text.clone()))
            }
            _ => {}
        }
        if !redraw {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
        if !consumed {
            if let Some(mut event) = pixie_ui_winit::convert_event(&event) {
                if let PixieEvent::CursorMoved { pos } = &mut event {
                    pos.x = (pos.x as f64 / self.scale) as i32;
                    pos.y = (pos.y as f64 / self.scale) as i32;
                }
                self.state.event(event);
            }
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let Some(surface) = self.surface.as_mut() {
                        surface.preferred_extent(mev::Extent2::new(size.width, size.height));
                    }
                    self.state.resize(
                        (size.width as f64 / self.scale).ceil() as i32,
                        (size.height as f64 / self.scale).ceil() as i32,
                    );
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => self.scale = scale_factor,
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.render() {
                    self.fail(event_loop, error);
                    return;
                }
                let pixie = Pixie::from(&mut self.state.world);
                let kind = pixie
                    .world()
                    .get_resource::<CursorIcon>()
                    .map(|icon| icon.kind)
                    .unwrap_or_default();
                let cursor = pixie.ui().cursor_set().get(kind).cloned();
                if let (Some(window), Some(cursor)) = (self.window.as_ref(), cursor) {
                    self.cursor_cache.apply(
                        event_loop,
                        window,
                        kind,
                        &cursor,
                        self.scale.round().max(1.0) as u16,
                    );
                }
                if self.smoke_test && self.frames >= 8 {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(33),
        ));
    }
}

#[derive(Default)]
struct Arguments {
    initial_path: Option<PathBuf>,
    preview: Option<PathBuf>,
    smoke_test: bool,
}

fn arguments() -> Result<Arguments> {
    let mut parsed = Arguments::default();
    let mut args = std::env::args_os().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--render-preview" {
            parsed.preview = Some(
                args.next()
                    .ok_or("--render-preview requires an output directory")?
                    .into(),
            );
        } else if argument == "--smoke-test" {
            parsed.smoke_test = true;
        } else if argument == "--help" || argument == "-h" {
            println!(
                "JKL Asset Workbench\n\nUsage: jkl-pixie [asset-path] [--smoke-test]\n       jkl-pixie --render-preview DIRECTORY\n\nCtrl+O: open asset; Ctrl+E: encode; Ctrl+S: save container; Ctrl+D: export decoded."
            );
            std::process::exit(0);
        } else if argument.to_string_lossy().starts_with('-') {
            return Err(format!("Unknown argument: {}", argument.to_string_lossy()).into());
        } else if parsed.initial_path.replace(argument.into()).is_some() {
            return Err("Only one startup asset can be opened".into());
        }
    }
    Ok(parsed)
}

fn run() -> Result<()> {
    let arguments = arguments()?;
    let instance = mev::Instance::load()?;
    let (device, mut queues) = instance.new_device(mev::DeviceDesc {
        idx: 0,
        queues: &[0],
        features: if arguments.preview.is_some() {
            mev::Features::empty()
        } else {
            mev::Features::SURFACE
        },
    })?;
    let mut queue = queues.pop().ok_or("Graphics queue is unavailable")?;
    if let Some(directory) = arguments.preview {
        render_previews(&device, &mut queue, &directory)?;
        return Ok(());
    }
    let renderer = pixie_ui_mev::Renderer::new(&mut queue)?;
    let cvt = mev::kernels::Cvt::new(&device);
    let mut state = State::new();
    if let Some(path) = arguments.initial_path {
        state.open(path, false);
    }
    let mut app = App {
        state,
        device,
        queue,
        renderer,
        cvt,
        surface: None,
        window: None,
        target: None,
        target_size: (0, 0),
        scale: 1.0,
        modifiers: ModifiersState::empty(),
        cursor_cache: pixie_ui_winit::CursorCache::new(),
        smoke_test: arguments.smoke_test,
        frames: 0,
        error: None,
    };
    winit::event_loop::EventLoop::new()?.run_app(&mut app)?;
    app.queue.wait_idle()?;
    if let Some(error) = app.error {
        return Err(error.into());
    }
    if app.smoke_test {
        println!("Native smoke passed: {} frames", app.frames);
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("JKL Workbench: {error}");
        std::process::exit(1);
    }
}

fn render_previews(device: &mev::Device, queue: &mut mev::Queue, directory: &Path) -> Result<()> {
    std::fs::create_dir_all(directory)?;
    for (name, kind) in [
        ("texture", backend::Kind::Texture),
        ("mesh", backend::Kind::Mesh),
        ("binary", backend::Kind::Blob),
    ] {
        for (suffix, width, height) in [("", 1360u32, 900u32), ("-narrow", 1040, 720)] {
            let mut state = State::new();
            state.resize(width as i32, height as i32);
            state.demo(kind);
            let mut renderer = pixie_ui_mev::Renderer::new(queue)?;
            let mut desc = mev::ImageDesc::new_d2_rt(width, height, mev::PixelFormat::Rgba8Unorm);
            desc.usage |= mev::ImageUsage::TRANSFER_SRC;
            let target = device.new_image(desc);
            draw_state(&mut state, device, queue, &mut renderer, &target)?;
            let mut readback = device.new_buffer(mev::BufferDesc {
                size: width as usize * height as usize * 4,
                usage: mev::BufferUsage::TRANSFER_DST | mev::BufferUsage::HOST_READ,
                name: "JKL screenshot readback",
            });
            let mut encoder = queue.new_command_encoder();
            {
                let mut copy = encoder.copy();
                copy.barrier(
                    mev::PipelineStages::COLOR_OUTPUT,
                    mev::PipelineStages::TRANSFER,
                );
                copy.copy_image_to_buffer(
                    &target,
                    &readback,
                    mev::TexelShape::new(mev::PixelFormat::Rgba8Unorm, target.extent()),
                );
            }
            queue.submit_checkpoint([encoder.finish()])?;
            queue.wait_idle()?;
            let mut rgba = vec![0; width as usize * height as usize * 4];
            readback.read(0, &mut rgba)?;
            let path = directory.join(format!("{name}{suffix}.png"));
            image::save_buffer(&path, &rgba, width, height, image::ColorType::Rgba8)?;
            println!("Rendered {}", path.display());
        }
    }
    Ok(())
}
