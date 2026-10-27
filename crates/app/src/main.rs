mod blob;
mod mesh;
mod texture;

use blob::BlobEditor;
use eframe::egui::{self, Color32, RichText};
use mesh::MeshEditor;
use std::{
    io::Read,
    path::{Path, PathBuf},
};
use texture::TextureEditor;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("JKL Studio")
            .with_inner_size([1360.0, 900.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };
    let initial_path = std::env::args_os().nth(1).map(PathBuf::from);
    eframe::run_native(
        "JKL Studio",
        options,
        Box::new(move |cc| {
            let mut visuals = egui::Visuals::dark();
            visuals.panel_fill = Color32::from_rgb(20, 24, 32);
            visuals.window_fill = Color32::from_rgb(26, 31, 41);
            visuals.selection.bg_fill = Color32::from_rgb(35, 104, 121);
            cc.egui_ctx.set_visuals(visuals);
            let mut style = (*cc.egui_ctx.style_of(egui::Theme::Dark)).clone();
            style.spacing.item_spacing = egui::vec2(10.0, 9.0);
            style.spacing.button_padding = egui::vec2(12.0, 7.0);
            cc.egui_ctx.set_style_of(egui::Theme::Dark, style);
            let mut app = Studio::default();
            if let Some(path) = initial_path {
                app.open(path);
            }
            Ok(Box::new(app))
        }),
    )
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Editor {
    #[default]
    Texture,
    Mesh,
    Blob,
}

#[derive(Default)]
struct Studio {
    selected: Editor,
    textures: TextureEditor,
    meshes: MeshEditor,
    blobs: BlobEditor,
    message: Option<String>,
}

fn classify(path: &Path) -> Editor {
    let mut magic = [0; 4];
    if std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok()
    {
        match &magic {
            b"JKLI" => return Editor::Texture,
            b"JKLM" => return Editor::Mesh,
            b"JKLB" => return Editor::Blob,
            _ => {}
        }
    }
    match path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "jkli" | "png" | "jpg" | "jpeg" | "bmp" | "tga" | "tif" | "tiff" | "webp" | "gif"
        | "hdr" | "exr" | "pnm" | "ppm" | "pgm" | "dds" | "avif" | "qoi" => Editor::Texture,
        "obj" | "jklm" => Editor::Mesh,
        _ => Editor::Blob,
    }
}

impl Studio {
    fn open(&mut self, path: PathBuf) {
        let target = classify(&path);
        let busy = match target {
            Editor::Texture => self.textures.busy(),
            Editor::Mesh => self.meshes.busy(),
            Editor::Blob => self.blobs.busy(),
        };
        if busy {
            self.message = Some(
                "This editor is working. Wait for its current job before opening another asset."
                    .into(),
            );
            return;
        }
        self.message = None;
        self.selected = target;
        match target {
            Editor::Texture => self.textures.open(path),
            Editor::Mesh => self.meshes.open(path),
            Editor::Blob => self.blobs.open(path),
        }
    }
}

impl eframe::App for Studio {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.textures.poll(&ctx);
        self.meshes.poll();
        self.blobs.poll();
        let dropped = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect::<Vec<_>>()
        });
        if let Some(path) = dropped.into_iter().next() {
            self.open(path);
        }
        let mut open = ctx.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::O));
        egui::Panel::top("studio_header").show(ui, |ui| {
            ui.add_space(7.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("JKL")
                        .size(25.0)
                        .strong()
                        .color(Color32::from_rgb(107, 219, 213)),
                );
                ui.label(RichText::new("STUDIO").size(18.0));
                ui.separator();
                ui.label("Encode · inspect · decode");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    open |= ui
                        .button("Open asset…")
                        .on_hover_text("Ctrl+O · or drop a file anywhere")
                        .clicked();
                });
            });
            ui.add_space(7.0);
        });
        egui::Panel::bottom("studio_footer").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.small("Native JKL codecs · CPU processing");
                if self.textures.busy() || self.meshes.busy() || self.blobs.busy() {
                    ui.spinner();
                    ui.small("Processing assets");
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                if let Some(message) = &self.message {
                    ui.colored_label(Color32::LIGHT_RED, message);
                }
            });
        });
        egui::Panel::left("studio_navigation").exact_size(160.0).resizable(false).show(ui, |ui| {
            ui.add_space(18.0);
            ui.small("ASSET WORKSPACE"); ui.add_space(10.0);
            for (editor, title, subtitle) in [(Editor::Texture,"Textures","Pixels & block codecs"),(Editor::Mesh,"Meshes","Geometry & attributes"),(Editor::Blob,"Binary","Lossless byte streams")] {
                ui.selectable_value(&mut self.selected, editor, RichText::new(title).size(17.0));
                ui.small(subtitle); ui.add_space(14.0);
            }
            ui.separator(); ui.add_space(10.0);
            ui.small("1  Open an asset\n\n2  Adjust encoding\n\n3  Encode and compare\n\n4  Save or decode");
        });
        egui::CentralPanel::default().show(ui, |ui| match self.selected {
            Editor::Texture => self.textures.show(&ctx, ui),
            Editor::Mesh => self.meshes.show(&ctx, ui),
            Editor::Blob => self.blobs.show(&ctx, ui),
        });
        if open {
            if let Some(path) = rfd::FileDialog::new()
                .set_title("Open an asset")
                .pick_file()
            {
                self.open(path);
            }
        }
    }
}
