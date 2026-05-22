use std::sync::Arc;

use egui::{Color32, Id, RichText, Ui, Vec2};
use jkl::jackal::image::Compression;

use crate::encode_state::{EncoderState, FORMAT_ALL, Image, ViewMode, format_label};

// ── Compressions list ─────────────────────────────────────────────────────

pub const ALL_COMPRESSIONS: &[(Compression, &str)] = &[
    (Compression::None, "None"),
    (Compression::Lz77, "LZ77"),
    (Compression::Ans, "ANS"),
    (Compression::Lz77Ans, "LZ77+ANS"),
    (Compression::RleAns, "RLE+ANS"),
];

// ── Fly-weight widget ──────────────────────────────────────────────────────

pub struct EncodingWidget<'a> {
    #[allow(dead_code)]
    id: Id,
    state: &'a mut EncoderState,
}

impl<'a> EncodingWidget<'a> {
    pub fn new(id_salt: impl std::hash::Hash, state: &'a mut EncoderState) -> Self {
        Self {
            id: Id::new(id_salt),
            state,
        }
    }

    /// Draw the widget. Call every frame while visible.
    pub fn show(self, ctx: &egui::Context, ui: &mut Ui) {
        let EncodingWidget { id: _, state } = self;

        state.ensure_texture(ctx);
        state.poll();

        let mut format = state.selected_format();
        let mut compression = state.selected_compression();

        // Trigger background work for the selected (format, compression).
        state.ensure_blocks(format);
        if state.compression_done(format) {
            state.ensure_jkli(format, compression);
        }
        if state.has_any_in_progress() {
            ctx.request_repaint();
        }

        // ── Snapshot display data before closures ─────────────────────────
        let input = state.input();
        let fmt_label = match input {
            Image::Rgb8(_) => "Rgb8",
            Image::Rgba8(_) => "Rgba8",
            _ => unreachable!(),
        };
        let dim_label = format!("{}x{} {}", input.width(), input.height(), fmt_label);

        let block_psnr = state.compression_psnr(format);
        let block_err = state.compression_error(format).map(str::to_owned);
        let blocks_running = state.compression_in_progress(format);
        let blocks_ready = state.compression_done(format);
        let jkli_data: Option<Arc<Vec<u8>>> = state.serialized_data(format, compression);
        let jkli_running = state.serialization_in_progress(format, compression);

        // Snapshot view state; commit back to state after the UI closures run.
        let mut view_mode = state.view_mode;
        let mut heatmap_threshold = state.heatmap_threshold;
        let mut error_gamma = state.error_gamma;

        // Generate comparison textures when encoded data is ready.
        if blocks_ready {
            state.ensure_view_textures(format, ctx);
        }
        let preview_tex_info = state.preview_texture(format);

        // ── Layout ────────────────────────────────────────────────────

        ui.horizontal_top(|ui| {
            // ── Left column: view selector + image preview ──────────────
            ui.vertical(|ui| {
                // View-mode toggle buttons
                ui.horizontal(|ui| {
                    for &mode in ViewMode::ALL {
                        let enabled = mode == ViewMode::Input || blocks_ready;
                        if ui
                            .add_enabled(
                                enabled,
                                egui::Button::selectable(view_mode == mode, mode.label()),
                            )
                            .clicked()
                        {
                            view_mode = mode;
                        }
                    }
                });

                // Threshold slider — only visible in Heatmap mode
                if view_mode == ViewMode::ErrorHeatmap && blocks_ready {
                    ui.add(
                        egui::Slider::new(&mut heatmap_threshold, 0.0..=1.0)
                            .text("Threshold")
                            .custom_formatter(|v, _| format!("{:.3}", v)),
                    );
                }

                // Gamma + palette controls — visible in Error or Heatmap mode
                if (view_mode == ViewMode::Error || view_mode == ViewMode::ErrorHeatmap)
                    && blocks_ready
                {
                    ui.add(
                        egui::Slider::new(&mut error_gamma, 0.1_f32..=2.0)
                            .text("Gamma")
                            .custom_formatter(|v, _| format!("{:.2}", v)),
                    );
                }

                // Image
                if let Some((tex_id, asp)) = preview_tex_info {
                    let avail = ui.available_size();
                    let max_w = (avail.x * 0.55).max(100.0);
                    let max_h = avail.y.max(100.0);
                    let (draw_w, draw_h) = if max_w / asp <= max_h {
                        (max_w, max_w / asp)
                    } else {
                        (max_h * asp, max_h)
                    };
                    ui.image((tex_id, Vec2::new(draw_w, draw_h)));
                }
            });

            ui.separator();

            ui.vertical(|ui| {
                ui.label(RichText::new(&dim_label).strong());
                ui.add_space(6.0);

                // Format toggles
                ui.label("Output format:");
                ui.horizontal_wrapped(|ui| {
                    for &fmt in FORMAT_ALL {
                        if ui
                            .selectable_label(format == fmt, format_label(fmt))
                            .clicked()
                        {
                            format = fmt;
                        }
                    }
                });

                ui.add_space(4.0);

                // Compression toggles
                ui.label("Compression:");
                ui.horizontal_wrapped(|ui| {
                    for &(comp, label) in ALL_COMPRESSIONS {
                        if ui.selectable_label(compression == comp, label).clicked() {
                            compression = comp;
                        }
                    }
                });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                // Status / metrics
                if let Some(err) = &block_err {
                    ui.colored_label(Color32::LIGHT_RED, format!("Block encode error: {err}"));
                    return;
                }
                if blocks_running || !blocks_ready {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Encoding blocks\u{2026}");
                    });
                    return;
                }
                if jkli_running {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Compressing\u{2026}");
                    });
                    return;
                }

                if let Some(data) = &jkli_data {
                    let bytes = data.len();
                    let size_str = if bytes >= 1024 * 1024 {
                        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
                    } else if bytes >= 1024 {
                        format!("{:.1} KB", bytes as f64 / 1024.0)
                    } else {
                        format!("{bytes} B")
                    };
                    ui.label(format!("Size: {size_str}"));

                    match block_psnr {
                        Some(psnr) if psnr.is_infinite() => {
                            ui.label("Quality: lossless");
                        }
                        Some(psnr) => {
                            ui.label(format!("PSNR: {psnr:.1} dB"));
                        }
                        None => {}
                    }

                    ui.add_space(8.0);
                } else {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Waiting\u{2026}");
                    });
                }
            });
        });

        state.set_selection(format, compression);
        state.view_mode = view_mode;
        state.heatmap_threshold = heatmap_threshold;
        state.error_gamma = error_gamma;
    }
}
