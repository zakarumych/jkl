//! Persistent Pixie workbench controls and asynchronous document operations.
mod preview;

use crate::backend::{
    self, BlobSettings, Encoded, Kind, Loaded, Settings, Source, TextureSettings,
};
use edict::{entity::EntityId, world::World};
use pixie_ui::{
    Pixie, Widget,
    align::Align2,
    color::Color,
    controls,
    draw::{AddressMode, Brush, Stroke, TextureScale},
    event::PixieEvent,
    font::FontId,
    layout::layout_system,
    margin::Margin,
    math::{Pos, Rect, Vec as V},
    style::{Anchor, Attributes, Style, WidgetPos, WidgetSize},
    text::Text,
    texture::TextureId,
    ui::InputState,
    widget::{Container, RootWidget, sync_widget_parents},
    widgets::{Button, Label, Panel, TextInput, Toggle},
};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, mpsc},
};

const BG: Color = Color::from_rgb(20, 23, 26);
const PANEL: Color = Color::from_rgb(29, 33, 37);
const CONTROL: Color = Color::from_rgb(43, 49, 54);
const LINE: Color = Color::from_rgb(61, 69, 74);
const INK: Color = Color::from_rgb(235, 233, 222);
const MUTED: Color = Color::from_rgb(157, 170, 175);
const ACCENT: Color = Color::from_rgb(216, 238, 138);
const ERROR: Color = Color::from_rgb(255, 164, 140);
const HEX_KEYS: [&str; 8] = [
    "hex0", "hex1", "hex2", "hex3", "hex4", "hex5", "hex6", "hex7",
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Action {
    Tab(usize),
    Open,
    OpenBytes,
    Encode,
    Save,
    Export,
    Mode(u8),
    Format(usize),
    Compression(usize),
    Advanced,
    Zoom(i32),
    Fit,
    Rotate(i32),
    Pitch(i32),
    Wire,
    Page(i32),
}

#[derive(Clone, Copy)]
enum Place {
    Fixed(i32, i32, i32, i32),
    Right(i32, i32, i32, i32),
    Bottom(i32, i32, i32, i32),
    Canvas,
    Inspector,
    Footer,
    Wide(i32, i32, i32, i32),
    CompareCaption,
    Note(i32),
}

struct Node {
    id: EntityId,
    place: Place,
}
struct Field {
    key: &'static str,
    id: EntityId,
    toggle: bool,
}
enum Completion {
    Open(Loaded, Vec<jkl::jackal::blob::Chunk>),
    Encode(Encoded, Vec<jkl::jackal::blob::Chunk>),
    Saved(String),
}
struct Document {
    loaded: Option<Loaded>,
    result: Option<Encoded>,
    settings: Settings,
    drafts: HashMap<&'static str, String>,
    job: Option<mpsc::Receiver<Result<Completion, String>>>,
    message: String,
    error: bool,
    stale: bool,
    mode: u8,
    zoom: f32,
    yaw: f32,
    pitch: f32,
    wire: bool,
    page: usize,
    chunks: Vec<jkl::jackal::blob::Chunk>,
}
impl Document {
    fn new(kind: Kind) -> Self {
        Self {
            loaded: None,
            result: None,
            settings: match kind {
                Kind::Texture => Settings::Texture(TextureSettings::default()),
                Kind::Mesh => Settings::Mesh(Default::default()),
                Kind::Blob => Settings::Blob(BlobSettings::default()),
            },
            drafts: HashMap::new(),
            job: None,
            message: "Open an asset to begin.".into(),
            error: false,
            stale: false,
            mode: 0,
            zoom: 1.,
            yaw: 0.55,
            pitch: -0.25,
            wire: false,
            page: 0,
            chunks: Vec::new(),
        }
    }
}

pub struct State {
    pub world: World,
    style: Style,
    root: Option<EntityId>,
    nodes: Vec<Node>,
    fields: Vec<Field>,
    handles: HashMap<&'static str, EntityId>,
    buttons: Vec<(EntityId, Action)>,
    documents: [Document; 3],
    selected: usize,
    width: i32,
    height: i32,
    advanced: bool,
    texture: TextureId,
    preview_revision: u64,
    preview_dirty: bool,
    preview_job: Option<mpsc::Receiver<(u64, u32, u32, Vec<u8>)>>,
    upload: Option<(TextureId, u32, u32, Vec<u8>)>,
    cursor: Pos,
    dragging: bool,
}

impl State {
    pub fn new() -> Self {
        let mut world = World::new();
        let texture = Pixie::from(&mut world).ui_mut().new_texture_id();
        let style = Style::new(Attributes {
            fg_brush: Some(Brush::Solid(INK)),
            text_scale: Some(V { x: 2, y: 2 }),
            font: Some(FontId(0)),
            ..Default::default()
        });
        let mut s = Self {
            world,
            style,
            root: None,
            nodes: vec![],
            fields: vec![],
            handles: HashMap::new(),
            buttons: vec![],
            documents: [
                Document::new(Kind::Texture),
                Document::new(Kind::Mesh),
                Document::new(Kind::Blob),
            ],
            selected: 0,
            width: 1360,
            height: 900,
            advanced: false,
            texture,
            preview_revision: 0,
            preview_dirty: true,
            preview_job: None,
            upload: None,
            cursor: Pos::ZERO,
            dragging: false,
        };
        s.build();
        s.prepare();
        s
    }
    fn doc(&self) -> &Document {
        &self.documents[self.selected]
    }
    fn doc_mut(&mut self) -> &mut Document {
        &mut self.documents[self.selected]
    }
    fn kind(&self) -> Kind {
        [Kind::Texture, Kind::Mesh, Kind::Blob][self.selected]
    }
    fn add(&mut self, widget: impl Widget, place: Place) -> EntityId {
        let id = Pixie::from(&mut self.world).new_widget(widget);
        if let Some(root) = self.root {
            self.world.remove::<RootWidget>(id).unwrap();
            self.world
                .get::<&mut Container>(root)
                .unwrap()
                .children
                .push(id);
        }
        self.nodes.push(Node { id, place });
        id
    }
    fn label(&mut self, text: impl Into<String>, place: Place, color: Color) -> EntityId {
        self.add(
            Label::new(text).with_attributes(Attributes {
                fg_brush: Some(Brush::Solid(color)),
                content_align: Some(Align2::LEFT),
                ..Default::default()
            }),
            place,
        )
    }
    fn button(&mut self, text: &str, action: Action, place: Place) -> EntityId {
        let id = self.add(
            Button::new(text)
                .on_click(action)
                .with_attributes(Attributes {
                    bg_brush: Some(Brush::Solid(CONTROL).into()),
                    fg_brush: Some(Brush::Solid(INK)),
                    inner_margin: Some(Margin::uniform(7)),
                    stroke: Some(Stroke::new(Brush::Solid(LINE), 1)),
                    ..Default::default()
                }),
            place,
        );
        self.buttons.push((id, action));
        id
    }
    fn note(&mut self, lines: &[&str], y: i32) {
        for (i, line) in lines.iter().enumerate() {
            self.label(*line, Place::Note(y + i as i32 * 22), MUTED);
        }
    }
    fn field(&mut self, key: &'static str, title: &str, value: String, y: i32, toggle: bool) {
        let value = self.doc().drafts.get(key).cloned().unwrap_or(value);
        self.label(title, Place::Right(316, y, 188, 28), MUTED);
        let attrs = Attributes {
            bg_brush: Some(Brush::Solid(CONTROL).into()),
            fg_brush: Some(Brush::Solid(if toggle { ACCENT } else { INK })),
            stroke: Some(Stroke::new(Brush::Solid(LINE), 1)),
            ..Default::default()
        };
        let id = if toggle {
            self.add(
                Toggle::new(value == "true").with_attributes(attrs),
                Place::Right(106, y + 4, 72, 22),
            )
        } else {
            self.add(
                TextInput::new(value.clone()).with_attributes(attrs),
                Place::Right(122, y, 94, 30),
            )
        };
        self.doc_mut().drafts.insert(key, value);
        self.fields.push(Field { key, id, toggle });
    }
    fn build(&mut self) {
        Pixie::from(&mut self.world)
            .ui_mut()
            .set_input(InputState::default());
        if let Some(root) = self.root.take() {
            let _ = self.world.despawn(root);
        }
        self.nodes.clear();
        self.fields.clear();
        self.handles.clear();
        self.buttons.clear();
        let root =
            Pixie::from(&mut self.world).new_widget(Panel::new().with_attributes(Attributes {
                bg_brush: Some(Brush::Solid(BG).into()),
                ..Default::default()
            }));
        self.root = Some(root);
        self.add(
            Panel::new().with_attributes(Attributes {
                bg_brush: Some(Brush::Solid(PANEL).into()),
                ..Default::default()
            }),
            Place::Inspector,
        );
        self.label("JKL / WORKBENCH", Place::Fixed(24, 18, 220, 30), ACCENT);
        for (i, title) in ["Textures", "Meshes", "Binary"].iter().enumerate() {
            self.button(
                title,
                Action::Tab(i),
                Place::Fixed(270 + i as i32 * 114, 16, 104, 34),
            );
        }
        self.button("Open asset", Action::Open, Place::Right(316, 16, 132, 34));
        self.button(
            "Open bytes",
            Action::OpenBytes,
            Place::Right(174, 16, 146, 34),
        );
        let name = self.label("No asset loaded", Place::Wide(24, 73, 370, 28), INK);
        self.handles.insert("name", name);
        let info = self.label(
            "Texture / mesh / binary",
            Place::Wide(24, 101, 370, 20),
            MUTED,
        );
        self.handles.insert("info", info);
        let modes = if self.selected == 2 {
            ["Decoded", "Source", "Encoded", "Chunks"]
        } else {
            ["Compare", "Source", "Result", "Error x8"]
        };
        for (i, title) in modes.iter().enumerate() {
            if (self.selected == 1 && i == 3) || (self.selected == 2 && i == 1) {
                continue;
            }
            self.button(
                title,
                Action::Mode(i as u8),
                Place::Fixed(
                    24 + (if self.selected == 2 && i > 1 {
                        i - 1
                    } else {
                        i
                    }) as i32
                        * 106,
                    136,
                    98,
                    30,
                ),
            );
        }
        if self.selected == 2 {
            self.button("<", Action::Page(-1), Place::Fixed(458, 136, 48, 30));
            self.button(">", Action::Page(1), Place::Fixed(514, 136, 48, 30));
        } else {
            self.button("-", Action::Zoom(-1), Place::Fixed(458, 136, 32, 30));
            self.button("Fit", Action::Fit, Place::Fixed(498, 136, 48, 30));
            self.button("+", Action::Zoom(1), Place::Fixed(554, 136, 32, 30));
        }
        let canvas = self.add(
            Panel::new().with_attributes(Attributes {
                bg_brush: Some(Brush::Solid(Color::from_rgb(13, 16, 18)).into()),
                stroke: Some(Stroke::new(Brush::Solid(LINE), 1)),
                ..Default::default()
            }),
            Place::Canvas,
        );
        self.handles.insert("canvas", canvas);
        let empty = self.label("DROP AN ASSET HERE", Place::Fixed(56, 264, 560, 30), INK);
        self.handles.insert("empty", empty);
        let id = self.label(
            "Open an image, triangle OBJ, or JKL container.",
            Place::Fixed(56, 308, 560, 24),
            MUTED,
        );
        self.handles.insert("empty_help", id);
        let id = self.label(
            "Tune settings, encode, then compare the result.",
            Place::Fixed(56, 340, 560, 24),
            MUTED,
        );
        self.handles.insert("empty_task", id);
        if self.selected == 2 {
            for (i, key) in HEX_KEYS.iter().enumerate() {
                let id = self.label("", Place::Fixed(40, 220 + i as i32 * 25, 650, 22), INK);
                self.world.get::<&mut Attributes>(id).unwrap().font = Some(FontId(1));
                self.handles.insert(key, id);
            }
        }
        let caption = self.label(
            "SOURCE                                      RECONSTRUCTION",
            Place::Wide(40, 190, 392, 22),
            MUTED,
        );
        self.handles.insert("caption", caption);
        let caption = self.label("", Place::CompareCaption, MUTED);
        self.handles.insert("caption_right", caption);
        if self.selected == 1 {
            self.button("Yaw -", Action::Rotate(-1), Place::Bottom(24, 184, 82, 30));
            self.button("Yaw +", Action::Rotate(1), Place::Bottom(114, 184, 82, 30));
            self.button("Tilt -", Action::Pitch(-1), Place::Bottom(204, 184, 82, 30));
            self.button("Tilt +", Action::Pitch(1), Place::Bottom(294, 184, 82, 30));
            self.button("Wireframe", Action::Wire, Place::Bottom(384, 184, 130, 30));
        }
        self.add(
            Panel::new().with_attributes(Attributes {
                bg_brush: Some(Brush::Solid(PANEL).into()),
                ..Default::default()
            }),
            Place::Footer,
        );
        let validity = self.label("NO OUTPUT", Place::Bottom(40, 132, 560, 24), ACCENT);
        self.handles.insert("validity", validity);
        let metrics = self.label("", Place::Bottom(40, 103, 620, 24), INK);
        self.handles.insert("metrics", metrics);
        let quality = self.label("", Place::Bottom(40, 75, 620, 20), MUTED);
        self.handles.insert("quality", quality);
        let status = self.label("", Place::Wide(24, -35, 48, 24), MUTED);
        self.handles.insert("status", status);
        self.label("ENCODING", Place::Right(316, 79, 270, 26), INK);
        self.label("APPLIED BY ENCODE", Place::Right(316, 109, 288, 20), MUTED);
        self.build_settings();
        self.button(
            "Encode asset",
            Action::Encode,
            Place::Right(316, -166, 288, 40),
        );
        self.button(
            "Save container",
            Action::Save,
            Place::Right(316, -114, 288, 34),
        );
        self.button(
            "Export decoded",
            Action::Export,
            Place::Right(316, -70, 288, 34),
        );
        self.resize_nodes();
        self.refresh();
    }
    fn build_settings(&mut self) {
        match self.doc().settings.clone() {
            Settings::Texture(s) => {
                self.button(
                    if self.advanced {
                        "< Format & compression"
                    } else {
                        "Tile & alpha settings >"
                    },
                    Action::Advanced,
                    Place::Right(316, 147, 288, 30),
                );
                if self.advanced {
                    self.label("TILE STRATEGY", Place::Right(316, 199, 270, 22), ACCENT);
                    self.field("fixed", "Fixed tile size", s.fixed.to_string(), 235, true);
                    self.field(
                        "tile_width",
                        "Tile width / px",
                        s.tile_width.to_string(),
                        277,
                        false,
                    );
                    self.field(
                        "tile_height",
                        "Tile height / px",
                        s.tile_height.to_string(),
                        319,
                        false,
                    );
                    self.field(
                        "flat_cost",
                        "Optimal flat cost",
                        s.flat_cost.to_string(),
                        361,
                        false,
                    );
                    self.field(
                        "size_cost",
                        "Optimal size cost",
                        s.size_cost.to_string(),
                        403,
                        false,
                    );
                    self.field(
                        "alpha_threshold",
                        "BC1 alpha cutoff",
                        s.alpha_threshold.to_string(),
                        455,
                        false,
                    );
                    self.note(
                        &[
                            "Tiles need block alignment.",
                            "Costs apply to optimal tiles.",
                        ],
                        505,
                    );
                } else {
                    self.label("PIXEL FORMAT", Place::Right(316, 199, 270, 22), ACCENT);
                    for (i, f) in backend::FORMATS.iter().enumerate() {
                        self.button(
                            &format!("{f:?}"),
                            Action::Format(i),
                            Place::Right(
                                316 - (i % 3) as i32 * 98,
                                231 + (i / 3) as i32 * 36,
                                90,
                                29,
                            ),
                        );
                    }
                    self.label(
                        "CONTAINER COMPRESSION",
                        Place::Right(316, 395, 290, 22),
                        ACCENT,
                    );
                    for (i, name) in ["Raw", "LZ77", "ANS", "LZ77+ANS", "RLE+ANS"]
                        .iter()
                        .enumerate()
                    {
                        self.button(
                            name,
                            Action::Compression(i),
                            Place::Right(
                                316 - (i % 2) as i32 * 148,
                                427 + (i / 2) as i32 * 36,
                                140,
                                29,
                            ),
                        );
                    }
                    self.note(
                        &["BC6: unsigned HDR / CPU", "Display & PNG clamp to 0-1."],
                        543,
                    );
                }
            }
            Settings::Mesh(s) => {
                self.label(
                    "PATCHES & PRECISION",
                    Place::Right(316, 157, 290, 22),
                    ACCENT,
                );
                self.field(
                    "max_vertices",
                    "Vertices / patch",
                    s.max_vertices.to_string(),
                    194,
                    false,
                );
                self.field(
                    "position_bits",
                    "Position bits",
                    s.position_bits.to_string(),
                    234,
                    false,
                );
                self.field("uv_bits", "UV bits", s.uv_bits.to_string(), 274, false);
                self.field(
                    "normal_bits",
                    "Normal bits",
                    s.normal_bits.to_string(),
                    314,
                    false,
                );
                self.field(
                    "tangent_bits",
                    "Tangent bits",
                    s.tangent_bits.to_string(),
                    354,
                    false,
                );
                self.field(
                    "tangent_uv_set",
                    "Tangent UV set",
                    s.tangent_uv_set.to_string(),
                    394,
                    false,
                );
                self.field(
                    "weld",
                    "Weld vertices",
                    s.weld_epsilon.is_some().to_string(),
                    446,
                    true,
                );
                self.field(
                    "weld_epsilon",
                    "Weld tolerance",
                    s.weld_epsilon.unwrap_or(1e-6).to_string(),
                    486,
                    false,
                );
                self.note(&["Triangle OBJ / JKLM", "Drag preview to orbit."], 538);
            }
            Settings::Blob(s) => {
                self.label("CHUNKING", Place::Right(316, 157, 290, 22), ACCENT);
                self.field(
                    "chunk_size",
                    "Chunk bytes",
                    s.chunk_size.to_string(),
                    194,
                    false,
                );
                self.label(
                    "1 byte to 16 MiB per chunk",
                    Place::Right(316, 233, 290, 22),
                    MUTED,
                );
                self.label("CODEC CANDIDATES", Place::Right(316, 277, 290, 22), ACCENT);
                for (i, (_, name)) in backend::CODECS.iter().enumerate() {
                    self.field(
                        ["raw", "lz77", "ans", "lz77ans", "rleans", "rle", "lz78"][i],
                        name,
                        s.allowed[i].to_string(),
                        313 + i as i32 * 33,
                        true,
                    );
                }
                self.note(
                    &["Smallest candidate wins.", "Encode verifies all bytes."],
                    557,
                );
            }
        }
    }
    fn rect_for(&self, p: Place) -> (i32, i32, i32, i32) {
        match p {
            Place::Fixed(x, y, w, h) => (x, y, w, h),
            Place::Right(x, y, w, h) => (
                self.width - x,
                if y < 0 { self.height + y } else { y },
                w,
                h,
            ),
            Place::Bottom(x, y, w, h) => (x, self.height - y, w, h),
            Place::Canvas => (24, 180, self.width - 384, (self.height - 380).max(160)),
            Place::Inspector => (self.width - 340, 64, 340, self.height - 108),
            Place::Footer => (24, self.height - 144, self.width - 384, 94),
            Place::Wide(x, y, gap, h) => (
                x,
                if y < 0 { self.height + y } else { y },
                self.width - gap,
                h,
            ),
            Place::CompareCaption => (
                24 + (self.width - 384) / 2 + 16,
                190,
                (self.width - 384) / 2 - 24,
                22,
            ),
            Place::Note(y) => (
                self.width - 316,
                if y + 20 > self.height - 180 { -1000 } else { y },
                288,
                20,
            ),
        }
    }
    fn resize_nodes(&mut self) {
        let (w, h) = (self.width, self.height);
        Pixie::from(&mut self.world)
            .ui_mut()
            .set_rect(Rect::from_pos_size(Pos::ZERO, V { x: w, y: h }));
        if let Some(root) = self.root {
            let a = self.world.get::<&mut Attributes>(root).unwrap();
            a.position = Some(WidgetPos::Fixed {
                offset: V::ZERO,
                anchor: Anchor::TopLeft,
            });
            a.size = Some(WidgetSize::Fixed(V { x: w, y: h }));
        }
        for node in &self.nodes {
            let (x, mut y, w, h) = self.rect_for(node.place);
            if self.height < 820 && matches!(node.place,Place::Right(_,yy,_,hh)if yy>=530&&hh==36) {
                y = -1000;
            }
            let a = self.world.get::<&mut Attributes>(node.id).unwrap();
            a.position = Some(WidgetPos::Fixed {
                offset: V { x, y },
                anchor: Anchor::TopLeft,
            });
            a.size = Some(WidgetSize::Fixed(V {
                x: w.max(0),
                y: h.max(0),
            }));
        }
    }
    pub fn resize(&mut self, w: i32, h: i32) {
        if self.width != w || self.height != h {
            self.width = w.max(680);
            self.height = h.max(520);
            self.resize_nodes();
            self.invalidate_preview();
        }
    }
    pub fn prepare(&mut self) {
        sync_widget_parents(&mut self.world);
        self.style.resolve_attributes(&mut self.world);
        layout_system(&mut self.world);
    }
    fn set_text(&mut self, key: &str, value: String) {
        if let Some(&id) = self.handles.get(key) {
            let text = self.world.get::<&mut Text>(id).unwrap();
            if text.string != value {
                text.string = value;
            }
        }
    }
    fn invalidate_preview(&mut self) {
        self.preview_revision = self.preview_revision.wrapping_add(1);
        self.preview_dirty = true;
        if let Some(&id) = self.handles.get("canvas") {
            self.world.get::<&mut Attributes>(id).unwrap().bg_brush =
                Some(Brush::Solid(Color::from_rgb(13, 16, 18)).into());
        }
    }
    fn capture_fields(&mut self) {
        let mut changed = false;
        for f in &self.fields {
            let value = if f.toggle {
                self.world
                    .get::<&controls::Toggle>(f.id)
                    .unwrap()
                    .checked()
                    .to_string()
            } else {
                self.world.get::<&Text>(f.id).unwrap().string.clone()
            };
            let d = &mut self.documents[self.selected];
            if d.drafts.get(f.key) != Some(&value) {
                d.drafts.insert(f.key, value);
                changed = true;
            }
        }
        if changed {
            self.doc_mut().stale = true;
            self.doc_mut().result = None;
            self.doc_mut().error = false;
            self.doc_mut().message = "Settings changed. Encode to create a current result.".into();
            self.invalidate_preview();
        }
    }
    fn parsed_settings(&self) -> Result<Settings, String> {
        let d = self.doc();
        let val = |key: &str, fallback: String| d.drafts.get(key).cloned().unwrap_or(fallback);
        fn number<T: std::str::FromStr>(s: String, name: &str) -> Result<T, String> {
            s.trim()
                .parse()
                .map_err(|_| format!("Enter a valid number for {name}."))
        }
        Ok(match d.settings.clone() {
            Settings::Texture(mut s) => {
                s.fixed = val("fixed", s.fixed.to_string()) == "true";
                s.tile_width = number(val("tile_width", s.tile_width.to_string()), "tile width")?;
                s.tile_height =
                    number(val("tile_height", s.tile_height.to_string()), "tile height")?;
                s.flat_cost = number(val("flat_cost", s.flat_cost.to_string()), "flat cost")?;
                s.size_cost = number(val("size_cost", s.size_cost.to_string()), "size cost")?;
                s.alpha_threshold = number(
                    val("alpha_threshold", s.alpha_threshold.to_string()),
                    "alpha cutoff",
                )?;
                if !s.alpha_threshold.is_finite() || !(0.0..=1.0).contains(&s.alpha_threshold) {
                    return Err("BC1 alpha cutoff must be between 0 and 1.".into());
                }
                Settings::Texture(s)
            }
            Settings::Mesh(mut s) => {
                s.max_vertices = number(
                    val("max_vertices", s.max_vertices.to_string()),
                    "vertices per patch",
                )?;
                s.position_bits = number(
                    val("position_bits", s.position_bits.to_string()),
                    "position bits",
                )?;
                s.uv_bits = number(val("uv_bits", s.uv_bits.to_string()), "UV bits")?;
                s.normal_bits =
                    number(val("normal_bits", s.normal_bits.to_string()), "normal bits")?;
                s.tangent_bits = number(
                    val("tangent_bits", s.tangent_bits.to_string()),
                    "tangent bits",
                )?;
                s.tangent_uv_set = number(
                    val("tangent_uv_set", s.tangent_uv_set.to_string()),
                    "tangent UV set",
                )?;
                s.weld_epsilon = if val("weld", s.weld_epsilon.is_some().to_string()) == "true" {
                    Some(number(
                        val("weld_epsilon", s.weld_epsilon.unwrap_or(1e-6).to_string()),
                        "weld tolerance",
                    )?)
                } else {
                    None
                };
                Settings::Mesh(s)
            }
            Settings::Blob(mut s) => {
                s.chunk_size = number(val("chunk_size", s.chunk_size.to_string()), "chunk bytes")?;
                for (i, k) in ["raw", "lz77", "ans", "lz77ans", "rleans", "rle", "lz78"]
                    .iter()
                    .enumerate()
                {
                    s.allowed[i] = val(k, s.allowed[i].to_string()) == "true";
                }
                Settings::Blob(s)
            }
        })
    }
    fn start(
        &mut self,
        slot: usize,
        message: &str,
        job: impl FnOnce() -> Result<Completion, String> + Send + 'static,
    ) {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(job());
        });
        let d = &mut self.documents[slot];
        d.job = Some(rx);
        d.message = message.into();
        d.error = false;
        self.refresh();
    }
    pub fn open(&mut self, path: PathBuf, force_blob: bool) {
        let kind = if force_blob {
            Kind::Blob
        } else {
            backend::classify(&path)
        };
        let slot = kind_index(kind);
        if self.documents[slot].job.is_some() {
            self.documents[slot].message =
                "Wait for this asset's current operation to finish.".into();
            return;
        }
        self.selected = slot;
        self.advanced = false;
        self.build();
        self.start(slot, "Opening and decoding asset...", move || {
            let loaded = backend::load(path, force_blob)?;
            let chunks = if matches!(loaded.source.as_ref(), Source::Blob(_)) {
                loaded
                    .container
                    .as_ref()
                    .map(|b| backend::blob_chunks(b))
                    .transpose()?
                    .unwrap_or_default()
            } else {
                vec![]
            };
            Ok(Completion::Open(loaded, chunks))
        });
    }
    fn fail(&mut self, message: String) {
        self.doc_mut().message = message;
        self.doc_mut().error = true;
        self.refresh();
    }
    fn encode(&mut self) {
        if self.doc().job.is_some() {
            return;
        }
        self.capture_fields();
        let settings = match self.parsed_settings() {
            Ok(s) => s,
            Err(e) => {
                self.fail(e);
                return;
            }
        };
        let Some(source) = self.doc().loaded.as_ref().map(|l| l.source.clone()) else {
            self.fail("Open an asset before encoding.".into());
            return;
        };
        self.doc_mut().settings = settings.clone();
        self.start(
            self.selected,
            "Encoding and reconstructing on CPU...",
            move || {
                let encoded = backend::encode(&source, &settings)?;
                let chunks = if matches!(source.as_ref(), Source::Blob(_)) {
                    backend::blob_chunks(&encoded.bytes)?
                } else {
                    vec![]
                };
                Ok(Completion::Encode(encoded, chunks))
            },
        );
    }
    fn save(&mut self, decoded: bool) {
        if self.doc().job.is_some() {
            return;
        }
        self.capture_fields();
        let Some(loaded) = self.doc().loaded.as_ref() else {
            self.fail("Open an asset first.".into());
            return;
        };
        let source_path = loaded.path.clone();
        if decoded {
            let source = if let Some(r) = &self.doc().result {
                r.decoded.clone()
            } else if loaded.container.is_some() && !self.doc().stale {
                loaded.source.clone()
            } else {
                self.fail(
                    "Encode first, or open a JKL container to export its decoded asset.".into(),
                );
                return;
            };
            let ext = match self.kind() {
                Kind::Texture => "png",
                Kind::Mesh => "obj",
                Kind::Blob => "bin",
            };
            let Some(path) = rfd::FileDialog::new()
                .set_title("Export decoded asset")
                .set_file_name(format!("decoded.{ext}"))
                .save_file()
            else {
                return;
            };
            self.start(self.selected, "Exporting decoded asset...", move || {
                backend::safe_export(&source, &path, &source_path)?;
                Ok(Completion::Saved(format!("Exported {}", path.display())))
            });
        } else {
            let bytes = if let Some(r) = &self.doc().result {
                r.bytes.clone()
            } else if !self.doc().stale {
                if let Some(b) = &loaded.container {
                    b.clone()
                } else {
                    self.fail("Encode before saving a container.".into());
                    return;
                }
            } else {
                self.fail("Settings changed. Encode before saving.".into());
                return;
            };
            let ext = match self.kind() {
                Kind::Texture => "jkli",
                Kind::Mesh => "jklm",
                Kind::Blob => "jklb",
            };
            let Some(path) = rfd::FileDialog::new()
                .set_title("Save JKL container")
                .set_file_name(format!("encoded.{ext}"))
                .save_file()
            else {
                return;
            };
            self.start(self.selected, "Saving container...", move || {
                backend::safe_write(&path, &source_path, &bytes)?;
                Ok(Completion::Saved(format!("Saved {}", path.display())))
            });
        }
    }
    fn action(&mut self, a: Action) {
        match a {
            Action::Tab(i) => {
                self.capture_fields();
                self.selected = i;
                self.advanced = false;
                self.build();
                self.invalidate_preview();
            }
            Action::Open | Action::OpenBytes => {
                if self.doc().job.is_none() {
                    if let Some(p) = rfd::FileDialog::new().pick_file() {
                        self.open(p, a == Action::OpenBytes);
                    }
                }
            }
            Action::Encode => self.encode(),
            Action::Save => self.save(false),
            Action::Export => self.save(true),
            Action::Format(i) | Action::Compression(i) => {
                if self.doc().job.is_some() {
                    return;
                }
                self.capture_fields();
                if let Settings::Texture(s) = &mut self.doc_mut().settings {
                    if matches!(a, Action::Format(_)) {
                        s.format = backend::FORMATS[i];
                    } else {
                        s.compression = backend::COMPRESSIONS[i];
                    }
                }
                self.doc_mut().stale = true;
                self.doc_mut().result = None;
                self.doc_mut().message =
                    "Settings changed. Encode to update reconstruction.".into();
                self.invalidate_preview();
            }
            Action::Advanced => {
                self.capture_fields();
                self.advanced = !self.advanced;
                self.build();
            }
            Action::Mode(m) => {
                self.doc_mut().mode = m;
                self.doc_mut().page = 0;
                self.invalidate_preview();
            }
            Action::Zoom(d) => {
                self.doc_mut().zoom = (self.doc().zoom * 1.25_f32.powi(d)).clamp(0.125, 16.);
                self.invalidate_preview();
            }
            Action::Fit => {
                self.doc_mut().zoom = 1.;
                self.invalidate_preview();
            }
            Action::Rotate(d) => {
                self.doc_mut().yaw += d as f32 * 0.2;
                self.invalidate_preview();
            }
            Action::Pitch(d) => {
                self.doc_mut().pitch = (self.doc().pitch + d as f32 * 0.15).clamp(-1.5, 1.5);
                self.invalidate_preview();
            }
            Action::Wire => {
                self.doc_mut().wire = !self.doc().wire;
                self.invalidate_preview();
            }
            Action::Page(d) => {
                let len = if self.doc().mode == 3 {
                    self.doc().chunks.len()
                } else {
                    self.blob_bytes().map_or(0, |b| b.len())
                };
                let page_size = if self.doc().mode == 3 { 8 } else { 64 };
                let last = len.saturating_sub(1) / page_size;
                self.doc_mut().page = self.doc().page.saturating_add_signed(d as isize).min(last);
            }
        }
        self.refresh();
    }
    pub fn shortcut(&mut self, c: char) {
        match c.to_ascii_lowercase() {
            'o' => self.action(Action::Open),
            'e' => self.action(Action::Encode),
            's' => self.action(Action::Save),
            'd' => self.action(Action::Export),
            _ => {}
        }
    }
    pub fn event(&mut self, event: PixieEvent) {
        if let PixieEvent::CursorMoved { pos } = event {
            if self.dragging && self.selected == 1 {
                let dx = pos.x - self.cursor.x;
                let dy = pos.y - self.cursor.y;
                self.doc_mut().yaw += dx as f32 * 0.009;
                self.doc_mut().pitch = (self.doc().pitch + dy as f32 * 0.009).clamp(-1.5, 1.5);
                self.invalidate_preview();
            }
            self.cursor = pos;
        }
        if event == PixieEvent::ButtonPressed
            && self.cursor.x > 24
            && self.cursor.x < self.width - 360
            && self.cursor.y > 180
            && self.cursor.y < self.height - 200
        {
            self.dragging = true;
        }
        if event == PixieEvent::ButtonReleased {
            self.dragging = false;
        }
        // Frozen settings cannot be mutated by keyboard or pointer while a codec snapshot is running.
        let input = Pixie::from(&mut self.world).input();
        let field_target = match event {
            PixieEvent::ButtonPressed | PixieEvent::ButtonReleased => {
                self.fields.iter().any(|f| Some(f.id) == input.hovered)
            }
            PixieEvent::KeyPressed(pixie_ui::event::Key::Tab | pixie_ui::event::Key::Escape) => {
                false
            }
            PixieEvent::KeyPressed(_) | PixieEvent::Paste(_) => {
                self.fields.iter().any(|f| Some(f.id) == input.focused)
            }
            _ => false,
        };
        if self.doc().job.is_some() && field_target {
            return;
        }
        self.prepare();
        let actions: Vec<_> = Pixie::from(&mut self.world)
            .handle_event_with_actions::<Action>(event)
            .collect();
        self.capture_fields();
        for action in actions {
            self.action(action);
        }
        self.refresh();
        self.prepare();
    }
    pub fn wheel(&mut self, delta: f32) {
        if self.cursor.x < self.width - 340 {
            if self.selected == 2 {
                self.action(Action::Page(if delta > 0. { -1 } else { 1 }));
            } else {
                self.action(Action::Zoom(if delta > 0. { 1 } else { -1 }));
            }
        }
    }
    pub fn tick(&mut self) {
        let mut changed = false;
        for i in 0..3 {
            let completion = self.documents[i]
                .job
                .as_ref()
                .and_then(|rx| match rx.try_recv() {
                    Ok(v) => Some(v),
                    Err(mpsc::TryRecvError::Disconnected) => Some(Err(
                        "Worker stopped before completing. Try the operation again.".into(),
                    )),
                    Err(mpsc::TryRecvError::Empty) => None,
                });
            if let Some(completion) = completion {
                let d = &mut self.documents[i];
                d.job = None;
                match completion {
                    Ok(Completion::Open(loaded, chunks)) => {
                        d.chunks = chunks;
                        d.loaded = Some(loaded);
                        d.result = None;
                        d.stale = false;
                        d.message = "Asset ready. Tune settings, then encode.".into();
                        d.error = false;
                        d.page = 0;
                    }
                    Ok(Completion::Encode(encoded, chunks)) => {
                        d.chunks = chunks;
                        d.result = Some(encoded);
                        d.stale = false;
                        d.message = "Encoding complete. Reconstruction is current.".into();
                        d.error = false;
                    }
                    Ok(Completion::Saved(message)) => {
                        d.message = message;
                        d.error = false;
                    }
                    Err(e) => {
                        d.message = e;
                        d.error = true;
                    }
                }
                changed = true;
            }
        }
        if changed {
            self.invalidate_preview();
            self.refresh();
        }
        if let Some(rx) = &self.preview_job {
            match rx.try_recv() {
                Ok((revision, w, h, pixels)) => {
                    self.preview_job = None;
                    if revision == self.preview_revision {
                        self.upload = Some((self.texture, w, h, pixels));
                        self.paint_preview();
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.preview_job = None;
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if self.preview_dirty && self.preview_job.is_none() {
            self.start_preview();
        }
    }
    fn start_preview(&mut self) {
        self.preview_dirty = false;
        if self.selected == 2 {
            return;
        }
        let Some(source) = self.doc().loaded.as_ref().map(|l| l.source.clone()) else {
            return;
        };
        let result = self
            .doc()
            .result
            .as_ref()
            .map(|r| r.decoded.clone())
            .or_else(|| {
                self.doc()
                    .loaded
                    .as_ref()
                    .filter(|l| l.container.is_some() && !self.doc().stale)
                    .map(|l| l.source.clone())
            });
        let config = self.preview_config();
        let revision = self.preview_revision;
        let (tx, rx) = mpsc::channel();
        self.preview_job = Some(rx);
        std::thread::spawn(move || {
            let (w, h, pixels) = preview::render(&source, result.as_deref(), config);
            let _ = tx.send((revision, w, h, pixels));
        });
    }
    fn preview_config(&self) -> preview::Config {
        let (_, _, w, h) = self.rect_for(Place::Canvas);
        preview::Config {
            width: w.max(1) as u32,
            height: h.max(1) as u32,
            mode: self.doc().mode,
            zoom: self.doc().zoom,
            yaw: self.doc().yaw,
            pitch: self.doc().pitch,
            wire: self.doc().wire,
        }
    }
    fn paint_preview(&mut self) {
        if let Some(&id) = self.handles.get("canvas") {
            self.world.get::<&mut Attributes>(id).unwrap().bg_brush = Some(
                Brush::Texture {
                    texture: self.texture,
                    scale: TextureScale::Stretch,
                    align: Align2::CENTER,
                    mode: AddressMode::Edge.into(),
                    source: None,
                }
                .into(),
            );
        }
    }
    pub fn preview_upload(&mut self) -> Option<(TextureId, u32, u32, Vec<u8>)> {
        self.upload.take()
    }
    fn blob_bytes(&self) -> Option<&[u8]> {
        if self.doc().mode == 2 {
            self.blob_container()
        } else {
            match self.doc().loaded.as_ref()?.source.as_ref() {
                Source::Blob(b) => Some(b),
                _ => None,
            }
        }
    }
    fn blob_container(&self) -> Option<&[u8]> {
        if self.doc().stale {
            return None;
        }
        self.doc()
            .result
            .as_ref()
            .map(|r| r.bytes.as_slice())
            .or_else(|| self.doc().loaded.as_ref()?.container.as_deref())
    }
    fn refresh(&mut self) {
        let d = self.doc();
        let name = d
            .loaded
            .as_ref()
            .map(|l| {
                l.path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_else(|| "No asset loaded".into());
        let info = d
            .loaded
            .as_ref()
            .map(|l| l.info.clone())
            .unwrap_or_else(|| "Open a source asset or decode a JKL container".into());
        let status = d.message.clone();
        let error = d.error;
        let loaded = d.loaded.is_some();
        let busy = d.job.is_some();
        let valid = if busy {
            "WORKING / CPU"
        } else if d.stale {
            "SETTINGS CHANGED / ENCODE REQUIRED"
        } else if d.result.is_some() {
            "CURRENT RECONSTRUCTION"
        } else if d.loaded.as_ref().is_some_and(|l| l.container.is_some()) {
            "DECODED CONTAINER"
        } else {
            "NO ENCODED OUTPUT"
        };
        let metrics = if let Some(r) = &d.result {
            format!(
                "{} bytes   /   {:.3} s   /   {:.1}% of input",
                r.bytes.len(),
                r.seconds,
                r.bytes.len() as f64
                    / self
                        .doc()
                        .loaded
                        .as_ref()
                        .map_or(1, |l| l.input_bytes.max(1)) as f64
                    * 100.
            )
        } else if let Some(l) = &d.loaded {
            format!(
                "Input: {} bytes   /   {}",
                l.input_bytes,
                if d.stale {
                    "Result invalidated"
                } else {
                    "Ready for encoding"
                }
            )
        } else {
            "Choose an asset. Its source remains untouched.".into()
        };
        let mode = d.mode;
        let has_result = d.result.is_some()
            || (!d.stale && d.loaded.as_ref().is_some_and(|l| l.container.is_some()));
        let wire = d.wire;
        let settings = d.settings.clone();
        let quality = d
            .result
            .as_ref()
            .map(|r| r.info.replace('·', "/"))
            .unwrap_or_default();
        self.set_text(
            "name",
            shorten(&name, ((self.width - 390) / 12).max(20) as usize),
        );
        self.set_text(
            "info",
            shorten(&info, ((self.width - 390) / 11).max(20) as usize),
        );
        self.set_text(
            "status",
            shorten(&status, ((self.width - 48) / 11).max(20) as usize),
        );
        self.set_text("validity", valid.into());
        self.set_text(
            "metrics",
            shorten(&metrics, ((self.width - 420) / 11).max(20) as usize),
        );
        self.set_text(
            "quality",
            shorten(&quality, ((self.width - 420) / 11).max(20) as usize),
        );
        self.set_text(
            "empty",
            if loaded {
                String::new()
            } else {
                "DROP AN ASSET HERE".into()
            },
        );
        self.set_text(
            "empty_help",
            if loaded {
                String::new()
            } else {
                "Open an image, triangle OBJ, or JKL container.".into()
            },
        );
        self.set_text(
            "empty_task",
            if loaded {
                String::new()
            } else {
                "Tune settings, encode, then compare the result.".into()
            },
        );
        self.set_text(
            "caption_right",
            if loaded && self.selected != 2 && mode == 0 {
                if has_result {
                    "RECONSTRUCTION"
                } else {
                    "ENCODE FOR RESULT"
                }
                .into()
            } else {
                String::new()
            },
        );
        self.set_text(
            "caption",
            if !loaded {
                String::new()
            } else if self.selected == 2 {
                format!(
                    "{}   /   PAGE {}",
                    if mode == 3 {
                        "CHUNKS / DECODED, PAYLOAD & CONTEXT BYTES"
                    } else if mode == 2 {
                        "ENCODED BYTES"
                    } else {
                        "DECODED SOURCE"
                    },
                    self.doc().page + 1
                )
            } else if !has_result && mode != 1 {
                if mode == 0 {
                    "SOURCE".into()
                } else {
                    "ENCODE TO INSPECT RECONSTRUCTION OR ERROR".into()
                }
            } else {
                match mode {
                    0 => "SOURCE",
                    1 => "SOURCE",
                    2 => "RECONSTRUCTION",
                    _ => "ERROR 8x / RGB WITH ALPHA ERROR AS WHITE",
                }
                .into()
            },
        );
        if let Some(id) = self.handles.get("status") {
            self.world.get::<&mut Attributes>(*id).unwrap().fg_brush =
                Some(Brush::Solid(if error { ERROR } else { MUTED }));
        }
        for &(id, action) in &self.buttons {
            let selected = match action {
                Action::Tab(i) => i == self.selected,
                Action::Mode(m) => m == mode,
                Action::Wire => wire,
                Action::Format(i) => {
                    matches!(&settings,Settings::Texture(s)if s.format==backend::FORMATS[i])
                }
                Action::Compression(i) => {
                    matches!(&settings,Settings::Texture(s)if s.compression==backend::COMPRESSIONS[i])
                }
                _ => false,
            };
            let primary = action == Action::Encode;
            let a = self.world.get::<&mut Attributes>(id).unwrap();
            a.bg_brush = Some(
                Brush::Solid(if selected {
                    Color::from_rgb(68, 79, 60)
                } else if primary {
                    ACCENT
                } else {
                    CONTROL
                })
                .into(),
            );
            a.fg_brush = Some(Brush::Solid(if primary {
                BG
            } else if selected {
                ACCENT
            } else {
                INK
            }));
            a.stroke = Some(Stroke::new(
                Brush::Solid(if selected { ACCENT } else { LINE }),
                1,
            ));
        }
        if self.selected == 2 {
            let chunks = if mode == 3 && !self.doc().stale {
                Some(&self.doc().chunks)
            } else {
                None
            };
            let rows: Vec<String> = (0..8)
                .map(|row| {
                    if mode == 3 {
                        let index = self.doc().page * 8 + row;
                        return chunks
                            .as_ref()
                            .and_then(|c| c.get(index))
                            .map(|c| {
                                format!(
                                    "#{index:03} {:?} raw:{} pack:{} ctx:{}",
                                    c.compression, c.decoded_len, c.payload_len, c.context_len
                                )
                            })
                            .map(|s| shorten(&s, ((self.width - 420) / 12).max(20) as usize))
                            .unwrap_or_default();
                    }
                    let start = self.doc().page * 64 + row * 8;
                    let Some(bytes) = self.blob_bytes() else {
                        return String::new();
                    };
                    if start >= bytes.len() {
                        return String::new();
                    }
                    let slice = &bytes[start..(start + 8).min(bytes.len())];
                    let hex = slice
                        .iter()
                        .map(|b| format!("{b:02X} "))
                        .collect::<String>();
                    let ascii = slice
                        .iter()
                        .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
                        .collect::<String>();
                    format!("{start:08X}   {hex:<24} {ascii}")
                })
                .collect();
            for (i, row) in rows.into_iter().enumerate() {
                self.set_text(HEX_KEYS[i], row);
            }
        }
    }
    pub fn demo(&mut self, kind: Kind) {
        self.selected = kind_index(kind);
        let source = preview::fixture(kind);
        let mut settings = self.doc().settings.clone();
        if let Settings::Texture(s) = &mut settings {
            s.format = jkl::image::format::Format::BC1;
        }
        let result = backend::encode(&source, &settings).ok();
        let name = match kind {
            Kind::Texture => "woven-spectrum.png",
            Kind::Mesh => "faceted-study.obj",
            Kind::Blob => "repeating-pattern.bin",
        };
        let loaded = Loaded {
            path: name.into(),
            source: Arc::new(source),
            input_bytes: match kind {
                Kind::Texture => 256 * 192 * 4,
                Kind::Mesh => 512,
                Kind::Blob => 4096,
            },
            container: None,
            info: match kind {
                Kind::Texture => "256 x 192 / RGBA source",
                Kind::Mesh => "8 vertices / 12 triangles",
                Kind::Blob => "4096 decoded bytes",
            }
            .into(),
        };
        let d = self.doc_mut();
        d.chunks = if kind == Kind::Blob {
            result
                .as_ref()
                .map(|r| backend::blob_chunks(&r.bytes).unwrap_or_default())
                .unwrap_or_default()
        } else {
            vec![]
        };
        d.loaded = Some(loaded);
        d.result = result;
        d.settings = settings;
        d.message = "Example asset / encoded with the current settings".into();
        d.stale = false;
        self.build();
        self.invalidate_preview();
        if kind != Kind::Blob {
            let d = self.doc();
            let (w, h, pixels) = preview::render(
                &d.loaded.as_ref().unwrap().source,
                d.result.as_ref().map(|r| r.decoded.as_ref()),
                self.preview_config(),
            );
            self.upload = Some((self.texture, w, h, pixels));
            self.preview_dirty = false;
            self.paint_preview();
        }
        self.prepare();
    }
}

fn kind_index(kind: Kind) -> usize {
    match kind {
        Kind::Texture => 0,
        Kind::Mesh => 1,
        Kind::Blob => 2,
    }
}
fn shorten(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.into()
    } else {
        text.chars().take(max.saturating_sub(3)).collect::<String>() + "..."
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn controls_survive_prepare_and_resize() {
        let mut s = State::new();
        let ids: Vec<_> = s.buttons.iter().map(|b| b.0).collect();
        s.prepare();
        s.resize(1040, 720);
        s.prepare();
        assert_eq!(ids, s.buttons.iter().map(|b| b.0).collect::<Vec<_>>());
    }
    #[test]
    fn invalid_draft_prevents_encode() {
        let mut s = State::new();
        s.doc_mut().drafts.insert("alpha_threshold", "NaN".into());
        assert!(s.parsed_settings().is_err());
        assert!(s.doc().job.is_none());
    }
    #[test]
    fn keyboard_button_routes_and_invalidates() {
        let mut s = State::new();
        let id = s
            .buttons
            .iter()
            .find(|(_, a)| *a == Action::Format(3))
            .unwrap()
            .0;
        Pixie::from(&mut s.world).ui_mut().set_focus(id);
        s.event(PixieEvent::KeyPressed(pixie_ui::event::Key::Char(' ')));
        assert!(s.doc().stale);
        assert!(matches!(&s.doc().settings,Settings::Texture(t)if t.format==backend::FORMATS[3]));
    }
    #[test]
    fn tabs_preserve_drafts() {
        let mut s = State::new();
        s.doc_mut().drafts.insert("flat_cost", "123.5".into());
        s.action(Action::Tab(1));
        s.action(Action::Tab(0));
        assert_eq!(s.doc().drafts["flat_cost"], "123.5");
    }
    #[test]
    fn busy_settings_freeze_without_trapping_navigation() {
        let mut s = State::new();
        s.action(Action::Advanced);
        let field = s.fields.iter().find(|f| f.key == "tile_width").unwrap().id;
        let before = s.world.get::<&Text>(field).unwrap().string.clone();
        let (_tx, rx) = mpsc::channel();
        s.doc_mut().job = Some(rx);
        Pixie::from(&mut s.world).ui_mut().set_focus(field);
        s.event(PixieEvent::KeyPressed(pixie_ui::event::Key::Char('9')));
        assert_eq!(s.world.get::<&Text>(field).unwrap().string, before);
        s.event(PixieEvent::KeyPressed(pixie_ui::event::Key::Tab));
        assert_ne!(Pixie::from(&mut s.world).input().focused, Some(field));
        let tab = s
            .buttons
            .iter()
            .find(|(_, a)| *a == Action::Tab(1))
            .unwrap()
            .0;
        Pixie::from(&mut s.world).ui_mut().set_focus(tab);
        s.event(PixieEvent::KeyPressed(pixie_ui::event::Key::Char(' ')));
        assert_eq!(s.selected, 1);
        assert!(s.documents[0].job.is_some());
    }
    #[test]
    fn inspector_controls_stay_above_output_actions_at_minimum_size() {
        let mut s = State::new();
        s.resize(1040, 720);
        for tab in 0..3 {
            s.action(Action::Tab(tab));
            for advanced in [false, true] {
                if tab != 0 && advanced {
                    continue;
                }
                if tab == 0 && s.advanced != advanced {
                    s.action(Action::Advanced);
                }
                for f in &s.fields {
                    let node = s.nodes.iter().find(|n| n.id == f.id).unwrap();
                    let (x, y, w, h) = s.rect_for(node.place);
                    assert!(x >= 700 && x + w <= 1040);
                    assert!(y + h <= 554, "{} overlaps encode button", f.key);
                }
                let root = s.world.get::<&Attributes>(s.root.unwrap()).unwrap();
                assert_eq!(root.size, Some(WidgetSize::Fixed(V { x: 1040, y: 720 })));
            }
        }
    }
}
