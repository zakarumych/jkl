use std::{
    collections::HashMap,
    fmt::Write as _,
    io::Cursor,
    path::PathBuf,
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use eframe::egui::{self, Color32, Pos2, Sense, Stroke, Vec2};
use jkl::{
    jackal::mesh::{JackalMeshReader, write_mesh},
    mesh::data::{EncodingOptions, MeshData},
};

struct Document {
    mesh: Arc<MeshData>,
    points: Vec<[f32; 3]>,
}

impl Document {
    fn new(mesh: MeshData) -> Result<Self, String> {
        mesh.validate().map_err(|e| e.to_string())?;
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in &mesh.positions {
            for i in 0..3 {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        let center = std::array::from_fn::<_, 3, _>(|i| (lo[i] as f64 + hi[i] as f64) * 0.5);
        let radius = (0..3)
            .map(|i| hi[i] as f64 - lo[i] as f64)
            .fold(0.0_f64, f64::max)
            .max(1e-20)
            * 0.5;
        let points = mesh
            .positions
            .iter()
            .map(|p| std::array::from_fn(|i| ((p[i] as f64 - center[i]) / radius) as f32))
            .collect();
        Ok(Self {
            mesh: Arc::new(mesh),
            points,
        })
    }
}

struct Encoded {
    data: Arc<Vec<u8>>,
    document: Document,
    patches: usize,
    elapsed: Duration,
}

enum Completed {
    Open {
        document: Document,
        name: String,
        bytes: usize,
        encoded: Option<Encoded>,
    },
    Encode(Encoded),
    Save(String),
}

pub struct MeshEditor {
    source: Option<Document>,
    name: String,
    source_bytes: usize,
    result: Option<Encoded>,
    options: EncodingOptions,
    job: Option<mpsc::Receiver<Result<Completed, String>>>,
    status: String,
    error: bool,
    reconstructed: bool,
    wireframe: bool,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    shapes: Vec<egui::Shape>,
    projection_key: Option<[u32; 6]>,
}

impl Default for MeshEditor {
    fn default() -> Self {
        Self {
            source: None,
            name: String::new(),
            source_bytes: 0,
            result: None,
            options: EncodingOptions::default(),
            job: None,
            status: "Open a triangle OBJ or JKLM mesh.".into(),
            error: false,
            reconstructed: false,
            wireframe: false,
            yaw: 0.5,
            pitch: -0.25,
            zoom: 1.0,
            shapes: Vec::new(),
            projection_key: None,
        }
    }
}

impl MeshEditor {
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    fn start(
        &mut self,
        label: &str,
        work: impl FnOnce() -> Result<Completed, String> + Send + 'static,
    ) {
        if self.busy() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.job = Some(rx);
        self.status = label.into();
        self.error = false;
        std::thread::spawn(move || {
            let _ = tx.send(work());
        });
    }

    pub fn open(&mut self, path: PathBuf) {
        if self.busy() {
            self.status = "Wait for the current mesh operation before opening another file.".into();
            return;
        }
        self.start("Loading mesh…", move || {
            let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let size = bytes.len();
            if bytes.starts_with(b"JKLM") {
                let mut reader =
                    JackalMeshReader::open(Cursor::new(&bytes)).map_err(|e| e.to_string())?;
                let patches = reader.patches().len();
                let mesh = reader.read_mesh().map_err(|e| e.to_string())?;
                let document = Document::new(mesh.clone())?;
                let encoded = Encoded {
                    data: Arc::new(bytes),
                    document: Document::new(mesh)?,
                    patches,
                    elapsed: Duration::ZERO,
                };
                Ok(Completed::Open {
                    document,
                    name,
                    bytes: size,
                    encoded: Some(encoded),
                })
            } else {
                let text = std::str::from_utf8(&bytes)
                    .map_err(|_| "Expected UTF-8 triangle OBJ or JKLM".to_string())?;
                Ok(Completed::Open {
                    document: Document::new(parse_obj(text)?)?,
                    name,
                    bytes: size,
                    encoded: None,
                })
            }
        });
    }

    pub(crate) fn poll(&mut self) {
        let Some(rx) = &self.job else {
            return;
        };
        let outcome = match rx.try_recv() {
            Ok(v) => v,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("Mesh worker stopped unexpectedly.".into())
            }
        };
        self.job = None;
        match outcome {
            Err(e) => {
                self.status = e;
                self.error = true;
            }
            Ok(Completed::Open {
                document,
                name,
                bytes,
                encoded,
            }) => {
                self.source = Some(document);
                self.name = name;
                self.source_bytes = bytes;
                self.result = encoded;
                self.reconstructed = false;
                self.zoom = 1.0;
                self.projection_key = None;
                self.status = "Mesh ready. Drag to orbit; scroll to zoom.".into();
                self.error = false;
            }
            Ok(Completed::Encode(result)) => {
                self.result = Some(result);
                self.reconstructed = true;
                self.projection_key = None;
                self.status = "Encoded and decoded successfully.".into();
                self.error = false;
            }
            Ok(Completed::Save(message)) => {
                self.status = message;
                self.error = false;
            }
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.poll();
        if self.busy() {
            ctx.request_repaint_after(Duration::from_millis(40));
        }
        ui.horizontal(|ui| {
            ui.heading("Mesh workshop");
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Open OBJ / JKLM…"))
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Mesh", &["obj", "jklm"])
                    .pick_file()
                {
                    self.open(path);
                }
            }
            if self.busy() {
                ui.spinner();
            }
        });
        if self.error {
            ui.colored_label(Color32::LIGHT_RED, &self.status);
        } else {
            ui.label(&self.status);
        }
        ui.separator();
        ui.small("OBJ import ignores object/group labels and smoothing groups; explicit normals and UV seams are retained. Triangle faces only; materials and unsupported attributes are rejected.");
        let Some(source) = &self.source else {
            ui.label("Triangle OBJ: positions, normals, one UV set. Material references, vertex colors, curves and polygons must be converted before import.");
            return;
        };
        ui.label(format!(
            "{} · {} vertices · {} triangles · {} bytes",
            self.name,
            source.mesh.positions.len(),
            source.mesh.triangles.len(),
            self.source_bytes
        ));
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(270.0);
                let old = self.options;
                ui.add_enabled_ui(!self.busy(), |ui| {
                    ui.strong("Encoding controls");
                    ui.add(
                        egui::Slider::new(&mut self.options.max_vertices, 1..=256)
                            .text("Patch vertices"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.options.position_bits, 2..=20)
                            .text("Position bits"),
                    );
                    ui.add(egui::Slider::new(&mut self.options.uv_bits, 2..=20).text("UV bits"));
                    ui.add(
                        egui::Slider::new(&mut self.options.normal_bits, 2..=20)
                            .text("Normal bits"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.options.tangent_bits, 2..=20)
                            .text("Tangent bits"),
                    );
                    ui.horizontal(|ui| {
                        ui.label("Tangent UV set");
                        ui.add(
                            egui::DragValue::new(&mut self.options.tangent_uv_set).range(0..=255),
                        );
                    });
                    let mut weld = self.options.weld_epsilon.is_some();
                    if ui.checkbox(&mut weld, "Weld matching vertices").changed() {
                        self.options.weld_epsilon = weld.then_some(1e-6);
                    }
                    if let Some(epsilon) = &mut self.options.weld_epsilon {
                        ui.add(
                            egui::Slider::new(epsilon, 1e-9..=1.0)
                                .logarithmic(true)
                                .text("Weld epsilon"),
                        );
                    }
                });
                if old != self.options {
                    self.result = None;
                    self.reconstructed = false;
                    self.projection_key = None;
                    self.status = "Settings changed. Encode to refresh the reconstruction.".into();
                }
                if ui
                    .add_enabled(!self.busy(), egui::Button::new("Encode mesh"))
                    .clicked()
                {
                    let mesh = Arc::clone(&self.source.as_ref().unwrap().mesh);
                    let options = self.options;
                    self.start("Encoding mesh…", move || {
                        encode(&mesh, options).map(Completed::Encode)
                    });
                }
                ui.separator();
                if let Some(result) = &self.result {
                    ui.label(format!(
                        "{} bytes · {} patches",
                        result.data.len(),
                        result.patches
                    ));
                    ui.label(format!(
                        "Encode + decode: {:.3} s",
                        result.elapsed.as_secs_f64()
                    ));
                    ui.label(format!(
                        "Reconstruction: {} vertices",
                        result.document.mesh.positions.len()
                    ));
                }
                if ui
                    .add_enabled(
                        !self.busy() && self.result.is_some(),
                        egui::Button::new("Save JKLM…"),
                    )
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Jackal mesh", &["jklm"])
                        .set_file_name("encoded.jklm")
                        .save_file()
                    {
                        let data = self.result.as_ref().unwrap().data.clone();
                        self.start("Saving JKLM…", move || {
                            std::fs::write(&path, &**data).map_err(|e| e.to_string())?;
                            Ok(Completed::Save(format!("Saved {}", path.display())))
                        });
                    }
                }
                let export_mesh = self
                    .result
                    .as_ref()
                    .map(|r| r.document.mesh.clone())
                    .or_else(|| self.source.as_ref().map(|s| s.mesh.clone()));
                let export_error = export_mesh.as_deref().and_then(obj_export_error);
                if let Some(reason) = &export_error {
                    ui.colored_label(Color32::YELLOW, reason);
                }
                if ui
                    .add_enabled(
                        !self.busy() && export_error.is_none(),
                        egui::Button::new("Export decoded OBJ…"),
                    )
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Wavefront OBJ", &["obj"])
                        .set_file_name("decoded.obj")
                        .save_file()
                    {
                        let mesh = export_mesh.unwrap();
                        self.start("Exporting OBJ…", move || {
                            let text = export_obj(&mesh)?;
                            std::fs::write(&path, text).map_err(|e| e.to_string())?;
                            Ok(Completed::Save(format!("Exported {}", path.display())))
                        });
                    }
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    if ui.selectable_label(!self.reconstructed, "Source").clicked() {
                        self.reconstructed = false;
                        self.projection_key = None;
                    }
                    if ui
                        .add_enabled(
                            self.result.is_some(),
                            egui::Button::selectable(self.reconstructed, "Reconstruction"),
                        )
                        .clicked()
                    {
                        self.reconstructed = true;
                        self.projection_key = None;
                    }
                    if ui.checkbox(&mut self.wireframe, "Wireframe").changed() {
                        self.projection_key = None;
                    }
                    if ui.button("Reset view").clicked() {
                        self.yaw = 0.5;
                        self.pitch = -0.25;
                        self.zoom = 1.0;
                    }
                });
                self.preview(ui);
            });
        });
    }

    fn preview(&mut self, ui: &mut egui::Ui) {
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size().max(Vec2::splat(100.0)), Sense::drag());
        if response.dragged() {
            let delta = ui.input(|i| i.pointer.delta());
            self.yaw += delta.x * 0.01;
            self.pitch = (self.pitch + delta.y * 0.01).clamp(-1.55, 1.55);
        }
        if response.hovered() {
            self.zoom = (self.zoom * (ui.input(|i| i.smooth_scroll_delta.y) * 0.002).exp())
                .clamp(0.05, 20.0);
        }
        let key = [
            rect.width().to_bits(),
            rect.height().to_bits(),
            self.yaw.to_bits(),
            self.pitch.to_bits(),
            self.zoom.to_bits(),
            self.wireframe as u32,
        ];
        if self.projection_key != Some(key) {
            let document = if self.reconstructed {
                self.result.as_ref().map(|r| &r.document)
            } else {
                self.source.as_ref()
            };
            self.shapes.clear();
            if let Some(document) = document {
                let (sy, cy) = self.yaw.sin_cos();
                let (sp, cp) = self.pitch.sin_cos();
                let scale = rect.width().min(rect.height()) * 0.34 * self.zoom;
                let points: Vec<[f32; 3]> = document
                    .points
                    .iter()
                    .map(|p| {
                        let x = cy * p[0] + sy * p[2];
                        let z = -sy * p[0] + cy * p[2];
                        [x, cp * p[1] - sp * z, sp * p[1] + cp * z]
                    })
                    .collect();
                let mut triangles: Vec<_> = document
                    .mesh
                    .triangles
                    .iter()
                    .map(|t| (*t, t.iter().map(|&i| points[i as usize][2]).sum::<f32>()))
                    .collect();
                triangles.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                for (t, _) in triangles {
                    let p = t.map(|i| points[i as usize]);
                    let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
                    let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
                    let n = [
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ];
                    let length = n.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-20);
                    let light = (0.25
                        + 0.75 * ((n[0] * 0.3 + n[1] * 0.5 + n[2] * 0.8) / length).abs())
                    .min(1.0);
                    let vertices = p.map(|p| {
                        Pos2::new(
                            rect.width() * 0.5 + p[0] * scale,
                            rect.height() * 0.5 - p[1] * scale,
                        )
                    });
                    self.shapes.push(egui::Shape::convex_polygon(
                        vertices.to_vec(),
                        if self.wireframe {
                            Color32::TRANSPARENT
                        } else {
                            Color32::from_rgb(
                                (70.0 * light) as u8,
                                (175.0 * light) as u8,
                                (220.0 * light) as u8,
                            )
                        },
                        if self.wireframe {
                            Stroke::new(0.7, Color32::from_rgb(99, 199, 230))
                        } else {
                            Stroke::NONE
                        },
                    ));
                }
            }
            self.projection_key = Some(key);
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 6.0, Color32::from_rgb(17, 23, 32));
        for shape in &self.shapes {
            let mut shape = shape.clone();
            shape.translate(rect.min.to_vec2());
            painter.add(shape);
        }
    }
}

fn encode(mesh: &MeshData, options: EncodingOptions) -> Result<Encoded, String> {
    options.validate(mesh).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let mut bytes = Vec::new();
    write_mesh(mesh, &options, &mut bytes).map_err(|e| e.to_string())?;
    let mut reader = JackalMeshReader::open(Cursor::new(&bytes)).map_err(|e| e.to_string())?;
    let patches = reader.patches().len();
    let decoded = reader.read_mesh().map_err(|e| e.to_string())?;
    Ok(Encoded {
        data: Arc::new(bytes),
        document: Document::new(decoded)?,
        patches,
        elapsed: start.elapsed(),
    })
}

fn obj_index(text: &str, len: usize) -> Result<usize, String> {
    let value: i64 = text.parse().map_err(|_| "Invalid OBJ index")?;
    let index = if value > 0 {
        value - 1
    } else if value < 0 {
        len as i64 + value
    } else {
        -1
    };
    if index < 0 || index >= len as i64 {
        Err("OBJ index out of range".into())
    } else {
        Ok(index as usize)
    }
}

fn parse_obj(text: &str) -> Result<MeshData, String> {
    let mut positions = Vec::<[f32; 3]>::new();
    let mut normals = Vec::<[f32; 3]>::new();
    let mut uvs = Vec::<[f32; 2]>::new();
    let mut mesh = MeshData::default();
    let mut map = HashMap::new();
    let mut attributes = None;
    for (line_no, line) in text.lines().enumerate() {
        let fields: Vec<_> = line
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        if fields.is_empty() {
            continue;
        }
        let error = |s: &str| format!("OBJ line {}: {s}", line_no + 1);
        match fields[0] {
            "v" | "vn" | "vt" => {
                let count = if fields[0] == "vt" { 2 } else { 3 };
                if fields.len() != count + 1 {
                    return Err(error(
                        "Only xyz positions/normals and two-component UVs are supported; vertex colors and homogeneous coordinates are not supported.",
                    ));
                }
                let values: Result<Vec<f32>, _> =
                    fields[1..].iter().map(|v| v.parse::<f32>()).collect();
                let values = values.map_err(|_| error("Invalid coordinate"))?;
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(error("Non-finite coordinate"));
                }
                match fields[0] {
                    "v" => positions.push([values[0], values[1], values[2]]),
                    "vn" => normals.push([values[0], values[1], values[2]]),
                    _ => uvs.push([values[0], values[1]]),
                }
            }
            "f" => {
                if fields.len() != 4 {
                    return Err(error(
                        "Only triangular faces are supported. Triangulate polygons before import.",
                    ));
                }
                let mut triangle = [0; 3];
                for (corner, token) in fields[1..].iter().enumerate() {
                    let indices: Vec<_> = token.split('/').collect();
                    if indices.len() > 3 {
                        return Err(error("Invalid face index"));
                    }
                    let p = obj_index(indices[0], positions.len()).map_err(|e| error(&e))?;
                    let uv = indices
                        .get(1)
                        .filter(|s| !s.is_empty())
                        .map(|s| obj_index(s, uvs.len()))
                        .transpose()
                        .map_err(|e| error(&e))?;
                    let normal = indices
                        .get(2)
                        .filter(|s| !s.is_empty())
                        .map(|s| obj_index(s, normals.len()))
                        .transpose()
                        .map_err(|e| error(&e))?;
                    let present = (uv.is_some(), normal.is_some());
                    if attributes.is_some_and(|a| a != present) {
                        return Err(error(
                            "Normals and UVs must be consistently present on all face corners.",
                        ));
                    }
                    attributes = Some(present);
                    let key = (p, uv, normal);
                    triangle[corner] = if let Some(&index) = map.get(&key) {
                        index
                    } else {
                        let index = u32::try_from(mesh.positions.len())
                            .map_err(|_| error("Too many vertices"))?;
                        mesh.positions.push(positions[p]);
                        if let Some(uv) = uv {
                            if mesh.tex_coords.is_empty() {
                                mesh.tex_coords.push(Vec::new());
                            }
                            mesh.tex_coords[0].push(uvs[uv]);
                        }
                        if let Some(n) = normal {
                            mesh.normals.get_or_insert_with(Vec::new).push(normals[n]);
                        }
                        map.insert(key, index);
                        index
                    };
                }
                mesh.triangles.push(triangle);
            }
            "o" | "g" | "s" => {}
            other => {
                return Err(error(&format!(
                    "Unsupported OBJ directive '{other}' (materials, colors, curves and lines are not imported)."
                )));
            }
        }
    }
    if mesh.triangles.is_empty() {
        return Err("OBJ contains no triangle faces".into());
    }
    mesh.validate().map_err(|e| e.to_string())?;
    Ok(mesh)
}

fn obj_export_error(mesh: &MeshData) -> Option<String> {
    if mesh.tangents.is_some()
        || mesh.colors.is_some()
        || mesh.tex_coords.len() > 1
        || !mesh.materials.is_empty()
        || mesh.primitive_ids.iter().any(|&id| id != 0)
    {
        Some("OBJ export unavailable: tangents, colors, multiple UV sets, or material/primitive metadata would be lost. Save JKLM to preserve them.".into())
    } else {
        None
    }
}

fn export_obj(mesh: &MeshData) -> Result<String, String> {
    mesh.validate().map_err(|e| e.to_string())?;
    if let Some(error) = obj_export_error(mesh) {
        return Err(error);
    }
    let mut text = String::from("# JKL decoded mesh\n");
    for p in &mesh.positions {
        writeln!(text, "v {} {} {}", p[0], p[1], p[2]).unwrap();
    }
    if let Some(uvs) = mesh.tex_coords.first() {
        for p in uvs {
            writeln!(text, "vt {} {}", p[0], p[1]).unwrap();
        }
    }
    if let Some(normals) = &mesh.normals {
        for p in normals {
            writeln!(text, "vn {} {} {}", p[0], p[1], p[2]).unwrap();
        }
    }
    for t in &mesh.triangles {
        text.push('f');
        for &i in t {
            let i = i + 1;
            match (!mesh.tex_coords.is_empty(), mesh.normals.is_some()) {
                (false, false) => write!(text, " {i}"),
                (true, false) => write!(text, " {i}/{i}"),
                (false, true) => write!(text, " {i}//{i}"),
                (true, true) => write!(text, " {i}/{i}/{i}"),
            }
            .unwrap();
        }
        text.push('\n');
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    const OBJ: &str =
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\nf -3/1/1 -2/2/1 -1/3/1\n";
    #[test]
    fn obj_roundtrip_preserves_attributes() {
        let mesh = parse_obj(OBJ).unwrap();
        assert_eq!(mesh, parse_obj(&export_obj(&mesh).unwrap()).unwrap());
    }
    #[test]
    fn obj_seams_split_vertices() {
        let mesh = parse_obj(&format!("{OBJ}f 1/2/1 2/2/1 3/3/1\n")).unwrap();
        assert_eq!(mesh.positions.len(), 4);
        assert_ne!(mesh.triangles[0][0], mesh.triangles[1][0]);
    }
    #[test]
    fn obj_rejects_unsupported_or_invalid_data() {
        for suffix in ["usemtl red\n", "f 0 1 2\n", "f 1 2 3 1\n", "f 1 2 3\n"] {
            assert!(parse_obj(&format!("{OBJ}{suffix}")).is_err());
        }
    }
    #[test]
    fn mesh_native_roundtrip() {
        let mesh = parse_obj(OBJ).unwrap();
        let result = encode(&mesh, EncodingOptions::default()).unwrap();
        assert_eq!(result.document.mesh.triangles.len(), 1);
        assert!(result.document.mesh.normals.is_some());
        assert_eq!(result.document.mesh.tex_coords.len(), 1);
        assert_eq!(result.patches, 1);
    }
    #[test]
    fn obj_export_refuses_attribute_loss() {
        let mut mesh = parse_obj(OBJ).unwrap();
        mesh.colors = Some(vec![[255; 4]; 3]);
        assert!(export_obj(&mesh).is_err());
    }
}

