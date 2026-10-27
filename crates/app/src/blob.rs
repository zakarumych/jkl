use eframe::egui::{self, Color32, RichText};
use jkl::jackal::blob::{self, Chunk, Compression, JackalBlobReader, Options};
use std::{
    io::Cursor,
    path::PathBuf,
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};

const CODECS: [(Compression, &str); 7] = [
    (Compression::None, "Raw"),
    (Compression::Lz77, "LZ77"),
    (Compression::Ans, "ANS"),
    (Compression::Lz77Ans, "LZ77 + ANS"),
    (Compression::RleAns, "RLE + ANS"),
    (Compression::Rle, "RLE"),
    (Compression::Lz78, "LZ78"),
];

struct Encoded {
    bytes: Arc<Vec<u8>>,
    chunks: Vec<Chunk>,
    elapsed: Duration,
}

struct Document {
    path: PathBuf,
    bytes: Arc<Vec<u8>>,
    input_size: usize,
    input_chunks: Vec<Chunk>,
}

enum Completed {
    Open(Document),
    Encode(Encoded),
    Save(PathBuf),
}

pub struct BlobEditor {
    document: Option<Document>,
    encoded: Option<Encoded>,
    pending: Option<mpsc::Receiver<Result<Completed, String>>>,
    chunk_size: usize,
    allowed: [bool; 7],
    status: String,
    error: Option<String>,
    preview_encoded: bool,
}

impl Default for BlobEditor {
    fn default() -> Self {
        Self {
            document: None,
            encoded: None,
            pending: None,
            chunk_size: 64 * 1024,
            allowed: [true; 7],
            status: "Open any file to compress its bytes, or a JKLB file to decode it.".into(),
            error: None,
            preview_encoded: false,
        }
    }
}

fn decode_container(bytes: &[u8]) -> Result<(Vec<u8>, Vec<Chunk>), String> {
    let mut reader = JackalBlobReader::open(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let chunks = reader.chunks().to_vec();
    let decoded = reader.read_blob().map_err(|e| e.to_string())?;
    Ok((decoded, chunks))
}

fn encode(bytes: &[u8], options: &Options) -> Result<Encoded, String> {
    if options.allowed_compressions.is_empty() {
        return Err("Select at least one compression candidate.".into());
    }
    if !(1..=blob::MAX_CHUNK_SIZE).contains(&options.chunk_size) {
        return Err("Chunk size must be between 1 byte and 16 MiB.".into());
    }
    let started = Instant::now();
    let mut output = Vec::new();
    blob::write_blob(bytes, options, &mut output).map_err(|e| e.to_string())?;
    let (decoded, chunks) = decode_container(&output)?;
    if decoded != bytes {
        return Err("Encoded data did not pass round-trip verification.".into());
    }
    Ok(Encoded {
        bytes: Arc::new(output),
        chunks,
        elapsed: started.elapsed(),
    })
}

impl BlobEditor {
    pub fn busy(&self) -> bool {
        self.pending.is_some()
    }

    fn start(&mut self, work: impl FnOnce() -> Result<Completed, String> + Send + 'static) {
        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);
        self.error = None;
        thread::spawn(move || {
            let _ = tx.send(work());
        });
    }

    pub fn open(&mut self, path: PathBuf) {
        if self.busy() {
            self.error =
                Some("Wait for the current binary job before opening another file.".into());
            return;
        }
        self.status = format!("Opening {}…", path.display());
        self.start(move || {
            let input =
                std::fs::read(&path).map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
            let input_size = input.len();
            let is_container = input.starts_with(b"JKLB")
                || path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("jklb"));
            let (bytes, input_chunks) = if is_container {
                decode_container(&input)?
            } else {
                (input, Vec::new())
            };
            Ok(Completed::Open(Document {
                path,
                bytes: Arc::new(bytes),
                input_size,
                input_chunks,
            }))
        });
    }

    pub(crate) fn poll(&mut self) {
        let Some(rx) = &self.pending else {
            return;
        };
        let result = match rx.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("Binary worker stopped before returning a result.".into())
            }
        };
        self.pending = None;
        match result {
            Ok(Completed::Open(document)) => {
                self.status = format!("Loaded {} decoded bytes", document.bytes.len());
                self.document = Some(document);
                self.encoded = None;
                self.preview_encoded = false;
            }
            Ok(Completed::Encode(encoded)) => {
                self.status =
                    "Encoding complete. Decoded bytes verified against the source.".into();
                self.encoded = Some(encoded);
            }
            Ok(Completed::Save(path)) => self.status = format!("Saved {}", path.display()),
            Err(error) => {
                self.status = "Job failed".into();
                self.error = Some(error);
            }
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.poll();
        ui.horizontal(|ui| {
            ui.heading("Binary assets");
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Open bytes…"))
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Open bytes or a JKLB container")
                    .pick_file()
                {
                    self.open(path);
                }
            }
        });
        ui.label("Lossless compression with independently decodable chunks");
        ui.separator();
        if self.busy() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(&self.status);
            });
            ctx.request_repaint_after(Duration::from_millis(80));
        } else {
            ui.label(&self.status);
        }
        if let Some(error) = &self.error {
            ui.colored_label(Color32::LIGHT_RED, error);
        }
        let Some(document) = &self.document else {
            ui.add_space(40.0);
            ui.heading("Every byte matters.");
            ui.label("Open or drop a file to compare lossless codecs and inspect their chunks.");
            ui.label("JKLB containers open as decoded bytes, ready to export or encode again.");
            return;
        };
        ui.label(RichText::new(document.path.display().to_string()).strong());
        ui.label(format!(
            "Decoded: {} bytes · Input file: {} bytes",
            document.bytes.len(),
            document.input_size
        ));
        let busy = self.busy();
        let mut changed = false;
        let mut encode_requested = false;
        ui.add_enabled_ui(!busy, |ui| {
            ui.group(|ui| {
                ui.label(RichText::new("Encoding settings").strong());
                ui.horizontal(|ui| {
                    ui.label("Chunk size (bytes)");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.chunk_size)
                                .range(1..=blob::MAX_CHUNK_SIZE)
                                .speed(1024.0),
                        )
                        .changed();
                    ui.small("1 B – 16 MiB");
                });
                ui.label("Candidate codecs · smallest result wins for each chunk");
                ui.horizontal_wrapped(|ui| {
                    for (index, (_, label)) in CODECS.iter().enumerate() {
                        changed |= ui.checkbox(&mut self.allowed[index], *label).changed();
                    }
                });
                if !self.allowed.iter().any(|b| *b) {
                    ui.colored_label(Color32::YELLOW, "Select at least one candidate.");
                }
                encode_requested = ui
                    .add_enabled(
                        self.allowed.iter().any(|b| *b),
                        egui::Button::new("Encode & verify"),
                    )
                    .clicked();
            });
        });
        if changed {
            self.encoded = None;
            self.preview_encoded = false;
            self.status = "Settings changed. Encode to produce a matching result.".into();
        }
        if encode_requested {
            let bytes = Arc::clone(&self.document.as_ref().unwrap().bytes);
            let options = Options {
                chunk_size: self.chunk_size,
                allowed_compressions: CODECS
                    .iter()
                    .enumerate()
                    .filter_map(|(i, (codec, _))| self.allowed[i].then_some(*codec))
                    .collect(),
            };
            self.status = "Encoding and verifying…".into();
            self.start(move || encode(&bytes, &options).map(Completed::Encode));
        }
        let mut save_encoded = false;
        let mut save_decoded = false;
        ui.horizontal(|ui| {
            save_encoded = ui
                .add_enabled(
                    !self.busy() && self.encoded.is_some(),
                    egui::Button::new("Save JKLB…"),
                )
                .clicked();
            save_decoded = ui
                .add_enabled(!self.busy(), egui::Button::new("Export decoded bytes…"))
                .clicked();
        });
        if let Some(encoded) = &self.encoded {
            let raw_len = self.document.as_ref().unwrap().bytes.len();
            let size = encoded.bytes.len();
            let ratio = if raw_len == 0 {
                "n/a".into()
            } else {
                format!(
                    "{:.1}% of decoded size",
                    size as f64 * 100.0 / raw_len as f64
                )
            };
            ui.label(format!(
                "Encoded: {size} bytes · {ratio} · {} chunks · {:.3} s including verification",
                encoded.chunks.len(),
                encoded.elapsed.as_secs_f64()
            ));
            ui.checkbox(&mut self.preview_encoded, "Preview encoded container bytes");
        }
        let document = self.document.as_ref().unwrap();
        let chunks = self
            .encoded
            .as_ref()
            .map_or(document.input_chunks.as_slice(), |e| e.chunks.as_slice());
        if !chunks.is_empty() {
            egui::CollapsingHeader::new(format!("Chunk metadata ({})", chunks.len())).show(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("blob_chunks").max_height(180.0).show_rows(ui, 20.0, chunks.len(), |ui, rows| {
                    for i in rows {
                        let chunk = &chunks[i];
                        ui.monospace(format!("{i:>5}  {:<9?} decoded {:>8}  payload {:>8}  context {:>7}  offset {:>10}", chunk.compression, chunk.decoded_len, chunk.payload_len, chunk.context_len, chunk.payload_offset));
                    }
                });
            });
        }
        ui.separator();
        ui.label(RichText::new("Byte preview").strong());
        let bytes = if self.preview_encoded {
            self.encoded
                .as_ref()
                .map_or(document.bytes.as_slice(), |e| e.bytes.as_slice())
        } else {
            document.bytes.as_slice()
        };
        egui::ScrollArea::both()
            .id_salt("blob_preview")
            .auto_shrink([false, false])
            .show_rows(ui, 18.0, bytes.len().div_ceil(16), |ui, rows| {
                for row in rows {
                    let offset = row * 16;
                    let slice = &bytes[offset..(offset + 16).min(bytes.len())];
                    let hex = slice
                        .iter()
                        .map(|b| format!("{b:02X} "))
                        .collect::<String>();
                    let ascii = slice
                        .iter()
                        .map(|&b| {
                            if b.is_ascii_graphic() || b == b' ' {
                                b as char
                            } else {
                                '.'
                            }
                        })
                        .collect::<String>();
                    ui.monospace(format!("{offset:08X}  {hex:<48}  {ascii}"));
                }
            });
        if save_encoded {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Jackal binary", &["jklb"])
                .set_file_name("asset.jklb")
                .save_file()
            {
                let bytes = Arc::clone(&self.encoded.as_ref().unwrap().bytes);
                self.status = "Saving container…".into();
                self.start(move || {
                    std::fs::write(&path, bytes.as_slice()).map_err(|e| e.to_string())?;
                    Ok(Completed::Save(path))
                });
            }
        } else if save_decoded {
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name("decoded.bin")
                .save_file()
            {
                let bytes = Arc::clone(&self.document.as_ref().unwrap().bytes);
                self.status = "Exporting decoded bytes…".into();
                self.start(move || {
                    std::fs::write(&path, bytes.as_slice()).map_err(|e| e.to_string())?;
                    Ok(Completed::Save(path))
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_codec_round_trips_across_chunk_boundaries() {
        let bytes: Vec<u8> = (0..777).map(|i| ((i / 11) % 17) as u8).collect();
        for (codec, _) in CODECS {
            let encoded = encode(
                &bytes,
                &Options {
                    chunk_size: 113,
                    allowed_compressions: vec![codec],
                },
            )
            .unwrap();
            assert_eq!(encoded.chunks.len(), 7);
            assert!(encoded.chunks.iter().all(|c| c.compression == codec));
            assert_eq!(decode_container(&encoded.bytes).unwrap().0, bytes);
        }
    }

    #[test]
    fn rejects_invalid_settings_and_truncated_containers() {
        assert!(
            encode(
                b"data",
                &Options {
                    chunk_size: 0,
                    ..Options::default()
                }
            )
            .is_err()
        );
        assert!(
            encode(
                b"data",
                &Options {
                    allowed_compressions: vec![],
                    ..Options::default()
                }
            )
            .is_err()
        );
        let encoded = encode(
            b"data",
            &Options {
                chunk_size: 2,
                allowed_compressions: vec![Compression::None],
            },
        )
        .unwrap();
        assert!(decode_container(&encoded.bytes[..encoded.bytes.len() - 1]).is_err());
    }

    #[test]
    fn empty_blob_is_valid() {
        let encoded = encode(&[], &Options::default()).unwrap();
        assert!(decode_container(&encoded.bytes).unwrap().0.is_empty());
    }
}
