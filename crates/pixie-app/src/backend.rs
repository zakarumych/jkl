//! CPU asset operations. Call these from a worker, outside the UI event loop.
#[path = "backend/mesh.rs"]
mod mesh;
#[path = "backend/texture.rs"]
mod texture;

use jkl::{
    jackal::{
        blob::{self, Compression, JackalBlobReader},
        mesh::{JackalMeshReader, write_mesh},
    },
    mesh::data::{EncodingOptions, MeshData},
};
use std::{
    io::{Cursor, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
pub use texture::{COMPRESSIONS, FORMATS, TextureSettings, TextureSource};

const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Texture,
    Mesh,
    Blob,
}

#[derive(Clone, Debug)]
pub enum Source {
    Texture(TextureSource),
    Mesh(MeshData),
    Blob(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlobSettings {
    pub chunk_size: usize,
    pub allowed: [bool; 7],
}
impl Default for BlobSettings {
    fn default() -> Self {
        Self {
            chunk_size: 64 * 1024,
            allowed: [true; 7],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Settings {
    Texture(TextureSettings),
    Mesh(EncodingOptions),
    Blob(BlobSettings),
}

pub struct Loaded {
    pub path: PathBuf,
    pub source: Arc<Source>,
    pub input_bytes: usize,
    pub container: Option<Vec<u8>>,
    pub info: String,
}
pub struct Encoded {
    pub bytes: Vec<u8>,
    pub decoded: Arc<Source>,
    pub info: String,
    pub seconds: f64,
}

pub const CODECS: [(Compression, &str); 7] = [
    (Compression::None, "Raw"),
    (Compression::Lz77, "LZ77"),
    (Compression::Ans, "ANS"),
    (Compression::Lz77Ans, "LZ77 + ANS"),
    (Compression::RleAns, "RLE + ANS"),
    (Compression::Rle, "RLE"),
    (Compression::Lz78, "LZ78"),
];

pub fn classify(path: &Path) -> Kind {
    let mut magic = [0; 4];
    if std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok()
    {
        match &magic {
            b"JKLI" => return Kind::Texture,
            b"JKLM" => return Kind::Mesh,
            b"JKLB" => return Kind::Blob,
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
        | "hdr" | "exr" | "pnm" | "ppm" | "pgm" | "dds" | "avif" | "qoi" => Kind::Texture,
        "obj" | "jklm" => Kind::Mesh,
        _ => Kind::Blob,
    }
}

pub fn load(path: PathBuf, force_blob: bool) -> Result<Loaded, String> {
    let kind = if force_blob {
        Kind::Blob
    } else {
        classify(&path)
    };
    if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() > MAX_INPUT_BYTES {
        return Err("This workbench accepts source files up to 512 MiB.".into());
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let input_bytes = bytes.len();
    let (source, info, native) = match kind {
        Kind::Texture if bytes.starts_with(b"JKLI") || extension(&path, "jkli") => {
            let (width, height, pixels, info) = texture::decode_jkli(&bytes)?;
            (
                Source::Texture(TextureSource {
                    width,
                    height,
                    pixels,
                }),
                info,
                true,
            )
        }
        Kind::Texture => {
            let reader = image::ImageReader::new(Cursor::new(&bytes))
                .with_guessed_format()
                .map_err(|e| e.to_string())?;
            let (width, height) = reader.into_dimensions().map_err(|e| e.to_string())?;
            let (width, height) = (width as usize, height as usize);
            if width == 0
                || height == 0
                || width
                    .checked_mul(height)
                    .is_none_or(|n| n > 64 * 1024 * 1024)
            {
                return Err("Texture must have 1–67,108,864 pixels.".into());
            }
            let image = image::ImageReader::new(Cursor::new(&bytes))
                .with_guessed_format()
                .map_err(|e| e.to_string())?
                .decode()
                .map_err(|e| e.to_string())?;
            let hdr = matches!(
                image,
                image::DynamicImage::ImageRgb32F(_) | image::DynamicImage::ImageRgba32F(_)
            );
            let pixels = image
                .to_rgba32f()
                .into_raw()
                .chunks_exact(4)
                .map(|p| [p[0], p[1], p[2], p[3]])
                .collect();
            (
                Source::Texture(TextureSource {
                    width,
                    height,
                    pixels,
                }),
                if hdr {
                    "HDR source; preview and PNG export clamp to 0–1.".into()
                } else {
                    format!("{width} × {height} pixels")
                },
                false,
            )
        }
        Kind::Mesh if bytes.starts_with(b"JKLM") || extension(&path, "jklm") => {
            let mut reader =
                JackalMeshReader::open(Cursor::new(&bytes)).map_err(|e| e.to_string())?;
            let patches = reader.patches().len();
            let mesh = reader.read_mesh().map_err(|e| e.to_string())?;
            mesh.validate().map_err(|e| e.to_string())?;
            (
                Source::Mesh(mesh),
                format!("Decoded JKLM · {patches} patches"),
                true,
            )
        }
        Kind::Mesh => {
            let text = std::str::from_utf8(&bytes)
                .map_err(|_| "Expected UTF-8 triangle OBJ or JKLM".to_string())?;
            let mesh = mesh::parse_obj(text)?;
            let info = format!(
                "{} vertices · {} triangles",
                mesh.positions.len(),
                mesh.triangles.len()
            );
            (Source::Mesh(mesh), info, false)
        }
        Kind::Blob if !force_blob && (bytes.starts_with(b"JKLB") || extension(&path, "jklb")) => {
            let mut reader =
                JackalBlobReader::open(Cursor::new(&bytes)).map_err(|e| e.to_string())?;
            if reader.len() > MAX_INPUT_BYTES {
                return Err("Decoded blob exceeds the 512 MiB workbench limit.".into());
            }
            let chunks = reader.chunks().len();
            let decoded = reader.read_blob().map_err(|e| e.to_string())?;
            let info = format!("Decoded JKLB · {} bytes · {chunks} chunks", decoded.len());
            (Source::Blob(decoded), info, true)
        }
        Kind::Blob => (
            Source::Blob(bytes.clone()),
            format!("{input_bytes} source bytes"),
            false,
        ),
    };
    Ok(Loaded {
        path,
        source: Arc::new(source),
        input_bytes,
        container: native.then_some(bytes),
        info,
    })
}

fn extension(path: &Path, ext: &str) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

pub fn encode(source: &Source, settings: &Settings) -> Result<Encoded, String> {
    let started = Instant::now();
    match (source, settings) {
        (Source::Texture(source), Settings::Texture(settings)) => {
            let encoded = texture::encode(source, settings.clone())?;
            let info = format!(
                "{:?} · {:?} · RGBA PSNR {:.2} dB",
                settings.format, settings.compression, encoded.psnr
            );
            Ok(Encoded {
                bytes: encoded.bytes,
                decoded: Arc::new(Source::Texture(TextureSource {
                    width: source.width,
                    height: source.height,
                    pixels: encoded.decoded,
                })),
                info,
                seconds: encoded.seconds,
            })
        }
        (Source::Mesh(source), Settings::Mesh(settings)) => {
            settings.validate(source).map_err(|e| e.to_string())?;
            let mut bytes = Vec::new();
            write_mesh(source, settings, &mut bytes).map_err(|e| e.to_string())?;
            let mut reader =
                JackalMeshReader::open(Cursor::new(&bytes)).map_err(|e| e.to_string())?;
            let patches = reader.patches().len();
            let mesh = reader.read_mesh().map_err(|e| e.to_string())?;
            mesh.validate().map_err(|e| e.to_string())?;
            let info = format!(
                "{} vertices · {} triangles · {patches} patches",
                mesh.positions.len(),
                mesh.triangles.len()
            );
            Ok(Encoded {
                bytes,
                decoded: Arc::new(Source::Mesh(mesh)),
                info,
                seconds: started.elapsed().as_secs_f64(),
            })
        }
        (Source::Blob(source), Settings::Blob(settings)) => {
            if !(1..=blob::MAX_CHUNK_SIZE).contains(&settings.chunk_size) {
                return Err("Chunk size must be between 1 byte and 16 MiB.".into());
            }
            let allowed_compressions: Vec<_> = CODECS
                .iter()
                .zip(settings.allowed)
                .filter_map(|(&(c, _), enabled)| enabled.then_some(c))
                .collect();
            if allowed_compressions.is_empty() {
                return Err("Select at least one compression candidate.".into());
            }
            let mut bytes = Vec::new();
            blob::write_blob(
                source,
                &blob::Options {
                    chunk_size: settings.chunk_size,
                    allowed_compressions,
                },
                &mut bytes,
            )
            .map_err(|e| e.to_string())?;
            let mut reader =
                JackalBlobReader::open(Cursor::new(&bytes)).map_err(|e| e.to_string())?;
            let chunks = reader.chunks().len();
            let decoded = reader.read_blob().map_err(|e| e.to_string())?;
            if decoded != *source {
                return Err("Encoded data failed byte-for-byte verification.".into());
            }
            Ok(Encoded {
                bytes,
                decoded: Arc::new(Source::Blob(decoded)),
                info: format!("Verified byte-for-byte · {chunks} chunks"),
                seconds: started.elapsed().as_secs_f64(),
            })
        }
        _ => Err("Encoding settings do not match this asset type.".into()),
    }
}

/// Inspect the native chunk table without decoding the payload again.
pub fn blob_chunks(bytes: &[u8]) -> Result<Vec<blob::Chunk>, String> {
    Ok(JackalBlobReader::open(Cursor::new(bytes))
        .map_err(|e| e.to_string())?
        .chunks()
        .to_vec())
}

fn export_bytes(source: &Source) -> Result<Vec<u8>, String> {
    match source {
        Source::Texture(source) => {
            let width =
                u32::try_from(source.width).map_err(|_| "Texture width exceeds PNG limits")?;
            let height =
                u32::try_from(source.height).map_err(|_| "Texture height exceeds PNG limits")?;
            let bytes: Vec<u8> = source
                .pixels
                .iter()
                .flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
                .collect();
            let image = image::RgbaImage::from_raw(width, height, bytes)
                .ok_or("Invalid texture dimensions")?;
            let mut output = Cursor::new(Vec::new());
            image
                .write_to(&mut output, image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            Ok(output.into_inner())
        }
        Source::Mesh(source) => Ok(mesh::export_obj(source)?.into_bytes()),
        Source::Blob(source) => Ok(source.clone()),
    }
}

pub fn safe_export(source: &Source, output: &Path, source_path: &Path) -> Result<(), String> {
    ensure_distinct(output, source_path)?;
    safe_write(output, source_path, &export_bytes(source)?)
}

pub fn safe_write(output: &Path, source_path: &Path, bytes: &[u8]) -> Result<(), String> {
    ensure_distinct(output, source_path)?;
    write_new(output, bytes)
}

fn write_new(output: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                "Output already exists. Choose a new filename.".into()
            } else {
                e.to_string()
            }
        })?;
    file.write_all(bytes)
        .and_then(|_| file.flush())
        .map_err(|e| e.to_string())
}

fn ensure_distinct(output: &Path, source: &Path) -> Result<(), String> {
    fn resolved(path: &Path) -> Result<PathBuf, String> {
        if path.exists() {
            return path.canonicalize().map_err(|e| e.to_string());
        }
        let absolute = std::path::absolute(path).map_err(|e| e.to_string())?;
        let parent = absolute
            .parent()
            .ok_or("Output path has no parent")?
            .canonicalize()
            .map_err(|e| e.to_string())?;
        Ok(parent.join(absolute.file_name().ok_or("Output path has no filename")?))
    }
    let a = resolved(output)?;
    let b = resolved(source)?;
    #[cfg(windows)]
    let same = a
        .to_string_lossy()
        .eq_ignore_ascii_case(&b.to_string_lossy());
    #[cfg(not(windows))]
    let same = a == b;
    if same {
        Err("Choose a different output filename to preserve the source asset.".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jkl::image::format::Format;

    const OBJ: &str =
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\nf -3/1/1 -2/2/1 -1/3/1\n";
    fn texture_source(w: usize, h: usize) -> Source {
        Source::Texture(TextureSource {
            width: w,
            height: h,
            pixels: vec![[0.2, 0.4, 0.7, 0.8]; w * h],
        })
    }
    fn fixture_dir() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "jkl-pixie-backend-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn all_texture_formats_decode_to_published_pixels() {
        for &format in FORMATS {
            let result = encode(
                &texture_source(4, 4),
                &Settings::Texture(TextureSettings {
                    format,
                    ..TextureSettings::default()
                }),
            )
            .unwrap();
            let (w, h, pixels, _) = texture::decode_jkli(&result.bytes).unwrap();
            let Source::Texture(decoded) = &*result.decoded else {
                panic!()
            };
            assert_eq!((w, h), (decoded.width, decoded.height));
            assert_eq!(pixels, decoded.pixels, "{format:?}");
        }
    }

    #[test]
    fn texture_compressions_and_odd_block_extents() {
        for &compression in COMPRESSIONS {
            for format in [Format::RGBA8, Format::BC1] {
                let result = encode(
                    &texture_source(5, 3),
                    &Settings::Texture(TextureSettings {
                        format,
                        compression,
                        ..TextureSettings::default()
                    }),
                )
                .unwrap();
                let (w, h, pixels, _) = texture::decode_jkli(&result.bytes).unwrap();
                assert_eq!(
                    (w, h),
                    if format == Format::BC1 {
                        (8, 4)
                    } else {
                        (5, 3)
                    }
                );
                let Source::Texture(decoded) = &*result.decoded else {
                    panic!()
                };
                assert_eq!((decoded.width, decoded.height), (5, 3));
                for y in 0..3 {
                    assert_eq!(&pixels[y * w..y * w + 5], &decoded.pixels[y * 5..y * 5 + 5]);
                }
            }
        }
    }

    #[test]
    fn invalid_texture_settings_and_payloads_fail() {
        for settings in [
            TextureSettings {
                fixed: true,
                tile_width: 3,
                ..TextureSettings::default()
            },
            TextureSettings {
                alpha_threshold: f32::NAN,
                ..TextureSettings::default()
            },
            TextureSettings {
                alpha_threshold: 1.1,
                ..TextureSettings::default()
            },
            TextureSettings {
                size_cost: 0.0,
                ..TextureSettings::default()
            },
        ] {
            assert!(encode(&texture_source(4, 4), &Settings::Texture(settings)).is_err());
        }
        assert!(texture::decode_jkli(b"JKLI").is_err());
        assert!(
            encode(
                &texture_source(0, 4),
                &Settings::Texture(TextureSettings::default())
            )
            .is_err()
        );
        assert!(
            encode(
                &Source::Blob(vec![]),
                &Settings::Texture(TextureSettings::default())
            )
            .is_err()
        );
    }

    #[test]
    fn every_blob_codec_roundtrips_across_chunks() {
        let data: Vec<_> = (0..259).map(|i| (i % 19) as u8).collect();
        for i in 0..CODECS.len() {
            let mut allowed = [false; 7];
            allowed[i] = true;
            let result = encode(
                &Source::Blob(data.clone()),
                &Settings::Blob(BlobSettings {
                    chunk_size: 63,
                    allowed,
                }),
            )
            .unwrap();
            let Source::Blob(decoded) = &*result.decoded else {
                panic!()
            };
            assert_eq!(decoded, &data);
            let chunks = blob_chunks(&result.bytes).unwrap();
            assert_eq!(chunks.len(), 5);
            assert!(chunks.iter().all(|c| c.compression == CODECS[i].0));
            assert!(JackalBlobReader::open(Cursor::new(&result.bytes[..10])).is_err());
        }
        for settings in [
            BlobSettings {
                chunk_size: 0,
                ..BlobSettings::default()
            },
            BlobSettings {
                allowed: [false; 7],
                ..BlobSettings::default()
            },
        ] {
            assert!(encode(&Source::Blob(data.clone()), &Settings::Blob(settings)).is_err());
        }
        let empty = encode(
            &Source::Blob(vec![]),
            &Settings::Blob(BlobSettings::default()),
        )
        .unwrap();
        assert!(blob_chunks(&empty.bytes).unwrap().is_empty());
    }

    #[test]
    fn mesh_attributes_and_seams_survive_import_encode_export() {
        let mesh = mesh::parse_obj(OBJ).unwrap();
        assert_eq!(
            mesh,
            mesh::parse_obj(&mesh::export_obj(&mesh).unwrap()).unwrap()
        );
        let result = encode(
            &Source::Mesh(mesh),
            &Settings::Mesh(EncodingOptions::default()),
        )
        .unwrap();
        let Source::Mesh(decoded) = &*result.decoded else {
            panic!()
        };
        assert_eq!(decoded.triangles.len(), 1);
        assert!(decoded.normals.is_some());
        assert_eq!(decoded.tex_coords.len(), 1);
        let seams = mesh::parse_obj(&format!("{OBJ}f 1/2/1 2/2/1 3/3/1\n")).unwrap();
        assert_eq!(seams.positions.len(), 4);
        assert_ne!(seams.triangles[0][0], seams.triangles[1][0]);
        let mut unsupported = seams;
        unsupported.colors = Some(vec![[255; 4]; 4]);
        assert!(mesh::export_obj(&unsupported).is_err());
        for suffix in ["usemtl red\n", "f 0 1 2\n", "f 1 2 3 1\n", "f 1 2 3\n"] {
            assert!(mesh::parse_obj(&format!("{OBJ}{suffix}")).is_err());
        }
    }

    #[test]
    fn disk_workflow_protects_source_and_existing_outputs() {
        let dir = fixture_dir();
        let input = dir.join("source.data");
        std::fs::write(&input, b"source bytes").unwrap();
        let loaded = load(input.clone(), false).unwrap();
        let encoded = encode(&loaded.source, &Settings::Blob(BlobSettings::default())).unwrap();
        assert!(safe_write(&input, &input, &encoded.bytes).is_err());
        assert_eq!(std::fs::read(&input).unwrap(), b"source bytes");
        let output = dir.join("encoded.jklb");
        safe_write(&output, &input, &encoded.bytes).unwrap();
        assert!(safe_write(&output, &input, b"replacement").is_err());
        let decoded = load(output.clone(), false).unwrap();
        assert!(decoded.container.is_some());
        let plain = dir.join("decoded.bin");
        safe_export(&decoded.source, &plain, &output).unwrap();
        assert_eq!(std::fs::read(&plain).unwrap(), b"source bytes");
        let forced = load(output, true).unwrap();
        let Source::Blob(raw) = &*forced.source else {
            panic!()
        };
        assert_eq!(raw, &encoded.bytes);
        let png = dir.join("source.png");
        safe_export(&texture_source(5, 3), &png, &input).unwrap();
        let loaded = load(png, false).unwrap();
        assert!(matches!(&*loaded.source, Source::Texture(_)));
        let Source::Texture(image) = &*loaded.source else {
            panic!()
        };
        assert_eq!((image.width, image.height), (5, 3));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
