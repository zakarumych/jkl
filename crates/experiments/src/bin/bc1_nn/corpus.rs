//! Explicit, family-separated RGB8 texture corpus sampling.
use super::Texels;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Validation,
    Test,
}

#[derive(Clone, Debug, Serialize)]
pub struct EvalGroup {
    pub name: String,
    pub start: usize,
    pub count: usize,
    pub family: String,
    pub path: String,
    pub width: u32,
    pub height: u32,
}

pub struct CorpusData {
    pub train: Vec<Texels>,
    pub evaluation: Vec<Texels>,
    pub groups: Vec<EvalGroup>,
    pub metadata: Value,
    pub train_families: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    textures: Vec<Entry>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    path: String,
    family: String,
    split: Split,
}

fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}

fn stream(seed: u64, entry: &Entry) -> u64 {
    // Length-prefix identity fields so separator characters cannot alias streams.
    let mut key = seed.to_le_bytes().to_vec();
    key.push(match entry.split {
        Split::Train => 0,
        Split::Validation => 1,
        Split::Test => 2,
    });
    for field in [&entry.family, &entry.path] {
        key.extend_from_slice(&(field.len() as u64).to_le_bytes());
        key.extend_from_slice(field.as_bytes());
    }
    hash(&key)
}

fn sample(img: &image::RgbImage, count: usize, mut state: u64, output: &mut Vec<Texels>) {
    let cols = u64::from(img.width().div_ceil(4));
    let rows = u64::from(img.height().div_ceil(4));
    let blocks = cols * rows;
    for _ in 0..count {
        // SplitMix64: reproducible sampling with replacement, including seed zero.
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^= value >> 31;
        let block = value % blocks;
        let x = (block % cols) as u32 * 4;
        let y = (block / cols) as u32 * 4;
        output.push(std::array::from_fn(|i| {
            img.get_pixel(
                (x + i as u32 % 4).min(img.width() - 1),
                (y + i as u32 / 4).min(img.height() - 1),
            )
            .0
        }));
    }
}

pub fn load(
    manifest: &Path,
    train_budget: usize,
    eval_per_texture: usize,
    split: Split,
    seed: u64,
    include_train: bool,
) -> Result<CorpusData, Box<dyn Error>> {
    if split == Split::Train {
        return Err("evaluation split must be validation or test".into());
    }
    let manifest_path =
        fs::canonicalize(manifest).map_err(|e| format!("manifest {}: {e}", manifest.display()))?;
    if fs::metadata(&manifest_path)?.len() > 16 * 1024 * 1024 {
        return Err("manifest exceeds 16 MiB limit".into());
    }
    let bytes = fs::read(&manifest_path)?;
    let mut parsed: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| format!("manifest {}: {e}", manifest.display()))?;
    if parsed.version != 1 {
        return Err(format!("unsupported corpus manifest version {}", parsed.version).into());
    }
    if parsed.textures.is_empty() {
        return Err("corpus manifest has no textures".into());
    }
    parsed
        .textures
        .sort_by(|a, b| (&a.family, &a.path).cmp(&(&b.family, &b.path)));
    let mut families = BTreeMap::new();
    let mut paths = BTreeSet::new();
    let mut resolved: Vec<PathBuf> = Vec::new();
    for entry in &parsed.textures {
        if entry.family.trim().is_empty() {
            return Err(format!("texture {} has empty family", entry.path).into());
        }
        if entry.path.is_empty() {
            return Err("texture path is empty".into());
        }
        if let Some(previous) = families.insert(entry.family.clone(), entry.split) {
            if previous != entry.split {
                return Err(format!("family {} belongs to multiple splits", entry.family).into());
            }
        }
        let path = fs::canonicalize(manifest_path.parent().unwrap().join(&entry.path))
            .map_err(|e| format!("texture {}: {e}", entry.path))?;
        if !path.is_file() {
            return Err(format!("texture {} is not a file", entry.path).into());
        }
        if !paths.insert(path.clone()) {
            return Err(format!("duplicate canonical texture path: {}", entry.path).into());
        }
        resolved.push(path);
    }
    let train_families: Vec<String> = families
        .iter()
        .filter(|(_, s)| **s == Split::Train)
        .map(|(f, _)| f.clone())
        .collect();
    let evaluation_files = parsed.textures.iter().filter(|e| e.split == split).count();
    if evaluation_files == 0 {
        return Err(format!("no textures in selected {split:?} split").into());
    }
    let eval_count = evaluation_files
        .checked_mul(eval_per_texture)
        .ok_or("evaluation sample count overflow")?;
    let train_count = if include_train { train_budget } else { 0 };
    let total = train_count
        .checked_add(eval_count)
        .and_then(|n| n.checked_mul(std::mem::size_of::<Texels>()))
        .ok_or("corpus sample memory overflow")?;
    if total > 512 * 1024 * 1024 {
        return Err("corpus samples exceed 512 MiB limit; reduce sample budgets".into());
    }
    let mut quotas = BTreeMap::new();
    if train_count > 0 {
        if train_families.is_empty() {
            return Err("training requested but manifest has no training families".into());
        }
        for (index, family) in train_families.iter().enumerate() {
            let images: Vec<_> = parsed
                .textures
                .iter()
                .filter(|e| &e.family == family)
                .collect();
            let quota = train_count / train_families.len()
                + usize::from(index < train_count % train_families.len());
            if quota < images.len() {
                return Err(format!(
                    "training budget too small to cover every image of family {family}"
                )
                .into());
            }
            for (index, entry) in images.iter().enumerate() {
                quotas.insert(
                    entry.path.clone(),
                    quota / images.len() + usize::from(index < quota % images.len()),
                );
            }
        }
    }
    let mut result = CorpusData {
        train: Vec::with_capacity(train_count),
        evaluation: Vec::with_capacity(eval_count),
        groups: Vec::new(),
        metadata: Value::Null,
        train_families,
    };
    let mut training_images = Vec::new();
    for (entry, path) in parsed.textures.iter().zip(resolved) {
        let train_quota = quotas.get(&entry.path).copied().unwrap_or(0);
        if entry.split != split && train_quota == 0 {
            continue;
        }
        let img = image::open(&path)
            .map_err(|e| format!("decode texture {}: {e}", entry.path))?
            .to_rgb8();
        if img.width() == 0 || img.height() == 0 {
            return Err(format!("texture {} has zero dimensions", entry.path).into());
        }
        if train_quota > 0 {
            sample(&img, train_quota, stream(seed, entry), &mut result.train);
            training_images.push(json!({"path":entry.path,"family":entry.family,"width":img.width(),"height":img.height(),"sample_count":train_quota}));
        } else {
            let start = result.evaluation.len();
            sample(
                &img,
                eval_per_texture,
                stream(seed, entry),
                &mut result.evaluation,
            );
            result.groups.push(EvalGroup {
                name: format!("texture:{}", entry.path),
                start,
                count: eval_per_texture,
                family: entry.family.clone(),
                path: entry.path.clone(),
                width: img.width(),
                height: img.height(),
            });
        }
    }
    result.metadata = json!({
        "manifest_version":1,"manifest_path":manifest_path.to_string_lossy(),
        "manifest_digest":{"algorithm":"fnv1a64 (noncryptographic)","raw_bytes":format!("{:016x}",hash(&bytes))},
        "seed":seed,"evaluation_split":split,"training_families":result.train_families,
        "training_images":training_images,"evaluation_images":result.groups,
        "training_sample_count":result.train.len(),"evaluation_sample_count":result.evaluation.len(),
        "sampling":"with replacement; balanced training families then images; per-image SplitMix64 streams",
        "conversion":"image crate decoder to_rgb8; source color/alpha conversion follows image crate",
        "padding":"4x4 block grid; nearest border pixel replication",
        "decoder_limit":"image crate default limits (512 MiB decode allocation)",
        "family_policy":"declared families authoritative; copy/resized-variant leakage audit deferred"
    });
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("build/corpus-tests");
            fs::create_dir_all(&root).unwrap();
            let path = root.join(format!(
                "{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn image(&self, name: &str, color: u8) {
            image::RgbImage::from_fn(7, 5, |x, y| image::Rgb([color, x as u8, y as u8]))
                .save(self.0.join(name))
                .unwrap();
        }
        fn manifest(&self, entries: Value) -> PathBuf {
            let path = self.0.join("manifest.json");
            fs::write(
                &path,
                serde_json::to_vec(&json!({"version":1,"textures":entries})).unwrap(),
            )
            .unwrap();
            path
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let root =
                fs::canonicalize(Path::new(env!("CARGO_MANIFEST_DIR")).join("build/corpus-tests"))
                    .unwrap();
            let path = fs::canonicalize(&self.0).unwrap();
            assert_eq!(path.parent(), Some(root.as_path()));
            fs::remove_dir_all(path).unwrap();
        }
    }
    fn entry(path: &str, family: &str, split: &str) -> Value {
        json!({"path":path,"family":family,"split":split})
    }

    #[test]
    fn rejects_cross_split_families_and_canonical_duplicates() {
        let f = Fixture::new();
        f.image("a.png", 1);
        f.image("b.png", 2);
        let m = f.manifest(json!([
            entry("a.png", "same", "train"),
            entry("b.png", "same", "validation")
        ]));
        assert!(
            load(&m, 4, 1, Split::Validation, 1, true)
                .err()
                .unwrap()
                .to_string()
                .contains("multiple splits")
        );
        let m = f.manifest(json!([
            entry("a.png", "a", "train"),
            entry("./a.png", "b", "validation")
        ]));
        assert!(
            load(&m, 4, 1, Split::Validation, 1, true)
                .err()
                .unwrap()
                .to_string()
                .contains("duplicate canonical")
        );
    }

    #[test]
    fn deterministic_order_independent_streams_and_balanced_families() {
        let f = Fixture::new();
        for (name, color) in [
            ("a.png", 1),
            ("b.png", 2),
            ("c.png", 3),
            ("v.png", 4),
            ("t.png", 5),
        ] {
            f.image(name, color);
        }
        let entries = vec![
            entry("a.png", "one", "train"),
            entry("b.png", "one", "train"),
            entry("c.png", "two", "train"),
            entry("v.png", "validation", "validation"),
            entry("t.png", "test", "test"),
        ];
        let m = f.manifest(json!(entries));
        let a = load(&m, 12, 2, Split::Validation, 42, true).unwrap();
        let m = f.manifest(json!(entries.into_iter().rev().collect::<Vec<_>>()));
        let b = load(&m, 12, 7, Split::Test, 42, true).unwrap();
        assert_eq!(a.train, b.train);
        assert_eq!(a.train.iter().filter(|block| block[0][0] == 1).count(), 3);
        assert_eq!(a.train.iter().filter(|block| block[0][0] == 2).count(), 3);
        assert_eq!(a.train.iter().filter(|block| block[0][0] == 3).count(), 6);
        let c = load(&m, 12, 7, Split::Validation, 42, true).unwrap();
        assert_eq!(a.evaluation, &c.evaluation[..2]);
        assert_eq!(c.groups[0].start, 0);
        assert_eq!(c.groups[0].count, 7);
        assert_eq!((c.groups[0].width, c.groups[0].height), (7, 5));
        assert!(load(&m, 2, 1, Split::Validation, 42, true).is_err());
    }

    #[test]
    fn decodes_only_requested_splits() {
        let f = Fixture::new();
        f.image("v.png", 1);
        f.image("t.png", 2);
        fs::write(f.0.join("bad.dat"), b"not an image").unwrap();
        let m = f.manifest(json!([
            entry("bad.dat", "train", "train"),
            entry("v.png", "v", "validation"),
            entry("t.png", "t", "test")
        ]));
        assert!(
            load(&m, 10, 3, Split::Validation, 0, false)
                .unwrap()
                .train
                .is_empty()
        );
        assert!(load(&m, 10, 3, Split::Test, 0, false).is_ok());
        assert!(
            load(&m, 10, 3, Split::Validation, 0, true)
                .err()
                .unwrap()
                .to_string()
                .contains("bad.dat")
        );
        let m = f.manifest(json!([
            entry("v.png", "v", "validation"),
            entry("bad.dat", "t", "test")
        ]));
        assert!(load(&m, 0, 1, Split::Validation, 0, false).is_ok());
    }

    #[test]
    fn tiny_and_partial_blocks_replicate_border() {
        let img = image::RgbImage::from_fn(1, 2, |_, y| image::Rgb([y as u8, 10, 20]));
        let mut samples = Vec::new();
        sample(&img, 1, 0, &mut samples);
        for (index, pixel) in samples[0].iter().enumerate() {
            assert_eq!(*pixel, [(index / 4).min(1) as u8, 10, 20]);
        }
        let img = image::RgbImage::from_fn(5, 5, |x, y| image::Rgb([x as u8, y as u8, 0]));
        let mut samples = Vec::new();
        sample(&img, 100, 0, &mut samples);
        assert!(samples.iter().any(|block| block[0] == [4, 4, 0]));
        for block in samples {
            for (index, pixel) in block.iter().enumerate() {
                assert_eq!(
                    *pixel,
                    [
                        (u32::from(block[0][0]) + index as u32 % 4).min(4) as u8,
                        (u32::from(block[0][1]) + index as u32 / 4).min(4) as u8,
                        0
                    ]
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_version_empty_family_and_excessive_budget() {
        let f = Fixture::new();
        f.image("v.png", 1);
        let m = f.manifest(json!([entry("v.png", " ", "validation")]));
        assert!(load(&m, 0, 1, Split::Validation, 0, false).is_err());
        let m = f.manifest(json!([entry("v.png", "v", "validation")]));
        assert!(load(&m, 0, usize::MAX, Split::Validation, 0, false).is_err());
        fs::write(&m, b"{\"version\":2,\"textures\":[]}").unwrap();
        assert!(
            load(&m, 0, 1, Split::Validation, 0, false)
                .err()
                .unwrap()
                .to_string()
                .contains("version 2")
        );
    }
}
