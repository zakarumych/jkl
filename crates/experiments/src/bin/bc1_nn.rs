//! Headless residual-endpoint BC1 experiment. See README for dataset and timing semantics.
use jkl::{
    image::block::bc1::Block,
    math::{Rgb32F, Rgb565, Vec3, Yiq32F, pca_axis3},
};
use mev::kernels::nn::{Activation, Adam, Batch, Encoding, Nn, NnDesc, Optimizer};
use mev::{Buffer, BufferDesc, BufferInitDesc, BufferUsage, Device, PipelineStages, Queue};
use serde_json::{Value, json};
use std::{error::Error, fs, hint::black_box, path::PathBuf, time::Instant};
#[path = "bc1_nn/corpus.rs"]
mod corpus;
#[path = "bc1_nn/format_color.rs"]
mod format_color;
#[path = "bc1_nn/format_experiment.rs"]
mod format_experiment;
#[path = "bc1_nn/format_scalar.rs"]
mod format_scalar;
#[path = "bc1_nn/gpu_encode.rs"]
mod gpu_encode;
#[path = "bc1_nn/gpu_search.rs"]
mod gpu_search;
#[path = "bc1_nn/gradient_refine.rs"]
mod gradient_refine;
#[path = "bc1_nn/neighborhood.rs"]
mod neighborhood;
#[path = "bc1_nn/reconstruction_loss.rs"]
mod reconstruction_loss;
#[path = "bc1_nn/scalar_nn.rs"]
mod scalar_nn;
type Texels = [[u8; 3]; 16];
type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Options {
    formats_train: bool,
    formats_compare: bool,
    scalar_models: Option<PathBuf>,
    samples: usize,
    steps: usize,
    batch: usize,
    repeats: usize,
    seed: u64,
    sizes: Vec<Vec<u32>>,
    images: Vec<PathBuf>,
    output: PathBuf,
    preview_size: u32,
    load_models: Option<PathBuf>,
    reconstruction_steps: usize,
    reconstruction_rate: f32,
    gd_compare: bool,
    neighbor_compare: bool,
    three_mode_compare: bool,
    principal_ls_sweep: bool,
    pipeline_neighbor_compare: bool,
    principal_ls_passes: Vec<u32>,
    gd_steps: Vec<u32>,
    gd_rates: Vec<f32>,
    corpus_manifest: Option<PathBuf>,
    corpus_split: corpus::Split,
    eval_blocks_per_texture: usize,
    sizes_explicit: bool,
    corpus_groups: Vec<corpus::EvalGroup>,
    corpus_metadata: Value,
    train_families: Vec<String>,
}
impl Options {
    fn parse() -> Result<Self> {
        Self::parse_args(std::env::args().skip(1))
    }
    fn parse_args(args: impl Iterator<Item = String>) -> Result<Self> {
        let mut o = Self {
            formats_train: false,
            formats_compare: false,
            scalar_models: None,
            samples: 4096,
            steps: 1000,
            batch: 64,
            repeats: 7,
            seed: 20261014,
            sizes: vec![vec![16, 6], vec![32, 6], vec![32, 16, 6], vec![64, 32, 6]],
            images: vec![],
            output: "bc1-nn-results".into(),
            preview_size: 1024,
            load_models: None,
            reconstruction_steps: 0,
            reconstruction_rate: 1e-4,
            gd_compare: false,
            neighbor_compare: false,
            three_mode_compare: false,
            principal_ls_sweep: false,
            pipeline_neighbor_compare: false,
            principal_ls_passes: vec![4, 8, 16, 32, 64],
            gd_steps: vec![4, 8, 16],
            gd_rates: vec![0.02, 0.1, 0.5],
            corpus_manifest: None,
            corpus_split: corpus::Split::Validation,
            eval_blocks_per_texture: 256,
            sizes_explicit: false,
            corpus_groups: Vec::new(),
            corpus_metadata: Value::Null,
            train_families: Vec::new(),
        };
        let mut args = args;
        let mut steps_explicit = false;
        while let Some(arg) = args.next() {
            if arg == "--formats-train" {
                o.formats_train = true;
                continue;
            }
            if arg == "--formats-compare" {
                o.formats_compare = true;
                continue;
            }
            if arg == "--pipeline-neighbor-compare" {
                o.pipeline_neighbor_compare = true;
                continue;
            }
            if arg == "--principal-ls-sweep" {
                o.principal_ls_sweep = true;
                continue;
            }
            if arg == "--three-mode-compare" {
                o.three_mode_compare = true;
                continue;
            }
            if arg == "--gd-compare" {
                o.gd_compare = true;
                continue;
            }
            if arg == "--ls-neighbor-compare" {
                o.neighbor_compare = true;
                continue;
            }
            if arg == "--help" {
                println!(
                    "BC2-5: --formats-train (validation split, endpoint MSE scalar training) or --formats-compare (test split, frozen RGB --load-models DIR and --scalar-models DIR); requires --corpus-manifest PATH."
                );
                println!(
                    "bc1_nn --samples 4096 --steps 1000 --batch 64 --repeats 7 --seed 20261014 --sizes 16,32,32x16,64x32 --image PATH [--image PATH] --output DIR --load-models DIR --reconstruction-steps 0 --reconstruction-rate 0.0001 --preview-size 1024 [--gd-compare --gd-steps 4,8,16 --gd-rates 0.02,0.1,0.5 | --ls-neighbor-compare]"
                );
                println!(
                    "Corpus: --corpus-manifest PATH --corpus-split validation|test --eval-blocks-per-texture 256. Test requires --load-models and a fresh empty output directory; weights/layers/mode come directly from selected.json; no training or comparison tuning."
                );
                println!(
                    "Frozen test --three-mode-compare: production GPU cluster fit, NN range guard + up to 2 LS, PCA extreme texels + up to 4 LS. Fixed methods; no model selection."
                );
                println!(
                    "Exploratory --principal-ls-sweep --principal-ls-passes 4,8,16,32,64: compare PCA LS budgets vs frozen NN+2LS; test split and saved model required."
                );
                println!(
                    "Exploratory --pipeline-neighbor-compare: NN+2LS and PCA+4LS, each with one/two RGB565 neighbor rounds and rebuilt/reused indices; frozen model, test split required."
                );
                std::process::exit(0);
            }
            let v = args.next().ok_or("missing option value")?;
            match arg.as_str() {
                "--scalar-models" => o.scalar_models = Some(v.into()),
                "--samples" => o.samples = v.parse()?,
                "--steps" => {
                    o.steps = v.parse()?;
                    steps_explicit = true;
                }
                "--batch" => o.batch = v.parse()?,
                "--repeats" => o.repeats = v.parse()?,
                "--seed" => o.seed = v.parse()?,
                "--output" => o.output = v.into(),
                "--image" => o.images.push(v.into()),
                "--preview-size" => o.preview_size = v.parse()?,
                "--load-models" => o.load_models = Some(v.into()),
                "--corpus-manifest" => o.corpus_manifest = Some(v.into()),
                "--corpus-split" => o.corpus_split = serde_json::from_value(json!(v))?,
                "--eval-blocks-per-texture" => o.eval_blocks_per_texture = v.parse()?,
                "--reconstruction-steps" => o.reconstruction_steps = v.parse()?,
                "--reconstruction-rate" => o.reconstruction_rate = v.parse()?,
                "--principal-ls-passes" => {
                    o.principal_ls_passes = v
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?
                }
                "--gd-steps" => {
                    o.gd_steps = v
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?
                }
                "--gd-rates" => {
                    o.gd_rates = v
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?
                }
                "--sizes" => {
                    o.sizes_explicit = true;
                    o.sizes = v
                        .split(',')
                        .map(|s| {
                            let mut l = s
                                .split('x')
                                .map(str::parse)
                                .collect::<std::result::Result<Vec<u32>, _>>()?;
                            l.push(6);
                            Ok(l)
                        })
                        .collect::<std::result::Result<_, std::num::ParseIntError>>()?
                }
                _ => return Err(format!("unknown option {arg}").into()),
            }
        }
        if o.samples < 16
            || o.batch == 0
            || o.batch > o.samples
            || o.repeats == 0
            || o.sizes.is_empty()
            || o.sizes.iter().any(|l| l.iter().any(|&n| n == 0 || n > 64))
            || !o.reconstruction_rate.is_finite()
            || o.reconstruction_rate <= 0.0
            || o.gd_steps.is_empty()
            || o.gd_steps.iter().any(|&s| s == 0 || s > 64)
            || o.gd_rates.is_empty()
            || o.gd_rates.iter().any(|&r| !r.is_finite() || r <= 0.0)
        {
            return Err(
                "require samples>=16, 0<batch<=samples, repeats>0, hidden widths 1..64".into(),
            );
        }
        if (o.gd_compare || o.neighbor_compare) && o.load_models.is_none() {
            return Err("comparison requires --load-models".into());
        }
        if o.gd_compare && o.neighbor_compare {
            return Err("choose --gd-compare or --ls-neighbor-compare".into());
        }
        if o.eval_blocks_per_texture == 0 {
            return Err("require --eval-blocks-per-texture > 0".into());
        }
        if o.corpus_manifest.is_some() && !o.images.is_empty() {
            return Err("choose --image or --corpus-manifest".into());
        }
        if o.corpus_split == corpus::Split::Train {
            return Err("evaluation --corpus-split must be validation or test".into());
        }
        if o.three_mode_compare && o.corpus_split != corpus::Split::Test {
            return Err("--three-mode-compare requires frozen corpus test split".into());
        }
        if o.principal_ls_passes.is_empty()
            || o.principal_ls_passes.iter().any(|&p| p == 0 || p > 64)
            || o.principal_ls_passes.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("principal LS budgets must be strictly increasing, within 1..64".into());
        }
        if o.principal_ls_sweep && o.corpus_split != corpus::Split::Test {
            return Err("--principal-ls-sweep requires frozen corpus test split".into());
        }
        if o.principal_ls_sweep && o.three_mode_compare {
            return Err("choose --principal-ls-sweep or --three-mode-compare".into());
        }
        if o.pipeline_neighbor_compare && o.corpus_split != corpus::Split::Test {
            return Err("--pipeline-neighbor-compare requires frozen corpus test split".into());
        }
        if o.pipeline_neighbor_compare && (o.principal_ls_sweep || o.three_mode_compare) {
            return Err("choose one frozen-test comparison".into());
        }
        if o.corpus_split == corpus::Split::Test {
            if o.corpus_manifest.is_none() || o.load_models.is_none() {
                return Err("locked test requires --corpus-manifest and --load-models".into());
            }
            if (steps_explicit && o.steps > 0)
                || o.reconstruction_steps > 0
                || o.gd_compare
                || o.neighbor_compare
            {
                return Err(
                    "locked test prohibits training, reconstruction, and comparison tuning".into(),
                );
            }
        }
        if o.formats_train || o.formats_compare {
            if o.formats_train == o.formats_compare || o.corpus_manifest.is_none() {
                return Err(
                    "choose --formats-train or --formats-compare with --corpus-manifest".into(),
                );
            }
            if o.gd_compare
                || o.neighbor_compare
                || o.three_mode_compare
                || o.principal_ls_sweep
                || o.pipeline_neighbor_compare
                || o.reconstruction_steps > 0
            {
                return Err("format experiments prohibit other comparison/training modes".into());
            }
            if o.formats_train && o.corpus_split != corpus::Split::Validation {
                return Err("scalar training requires validation split".into());
            }
            if o.formats_compare
                && (o.corpus_split != corpus::Split::Test
                    || o.scalar_models.is_none()
                    || o.load_models.is_none())
            {
                return Err(
                    "format comparison requires test split, --load-models and --scalar-models"
                        .into(),
                );
            }
        }
        Ok(o)
    }
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 16777216.0
    }
}
fn rgb(c: [u8; 3]) -> Rgb32F {
    Rgb32F(c.map(|v| v as f32 / 255.0))
}
fn distance(a: Rgb32F, b: Rgb32F) -> f32 {
    (0..3)
        .map(|c| (a.0[c] - b.0[c]).powi(2))
        .sum::<f32>()
        .sqrt()
}
fn production(t: &Texels) -> Block {
    Block::encode(std::array::from_fn(|y| {
        std::array::from_fn(|x| (rgb(t[y * 4 + x]), true))
    }))
}
fn error(t: &Texels, b: Block) -> f32 {
    let d = b.decode();
    (0..16)
        .map(|i| distance(rgb(t[i]), d[i / 4][i % 4]))
        .sum::<f32>()
        / 16.0
}
fn endpoints(t: &Texels, b: Block) -> [f32; 6] {
    let axis = principal_axis(t);
    let mut a = b.color0.into_f32().0;
    let mut c = b.color1.into_f32().0;
    let pa = Vec3(a).dot(axis);
    let pc = Vec3(c).dot(axis);
    if pa > pc || (pa == pc && a > c) {
        std::mem::swap(&mut a, &mut c);
    }
    [a[0], a[1], a[2], c[0], c[1], c[2]]
}
fn principal_axis(t: &Texels) -> Vec3 {
    let mut colors = *t;
    colors.sort_unstable();
    let samples = colors.map(|c| Vec3(rgb(c).0));
    let mut axis = pca_axis3(&samples);
    if !axis.0.iter().all(|x| x.is_finite()) || axis.length_squared() < 1e-12 {
        axis = Vec3([1.0, 0.0, 0.0]);
    }
    let dominant = (0..3)
        .max_by(|&a, &b| axis.0[a].abs().total_cmp(&axis.0[b].abs()).then(b.cmp(&a)))
        .unwrap();
    if axis.0[dominant] < 0.0 {
        axis *= -1.0;
    }
    axis
}
fn canonical(t: &Texels) -> [f32; 48] {
    let mut colors = *t;
    let axis = principal_axis(t);
    colors.sort_unstable_by(|a, b| {
        Vec3(rgb(*a).0)
            .dot(axis)
            .total_cmp(&Vec3(rgb(*b).0).dot(axis))
            .then(a.cmp(b))
    });
    std::array::from_fn(|i| colors[i / 3][i % 3] as f32 / 255.0)
}
fn baseline(input: &[f32]) -> [f32; 6] {
    [
        input[0], input[1], input[2], input[45], input[46], input[47],
    ]
}
fn quantize(c: [f32; 3]) -> Rgb565 {
    Rgb565::new(
        (c[0].clamp(0.0, 1.0) * 31.0).round() as u8,
        (c[1].clamp(0.0, 1.0) * 63.0).round() as u8,
        (c[2].clamp(0.0, 1.0) * 31.0).round() as u8,
    )
}
fn pack(t: &Texels, e: [f32; 6]) -> Block {
    let mut a = quantize([e[0], e[1], e[2]]);
    let mut b = quantize([e[3], e[4], e[5]]);
    if a.bits() < b.bits() {
        std::mem::swap(&mut a, &mut b);
    }
    let mut block = Block {
        color0: a,
        color1: b,
        indices: [0; 4],
    };
    let palette = Block {
        indices: [0xe4; 4],
        ..block
    }
    .decode()[0];
    for i in 0..16 {
        // Equal endpoints use only their opaque color; index 3 is transparent.
        let n = if a == b { 1 } else { 4 };
        let idx = (0..n)
            .min_by(|&x, &y| {
                distance(rgb(t[i]), palette[x]).total_cmp(&distance(rgb(t[i]), palette[y]))
            })
            .unwrap();
        block.indices[i / 4] |= (idx as u8) << ((i % 4) * 2);
    }
    block
}
fn fit(t: &Texels, weights: &[f32; 16]) -> Option<[f32; 6]> {
    let mut aa = 0.0;
    let mut ab = 0.0;
    let mut bb = 0.0;
    let mut ax = [0.0; 3];
    let mut bx = [0.0; 3];
    for i in 0..16 {
        let b = weights[i];
        let a = 1.0 - b;
        aa += a * a;
        ab += a * b;
        bb += b * b;
        for c in 0..3 {
            let v = t[i][c] as f32 / 255.0;
            ax[c] += a * v;
            bx[c] += b * v;
        }
    }
    let det = aa * bb - ab * ab;
    if det.abs() < 1e-8 {
        return None;
    }
    Some(std::array::from_fn(|i| {
        if i < 3 {
            (bb * ax[i] - ab * bx[i]) / det
        } else {
            (aa * bx[i - 3] - ab * ax[i - 3]) / det
        }
    }))
}
fn refine(t: &Texels, b: Block) -> Block {
    refine_passes(t, b, 2)
}
fn refine_passes(t: &Texels, mut b: Block, passes: usize) -> Block {
    for _ in 0..passes {
        let weights = std::array::from_fn(|i| match (b.indices[i / 4] >> ((i % 4) * 2)) & 3 {
            0 => 0.0,
            1 => 1.0,
            2 => 1.0 / 3.0,
            _ => 2.0 / 3.0,
        });
        if let Some(e) = fit(t, &weights) {
            let c = pack(t, e);
            if error(t, c) < error(t, b) {
                b = c;
            } else {
                break;
            }
        }
    }
    b
}
fn guarded(t: &Texels, input: &[f32], residual: &[f32], refined: bool) -> Block {
    let base = baseline(input);
    let e = std::array::from_fn(|i| base[i] + residual[i]);
    let a = pack(t, e);
    let b = pack(t, base);
    let best = if error(t, a) < error(t, b) { a } else { b };
    if refined { refine(t, best) } else { best }
}
fn teacher(t: &Texels) -> Block {
    let input = canonical(t);
    let mut prefix = [[0.0f32; 3]; 17];
    for i in 0..16 {
        for c in 0..3 {
            prefix[i + 1][c] = prefix[i][c] + input[i * 3 + c];
        }
    }
    let mut best = production(t);
    let mut best_error = error(t, best);
    // All four sorted clusters, including empty clusters. Prefix sums keep each fit O(1).
    for i in 0..=16 {
        for j in i..=16 {
            for k in j..=16 {
                let cuts = [0, i, j, k, 16];
                let mut aa = 0.0;
                let mut ab = 0.0;
                let mut bb = 0.0;
                let mut ax = [0.0; 3];
                let mut bx = [0.0; 3];
                for g in 0..4 {
                    let b = g as f32 / 3.0;
                    let a = 1.0 - b;
                    let n = (cuts[g + 1] - cuts[g]) as f32;
                    aa += n * a * a;
                    ab += n * a * b;
                    bb += n * b * b;
                    for c in 0..3 {
                        let sum = prefix[cuts[g + 1]][c] - prefix[cuts[g]][c];
                        ax[c] += a * sum;
                        bx[c] += b * sum;
                    }
                }
                let det = aa * bb - ab * ab;
                if det < 1e-8 {
                    continue;
                }
                let e = std::array::from_fn(|c| {
                    if c < 3 {
                        (bb * ax[c] - ab * bx[c]) / det
                    } else {
                        (aa * bx[c - 3] - ab * ax[c - 3]) / det
                    }
                });
                let block = pack(t, e);
                let err = error(t, block);
                if err < best_error {
                    best = block;
                    best_error = err;
                }
            }
        }
    }
    best
}
fn synthetic(rng: &mut Rng, index: usize) -> Texels {
    let a = [rng.unit(), rng.unit(), rng.unit()];
    let b = [rng.unit(), rng.unit(), rng.unit()];
    std::array::from_fn(|i| {
        let w = match index % 6 {
            0 => 0.0,
            1 => i as f32 / 15.0,
            2 => {
                if i < 8 {
                    0.0
                } else {
                    1.0
                }
            }
            3 => (i % 4) as f32 / 3.0,
            _ => rng.unit(),
        };
        std::array::from_fn(|c| {
            let v = match index % 6 {
                4 => rng.unit(),
                5 => a[c] + (rng.unit() - 0.5) * 0.06,
                _ => a[c] * (1.0 - w) + b[c] * w,
            };
            (v.clamp(0.0, 1.0) * 255.0).round() as u8
        })
    })
}
fn image_blocks(img: &image::RgbImage) -> Vec<Texels> {
    let (w, h) = img.dimensions();
    (0..h.div_ceil(4))
        .flat_map(|y| {
            (0..w.div_ceil(4)).map(move |x| {
                std::array::from_fn(|i| {
                    img.get_pixel(
                        (x * 4 + i as u32 % 4).min(w - 1),
                        (y * 4 + i as u32 / 4).min(h - 1),
                    )
                    .0
                })
            })
        })
        .collect()
}
fn data(o: &mut Options) -> Result<(Vec<Texels>, Vec<Texels>)> {
    if let Some(path) = &o.corpus_manifest {
        let loaded = corpus::load(
            path,
            o.samples,
            o.eval_blocks_per_texture,
            o.corpus_split,
            o.seed,
            o.formats_train || o.load_models.is_none() || o.reconstruction_steps > 0,
        )?;
        o.corpus_groups = loaded.groups;
        o.corpus_metadata = loaded.metadata;
        o.train_families = loaded.train_families;
        if (o.formats_train || o.load_models.is_none() || o.reconstruction_steps > 0)
            && o.batch > loaded.train.len()
        {
            return Err(format!(
                "training batch {} exceeds actual corpus train samples {}",
                o.batch,
                loaded.train.len()
            )
            .into());
        }
        return Ok((loaded.train, loaded.evaluation));
    }
    let mut rng = Rng(o.seed.max(1));
    let mut train = (0..o.samples)
        .map(|i| synthetic(&mut rng, i))
        .collect::<Vec<_>>();
    let mut test = (0..(o.samples / 4).max(64))
        .map(|i| synthetic(&mut rng, i))
        .collect::<Vec<_>>();
    for (image_index, path) in o.images.iter().enumerate() {
        let img = image::open(path)?.to_rgb8();
        let blocks = image_blocks(&img);
        let rows = img.height().div_ceil(4) as usize;
        let cols = img.width().div_ceil(4) as usize;
        // Treat all supplied textures as one source family. Only first source's top 40% trains;
        // all bottom 40% may evaluate, leaving a 20% source-region gap.
        let top = rows * 2 / 5 * cols;
        let bottom = rows * 3 / 5 * cols;
        if image_index == 0 && top > 0 {
            for i in 0..train.len() / 2 {
                train[i] = blocks[(rng.next() as usize) % top];
            }
        }
        if bottom < blocks.len() {
            let count = (o.samples / 4).max(64);
            for _ in 0..count {
                test.push(blocks[bottom + rng.next() as usize % (blocks.len() - bottom)]);
            }
        }
    }
    Ok((train, test))
}
fn storage(device: &Device, values: &[f32], name: &str) -> Buffer {
    device.new_buffer_init(BufferInitDesc {
        name,
        data: bytemuck::cast_slice::<f32, u8>(values),
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
    })
}
fn read(device: &Device, queue: &mut Queue, source: &Buffer, count: usize) -> Result<Vec<f32>> {
    let mut stage = device.new_buffer(BufferDesc {
        name: "NN weights readback",
        size: count * 4,
        usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
    });
    let mut enc = queue.new_command_encoder();
    enc.barrier(
        PipelineStages::COMPUTE_SHADER | PipelineStages::TRANSFER,
        PipelineStages::TRANSFER,
    );
    enc.copy().copy_buffer_to_buffer(source, &stage);
    queue.submit_checkpoint([enc.finish()])?;
    queue.wait_idle()?;
    let mut values = vec![0.0; count];
    stage.read(0, bytemuck::cast_slice_mut(&mut values))?;
    Ok(values)
}
#[derive(Clone)]
struct Dense {
    layers: Vec<u32>,
    weights: Vec<f32>,
}
impl Dense {
    fn new(layers: Vec<u32>, rng: &mut Rng) -> Self {
        let mut weights = vec![];
        let mut n = 48;
        for &m in &layers {
            let scale = (6.0 / (n + m) as f32).sqrt();
            weights.extend((0..n * m).map(|_| (rng.unit() * 2.0 - 1.0) * scale));
            weights.extend((0..m).map(|_| 0.0));
            n = m;
        }
        Self { layers, weights }
    }
    fn forward(&self, input: &[f32]) -> [f32; 6] {
        let mut a = [0.0; 64];
        a[..48].copy_from_slice(input);
        let mut n = 48;
        let mut offset = 0;
        for (layer, &m) in self.layers.iter().enumerate() {
            let m = m as usize;
            let mut b = [0.0; 64];
            for o in 0..m {
                let mut v = self.weights[offset + m * n + o];
                for i in 0..n {
                    v += self.weights[offset + o * n + i] * a[i];
                }
                b[o] = if layer + 1 == self.layers.len() {
                    v
                } else {
                    v.max(0.0)
                };
            }
            offset += m * (n + 1);
            n = m;
            a = b;
        }
        a[..6].try_into().unwrap()
    }
}
struct InferenceBuffers {
    input: Buffer,
    output: Buffer,
    stage: Buffer,
    inputs: Vec<f32>,
    outputs: Vec<f32>,
}
impl InferenceBuffers {
    fn new(device: &Device, n: usize) -> Self {
        Self {
            input: storage(device, &vec![0.0; n * 48], "NN inputs"),
            output: storage(device, &vec![0.0; n * 6], "NN endpoints"),
            stage: device.new_buffer(BufferDesc {
                name: "NN readback",
                size: n * 6 * 4,
                usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
            }),
            inputs: vec![0.0; n * 48],
            outputs: vec![0.0; n * 6],
        }
    }
    fn upload(&self, queue: &mut Queue) -> Result<()> {
        let mut enc = queue.new_command_encoder();
        enc.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        enc.copy()
            .write_buffer_raw(&self.input, bytemuck::cast_slice(&self.inputs));
        enc.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
        queue.submit([enc.finish()])?;
        Ok(())
    }
    fn infer(&mut self, queue: &mut Queue, nn: &Nn, readback: bool) -> Result<()> {
        nn.inference(
            queue,
            (&self.input).into(),
            (&self.output).into(),
            &[],
            Batch::new((self.inputs.len() / 48) as u32),
        )?;
        if readback {
            let mut enc = queue.new_command_encoder();
            enc.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
            enc.copy().copy_buffer_to_buffer(&self.output, &self.stage);
            queue.submit_checkpoint([enc.finish()])?;
        }
        queue.wait_idle()?;
        if readback {
            self.stage
                .read(0, bytemuck::cast_slice_mut(&mut self.outputs))?;
        }
        Ok(())
    }
}
fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}
fn benchmark(mut f: impl FnMut() -> Result<()>, repeats: usize) -> Result<f64> {
    f()?;
    let mut times = vec![];
    for _ in 0..repeats {
        let start = Instant::now();
        f()?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median(times))
}
fn quality(t: &[Texels], b: &[Block]) -> Value {
    let errors = t
        .iter()
        .zip(b)
        .map(|(t, &b)| error(t, b))
        .collect::<Vec<_>>();
    let mut sorted = errors.clone();
    sorted.sort_by(f32::total_cmp);
    let yiq = t
        .iter()
        .zip(b)
        .map(|(t, &b)| {
            let d = b.decode();
            (0..16)
                .map(|i| {
                    Yiq32F::perceptual_distance(
                        Yiq32F::from_rgb(rgb(t[i])),
                        Yiq32F::from_rgb(d[i / 4][i % 4]),
                    )
                })
                .sum::<f32>()
                / 16.0
        })
        .sum::<f32>()
        / t.len() as f32;
    let mse = t
        .iter()
        .zip(b)
        .map(|(t, &b)| {
            let d = b.decode();
            (0..16)
                .flat_map(|i| (0..3).map(move |c| (rgb(t[i]).0[c] - d[i / 4][i % 4].0[c]).powi(2)))
                .sum::<f32>()
                / 48.0
        })
        .sum::<f32>()
        / t.len() as f32;
    json!({"mean_rgb_distance":errors.iter().sum::<f32>()/errors.len() as f32,"p95_rgb_distance":sorted[(sorted.len()-1)*95/100],"mean_yiq_distance":yiq,"rgb_mse":mse,"rgb_psnr_db":-10.0*mse.max(1e-20).log10()})
}
fn grouped_quality(o: &Options, t: &[Texels], b: &[Block]) -> Value {
    let mut result = quality(t, b);
    result["blocks"] = json!(t.len());
    if o.corpus_manifest.is_some() {
        let mut groups = serde_json::Map::new();
        for group in &o.corpus_groups {
            let end = group.start + group.count;
            let mut q = quality(&t[group.start..end], &b[group.start..end]);
            q["blocks"] = json!(group.count);
            q["family"] = json!(group.family);
            q["path"] = json!(group.path);
            q["width"] = json!(group.width);
            q["height"] = json!(group.height);
            groups.insert(group.name.clone(), q);
        }
        result["groups"] = Value::Object(groups);
        return result;
    }
    let count = (o.samples / 4).max(64);
    let mut groups = serde_json::Map::new();
    groups.insert("synthetic".into(), quality(&t[..count], &b[..count]));
    for category in 0..6 {
        let indices = (0..count).filter(|i| i % 6 == category).collect::<Vec<_>>();
        let texels = indices.iter().map(|&i| t[i]).collect::<Vec<_>>();
        let blocks = indices.iter().map(|&i| b[i]).collect::<Vec<_>>();
        groups.insert(
            format!("synthetic_category_{category}"),
            quality(&texels, &blocks),
        );
    }
    for (i, path) in o.images.iter().enumerate() {
        let start = count * (i + 1);
        let end = (start + count).min(t.len());
        if start < end {
            groups.insert(
                path.display().to_string(),
                quality(&t[start..end], &b[start..end]),
            );
        }
    }
    result["groups"] = Value::Object(groups);
    result
}
fn prediction(t: &Texels, input: &[f32], residual: &[f32], mode: usize) -> Block {
    if mode == 0 {
        let base = baseline(input);
        pack(t, std::array::from_fn(|c| base[c] + residual[c]))
    } else {
        guarded(t, input, residual, mode == 2)
    }
}
fn load_dense(directory: &PathBuf, layers: &[u32], name: &str) -> Result<Dense> {
    let metadata: Value = serde_json::from_slice(&fs::read(directory.join("selected.json"))?)?;
    validate_model_identity(&metadata)?;
    let model: Value =
        serde_json::from_slice(&fs::read(directory.join(format!("48-{name}-6.json")))?)?;
    let model_layers: Vec<u32> = serde_json::from_value(model["config"]["layers"].clone())?;
    let weights: Vec<f32> = serde_json::from_value(model["weights"].clone())?;
    validate_dense(model_layers, weights, Some(layers))
}
fn validate_model_identity(metadata: &Value) -> Result<()> {
    if metadata["input_size"] != 48
        || metadata["activation"] != "relu"
        || metadata["output"] != "range endpoint residual"
        || !metadata["preprocessing"]["axis"]
            .as_str()
            .is_some_and(|s| s.contains("pca_axis3"))
    {
        return Err(
            "loaded model requires matching PCA RGB8 preprocessing and residual endpoint identity"
                .into(),
        );
    }
    Ok(())
}
fn validate_dense(
    model_layers: Vec<u32>,
    weights: Vec<f32>,
    expected: Option<&[u32]>,
) -> Result<Dense> {
    if model_layers.is_empty()
        || model_layers.last() != Some(&6)
        || model_layers.iter().any(|&n| n == 0 || n > 64)
        || expected.is_some_and(|layers| layers != model_layers)
    {
        return Err("loaded model layers invalid or mismatched".into());
    }
    let mut input = 48usize;
    let mut count = 0;
    for &width in &model_layers {
        count += width as usize * (input + 1);
        input = width as usize;
    }
    if weights.len() != count || !weights.iter().all(|v| v.is_finite()) {
        return Err("loaded model layers, parameter count, or finite weights invalid".into());
    }
    Ok(Dense {
        layers: model_layers,
        weights,
    })
}
fn frozen_model(directory: &PathBuf, sizes: Option<&[Vec<u32>]>) -> Result<(Dense, u32, Value)> {
    let metadata: Value = serde_json::from_slice(&fs::read(directory.join("selected.json"))?)?;
    validate_model_identity(&metadata)?;
    let layers: Vec<u32> = serde_json::from_value(metadata["layers"].clone())?;
    if sizes.is_some_and(|sizes| sizes != [layers.clone()]) {
        return Err("locked test --sizes must match selected.json authoritative layers".into());
    }
    let mode = match metadata["mode"].as_str() {
        Some("raw") => 0,
        Some("range_guard") => 1,
        Some("range_guard_two_ls") => 2,
        _ => return Err("locked test selected.json mode unsupported; supported raw, range_guard, range_guard_two_ls".into()),
    };
    let weights = serde_json::from_value(metadata["weights"].clone())?;
    Ok((validate_dense(layers, weights, None)?, mode, metadata))
}
fn check_test_provenance(metadata: &Value, groups: &[corpus::EvalGroup]) -> Result<()> {
    if let Some(families) = metadata["provenance"]["train_families"].as_array() {
        for group in groups {
            if families
                .iter()
                .any(|family| family.as_str() == Some(group.family.as_str()))
            {
                return Err(format!(
                    "test family {} overlaps model training provenance",
                    group.family
                )
                .into());
            }
        }
    }
    Ok(())
}
fn training_provenance(o: &Options) -> Result<Value> {
    let mut priors = Vec::new();
    if let Some(directory) = &o.load_models {
        let artifact: Value = serde_json::from_slice(&fs::read(directory.join("selected.json"))?)?;
        priors.push(artifact["provenance"].clone());
        for layers in &o.sizes {
            let size = layers[..layers.len() - 1]
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("x");
            let model: Value =
                serde_json::from_slice(&fs::read(directory.join(format!("48-{size}-6.json")))?)?;
            priors.push(model["provenance"].clone());
        }
    }
    let training = o.load_models.is_none() || o.reconstruction_steps > 0;
    let mut families = priors
        .iter()
        .flat_map(|prior| prior["train_families"].as_array().into_iter().flatten())
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if training && o.corpus_manifest.is_some() {
        families.extend(o.train_families.iter().cloned());
    }
    families.sort();
    families.dedup();
    let known_prior = priors
        .iter()
        .all(|prior| prior["status"] == "known" && prior["train_families"].is_array());
    let known = known_prior && (!training || o.corpus_manifest.is_some());
    let mut manifests = Vec::new();
    for manifest in priors
        .iter()
        .flat_map(|prior| prior["training_manifests"].as_array().into_iter().flatten())
    {
        if !manifests.contains(manifest) {
            manifests.push(manifest.clone());
        }
    }
    if training && o.corpus_manifest.is_some() {
        manifests.push(o.corpus_metadata.clone());
    }
    Ok(
        json!({"status":if known {"known"} else {"unknown"},"train_families":families,"training_manifests":manifests,"scope":"declared manifest families; finite sampled corpus, no universal generalization guarantee"}),
    )
}
fn mode_name(mode: u32) -> &'static str {
    match mode {
        0 => "raw",
        1 => "range_guard",
        2 => "range_guard_two_ls",
        _ => "range_two_ls_without_nn",
    }
}
fn quality_gate(q: &Value, reference: &Value) -> (bool, f64) {
    let ratio = q["mean_rgb_distance"].as_f64().unwrap()
        / reference["mean_rgb_distance"].as_f64().unwrap().max(1e-12);
    let mut qualified = ratio <= 1.01;
    for (name, group) in q["groups"].as_object().unwrap() {
        if name.starts_with("synthetic_category_") {
            continue;
        }
        let actual = group["mean_rgb_distance"].as_f64().unwrap();
        let expected = reference["groups"][name]["mean_rgb_distance"]
            .as_f64()
            .unwrap();
        qualified &= actual <= expected * 1.01 + 1e-7;
    }
    (qualified, ratio)
}
fn select_device() -> Result<(Device, Queue, String)> {
    let instance = mev::Instance::load()?;
    let (idx, family) = instance
        .capabilities()
        .devices
        .iter()
        .enumerate()
        .filter(|(_, d)| d.features.contains(mev::Features::BUFFER_WRITE_COMMAND))
        .find_map(|(i, d)| {
            d.families
                .iter()
                .position(|f| f.queue_flags.contains(mev::QueueFlags::COMPUTE))
                .map(|f| (i, f as u32))
        })
        .ok_or("no compute queue")?;
    let name = instance.capabilities().devices[idx].name.clone();
    let (device, queues) = instance.new_device(mev::DeviceDesc {
        idx,
        queues: &[family],
        features: mev::Features::BUFFER_WRITE_COMMAND,
    })?;
    Ok((device, queues.into_iter().next().unwrap(), name))
}
fn refinement_comparison(
    test: &[Texels],
    actual: &[Block],
    start: &[Block],
    ls: &[Block],
) -> Result<Value> {
    let mut wins = 0;
    let mut ties = 0;
    let mut losses = 0;
    let mut maximum_start_increase = 0.0f32;
    for (((t, &actual), &start), &ls) in test.iter().zip(actual).zip(start).zip(ls) {
        let a = error(t, actual);
        let s = error(t, start);
        let l = error(t, ls);
        if !a.is_finite() || a > s + 2e-7 {
            return Err(format!("refinement worsens its quantized start: {a} > {s}").into());
        }
        maximum_start_increase = maximum_start_increase.max(a - s);
        if a < l - 1e-7 {
            wins += 1;
        } else if a > l + 1e-7 {
            losses += 1;
        } else {
            ties += 1;
        }
    }
    Ok(
        json!({"wins_vs_ls2":wins,"ties_vs_ls2":ties,"losses_vs_ls2":losses,"tie_rgb_distance_tolerance":1e-7,"max_rgb_distance_increase_vs_start":maximum_start_increase}),
    )
}
fn gradient_comparison(
    o: &Options,
    device: &Device,
    queue: &mut Queue,
    test: &[Texels],
    gpu_name: &str,
) -> Result<()> {
    let mut encoder = gpu_encode::GpuEncoder::new(device, test)?;
    let mut models = Vec::new();
    for layers in &o.sizes {
        let name = layers[..layers.len() - 1]
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join("x");
        let dense = load_dense(o.load_models.as_ref().unwrap(), layers, &name)?;
        let mut desc = NnDesc::new(&[Encoding::Identity { arity: 48 }], layers);
        desc.activation = Activation::Relu;
        let nn = Nn::new(device, desc, &dense.weights)?;
        let start = encoder.benchmark(queue, Some(&nn), test, o.repeats, 1)?;
        encoder.check_relaxed(queue)?;
        let checked = read(device, queue, &encoder.input, test.len() * 48)?;
        let mut gradient_difference = 0.0f32;
        for i in 0..test.len().min(8) {
            let block = start.blocks[i];
            let (value, ga, gb) = gradient_refine::relaxed(
                &test[i],
                block.color0.into_f32().0,
                block.color1.into_f32().0,
            );
            let expected = [value, ga[0], ga[1], ga[2], gb[0], gb[1], gb[2]];
            for c in 0..7 {
                let gpu = checked[i * 48 + c];
                if !gpu.is_finite() {
                    return Err("GPU relaxed gradient nonfinite".into());
                }
                gradient_difference = gradient_difference.max((gpu - expected[c]).abs());
            }
        }
        if gradient_difference > 2e-5 {
            return Err(format!(
                "GPU relaxed objective/gradient reference mismatch {gradient_difference}"
            )
            .into());
        }
        let ls = encoder.benchmark(queue, Some(&nn), test, o.repeats, 2)?;
        refinement_comparison(test, &ls.blocks, &start.blocks, &ls.blocks)?;
        let mut rows = Vec::new();
        let mut best = (f64::INFINITY, f64::INFINITY);
        let mut best_setting = (4u32, 0.02f32);
        for &steps in &o.gd_steps {
            for &rate in &o.gd_rates {
                println!("GD 48-{name}-6: {steps} iterations, rate {rate}");
                let result = encoder.benchmark_gd(queue, &nn, test, o.repeats, steps, rate)?;
                let comparisons =
                    refinement_comparison(test, &result.blocks, &start.blocks, &ls.blocks)?;
                let q = grouped_quality(o, test, &result.blocks);
                // CPU reference is a numerical check, outside every GPU timing region.
                let reference_count = test.len().min(64);
                let mut reference_difference = 0.0f32;
                for i in 0..reference_count {
                    let cpu = gradient_refine::refine(&test[i], start.blocks[i], steps, rate);
                    let difference =
                        (error(&test[i], cpu) - error(&test[i], result.blocks[i])).abs();
                    reference_difference = reference_difference.max(difference);
                }
                let key = (q["mean_rgb_distance"].as_f64().unwrap(), result.host_ms);
                if key < best {
                    best = key;
                    best_setting = (steps, rate);
                }
                rows.push(json!({"iterations":steps,"learning_rate":rate,"max_continuous_objective_evaluations_per_block_within_refinement_loop":steps*10,"max_quantized_candidate_evaluations_per_block_within_refinement_loop":steps,"quality":q,"resident_submit_wait_ms":result.resident_ms,"host_end_to_end_ms":result.host_ms,"host_time_ratio_to_ls2":result.host_ms/ls.host_ms,"per_block":comparisons,"cpu_reference_blocks":reference_count,"cpu_gpu_max_decoded_error_difference":reference_difference,"cpu_gpu_trajectory_agrees":reference_difference<=2e-4,"cpu_gpu_trajectory_tolerance":2e-4,"trajectory_comparison":"diagnostic only; branch and quantization decisions can diverge; direct objective and gradient checked separately"}));
            }
        }
        let range_start = encoder.benchmark_range_start(queue, test, o.repeats)?;
        let range_ls = encoder.benchmark(queue, None, test, o.repeats, 3)?;
        let range_gd =
            encoder.benchmark_range_gd(queue, test, o.repeats, best_setting.0, best_setting.1)?;
        let range_comparisons = refinement_comparison(
            test,
            &range_gd.blocks,
            &range_start.blocks,
            &range_ls.blocks,
        )?;
        models.push(json!({"model":format!("48-{name}-6"),"direct_gpu_cpu_relaxed_objective_gradient_max_difference":gradient_difference,"direct_gpu_cpu_relaxed_objective_gradient_tolerance":2e-5,"direct_gradient_check_blocks":test.len().min(8),"starting_candidate":{"quality":grouped_quality(o,test,&start.blocks),"resident_submit_wait_ms":start.resident_ms,"host_end_to_end_ms":start.host_ms},"ls2":{"quality":grouped_quality(o,test,&ls.blocks),"resident_submit_wait_ms":ls.resident_ms,"host_end_to_end_ms":ls.host_ms},"gradient_descent":rows,"range_control":{"setting_selection":"lowest NN-start decoded mean RGB distance, time breaks ties","iterations":best_setting.0,"learning_rate":best_setting.1,"start_quality":grouped_quality(o,test,&range_start.blocks),"ls2_quality":grouped_quality(o,test,&range_ls.blocks),"ls2_resident_ms":range_ls.resident_ms,"ls2_host_ms":range_ls.host_ms,"gd_quality":grouped_quality(o,test,&range_gd.blocks),"gd_resident_ms":range_gd.resident_ms,"gd_host_ms":range_gd.host_ms,"per_block":range_comparisons}}));
    }
    let report = json!({"gpu":gpu_name,"seed":o.seed,"validation_blocks":test.len(),"repeats":o.repeats,"loaded_models":o.load_models,"objective":"actual decoded quantized mean Euclidean RGB distance for output selection; continuous smooth nearest-palette RGB norm relaxation for GD gradients","gd_start":"same NN-guarded packed candidate as LS2, dequantized endpoints","gradient":"nearest assignments frozen for each gradient; recomputed at every proposed continuous point","line_search":"projected clamp [0,1], strict relaxed objective decrease, initial trial plus at most eight halvings (nine trials); not Armijo","quantization":"continuous trajectory retained across RGB565 plateaus; best quantized candidate retained using actual decoded distance","ls2":"two fixed-assignment RGB squared-error least-squares solves, accepted only if actual decoded RGB norm decreases","budget":"evaluation bounds cover refinement loop only; common guard/start costs are outside these bounds; fixed iteration and backtracking limits imply no convergence or local-minimum guarantee","timing":"same resident packed-RGB to packed-BC1 and full host upload/readback boundaries for all methods; CPU reference excluded","models":models});
    fs::write(
        o.output.join("gradient-comparison.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "Gradient comparison saved {}",
        o.output.join("gradient-comparison.json").display()
    );
    Ok(())
}
fn neighbor_comparison(
    o: &Options,
    device: &Device,
    queue: &mut Queue,
    test: &[Texels],
    gpu_name: &str,
) -> Result<()> {
    let mut encoder = gpu_encode::GpuEncoder::new(device, test)?;
    let mut models = Vec::new();
    let check_opaque = |blocks: &[Block]| -> Result<()> {
        if blocks.iter().any(|b| {
            b.color0.bits() < b.color1.bits()
                || b.decode_with_alpha().iter().flatten().any(|c| c.a() != 1.0)
        }) {
            return Err("neighbor comparison produced unordered or transparent BC1 output".into());
        }
        Ok(())
    };
    for layers in &o.sizes {
        let name = layers[..layers.len() - 1]
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join("x");
        let dense = load_dense(o.load_models.as_ref().unwrap(), layers, &name)?;
        let mut desc = NnDesc::new(&[Encoding::Identity { arity: 48 }], layers);
        desc.activation = Activation::Relu;
        let nn = Nn::new(device, desc, &dense.weights)?;
        let start = encoder.benchmark(queue, Some(&nn), test, o.repeats, 1)?;
        let ls = encoder.benchmark(queue, Some(&nn), test, o.repeats, 2)?;
        check_opaque(&ls.blocks)?;
        refinement_comparison(test, &ls.blocks, &start.blocks, &ls.blocks)?;
        let mut rows = Vec::new();
        for reuse in [false, true] {
            let proposals = ls
                .blocks
                .iter()
                .map(|&b| neighborhood::proposal_count(b, reuse))
                .collect::<Vec<_>>();
            let pool = json!({"minimum":proposals.iter().min(),"maximum":proposals.iter().max(),"mean":proposals.iter().sum::<usize>() as f64/proposals.len() as f64,"total":proposals.iter().sum::<usize>(),"count_semantics":"valid endpoint proposals, not unique packed BC1 blocks; both endpoint pools use the same LS2 seed"});
            for rounds in [1, 2] {
                println!("Neighbors 48-{name}-6: {rounds} rounds, reuse indices {reuse}");
                let result =
                    encoder.benchmark_neighbors(queue, &nn, test, o.repeats, rounds, reuse)?;
                check_opaque(&result.blocks)?;
                let comparisons =
                    refinement_comparison(test, &result.blocks, &ls.blocks, &ls.blocks)?;
                // Numerical reference checks stay outside GPU timing regions.
                let reference_count = test.len().min(256);
                let mut reference_difference = 0.0f32;
                let mut different_blocks = 0;
                for i in 0..reference_count {
                    let cpu = neighborhood::refine(&test[i], ls.blocks[i], rounds, reuse);
                    reference_difference = reference_difference
                        .max((error(&test[i], cpu) - error(&test[i], result.blocks[i])).abs());
                    different_blocks += usize::from(cpu != result.blocks[i]);
                }
                rows.push(json!({"rounds":rounds,"index_policy":if reuse {"reuse"} else {"rebuild"},"max_endpoint_proposals_per_block_per_round":52,"max_endpoint_proposals_per_block":52*rounds,"initial_valid_pool_proposal_counts":pool,"quality":grouped_quality(o,test,&result.blocks),"resident_submit_wait_ms":result.resident_ms,"host_end_to_end_ms":result.host_ms,"host_time_ratio_to_ls2":result.host_ms/ls.host_ms,"per_block":comparisons,"all_outputs_opaque":true,"cpu_reference_blocks":reference_count,"cpu_gpu_different_packed_blocks":different_blocks,"cpu_gpu_max_decoded_error_difference":reference_difference,"cpu_gpu_error_tolerance":2e-6,"cpu_gpu_error_agrees":reference_difference<=2e-6}));
            }
        }
        let gd = encoder.benchmark_gd(queue, &nn, test, o.repeats, 4, 0.5)?;
        check_opaque(&gd.blocks)?;
        let gd_comparisons = refinement_comparison(test, &gd.blocks, &start.blocks, &ls.blocks)?;
        let mut measure = |method: usize| -> Result<gpu_encode::EncodeResult> {
            match method {
                0 => encoder.benchmark(queue, Some(&nn), test, 1, 2),
                1..=4 => encoder.benchmark_neighbors(
                    queue,
                    &nn,
                    test,
                    1,
                    if method % 2 == 1 { 1 } else { 2 },
                    method >= 3,
                ),
                _ => encoder.benchmark_gd(queue, &nn, test, 1, 4, 0.5),
            }
        };
        let warmup = Instant::now();
        let mut warmup_calls = 0;
        while warmup.elapsed().as_millis() < 500 {
            black_box(measure(warmup_calls % 6)?);
            warmup_calls += 1;
        }
        let warmup_ms = warmup.elapsed().as_secs_f64() * 1000.0;
        let mut resident_samples: [Vec<f64>; 6] =
            std::array::from_fn(|_| Vec::with_capacity(o.repeats));
        let mut host_samples: [Vec<f64>; 6] =
            std::array::from_fn(|_| Vec::with_capacity(o.repeats));
        for round in 0..o.repeats {
            for offset in 0..6 {
                let method = (round + offset) % 6;
                let result = measure(method)?;
                resident_samples[method].push(result.resident_ms);
                host_samples[method].push(result.host_ms);
            }
        }
        let resident_medians = resident_samples
            .each_ref()
            .map(|samples| median(samples.clone()));
        let host_medians = host_samples
            .each_ref()
            .map(|samples| median(samples.clone()));
        for (index, row) in rows.iter_mut().enumerate() {
            row["resident_submit_wait_ms"] = json!(resident_medians[index + 1]);
            row["host_end_to_end_ms"] = json!(host_medians[index + 1]);
            row["host_time_ratio_to_ls2"] = json!(host_medians[index + 1] / host_medians[0]);
        }
        models.push(json!({"model":format!("48-{name}-6"),"timing_measurements":{"warmup_minimum_ms":500,"warmup_actual_ms":warmup_ms,"warmup_calls":warmup_calls,"order":"interleaved; starting method rotates by one each round","rounds":o.repeats,"method_order":["ls2","rebuild1","rebuild2","reuse1","reuse2","gd_only4_rate0.5"],"resident_submit_wait_samples_ms":resident_samples,"host_end_to_end_samples_ms":host_samples,"aggregation":"per-method median across rounds; each method call includes one untimed warmup before one resident and one full-host sample"},"ls2":{"quality":grouped_quality(o,test,&ls.blocks),"resident_submit_wait_ms":resident_medians[0],"host_end_to_end_ms":host_medians[0],"all_outputs_opaque":true},"neighbors":rows,"gd_only":{"iterations":4,"learning_rate":0.5,"start":"NN guarded candidate without LS","quality":grouped_quality(o,test,&gd.blocks),"resident_submit_wait_ms":resident_medians[5],"host_end_to_end_ms":host_medians[5],"per_block":gd_comparisons}}));
    }
    let report = json!({"gpu":gpu_name,"seed":o.seed,"validation_blocks":test.len(),"repeats":o.repeats,"loaded_models":o.load_models,"objective":"actual decoded quantized mean Euclidean RGB distance","ls2":"NN guarded packed candidate followed by two fixed-assignment RGB squared-error least-squares solves, accepted only on decoded RGB distance improvement","neighborhood":"each endpoint independently explores nonzero RGB565 channel offsets in {-1,0,1} cubed; invalid channels rejected; both pools centered on fixed round seed before committing strict best improvement; stop on unchanged winner","rebuild_indices":"nearest palette assignment for every proposal and its output","reuse_indices":"round seed assignments retained during proposal scoring and output; endpoint swaps remap indices with XOR 0x55555555; collapsed endpoints rejected when index 3 would introduce transparency","budget":"at most 26 proposals per endpoint, 52 per round, 104 for two rounds; proposal counts may include duplicate packed blocks; common NN guard and LS costs excluded from proposal bounds","timing":"same resident packed-RGB to packed-BC1 and full host upload/readback boundaries for every method; CPU reference excluded","non_worsening_tolerance":2e-7,"models":models});
    let path = o.output.join("neighbor-comparison.json");
    fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!("Neighbor comparison saved {}", path.display());
    Ok(())
}
fn timing_distribution(samples: &[f64]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
    } else {
        sorted[n / 2]
    };
    json!({"median_ms":median,"min_ms":sorted[0],"p10_ms":sorted[(n-1)/10],"p90_ms":sorted[(n-1)*9/10],"max_ms":sorted[n-1],"samples_ms":samples})
}
fn block_mse(texels: &Texels, block: Block) -> f64 {
    let decoded = block.decode();
    (0..16)
        .map(|i| {
            (0..3)
                .map(|c| {
                    let difference =
                        texels[i][c] as f64 / 255.0 - decoded[i / 4][i % 4].0[c] as f64;
                    difference * difference
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / 48.0
}
fn packed_comparison(test: &[Texels], actual: &[Block], reference: &[Block]) -> Value {
    let (
        mut rgb_wins,
        mut rgb_losses,
        mut rgb_ties,
        mut mse_wins,
        mut mse_losses,
        mut mse_ties,
        mut changed,
    ) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut maximum_rgb_increase = 0.0f32;
    for ((texels, &actual), &reference) in test.iter().zip(actual).zip(reference) {
        let difference = error(texels, actual) - error(texels, reference);
        maximum_rgb_increase = maximum_rgb_increase.max(difference);
        if difference < -1e-7 {
            rgb_wins += 1;
        } else if difference > 1e-7 {
            rgb_losses += 1;
        } else {
            rgb_ties += 1;
        }
        let mse_difference = block_mse(texels, actual) - block_mse(texels, reference);
        if mse_difference < -1e-12 {
            mse_wins += 1;
        } else if mse_difference > 1e-12 {
            mse_losses += 1;
        } else {
            mse_ties += 1;
        }
        changed += usize::from(actual != reference);
    }
    json!({"changed_packed_blocks":changed,"rgb_distance_wins":rgb_wins,"rgb_distance_losses":rgb_losses,"rgb_distance_ties":rgb_ties,"rgb_distance_tie_tolerance":1e-7,"max_rgb_distance_increase":maximum_rgb_increase,"mse_wins":mse_wins,"mse_losses":mse_losses,"mse_ties":mse_ties,"mse_tie_tolerance":1e-12})
}
fn check_opaque_blocks(blocks: &[Block]) -> Result<()> {
    if blocks.iter().any(|b| {
        b.color0.bits() < b.color1.bits()
            || b.decode_with_alpha().iter().flatten().any(|c| c.a() != 1.0)
    }) {
        return Err(
            "pipeline neighborhood comparison produced unordered or transparent output".into(),
        );
    }
    Ok(())
}
fn pipeline_neighbor_method(
    encoder: &mut gpu_encode::GpuEncoder,
    queue: &mut Queue,
    nn: &Nn,
    test: &[Texels],
    method: usize,
) -> Result<gpu_encode::EncodeResult> {
    match method {
        0 => encoder.benchmark(queue, Some(nn), test, 1, 2),
        1 => encoder.benchmark(queue, None, test, 1, 11),
        2..=5 => encoder.benchmark_neighbors(
            queue,
            nn,
            test,
            1,
            if method % 2 == 0 { 1 } else { 2 },
            method >= 4,
        ),
        6..=9 => encoder.benchmark_principal_neighbors(
            queue,
            test,
            1,
            if method % 2 == 0 { 1 } else { 2 },
            method >= 8,
        ),
        _ => Err("invalid pipeline neighbor comparison method".into()),
    }
}
fn pipeline_neighbor_compare(o: &Options, test: &[Texels]) -> Result<()> {
    let (dense, _, metadata) = frozen_model(
        o.load_models.as_ref().unwrap(),
        o.sizes_explicit.then_some(o.sizes.as_slice()),
    )?;
    check_test_provenance(&metadata, &o.corpus_groups)?;
    if o.output.exists() && fs::read_dir(&o.output)?.next().is_some() {
        return Err("pipeline neighbor comparison requires a fresh empty output directory".into());
    }
    let (device, mut queue, gpu_name) = select_device()?;
    let mut desc = NnDesc::new(&[Encoding::Identity { arity: 48 }], &dense.layers);
    desc.activation = Activation::Relu;
    let nn = Nn::new(&device, desc, &dense.weights)?;
    let names = [
        "nn_two_ls",
        "principal_four_ls",
        "nn_rebuild_1",
        "nn_rebuild_2",
        "nn_reuse_1",
        "nn_reuse_2",
        "principal_rebuild_1",
        "principal_rebuild_2",
        "principal_reuse_1",
        "principal_reuse_2",
    ];
    let smoke_texels = &test[..test.len().min(65)];
    let mut smoke_encoder = gpu_encode::GpuEncoder::new(&device, smoke_texels)?;
    let mut smoke_outputs: Vec<Vec<Block>> = Vec::new();
    let mut smoke_checks = Vec::new();
    for method in 0..10 {
        let result =
            pipeline_neighbor_method(&mut smoke_encoder, &mut queue, &nn, smoke_texels, method)?;
        check_opaque_blocks(&result.blocks)?;
        let seed_method = if method == 1 || method >= 6 { 1 } else { 0 };
        let mut maximum_difference = 0.0f32;
        let mut different_blocks = 0;
        for (i, (texels, &actual)) in smoke_texels.iter().zip(&result.blocks).enumerate() {
            let reference = if method <= 1 {
                let input = canonical(texels);
                if method == 0 {
                    guarded(texels, &input, &dense.forward(&input), true)
                } else {
                    refine_passes(texels, pack(texels, baseline(&input)), 4)
                }
            } else {
                neighborhood::refine(
                    texels,
                    smoke_outputs[seed_method][i],
                    if method % 2 == 0 { 1 } else { 2 },
                    method == 4 || method == 5 || method >= 8,
                )
            };
            let difference = (error(texels, actual) - error(texels, reference)).abs();
            maximum_difference = maximum_difference.max(difference);
            different_blocks += usize::from(actual != reference);
            let tolerance = if method <= 1 { 2e-5 } else { 2e-6 };
            if difference > tolerance {
                return Err(format!(
                    "{} GPU/CPU neighborhood check failed: {difference}",
                    names[method]
                )
                .into());
            }
            if method >= 2
                && error(texels, actual) > error(texels, smoke_outputs[seed_method][i]) + 2e-7
            {
                return Err("smoke neighborhood output worsens LS seed".into());
            }
        }
        smoke_checks.push(json!({"method":names[method],"blocks":smoke_texels.len(),"cpu_gpu_max_rgb_distance_difference":maximum_difference,"cpu_gpu_different_packed_blocks":different_blocks,"reference":"CPU neighborhood::refine applied to matching GPU LS seed; LS controls independently checked against CPU preprocessing/NN/LS"}));
        smoke_outputs.push(result.blocks);
    }
    let mut encoder = gpu_encode::GpuEncoder::new(&device, test)?;
    let mut resident = vec![Vec::new(); 10];
    let mut host = vec![Vec::new(); 10];
    let mut outputs = vec![Vec::new(); 10];
    let warmup = Instant::now();
    let mut warmup_cycles = 0;
    while warmup.elapsed().as_millis() < 500 || warmup_cycles < 3 {
        for method in 0..10 {
            pipeline_neighbor_method(&mut encoder, &mut queue, &nn, test, method)?;
        }
        warmup_cycles += 1;
    }
    let warmup_ms = warmup.elapsed().as_secs_f64() * 1000.0;
    for round in 0..o.repeats {
        for offset in 0..10 {
            let method = (round + offset) % 10;
            let result = pipeline_neighbor_method(&mut encoder, &mut queue, &nn, test, method)?;
            resident[method].push(result.resident_ms);
            host[method].push(result.host_ms);
            outputs[method] = result.blocks;
        }
        println!(
            "Pipeline neighbors timing round {}/{}",
            round + 1,
            o.repeats
        );
    }
    let qualities: Vec<Value> = outputs
        .iter()
        .map(|b| grouped_quality(o, test, b))
        .collect();
    let mut methods = serde_json::Map::new();
    for method in 0..10 {
        check_opaque_blocks(&outputs[method])?;
        let seed_method = if method == 1 || method >= 6 { 1 } else { 0 };
        let comparison = packed_comparison(test, &outputs[method], &outputs[seed_method]);
        if comparison["max_rgb_distance_increase"].as_f64().unwrap() > 2e-7 {
            return Err(format!("{} full output worsens LS seed", names[method]).into());
        }
        let mut deltas = serde_json::Map::new();
        for group in &o.corpus_groups {
            deltas.insert(
                group.name.clone(),
                json!(
                    qualities[method]["groups"][&group.name]["rgb_psnr_db"]
                        .as_f64()
                        .unwrap()
                        - qualities[seed_method]["groups"][&group.name]["rgb_psnr_db"]
                            .as_f64()
                            .unwrap()
                ),
            );
        }
        methods.insert(names[method].into(),json!({"pipeline":if seed_method==0 {"NN range guard + up to 2 LS"} else {"PCA extreme texels + up to 4 LS, no NN"},"neighbor_rounds":if method<=1 {0} else if method%2==0 {1} else {2},"index_policy":if method<=1 {"LS control"} else if method==4 || method==5 || method>=8 {"reuse"} else {"rebuild"},"quality":qualities[method],"ls_control":names[seed_method],"aggregate_psnr_delta_vs_ls_db":qualities[method]["rgb_psnr_db"].as_f64().unwrap()-qualities[seed_method]["rgb_psnr_db"].as_f64().unwrap(),"per_texture_psnr_delta_vs_ls_db":deltas,"per_block_vs_ls":comparison,"all_outputs_opaque":true,"resident_submit_wait":timing_distribution(&resident[method]),"host_end_to_end":timing_distribution(&host[method])}));
    }
    let mut pairs = Vec::new();
    for (nn_method, pca_method) in [(0, 1), (2, 6), (3, 7), (4, 8), (5, 9)] {
        let mut deltas = serde_json::Map::new();
        let mut values = Vec::new();
        for group in &o.corpus_groups {
            let delta = qualities[pca_method]["groups"][&group.name]["rgb_psnr_db"]
                .as_f64()
                .unwrap()
                - qualities[nn_method]["groups"][&group.name]["rgb_psnr_db"]
                    .as_f64()
                    .unwrap();
            values.push(delta);
            deltas.insert(group.name.clone(), json!(delta));
        }
        values.sort_by(f64::total_cmp);
        let n = values.len();
        pairs.push(json!({"nn_method":names[nn_method],"principal_method":names[pca_method],"aggregate_principal_minus_nn_psnr_db":qualities[pca_method]["rgb_psnr_db"].as_f64().unwrap()-qualities[nn_method]["rgb_psnr_db"].as_f64().unwrap(),"per_texture_principal_minus_nn_psnr_db":deltas,"texture_delta_distribution":if n==0 {Value::Null} else {json!({"min_db":values[0],"p10_db":values[(n-1)/10],"median_db":values[n/2],"p90_db":values[(n-1)*9/10],"max_db":values[n-1],"principal_wins":values.iter().filter(|&&v|v>0.0).count(),"principal_losses":values.iter().filter(|&&v|v<0.0).count()})},"per_block_principal_vs_nn":packed_comparison(test,&outputs[pca_method],&outputs[nn_method])}));
    }
    let report = json!({"gpu":gpu_name,"seed":o.seed,"split":"test","exploratory_post_test_refinement_comparison":true,"independence_note":"user-authorized neighborhood exploration on previously evaluated held-out textures; no untouched-final-test selection claim","sampled_blocks":test.len(),"corpus":o.corpus_metadata,"model_source":o.load_models,"model_layers":dense.layers,"model_provenance":metadata.get("provenance").cloned().unwrap_or(json!({"status":"unknown"})),"methods":methods,"paired_comparisons":pairs,"gpu_cpu_smoke":smoke_checks,"neighborhood":"26 nonzero RGB565 offsets per endpoint maximum; both pools centered on fixed round seed, at most 52 proposals per round; no joint endpoint combinations, invalid channels rejected; retain strict decoded RGB norm improvement; stop unchanged winner","index_policy":"rebuild: nearest assignments for every proposal/output; reuse: seed assignments retained in scoring/output, endpoint swaps XOR 0x55555555; collapsed endpoints with transparent index3 rejected","gpu_parallelism":"blocks parallel; all neighbor proposals sequential within each GPU invocation, no candidate-lane parallelism","quality_metric":"source RGB8 PSNR from aggregate sampled normalized-channel MSE, not average texture dB; decoded RGB norm acceptance does not guarantee per-block PSNR improvement","timing":"same full RGB->BC1 resident and host boundaries for every method; CPU wall clock includes recording/submit/wait, host also packing/upload/readback. Allocation/shader compilation excluded, untimed per-call warmup. No GPU timestamps; clock/CPU variation remains.","timing_order":"10 methods interleaved each round, starting method rotates","method_order":names,"warmup_ms":warmup_ms,"warmup_cycles":warmup_cycles,"repeats":o.repeats,"training_steps":0,"reconstruction_steps":0,"selection_performed":false});
    fs::create_dir_all(&o.output)?;
    let path = o.output.join("pipeline-neighbor-report.json");
    fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!("Pipeline neighbor comparison saved {}", path.display());
    Ok(())
}
fn principal_ls_sweep(o: &Options, test: &[Texels]) -> Result<()> {
    let (dense, _, metadata) = frozen_model(
        o.load_models.as_ref().unwrap(),
        o.sizes_explicit.then_some(o.sizes.as_slice()),
    )?;
    check_test_provenance(&metadata, &o.corpus_groups)?;
    if o.output.exists() && fs::read_dir(&o.output)?.next().is_some() {
        return Err("LS sweep requires a fresh empty output directory".into());
    }
    let (device, mut queue, gpu_name) = select_device()?;
    let mut desc = NnDesc::new(&[Encoding::Identity { arity: 48 }], &dense.layers);
    desc.activation = Activation::Relu;
    let nn = Nn::new(&device, desc, &dense.weights)?;
    let smoke_texels = &test[..test.len().min(65)];
    let mut smoke_encoder = gpu_encode::GpuEncoder::new(&device, smoke_texels)?;
    let mut smoke_checks = Vec::new();
    for &passes in &o.principal_ls_passes {
        let result = smoke_encoder.benchmark_principal_ls(&mut queue, smoke_texels, 1, passes)?;
        let mut max_difference = 0.0f32;
        for (texels, &block) in smoke_texels.iter().zip(&result.blocks) {
            let initial = pack(texels, baseline(&canonical(texels)));
            let reference = refine_passes(texels, initial, passes as usize);
            let difference = (error(texels, reference) - error(texels, block)).abs();
            max_difference = max_difference.max(difference);
            if difference > 2e-5 || error(texels, block) > error(texels, initial) + 2e-7 {
                return Err(format!("principal LS {passes}-pass GPU/CPU smoke failed").into());
            }
            if block.color0 == block.color1 && block.indices.iter().any(|&i| i != 0) {
                return Err("principal LS sweep produced nonopaque collapsed block".into());
            }
        }
        smoke_checks.push(json!({"passes":passes,"blocks":smoke_texels.len(),"gpu_cpu_max_rgb_distance_difference":max_difference}));
    }
    let mut encoder = gpu_encode::GpuEncoder::new(&device, test)?;
    let method_count = o.principal_ls_passes.len() + 2;
    let mut names = vec![
        "nn_two_ls".to_owned(),
        "principal_four_ls_literal".to_owned(),
    ];
    names.extend(
        o.principal_ls_passes
            .iter()
            .map(|p| format!("principal_ls_{p}")),
    );
    let mut resident = vec![Vec::new(); method_count];
    let mut host = vec![Vec::new(); method_count];
    let mut outputs = vec![Vec::new(); method_count];
    let mut run = |method: usize| -> Result<gpu_encode::EncodeResult> {
        match method {
            0 => encoder.benchmark(&mut queue, Some(&nn), test, 1, 2),
            1 => encoder.benchmark(&mut queue, None, test, 1, 11),
            _ => encoder.benchmark_principal_ls(
                &mut queue,
                test,
                1,
                o.principal_ls_passes[method - 2],
            ),
        }
    };
    let warmup = Instant::now();
    let mut warmup_cycles = 0;
    while warmup.elapsed().as_millis() < 500 || warmup_cycles < 3 {
        for method in 0..method_count {
            run(method)?;
        }
        warmup_cycles += 1;
    }
    let warmup_ms = warmup.elapsed().as_secs_f64() * 1000.0;
    for round in 0..o.repeats {
        for offset in 0..method_count {
            let method = (round + offset) % method_count;
            let r = run(method)?;
            resident[method].push(r.resident_ms);
            host[method].push(r.host_ms);
            outputs[method] = r.blocks;
        }
    }
    let nn_quality = grouped_quality(o, test, &outputs[0]);
    let mut methods = serde_json::Map::new();
    for method in 0..method_count {
        let quality = if method == 0 {
            nn_quality.clone()
        } else {
            grouped_quality(o, test, &outputs[method])
        };
        let mut deltas = serde_json::Map::new();
        let mut delta_values = Vec::new();
        for group in &o.corpus_groups {
            let delta = quality["groups"][&group.name]["rgb_psnr_db"]
                .as_f64()
                .unwrap()
                - nn_quality["groups"][&group.name]["rgb_psnr_db"]
                    .as_f64()
                    .unwrap();
            deltas.insert(group.name.clone(), json!(delta));
            delta_values.push(delta);
        }
        delta_values.sort_by(f64::total_cmp);
        let n = delta_values.len();
        let distribution = if n == 0 {
            Value::Null
        } else {
            json!({"min_db":delta_values[0],"p10_db":delta_values[(n-1)/10],"median_db":delta_values[n/2],"p90_db":delta_values[(n-1)*9/10],"max_db":delta_values[n-1],"textures_below_nn":delta_values.iter().filter(|&&x|x<0.0).count()})
        };
        let previous = if method <= 2 { 1 } else { method - 1 };
        let different_from_literal_four = outputs[method]
            .iter()
            .zip(&outputs[1])
            .filter(|(a, b)| a.bytes() != b.bytes())
            .count();
        let different_from_previous = outputs[method]
            .iter()
            .zip(&outputs[previous])
            .filter(|(a, b)| a.bytes() != b.bytes())
            .count();
        let (mut improved, mut worsened) = (0usize, 0usize);
        if method >= 2 {
            for ((texels, &current), &prior) in
                test.iter().zip(&outputs[method]).zip(&outputs[previous])
            {
                let difference = error(texels, current) - error(texels, prior);
                if difference < -2e-7 {
                    improved += 1;
                }
                if difference > 2e-7 {
                    worsened += 1;
                }
            }
        }
        methods.insert(names[method].clone(),json!({"max_ls_passes":if method==0 {2} else if method==1 {4} else {o.principal_ls_passes[method-2]},"budget_form":if method<=1 {"literal"} else {"uniform"},"quality":quality,"aggregate_psnr_delta_vs_nn_db":quality["rgb_psnr_db"].as_f64().unwrap()-nn_quality["rgb_psnr_db"].as_f64().unwrap(),"per_texture_psnr_delta_vs_nn_db":deltas,"texture_psnr_delta_distribution":distribution,"resident_submit_wait":timing_distribution(&resident[method]),"host_end_to_end":timing_distribution(&host[method]),"different_packed_blocks_vs_literal_four":different_from_literal_four,"previous_method":names[previous],"different_packed_blocks_vs_previous":different_from_previous,"rgb_distance_improved_vs_previous_blocks":improved,"rgb_distance_worsened_vs_previous_blocks":worsened}));
    }
    let report = json!({"gpu":gpu_name,"seed":o.seed,"split":"test","exploratory_post_test_budget_sweep":true,"independence_note":"user-authorized refinement-budget exploration on previously evaluated held-out textures; this is not untouched final-test model or policy selection","sampled_blocks":test.len(),"corpus":o.corpus_metadata,"model_source":o.load_models,"model_layers":dense.layers,"model_provenance":metadata.get("provenance").cloned().unwrap_or(json!({"status":"unknown"})),"max_ls_passes":o.principal_ls_passes,"methods":methods,"gpu_cpu_smoke":smoke_checks,"quality_metric":"PSNR against original RGB8 source; aggregate PSNR from all sampled normalized RGB channel MSE, not average texture dB","policy":"PCA projection extreme texel endpoints; quantized fixed-assignment squared-error LS, nearest indices rebuilt each proposal, strict decoded RGB norm acceptance and early termination on no improvement or singular fit. Passes are maximum budgets; no forced iterations.","uniform_budget_control":"principal_four_ls_literal vs principal_ls_4 compare current literal-four branch with uniform-four budget; shader compilation changes may affect absolute timings versus earlier experiment","timing":"CPU wall clock command recording, submit/wait; resident RGB-to-BC1 GPU buffers, host includes packing/upload/readback. Resource allocation and shader compilation excluded; untimed per-call warmup. No GPU timestamps; clock and CPU variation remain.","timing_order":"interleaved methods, starting method rotates each round","warmup_ms":warmup_ms,"warmup_cycles":warmup_cycles,"repeats":o.repeats,"training_steps":0,"reconstruction_steps":0,"selection_performed":false});
    fs::create_dir_all(&o.output)?;
    let path = o.output.join("principal-ls-sweep-report.json");
    fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!("Principal LS sweep saved {}", path.display());
    Ok(())
}
fn three_mode_compare(o: &Options, test: &[Texels]) -> Result<()> {
    let (dense, _, metadata) = frozen_model(
        o.load_models.as_ref().unwrap(),
        o.sizes_explicit.then_some(o.sizes.as_slice()),
    )?;
    check_test_provenance(&metadata, &o.corpus_groups)?;
    if o.output.exists() && fs::read_dir(&o.output)?.next().is_some() {
        return Err("locked test requires a fresh empty output directory".into());
    }
    let (device, mut queue, gpu_name) = select_device()?;
    let mut desc = NnDesc::new(&[Encoding::Identity { arity: 48 }], &dense.layers);
    desc.activation = Activation::Relu;
    let nn = Nn::new(&device, desc, &dense.weights)?;
    let mut search = gpu_search::PreparedSearch::new(&device, test)?;
    let mut encoder = gpu_encode::GpuEncoder::new(&device, test)?;
    let smoke_texels = &test[..test.len().min(65)];
    let mut smoke_encoder = gpu_encode::GpuEncoder::new(&device, smoke_texels)?;
    let smoke = smoke_encoder.benchmark(&mut queue, None, smoke_texels, 1, 11)?;
    let mut principal_axis_cpu_gpu_max_error_difference = 0.0f32;
    for (texels, block) in smoke_texels.iter().zip(&smoke.blocks) {
        let initial = pack(texels, baseline(&canonical(texels)));
        let reference = refine_passes(texels, initial, 4);
        let difference = (error(texels, reference) - error(texels, *block)).abs();
        principal_axis_cpu_gpu_max_error_difference =
            principal_axis_cpu_gpu_max_error_difference.max(difference);
        if difference > 2e-5 || error(texels, *block) > error(texels, initial) + 2e-7 {
            return Err("principal axis four-LS GPU/CPU smoke check failed".into());
        }
        if block.color0 == block.color1 && block.indices.iter().any(|&i| i != 0) {
            return Err("principal axis equal-endpoint output must use opaque index zero".into());
        }
    }
    let mut resident: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    let mut host: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    let mut outputs: [Vec<Block>; 3] = std::array::from_fn(|_| Vec::new());
    let warmup = Instant::now();
    let mut warmup_cycles = 0;
    while warmup.elapsed().as_millis() < 500 || warmup_cycles < 3 {
        search.benchmark(&mut queue, test, 1)?;
        encoder.benchmark(&mut queue, Some(&nn), test, 1, 2)?;
        encoder.benchmark(&mut queue, None, test, 1, 11)?;
        warmup_cycles += 1;
    }
    let warmup_ms = warmup.elapsed().as_secs_f64() * 1000.0;
    for round in 0..o.repeats {
        for offset in 0..3 {
            let method = (round + offset) % 3;
            if method == 0 {
                let r = search.benchmark(&mut queue, test, 1)?;
                resident[method].push(r.resident_ms);
                host[method].push(r.end_to_end_ms);
                outputs[method] = r.blocks;
            } else {
                let r = encoder.benchmark(
                    &mut queue,
                    if method == 1 { Some(&nn) } else { None },
                    test,
                    1,
                    if method == 1 { 2 } else { 11 },
                )?;
                resident[method].push(r.resident_ms);
                host[method].push(r.host_ms);
                outputs[method] = r.blocks;
            }
        }
    }
    let qualities: [Value; 3] = std::array::from_fn(|i| grouped_quality(o, test, &outputs[i]));
    let names = ["gpu_cluster_fit", "nn_two_ls", "principal_axis_four_ls"];
    let mut modes = serde_json::Map::new();
    for i in 0..3 {
        let mut deltas = serde_json::Map::new();
        let mut values = Vec::new();
        for group in &o.corpus_groups {
            let delta = qualities[i]["groups"][&group.name]["rgb_psnr_db"]
                .as_f64()
                .unwrap()
                - qualities[0]["groups"][&group.name]["rgb_psnr_db"]
                    .as_f64()
                    .unwrap();
            deltas.insert(group.name.clone(), json!(delta));
            values.push(delta);
        }
        values.sort_by(f64::total_cmp);
        let distribution = if values.is_empty() {
            Value::Null
        } else {
            let n = values.len();
            json!({"min_db":values[0],"p10_db":values[(n-1)/10],"median_db":values[n/2],"p90_db":values[(n-1)*9/10],"max_db":values[n-1],"textures_below_cluster":values.iter().filter(|&&x|x<0.0).count()})
        };
        modes.insert(names[i].into(),json!({"quality":qualities[i],"aggregate_psnr_delta_vs_cluster_db":qualities[i]["rgb_psnr_db"].as_f64().unwrap()-qualities[0]["rgb_psnr_db"].as_f64().unwrap(),"per_texture_psnr_delta_vs_cluster_db":deltas,"texture_psnr_delta_distribution":distribution,"resident_submit_wait":timing_distribution(&resident[i]),"host_end_to_end":timing_distribution(&host[i])}));
    }
    let provenance = metadata
        .get("provenance")
        .cloned()
        .unwrap_or(json!({"status":"unknown"}));
    let report = json!({"gpu":gpu_name,"seed":o.seed,"split":"test","sampled_blocks":test.len(),"corpus":o.corpus_metadata,"model_source":o.load_models,"model_layers":dense.layers,"model_provenance":provenance,"modes":modes,"quality_metric":"PSNR against original RGB8 source; normalized RGB squared error averaged over all sampled channels, then -10log10(MSE); aggregate is block weighted, not average of texture dB","evaluation_scope":"bounded sampled blocks per texture; whole declared families held out","cluster_fit":"unmodified production WGSL, 120 partition iterations / 64 lanes for opaque 16-texel blocks, at most 1024 blocks per submission","nn_two_ls":"NN endpoint prediction with PCA extreme-texel range guard; up to two quantized fixed-assignment squared-error LS passes; nearest indices rebuilt after each; accept only decoded mean RGB norm improvement","principal_axis_four_ls":"PCA projection extreme texel RGB endpoints; no NN; up to four LS passes with same acceptance and reassignment policy","timing":"CPU wall clock including command recording, submit/wait; resident input and output GPU buffers; host includes RGB packing/upload and BC1 readback. Allocations and shader compilation excluded. Each method call has untimed warmup. No GPU timestamps; CPU overhead and GPU clock variation remain.","timing_order":"methods interleaved each round; starting method rotates","warmup_ms":warmup_ms,"warmup_cycles":warmup_cycles,"principal_axis_gpu_cpu_smoke_blocks":smoke_texels.len(),"principal_axis_gpu_cpu_max_rgb_distance_difference":principal_axis_cpu_gpu_max_error_difference,"repeats":o.repeats,"training_steps":0,"reconstruction_steps":0,"selection_performed":false});
    fs::create_dir_all(&o.output)?;
    let path = o.output.join("three-mode-report.json");
    fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!("Three-mode comparison saved {}", path.display());
    Ok(())
}
fn locked_test(o: &Options, test: &[Texels]) -> Result<()> {
    let (dense, mode, metadata) = frozen_model(
        o.load_models.as_ref().unwrap(),
        o.sizes_explicit.then_some(o.sizes.as_slice()),
    )?;
    check_test_provenance(&metadata, &o.corpus_groups)?;
    if o.output.exists() && fs::read_dir(&o.output)?.next().is_some() {
        return Err("locked test requires a fresh empty output directory".into());
    }
    let (device, mut queue, gpu_name) = select_device()?;
    let mut desc = NnDesc::new(&[Encoding::Identity { arity: 48 }], &dense.layers);
    desc.activation = Activation::Relu;
    let nn = Nn::new(&device, desc, &dense.weights)?;
    let search = gpu_search::run(&device, &mut queue, test, o.repeats)?;
    let mut encoder = gpu_encode::GpuEncoder::new(&device, test)?;
    let result = encoder.benchmark(&mut queue, Some(&nn), test, o.repeats, mode)?;
    let baseline = encoder.benchmark(&mut queue, None, test, o.repeats, 3)?;
    let q = grouped_quality(o, test, &result.blocks);
    let reference = grouped_quality(o, test, &search.blocks);
    let (qualified, ratio) = quality_gate(&q, &reference);
    let mut ratios = serde_json::Map::new();
    let mut worst_ratio = 0.0f64;
    let mut worst_texture = String::new();
    for group in &o.corpus_groups {
        let name = &group.name;
        let ratio = q["groups"][name]["mean_rgb_distance"].as_f64().unwrap()
            / reference["groups"][name]["mean_rgb_distance"]
                .as_f64()
                .unwrap()
                .max(1e-12);
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_texture = name.clone();
        }
        ratios.insert(name.clone(), json!(ratio));
    }
    let provenance = if metadata["provenance"].is_object() {
        metadata["provenance"].clone()
    } else {
        json!({"status":"unknown","train_families":[]})
    };
    let speed = search.end_to_end_ms / result.host_ms;
    let report = json!({
        "gpu":gpu_name,"seed":o.seed,"split":"test","sampled_blocks":test.len(),
        "evaluation_scope":"bounded sampled blocks per declared test texture; no full-image, global-optimum, or universal-generalization certification",
        "corpus":o.corpus_metadata,"model_source":o.load_models,"model_layers":dense.layers,
        "model_mode":mode_name(mode),"model_provenance":provenance,
        "test_independence":if provenance["status"] == "known" {"declared test families disjoint from recorded training families"} else {"unknown prior training provenance; untouched-test claim unavailable"},
        "frozen_model_quality":q,"gpu_search":{"label":search.label,"quality":reference,"resident_submit_wait_ms":search.resident_ms,"host_end_to_end_ms":search.end_to_end_ms},
        "cheap_range_two_ls":{"quality":grouped_quality(o,test,&baseline.blocks),"resident_submit_wait_ms":baseline.resident_ms,"host_end_to_end_ms":baseline.host_ms},
        "resident_submit_wait_ms":result.resident_ms,"host_end_to_end_ms":result.host_ms,
        "gpu_search_host_speedup":speed,"rgb_distance_ratio_to_gpu_search":ratio,
        "per_texture_rgb_distance_ratios":ratios,"worst_texture":worst_texture,"worst_texture_rgb_distance_ratio":worst_ratio,
        "quality_qualified":qualified,"meets_quality_and_5x":qualified && speed >= 5.0,
        "quality_gate":"within 1% mean decoded RGB distance overall and every sampled texture versus current GPU production cluster fit; fixed gate, no model or mode selection",
        "timing":"CPU wall-clock medians include submit/wait overhead; host samples include upload/readback; no GPU timestamp measurements, speed threshold evidence has timing uncertainty",
        "repeats":o.repeats,"training_steps":0,"reconstruction_steps":0,"selection_performed":false
    });
    fs::create_dir_all(&o.output)?;
    fs::write(
        o.output.join("corpus-test-report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "Frozen test: RGB ratio {ratio:.3}, host speedup {speed:.2}x, quality qualified {qualified}; {}",
        o.output.join("corpus-test-report.json").display()
    );
    Ok(())
}
fn main() -> Result<()> {
    let mut o = Options::parse()?;
    let (train, test) = data(&mut o)?;
    if o.formats_train {
        return scalar_nn::train(&o, &train, &test);
    }
    if o.formats_compare {
        return format_experiment::compare(&o, &test);
    }
    if o.pipeline_neighbor_compare {
        return pipeline_neighbor_compare(&o, &test);
    }
    if o.principal_ls_sweep {
        return principal_ls_sweep(&o, &test);
    }
    if o.three_mode_compare {
        return three_mode_compare(&o, &test);
    }
    if o.corpus_split == corpus::Split::Test {
        return locked_test(&o, &test);
    }
    fs::create_dir_all(&o.output)?;
    let provenance = training_provenance(&o)?;
    println!(
        "Preparing {} independent teacher labels; {} held-out blocks",
        train.len(),
        test.len()
    );
    let (device, mut queue, name) = select_device()?;
    println!("GPU {name}");
    if o.gd_compare {
        return gradient_comparison(&o, &device, &mut queue, &test, &name);
    }
    if o.neighbor_compare {
        return neighbor_comparison(&o, &device, &mut queue, &test, &name);
    }
    let loss_parity = if o.reconstruction_steps > 0 {
        {
            reconstruction_loss::verify_gpu(&device, &mut queue)?;
            Some(true)
        }
    } else {
        None
    };
    let (inputs, targets) = if o.load_models.is_none() || o.reconstruction_steps > 0 {
        let train_search = gpu_search::run(&device, &mut queue, &train, 1)?;
        let inputs = train.iter().flat_map(canonical).collect::<Vec<_>>();
        let targets = train
            .iter()
            .enumerate()
            .flat_map(|(i, t)| {
                if i % 512 == 0 {
                    println!("labels {i}/{}", train.len());
                }
                let a = teacher(t);
                let b = train_search.blocks[i];
                let e = endpoints(t, if error(t, a) < error(t, b) { a } else { b });
                let base = baseline(&inputs[i * 48..i * 48 + 48]);
                std::array::from_fn::<_, 6, _>(|c| e[c] - base[c])
            })
            .collect::<Vec<_>>();
        (inputs, targets)
    } else {
        (Vec::new(), Vec::new())
    };
    let search = gpu_search::run(&device, &mut queue, &test, o.repeats)?;
    let cpu_blocks = test.iter().map(production).collect::<Vec<_>>();
    let mut scratch = vec![Block::BLACK; test.len()];
    let cpu_search_ms = benchmark(
        || {
            for (t, b) in test.iter().zip(&mut scratch) {
                *b = production(black_box(t));
            }
            black_box(&scratch);
            Ok(())
        },
        o.repeats,
    )?;
    let range = test
        .iter()
        .map(|t| pack(t, baseline(&canonical(t))))
        .collect::<Vec<_>>();
    let refined_range = test
        .iter()
        .zip(&range)
        .map(|(t, &b)| refine(t, b))
        .collect::<Vec<_>>();
    let range_ms = benchmark(
        || {
            for (t, b) in test.iter().zip(&mut scratch) {
                *b = refine(t, pack(t, baseline(&canonical(t))));
            }
            black_box(&scratch);
            Ok(())
        },
        o.repeats,
    )?;
    let mut report = json!({"gpu":name,"seed":o.seed,"train_samples":train.len(),"validation_blocks":test.len(),"steps_per_size":if o.load_models.is_some(){0}else{o.steps},"training_batch":o.batch,"benchmark_batch":test.len(),"gpu_search_dispatched_count":search.dispatched_count,"gpu_search_count":search.count,"repeats":o.repeats,"precision":"f32","input":"48 RGB8 values /255, sorted PCA projection then RGB","target":"6 residual endpoints relative to sorted range; canonical PCA axis orientation","teacher":"all sorted four-cluster partitions with prefix-sum LS, rounded RGB565, decoded Euclidean RGB selection; retains CPU and current GPU production candidates; not global RGB565 optimum","holdout":"seeded synthetic; first image top40% training, all image bottom40% evaluation, 20% spatial gap; supplied images treated as same source family","gpu_search":{"label":search.label,"quality":grouped_quality(&o,&test,&search.blocks),"resident_submit_wait_ms":search.resident_ms,"host_end_to_end_ms":search.end_to_end_ms},"cpu_production":{"quality":grouped_quality(&o,&test,&cpu_blocks),"ms":cpu_search_ms},"range":{"quality":grouped_quality(&o,&test,&range)},"range_refined":{"quality":grouped_quality(&o,&test,&refined_range),"cpu_ms":range_ms},"sizes":[]});
    let mut gpu_encoder = gpu_encode::GpuEncoder::new(&device, &test)?;
    let gpu_range = gpu_encoder.benchmark(&mut queue, None, &test, o.repeats, 3)?;
    report["gpu_range_refined"] = json!({"quality":grouped_quality(&o,&test,&gpu_range.blocks),"resident_submit_wait_ms":gpu_range.resident_ms,"host_end_to_end_ms":gpu_range.host_ms,"gpu_search_host_speedup":search.end_to_end_ms/gpu_range.host_ms});
    report["loaded_models"] = json!(o.load_models);
    report["corpus"] = o.corpus_metadata.clone();
    report["model_provenance"] = provenance.clone();
    if o.corpus_manifest.is_some() {
        report["holdout"] = json!(
            "declared validation families; sampled blocks per texture; training uses only train split, no synthetic stress blocks mixed into evaluation"
        );
        report["evaluation_scope"] = json!(
            "bounded sampled corpus; no full-image, global-optimum, or universal-generalization certification"
        );
    }
    report["reconstruction_steps_per_size"] = json!(o.reconstruction_steps);
    report["reconstruction_learning_rate"] = json!(o.reconstruction_rate);
    report["reconstruction_loss"] = json!(
        "smooth decoded nearest quantized RGB palette distance +0.005 endpoint teacher anchor; frozen assignments, rounding STE"
    );
    report["reconstruction_gpu_cpu_derivative_verified"] = json!(loss_parity);
    let mut rng = Rng(o.seed.max(1));
    let mut best_score = (2u8, f64::INFINITY, f64::INFINITY);
    let mut best = None;
    for layers in &o.sizes {
        let size = layers[..layers.len() - 1]
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join("x");
        println!(
            "{} 48-{size}-6",
            if o.load_models.is_some() {
                "Evaluating"
            } else {
                "Training"
            }
        );
        let mut dense = if let Some(directory) = &o.load_models {
            load_dense(directory, layers, &size)?
        } else {
            Dense::new(layers.clone(), &mut rng)
        };
        let mut desc = NnDesc::new(&[Encoding::Identity { arity: 48 }], layers);
        desc.activation = Activation::Relu;
        let mut nn = Nn::new(&device, desc, &dense.weights)?;
        let start = Instant::now();
        if o.load_models.is_none() {
            let mut training = InferenceBuffers::new(&device, o.batch);
            let target_buffer = storage(&device, &vec![0.0; o.batch * 6], "NN targets");
            let mut batch_targets = vec![0.0; o.batch * 6];
            for step in 0..o.steps {
                for b in 0..o.batch {
                    let i = rng.next() as usize % train.len();
                    training.inputs[b * 48..b * 48 + 48]
                        .copy_from_slice(&inputs[i * 48..i * 48 + 48]);
                    batch_targets[b * 6..b * 6 + 6].copy_from_slice(&targets[i * 6..i * 6 + 6]);
                }
                let mut enc = queue.new_command_encoder();
                enc.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
                {
                    let mut copy = enc.copy();
                    copy.write_buffer_raw(&training.input, bytemuck::cast_slice(&training.inputs));
                    copy.write_buffer_raw(&target_buffer, bytemuck::cast_slice(&batch_targets));
                }
                enc.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
                queue.submit([enc.finish()])?;
                nn.training(
                    &mut queue,
                    (&training.input).into(),
                    (&target_buffer).into(),
                    (&training.output).into(),
                    &[],
                    Batch::new(o.batch as u32),
                )?;
                if step % 250 == 0 {
                    println!("  step {step}/{}", o.steps);
                }
            }
            queue.wait_idle()?;
            dense.weights = read(&device, &mut queue, nn.parameters(), dense.weights.len())?;
        }
        let train_seconds = if o.load_models.is_some() {
            0.0
        } else {
            start.elapsed().as_secs_f64()
        };
        let reconstruction_start = Instant::now();
        if o.reconstruction_steps > 0 {
            let mut desc = NnDesc::new(&[Encoding::Identity { arity: 48 }], layers);
            desc.activation = Activation::Relu;
            desc.loss = reconstruction_loss::loss();
            desc.optimizer = Optimizer::Adam(Adam {
                learning_rate: o.reconstruction_rate,
                ..Adam::default()
            });
            nn = Nn::new(&device, desc, &dense.weights)?;
            let mut training = InferenceBuffers::new(&device, o.batch);
            let target_buffer = storage(
                &device,
                &vec![0.0; o.batch * 54],
                "BC1 reconstruction targets",
            );
            let mut batch_targets = vec![0.0; o.batch * 54];
            for step in 0..o.reconstruction_steps {
                for b in 0..o.batch {
                    let i = rng.next() as usize % train.len();
                    let input = &inputs[i * 48..i * 48 + 48];
                    training.inputs[b * 48..b * 48 + 48].copy_from_slice(input);
                    batch_targets[b * 54..b * 54 + 54].copy_from_slice(
                        &reconstruction_loss::target(input, &targets[i * 6..i * 6 + 6]),
                    );
                }
                let mut encoder = queue.new_command_encoder();
                encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
                {
                    let mut copy = encoder.copy();
                    copy.write_buffer_raw(&training.input, bytemuck::cast_slice(&training.inputs));
                    copy.write_buffer_raw(&target_buffer, bytemuck::cast_slice(&batch_targets));
                }
                encoder.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
                queue.submit([encoder.finish()])?;
                nn.training(
                    &mut queue,
                    (&training.input).into(),
                    (&target_buffer).into(),
                    (&training.output).into(),
                    &[],
                    Batch::new(o.batch as u32),
                )?;
                if step % 10000 == 0 {
                    println!("  reconstruction step {step}/{}", o.reconstruction_steps);
                }
            }
            queue.wait_idle()?;
            dense.weights = read(&device, &mut queue, nn.parameters(), dense.weights.len())?;
        }
        let reconstruction_seconds = if o.reconstruction_steps > 0 {
            reconstruction_start.elapsed().as_secs_f64()
        } else {
            0.0
        };
        if !dense.weights.iter().all(|v| v.is_finite()) {
            return Err("trained weights contain nonfinite values".into());
        }
        let mut inference = InferenceBuffers::new(&device, test.len());
        for (t, input) in test.iter().zip(inference.inputs.chunks_exact_mut(48)) {
            input.copy_from_slice(&canonical(t));
        }
        inference.upload(&mut queue)?;
        inference.infer(&mut queue, &nn, true)?;
        let mut max_diff = 0.0f32;
        for (input, gpu) in inference
            .inputs
            .chunks_exact(48)
            .zip(inference.outputs.chunks_exact(6))
        {
            for (cpu, &gpu) in dense.forward(input).iter().zip(gpu) {
                if !cpu.is_finite() || !gpu.is_finite() {
                    return Err("GPU/CPU parity contains nonfinite values".into());
                }
                max_diff = max_diff.max((cpu - gpu).abs());
            }
        }
        if !max_diff.is_finite() || max_diff > 2e-4 {
            return Err(format!("GPU/CPU parity failed {max_diff}").into());
        }
        let raw = test
            .iter()
            .zip(inference.inputs.chunks_exact(48))
            .zip(inference.outputs.chunks_exact(6))
            .map(|((t, i), e)| pack(t, std::array::from_fn(|c| baseline(i)[c] + e[c])))
            .collect::<Vec<_>>();
        let guarded_blocks = test
            .iter()
            .zip(inference.inputs.chunks_exact(48))
            .zip(inference.outputs.chunks_exact(6))
            .map(|((t, i), e)| guarded(t, i, e, false))
            .collect::<Vec<_>>();
        let refined = test
            .iter()
            .zip(inference.inputs.chunks_exact(48))
            .zip(inference.outputs.chunks_exact(6))
            .map(|((t, i), e)| guarded(t, i, e, true))
            .collect::<Vec<_>>();
        let mut cpu_timings = Vec::new();
        for mode in 0..3 {
            cpu_timings.push(benchmark(
                || {
                    for (t, b) in test.iter().zip(&mut scratch) {
                        let input = canonical(t);
                        *b = prediction(t, &input, &dense.forward(&input), mode);
                    }
                    black_box(&scratch);
                    Ok(())
                },
                o.repeats,
            )?);
        }
        let gpu_infer_ms = benchmark(|| inference.infer(&mut queue, &nn, false), o.repeats)?;
        let mut gpu_timings = Vec::new();
        for mode in 0..3 {
            gpu_timings.push(benchmark(
                || {
                    for (t, input) in test.iter().zip(inference.inputs.chunks_exact_mut(48)) {
                        input.copy_from_slice(&canonical(t));
                    }
                    inference.upload(&mut queue)?;
                    inference.infer(&mut queue, &nn, true)?;
                    for (((t, input), e), b) in test
                        .iter()
                        .zip(inference.inputs.chunks_exact(48))
                        .zip(inference.outputs.chunks_exact(6))
                        .zip(&mut scratch)
                    {
                        *b = prediction(t, input, e, mode);
                    }
                    black_box(&scratch);
                    Ok(())
                },
                o.repeats,
            )?);
        }
        let mut gpu_results = Vec::new();
        let mut modes = Vec::new();
        let mut network_best = (2u8, f64::INFINITY, f64::INFINITY);
        let mut network_mode = 2usize;
        for mode in 0..3u32 {
            let result = gpu_encoder.benchmark(&mut queue, Some(&nn), &test, o.repeats, mode)?;
            let q = grouped_quality(&o, &test, &result.blocks);
            let (qualified, ratio) = quality_gate(&q, &report["gpu_search"]["quality"]);
            let speed = search.end_to_end_ms / result.host_ms;
            let score = if qualified {
                (0u8, result.host_ms, 0.0)
            } else {
                (1u8, ratio, result.host_ms)
            };
            if score < network_best {
                network_best = score;
                network_mode = mode as usize;
            }
            if score < best_score {
                best_score = score;
                best = Some((
                    dense.clone(),
                    size.clone(),
                    mode,
                    qualified,
                    qualified && speed >= 5.0,
                ));
            }
            modes.push(json!({"mode":mode_name(mode),"quality":q,"resident_submit_wait_ms":result.resident_ms,"host_end_to_end_ms":result.host_ms,"gpu_search_host_speedup":speed,"gpu_search_resident_speedup":search.resident_ms/result.resident_ms,"rgb_distance_ratio_to_gpu_search":ratio,"quality_qualified":qualified,"meets_quality_and_5x":qualified&&speed>=5.0}));
            if o.corpus_manifest.is_some() {
                let mut ratios = serde_json::Map::new();
                let mut worst = (String::new(), 0.0f64);
                for group in &o.corpus_groups {
                    let reference =
                        report["gpu_search"]["quality"]["groups"][&group.name]["mean_rgb_distance"]
                            .as_f64()
                            .unwrap();
                    let ratio = q["groups"][&group.name]["mean_rgb_distance"]
                        .as_f64()
                        .unwrap()
                        / reference.max(1e-12);
                    if ratio > worst.1 {
                        worst = (group.name.clone(), ratio);
                    }
                    ratios.insert(group.name.clone(), json!(ratio));
                }
                let row = modes.last_mut().unwrap();
                row["per_texture_rgb_distance_ratios"] = Value::Object(ratios);
                row["worst_texture"] = json!(worst.0);
                row["worst_texture_rgb_distance_ratio"] = json!(worst.1);
            }
            gpu_results.push(result);
        }
        // CPU forward on the actual GPU-ordered colors checks GPU preprocessing.
        let gpu_inputs = read(&device, &mut queue, &gpu_encoder.input, test.len() * 48)?;
        let gpu_residuals = read(
            &device,
            &mut queue,
            &gpu_encoder.output_residual,
            test.len() * 6,
        )?;
        let mut complete_parity = 0.0f32;
        for ((t, input), residual) in test
            .iter()
            .zip(gpu_inputs.chunks_exact(48))
            .zip(gpu_residuals.chunks_exact(6))
        {
            let mut actual: Texels = std::array::from_fn(|i| {
                std::array::from_fn(|c| (input[i * 3 + c] * 255.0).round() as u8)
            });
            actual.sort_unstable();
            let mut expected = *t;
            expected.sort_unstable();
            if actual != expected || input.iter().any(|v| !v.is_finite()) {
                return Err("GPU preprocessing corrupts block color multiset".into());
            }
            for (cpu, &gpu) in dense.forward(input).iter().zip(residual) {
                if !cpu.is_finite() || !gpu.is_finite() {
                    return Err("complete GPU inference contains nonfinite output".into());
                }
                complete_parity = complete_parity.max((cpu - gpu).abs());
            }
        }
        if complete_parity > 2e-4 {
            return Err(
                format!("complete GPU/CPU inference parity failed {complete_parity}").into(),
            );
        }
        let chosen = &modes[network_mode];
        let ratio = chosen["rgb_distance_ratio_to_gpu_search"].as_f64().unwrap();
        let speed = chosen["gpu_search_host_speedup"].as_f64().unwrap();
        let origin = if let Some(directory) = &o.load_models {
            serde_json::from_slice::<Value>(&fs::read(
                directory.join(format!("48-{size}-6.json")),
            )?)?["config"]
                .clone()
        } else {
            Value::Null
        };
        let row = json!({"layers":layers,"parameters":dense.weights.len(),"endpoint_training_steps":if o.load_models.is_some(){0}else{o.steps},"training_seconds":train_seconds,"reconstruction_steps":o.reconstruction_steps,"reconstruction_seconds":reconstruction_seconds,"reconstruction_learning_rate":o.reconstruction_rate,"loaded_config":origin,"gpu_cpu_max_abs_difference":max_diff,"complete_gpu_cpu_max_abs_difference":complete_parity,"raw_quality":modes[0]["quality"],"range_guard_quality":modes[1]["quality"],"range_guard_two_ls_quality":modes[2]["quality"],"cpu_canonical_quality":{"raw":grouped_quality(&o,&test,&raw),"range_guard":grouped_quality(&o,&test,&guarded_blocks),"range_guard_two_ls":grouped_quality(&o,&test,&refined)},"cpu_full_ms":cpu_timings[2],"cpu_timings_ms":{"raw":cpu_timings[0],"range_guard":cpu_timings[1],"range_guard_two_ls":cpu_timings[2]},"gpu_inference_with_cpu_postprocess_host_timings_ms":{"raw":gpu_timings[0],"range_guard":gpu_timings[1],"range_guard_two_ls":gpu_timings[2]},"gpu_inference_submit_wait_ms":gpu_infer_ms,"gpu_complete_modes":modes,"best_mode":mode_name(network_mode as u32),"gpu_full_host_ms":chosen["host_end_to_end_ms"],"gpu_search_host_speedup":speed,"rgb_distance_ratio_to_gpu_search":ratio,"meets_quality_and_5x":chosen["meets_quality_and_5x"]});
        println!(
            "48-{size}-6: RGB distance ratio {ratio:.3}, GPU full speedup {speed:.2}x, parity {max_diff:e}"
        );
        fs::write(
            o.output.join(format!("48-{size}-6.json")),
            serde_json::to_vec_pretty(
                &json!({"config":row,"weights":dense.weights,"provenance":provenance}),
            )?,
        )?;
        report["sizes"].as_array_mut().unwrap().push(row);
    }
    let (dense, size, selected_mode, quality_qualified, meets_quality_and_5x) = best.unwrap();
    report["selected"] = json!(format!("48-{size}-6"));
    report["selected_mode"] = json!(mode_name(selected_mode));
    report["selected_quality_qualified"] = json!(quality_qualified);
    report["selected_meets_quality_and_5x"] = json!(meets_quality_and_5x);
    report["selection_rule"] = json!(
        "fastest complete GPU host end-to-end within 1% decoded RGB distance of current GPU production cluster fit overall and in each source group; if none qualifies, smallest RGB distance ratio, marked unqualified"
    );
    fs::write(
        o.output.join("metrics.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    fs::write(
        o.output.join("selected.json"),
        serde_json::to_vec_pretty(
            &json!({"input_size":48,"layers":dense.layers,"activation":"relu","output":"range endpoint residual","preprocessing":{"normalization":"RGB8 /255","axis":"jkl::math::pca_axis3, ten covariance power iterations, RGB lexicographic accumulation order","axis_sign":"largest absolute component positive; ties prefer red then green then blue","flat_axis":[1,0,0],"order":"ascending projection, RGB lexicographic ties"},"mode":mode_name(selected_mode),"quality_qualified":quality_qualified,"meets_quality_and_5x":meets_quality_and_5x,"postprocess":mode_name(selected_mode),"weights":dense.weights,"metrics":report["sizes"],"provenance":provenance,"selection_corpus":o.corpus_metadata}),
        )?,
    )?;
    let mut selected_desc = NnDesc::new(&[Encoding::Identity { arity: 48 }], &dense.layers);
    selected_desc.activation = Activation::Relu;
    let selected_nn = Nn::new(&device, selected_desc, &dense.weights)?;
    for (i, path) in o.images.iter().enumerate() {
        let source = image::open(path)?.to_rgb8();
        let img = if o.preview_size > 0 {
            image::imageops::crop_imm(
                &source,
                0,
                0,
                source.width().min(o.preview_size),
                source.height().min(o.preview_size),
            )
            .to_image()
        } else {
            source
        };
        let texels = image_blocks(&img);
        let mut preview_encoder = gpu_encode::GpuEncoder::new(&device, &texels)?;
        let blocks = preview_encoder
            .benchmark(&mut queue, Some(&selected_nn), &texels, 1, selected_mode)?
            .blocks;
        let bytes = blocks.iter().flat_map(Block::bytes).collect::<Vec<_>>();
        fs::write(o.output.join(format!("image-{i}.bc1")), bytes)?;
        let mut preview = image::RgbImage::new(img.width(), img.height());
        let cols = img.width().div_ceil(4) as usize;
        for (j, b) in blocks.iter().enumerate() {
            let d = b.decode();
            for y in 0..4 {
                for x in 0..4 {
                    let px = (j % cols) * 4 + x;
                    let py = (j / cols) * 4 + y;
                    if px < img.width() as usize && py < img.height() as usize {
                        preview.put_pixel(
                            px as u32,
                            py as u32,
                            image::Rgb(
                                d[y][x].0.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8),
                            ),
                        );
                    }
                }
            }
        }
        preview.save(o.output.join(format!("image-{i}-decoded.png")))?;
        fs::write(
            o.output.join(format!("image-{i}.json")),
            serde_json::to_vec_pretty(
                &json!({"source":path,"width":img.width(),"height":img.height(),"block_width":cols,"block_order":"row major; edge texels replicated; raw little endian BC1"}),
            )?,
        )?;
    }
    println!(
        "Selected 48-{size}-6; metrics {}",
        o.output.join("metrics.json").display()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pipeline_neighbors_require_frozen_test_and_exclude_training() {
        let base = [
            "--pipeline-neighbor-compare",
            "--corpus-manifest",
            "corpus.json",
            "--corpus-split",
            "test",
            "--load-models",
            "models",
        ];
        assert!(options(&base).is_ok());
        assert!(options(&["--pipeline-neighbor-compare"]).is_err());
        for extra in [
            vec!["--steps", "1"],
            vec!["--reconstruction-steps", "1"],
            vec!["--three-mode-compare"],
            vec!["--principal-ls-sweep"],
            vec!["--ls-neighbor-compare"],
            vec!["--gd-compare"],
        ] {
            let mut args = base.to_vec();
            args.extend(extra);
            assert!(options(&args).is_err());
        }
    }
    #[test]
    fn principal_ls_neighbors_are_nonworsening_and_opaque() {
        let mut rng = Rng(654321);
        let mut texels = vec![[[0; 3]; 16], [[255; 3]; 16], [[120, 80, 160]; 16]];
        for _ in 0..64 {
            texels.push(std::array::from_fn(|_| {
                std::array::from_fn(|_| rng.next() as u8)
            }));
        }
        for t in texels {
            let seed = refine_passes(&t, pack(&t, baseline(&canonical(&t))), 4);
            for reuse in [false, true] {
                for rounds in [1, 2] {
                    let result = neighborhood::refine(&t, seed, rounds, reuse);
                    assert!(error(&t, result) <= error(&t, seed) + 2e-7);
                    check_opaque_blocks(&[result]).unwrap();
                }
            }
        }
    }
    #[test]
    fn principal_axis_four_ls_is_nonworsening_and_opaque() {
        let mut rng = Rng(1234567);
        let mut fixtures = vec![[[0; 3]; 16], [[255; 3]; 16], [[80, 140, 200]; 16]];
        for _ in 0..256 {
            fixtures.push(std::array::from_fn(|_| {
                std::array::from_fn(|_| rng.next() as u8)
            }));
        }
        for texels in fixtures {
            let initial = pack(&texels, baseline(&canonical(&texels)));
            let two = refine_passes(&texels, initial, 2);
            let mut previous = two;
            for passes in [4, 8, 16, 32, 64] {
                let refined = refine_passes(&texels, initial, passes);
                assert!(error(&texels, refined) <= error(&texels, previous) + 1e-7);
                assert!(error(&texels, refined) <= error(&texels, initial) + 1e-7);
                if refined.color0 == refined.color1 {
                    assert_eq!(refined.indices, [0; 4]);
                }
                previous = refined;
            }
        }
    }
    #[test]
    fn principal_ls_sweep_rejects_invalid_budget_and_training() {
        let base = [
            "--principal-ls-sweep",
            "--corpus-manifest",
            "corpus.json",
            "--corpus-split",
            "test",
            "--load-models",
            "models",
        ];
        assert!(options(&base).is_ok());
        assert!(options(&["--principal-ls-sweep"]).is_err());
        for extra in [
            vec!["--steps", "1"],
            vec!["--reconstruction-steps", "1"],
            vec!["--three-mode-compare"],
            vec!["--principal-ls-passes", "0,4"],
            vec!["--principal-ls-passes", "4,65"],
            vec!["--principal-ls-passes", "8,4"],
            vec!["--principal-ls-passes", "4,4"],
        ] {
            let mut args = base.to_vec();
            args.extend(extra);
            assert!(options(&args).is_err());
        }
    }
    #[test]
    fn three_mode_comparison_requires_frozen_test() {
        assert!(options(&["--three-mode-compare"]).is_err());
        assert!(
            options(&[
                "--three-mode-compare",
                "--corpus-manifest",
                "corpus.json",
                "--corpus-split",
                "test",
                "--load-models",
                "models"
            ])
            .is_ok()
        );
        assert!(
            options(&[
                "--three-mode-compare",
                "--corpus-manifest",
                "corpus.json",
                "--corpus-split",
                "test",
                "--load-models",
                "models",
                "--steps",
                "1"
            ])
            .is_err()
        );
    }
    fn options(args: &[&str]) -> Result<Options> {
        Options::parse_args(args.iter().map(|s| s.to_string()))
    }
    fn temporary_directory(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("bc1-nn-{name}-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }
    fn artifact() -> Value {
        json!({"input_size":48,"layers":[16,6],"activation":"relu","output":"range endpoint residual","preprocessing":{"axis":"jkl::math::pca_axis3"},"mode":"range_guard_two_ls","weights":vec![0.125f32;16*49+6*17]})
    }
    #[test]
    fn locked_test_options_reject_training_and_tuning() {
        let base = [
            "--corpus-manifest",
            "corpus.json",
            "--corpus-split",
            "test",
            "--load-models",
            "models",
        ];
        assert!(options(&base).is_ok());
        for flags in [
            vec!["--steps", "1"],
            vec!["--reconstruction-steps", "1"],
            vec!["--gd-compare"],
            vec!["--ls-neighbor-compare"],
        ] {
            let mut args = base.to_vec();
            args.extend(flags);
            assert!(options(&args).is_err());
        }
        assert!(options(&["--corpus-split", "test"]).is_err());
        assert!(options(&["--corpus-manifest", "manifest.json", "--image", "image.png"]).is_err());
        assert!(options(&["--corpus-split", "train"]).is_err());
    }
    #[test]
    fn frozen_weights_are_authoritative_and_validated() {
        let path = temporary_directory("frozen");
        let mut value = artifact();
        fs::write(
            path.join("selected.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        fs::write(
            path.join("48-16-6.json"),
            b"invalid unrelated per-size file",
        )
        .unwrap();
        let (dense, mode, _) = frozen_model(&path, None).unwrap();
        assert_eq!(dense.layers, vec![16, 6]);
        assert!(dense.weights.iter().all(|&w| w == 0.125));
        assert_eq!(mode, 2);
        assert!(frozen_model(&path, Some(&[vec![32, 6]])).is_err());
        value["mode"] = json!("unknown_neighborhood");
        fs::write(
            path.join("selected.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        assert!(frozen_model(&path, None).is_err());
        value["mode"] = json!("raw");
        value["weights"] = json!([0.0]);
        fs::write(
            path.join("selected.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        assert!(frozen_model(&path, None).is_err());
        assert!(validate_dense(vec![6], vec![f32::INFINITY; 294], None).is_err());
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn training_provenance_preserves_unknown_and_known_family_union() {
        let path = temporary_directory("provenance");
        let mut value = artifact();
        value["provenance"] =
            json!({"status":"unknown","train_families":["prior"],"training_manifests":[]});
        fs::write(
            path.join("selected.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        let mut o = options(&[]).unwrap();
        o.sizes = vec![vec![16, 6]];
        fs::write(
            path.join("48-16-6.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        o.load_models = Some(path.clone());
        o.corpus_manifest = Some("manifest.json".into());
        o.reconstruction_steps = 1;
        o.train_families = vec!["new".into()];
        let provenance = training_provenance(&o).unwrap();
        assert_eq!(provenance["status"], "unknown");
        assert_eq!(provenance["train_families"], json!(["new", "prior"]));
        let group = corpus::EvalGroup {
            name: "test".into(),
            start: 0,
            count: 1,
            family: "prior".into(),
            path: "test.png".into(),
            width: 4,
            height: 4,
        };
        assert!(check_test_provenance(&json!({"provenance":provenance}), &[group]).is_err());
        o.load_models = None;
        assert_eq!(training_provenance(&o).unwrap()["status"], "known");
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn legacy_grouped_quality_keeps_synthetic_segments() {
        let o = options(&["--samples", "64"]).unwrap();
        let mut rng = Rng(42);
        let t = (0..64).map(|i| synthetic(&mut rng, i)).collect::<Vec<_>>();
        let b = t.iter().map(production).collect::<Vec<_>>();
        let q = grouped_quality(&o, &t, &b);
        assert_eq!(q["groups"].as_object().unwrap().len(), 7);
        assert_eq!(q["groups"]["synthetic"], quality(&t, &b));
    }
    #[test]
    fn teacher_retains_production_and_valid_blocks() {
        let mut r = Rng(123);
        for i in 0..12 {
            let t = synthetic(&mut r, i);
            let b = teacher(&t);
            assert!(error(&t, b) <= error(&t, production(&t)) + 1e-7);
            assert_eq!(b, Block::from_bytes(b.bytes()));
            assert!(
                b.decode()
                    .iter()
                    .flatten()
                    .all(|c| c.0.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0))
            );
        }
    }
    #[test]
    fn canonical_color_order_invariant() {
        let mut r = Rng(123);
        for i in 0..12 {
            let mut t = synthetic(&mut r, i);
            let a = canonical(&t);
            t.reverse();
            assert_eq!(a, canonical(&t));
            t.rotate_left(3);
            assert_eq!(a, canonical(&t));
        }
    }
    #[test]
    fn flat_blocks_remain_opaque() {
        for c in [[0, 0, 0], [255, 255, 255], [120, 30, 180]] {
            let t = [c; 16];
            let b = pack(&t, baseline(&canonical(&t)));
            assert!(b.decode_with_alpha().iter().flatten().all(|c| c.a() == 1.0));
        }
    }
}
