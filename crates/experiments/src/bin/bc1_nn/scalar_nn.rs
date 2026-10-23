//! Sorted scalar residual-endpoint neural model and validation-only training.
use super::*;
#[derive(Clone)]
pub struct Dense16 {
    pub layers: Vec<u32>,
    pub weights: Vec<f32>,
}
impl Dense16 {
    fn new(width: u32, rng: &mut Rng) -> Self {
        let layers = vec![width, 2];
        let mut weights = Vec::new();
        let mut n = 16;
        for &m in &layers {
            let scale = (6.0 / (n + m) as f32).sqrt();
            weights.extend((0..n * m).map(|_| (rng.unit() * 2.0 - 1.0) * scale));
            weights.extend((0..m).map(|_| 0.0));
            n = m;
        }
        Self { layers, weights }
    }
    pub fn forward(&self, input: &[f32]) -> [f32; 2] {
        let mut a = input.to_vec();
        let mut offset = 0;
        for (layer, &m) in self.layers.iter().enumerate() {
            let n = a.len();
            let m = m as usize;
            let mut b = vec![0.0; m];
            for j in 0..m {
                let mut v = self.weights[offset + m * n + j];
                for i in 0..n {
                    v += self.weights[offset + j * n + i] * a[i];
                }
                b[j] = if layer + 1 == self.layers.len() {
                    v
                } else {
                    v.max(0.0)
                };
            }
            offset += m * (n + 1);
            a = b;
        }
        a[..2].try_into().unwrap()
    }
    pub fn nn(&self, device: &Device) -> Result<Nn> {
        let mut d = NnDesc::new(&[Encoding::Identity { arity: 16 }], &self.layers);
        d.activation = Activation::Relu;
        Ok(Nn::new(device, d, &self.weights)?)
    }
}
pub fn load(path: &PathBuf) -> Result<(Dense16, Value)> {
    let v: Value = serde_json::from_slice(&fs::read(path.join("scalar-selected.json"))?)?;
    if v["input_arity"] != 16
        || v["preprocessing"] != "sort ascending scalar RGB8 channel /255"
        || v["mode"] != "minmax_guard_two_ls_one_rebuilt_neighbor"
    {
        return Err("unsupported scalar model identity".into());
    }
    let layers: Vec<u32> = serde_json::from_value(v["layers"].clone())?;
    if layers.len() != 2 || layers[1] != 2 || ![8, 16, 32].contains(&layers[0]) {
        return Err("invalid scalar layers".into());
    }
    let weights: Vec<f32> = serde_json::from_value(v["weights"].clone())?;
    let count = (16 + 1) * layers[0] as usize + (layers[0] as usize + 1) * 2;
    if count != weights.len() || weights.iter().any(|w| !w.is_finite()) {
        return Err("invalid scalar weights".into());
    }
    Ok((Dense16 { layers, weights }, v))
}
fn inputs(blocks: &[Texels]) -> Vec<f32> {
    blocks
        .iter()
        .flat_map(|t| {
            (0..2).flat_map(move |c| {
                let mut a = t.map(|p| p[c]);
                a.sort_unstable();
                a.map(|v| v as f32 / 255.0)
            })
        })
        .collect()
}
pub fn parity(
    device: &Device,
    queue: &mut Queue,
    dense: &Dense16,
    nn: &Nn,
    blocks: &[Texels],
) -> Result<f32> {
    let blocks = &blocks[..blocks.len().min(65)];
    let mut encoder = format_scalar::ScalarEncoder::new(device, blocks, 2)?;
    encoder.upload(queue, blocks)?;
    encoder.preprocess(queue)?;
    let sorted = read(device, queue, encoder.canonical_input(), blocks.len() * 32)?;
    let expected = inputs(blocks);
    if sorted
        .iter()
        .zip(&expected)
        .any(|(a, b)| (a - b).abs() > 1e-7)
    {
        return Err("scalar GPU preprocessing mismatch".into());
    }
    nn.inference(
        queue,
        encoder.canonical_input().into(),
        encoder.residual_buffer().into(),
        &[],
        Batch::new((blocks.len() * 2) as u32),
    )?;
    let output = read(device, queue, encoder.residual_buffer(), blocks.len() * 4)?;
    let cpu = expected
        .chunks_exact(16)
        .flat_map(|x| dense.forward(x))
        .collect::<Vec<_>>();
    let max = cpu
        .iter()
        .zip(output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    if max > 2e-4 {
        return Err(format!("scalar NN GPU/CPU deviation {max}").into());
    }
    Ok(max)
}
pub fn train(o: &Options, train: &[Texels], validation: &[Texels]) -> Result<()> {
    if train.is_empty() || validation.is_empty() {
        return Err("scalar training needs train and validation blocks".into());
    }
    fs::create_dir_all(&o.output)?;
    let (device, mut queue, name) = select_device()?;
    let baseline_parity = format_scalar::verify_gpu(&device, &mut queue, train, None)?;
    let mut encoder = format_scalar::ScalarEncoder::new(&device, train, 2)?;
    let labels = encoder.teacher(&mut queue, train)?;
    let input = inputs(train);
    let mut targets = Vec::with_capacity(labels.len() * 2);
    for (i, b) in labels.iter().enumerate() {
        let mut e = [
            b.color0.bits() as f32 / 255.0,
            b.color1.bits() as f32 / 255.0,
        ];
        e.sort_by(f32::total_cmp);
        targets.extend([e[0] - input[i * 16], e[1] - input[i * 16 + 15]]);
    }
    let mut rng = Rng(o.seed.max(1));
    let mut candidates = Vec::new();
    let mut models = Vec::new();
    let mut eval = format_scalar::ScalarEncoder::new(&device, validation, 2)?;
    for width in [8, 16, 32] {
        println!("Scalar training 16-{width}-2, {} steps", o.steps);
        let mut dense = Dense16::new(width, &mut rng);
        let mut nn = dense.nn(&device)?;
        let x = storage(&device, &vec![0.0; o.batch * 16], "scalar training input");
        let y = storage(&device, &vec![0.0; o.batch * 2], "scalar targets");
        let z = storage(&device, &vec![0.0; o.batch * 2], "scalar prediction");
        let mut bx = vec![0.0; o.batch * 16];
        let mut by = vec![0.0; o.batch * 2];
        let start = Instant::now();
        for step in 0..o.steps {
            for b in 0..o.batch {
                let i = rng.next() as usize % labels.len();
                bx[b * 16..b * 16 + 16].copy_from_slice(&input[i * 16..i * 16 + 16]);
                by[b * 2..b * 2 + 2].copy_from_slice(&targets[i * 2..i * 2 + 2]);
            }
            let mut enc = queue.new_command_encoder();
            enc.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
            {
                let mut copy = enc.copy();
                copy.write_buffer_raw(&x, bytemuck::cast_slice(&bx));
                copy.write_buffer_raw(&y, bytemuck::cast_slice(&by));
            }
            enc.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
            queue.submit([enc.finish()])?;
            nn.training(
                &mut queue,
                (&x).into(),
                (&y).into(),
                (&z).into(),
                &[],
                Batch::new(o.batch as u32),
            )?;
            if step % 1000 == 0 {
                println!("scalar {width}: {step}/{}", o.steps);
            }
        }
        queue.wait_idle()?;
        dense.weights = read(&device, &mut queue, nn.parameters(), dense.weights.len())?;
        let max = parity(&device, &mut queue, &dense, &nn, validation)?;
        let result = eval.benchmark(
            &mut queue,
            Some(&nn),
            validation,
            1,
            format_scalar::Method::Neural,
        )?;
        let quality = format_experiment::quality(validation, &result.blocks, 5);
        let model = json!({"input_arity":16,"layers":dense.layers,"weights":dense.weights,"preprocessing":"sort ascending scalar RGB8 channel /255","mode":"minmax_guard_two_ls_one_rebuilt_neighbor","objective":"endpoint residual MSE","teacher":"GPU minmax, both 6/8 palettes, LS64 plus 16 rebuilt neighbor rounds, retains corrected current candidate; heuristic, not global optimum","provenance":{"status":"known","train_families":o.train_families,"training_manifests":[o.corpus_metadata]},"validation_quality":quality,"training_steps":o.steps,"gpu_cpu_max_deviation":max});
        fs::write(
            o.output.join(format!("scalar-16-{width}-2.json")),
            serde_json::to_vec_pretty(&model)?,
        )?;
        candidates.push(json!({"width":width,"parameters":dense.weights.len(),"quality":quality,"training_seconds":start.elapsed().as_secs_f64(),"gpu_cpu_max_deviation":max}));
        models.push(model);
    }
    let best = models
        .iter()
        .map(|m| {
            m["validation_quality"]["psnr"]
                .as_f64()
                .unwrap_or(f64::INFINITY)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let selected = models
        .iter()
        .find(|m| {
            m["validation_quality"]["psnr"]
                .as_f64()
                .unwrap_or(f64::INFINITY)
                >= best - 0.1
        })
        .unwrap();
    fs::write(
        o.output.join("scalar-selected.json"),
        serde_json::to_vec_pretty(selected)?,
    )?;
    fs::write(
        o.output.join("scalar-training-report.json"),
        serde_json::to_vec_pretty(
            &json!({"gpu":name,"baseline_parity":baseline_parity,"train_blocks":train.len(),"scalar_examples":labels.len(),"validation_blocks":validation.len(),"selection":"smallest model within 0.1 dB best RG validation PSNR after fixed NN2LS+one neighbor round","candidates":candidates,"selected_layers":selected["layers"]}),
        )?,
    )?;
    Ok(())
}
