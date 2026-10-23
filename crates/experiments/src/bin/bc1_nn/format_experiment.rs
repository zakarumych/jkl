//! Frozen BC2-5 sampled-corpus comparison; no training or model selection.
use super::*;
use jkl::image::block::{bc2, bc3, bc4, bc5};
fn decoded(bytes: [u8; 16], format: u32) -> [[f32; 3]; 16] {
    match format {
        2 => {
            let d = bc2::Block::from_bytes(bytes).decode();
            std::array::from_fn(|i| d[i / 4][i % 4].0)
        }
        3 => {
            let d = bc3::Block::from_bytes(bytes).decode();
            std::array::from_fn(|i| d[i / 4][i % 4].0)
        }
        4 => {
            let d = bc4::Block::from_bytes(bytes[..8].try_into().unwrap()).decode();
            std::array::from_fn(|i| [d[i / 4][i % 4].r(), 0.0, 0.0])
        }
        5 => {
            let d = bc5::Block::from_bytes(bytes).decode();
            std::array::from_fn(|i| [d[i / 4][i % 4].r(), d[i / 4][i % 4].g(), 0.0])
        }
        _ => unreachable!(),
    }
}
pub fn quality(texels: &[Texels], blocks: &[[u8; 16]], format: u32) -> Value {
    assert_eq!(texels.len(), blocks.len());
    let channels = if format <= 3 {
        3
    } else {
        (format - 3) as usize
    };
    let mut sse = 0.0;
    for (t, &b) in texels.iter().zip(blocks) {
        let d = decoded(b, format);
        for i in 0..16 {
            for c in 0..channels {
                sse += (t[i][c] as f64 / 255.0 - d[i][c] as f64).powi(2);
            }
        }
    }
    let mse = sse / (texels.len() * 16 * channels) as f64;
    json!({"psnr":if mse==0.0{None}else{Some(-10.0*mse.log10())},"mse":mse,"sampled_blocks":texels.len(),"channels":channels,"perfect":mse==0.0})
}
fn grouped(o: &Options, t: &[Texels], b: &[[u8; 16]], format: u32) -> Value {
    json!({"aggregate":quality(t,b,format),"textures":o.corpus_groups.iter().map(|g|json!({"name":g.name,"family":g.family,"path":g.path,"quality":quality(&t[g.start..g.start+g.count],&b[g.start..g.start+g.count],format)})).collect::<Vec<_>>()})
}
fn scalar_cpu(t: &Texels, format: u32, corrected: bool) -> [u8; 16] {
    let mut b = [0; 16];
    b[..8].copy_from_slice(&format_scalar::current_cpu(t, 0, corrected).bytes());
    if format == 5 {
        b[8..].copy_from_slice(&format_scalar::current_cpu(t, 1, corrected).bytes());
    }
    b
}
fn stats(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({"median":sorted[sorted.len()/2],"min":sorted[0],"max":sorted[sorted.len()-1],"samples":values})
}
fn row(
    o: &Options,
    t: &[Texels],
    b: &[[u8; 16]],
    format: u32,
    name: &str,
    resident: &[f64],
    host: &[f64],
) -> Value {
    json!({"name":name,"quality":grouped(o,t,b,format),"resident_submit_wait_ms":stats(resident),"host_end_to_end_ms":stats(host)})
}
pub fn compare(o: &Options, test: &[Texels]) -> Result<()> {
    if o.output.exists() && fs::read_dir(&o.output)?.next().is_some() {
        return Err("frozen format comparison requires fresh empty output directory".into());
    }
    let rgbpath = o.load_models.as_ref().unwrap();
    let scalarpath = o.scalar_models.as_ref().unwrap();
    let rgbbytes = fs::read(rgbpath.join("selected.json"))?;
    let scalarbytes = fs::read(scalarpath.join("scalar-selected.json"))?;
    let (rgb, _, rgbmeta) = frozen_model(rgbpath, None)?;
    check_test_provenance(&rgbmeta, &o.corpus_groups)?;
    let (scalar, scalarmeta) = scalar_nn::load(scalarpath)?;
    check_test_provenance(&scalarmeta, &o.corpus_groups)?;
    let (device, mut queue, gpu) = select_device()?;
    let rgbnn = {
        let mut d = NnDesc::new(&[Encoding::Identity { arity: 48 }], &rgb.layers);
        d.activation = Activation::Relu;
        Nn::new(&device, d, &rgb.weights)?
    };
    let scalarnn = scalar.nn(&device)?;
    let scalar_forward_deviation =
        scalar_nn::parity(&device, &mut queue, &scalar, &scalarnn, test)?;
    let scalar_parity = format_scalar::verify_gpu(&device, &mut queue, test, Some(&scalarnn))?;
    let color_parity = format_color::verify_gpu(&device, &mut queue, test, Some(&rgbnn))?;
    let mut formats = Vec::new();
    for format in 2..=5 {
        println!("BC{format}: {} held-out sampled blocks", test.len());
        let mut rows = Vec::new();
        let mut results = Vec::<Vec<[u8; 16]>>::new();
        if format <= 3 {
            let fmt = if format == 2 {
                format_color::Format::Bc2
            } else {
                format_color::Format::Bc3
            };
            let mut encoder = format_color::ColorEncoder::new(&device, test, fmt)?;
            let methods = [
                format_color::Method::CurrentGpu,
                format_color::Method::Neural,
                format_color::Method::Extrema,
            ];
            let names = [
                if format == 2 {
                    "native_current_gpu"
                } else {
                    "experimental_gpu_port_current_cpu"
                },
                "nn_2ls_one_rebuilt_neighbor",
                "principal_axis_extrema_4ls_one_rebuilt_neighbor",
            ];
            let mut resident = vec![Vec::new(); 3];
            let mut host = vec![Vec::new(); 3];
            results.resize_with(3, Vec::new);
            for &m in &methods {
                encoder.benchmark(&mut queue, Some(&rgbnn), test, 1, m)?;
            }
            for round in 0..o.repeats {
                for step in 0..3 {
                    let i = (round + step) % 3;
                    let r = encoder.benchmark(&mut queue, Some(&rgbnn), test, 1, methods[i])?;
                    resident[i].push(r.resident_ms);
                    host[i].push(r.host_ms);
                    if round == 0 {
                        results[i] = r.blocks;
                    }
                }
            }
            for i in 0..3 {
                rows.push(row(
                    o,
                    test,
                    &results[i],
                    format,
                    names[i],
                    &resident[i],
                    &host[i],
                ));
            }
            let cpu = test
                .iter()
                .map(|t| format_color::current_cpu(t, fmt))
                .collect::<Vec<_>>();
            let mut scratch = vec![[0; 16]; test.len()];
            let ms = benchmark(
                || {
                    for (t, b) in test.iter().zip(&mut scratch) {
                        *b = format_color::current_cpu(black_box(t), fmt);
                    }
                    black_box(&scratch);
                    Ok(())
                },
                3,
            )?;
            rows.push(json!({"name":"actual_current_cpu","quality":grouped(o,test,&cpu,format),"cpu_ms":ms}));
        } else {
            let mut encoder = format_scalar::ScalarEncoder::new(&device, test, format - 3)?;
            let methods = [
                format_scalar::Method::CurrentBuggy,
                format_scalar::Method::CurrentCorrected,
                format_scalar::Method::Neural,
                format_scalar::Method::Extrema,
            ];
            let names = [
                "experimental_gpu_port_current_cpu_buggy_indices",
                "experimental_gpu_port_corrected_current_cpu",
                "nn_2ls_one_rebuilt_neighbor",
                "minmax_extrema_4ls_one_rebuilt_neighbor",
            ];
            let mut resident = vec![Vec::new(); 4];
            let mut host = vec![Vec::new(); 4];
            results.resize_with(4, Vec::new);
            for &m in &methods {
                encoder.benchmark(&mut queue, Some(&scalarnn), test, 1, m)?;
            }
            for round in 0..o.repeats {
                for step in 0..4 {
                    let i = (round + step) % 4;
                    let r = encoder.benchmark(&mut queue, Some(&scalarnn), test, 1, methods[i])?;
                    resident[i].push(r.resident_ms);
                    host[i].push(r.host_ms);
                    if round == 0 {
                        results[i] = r.blocks;
                    }
                }
            }
            for i in 0..4 {
                rows.push(row(
                    o,
                    test,
                    &results[i],
                    format,
                    names[i],
                    &resident[i],
                    &host[i],
                ));
            }
            for corrected in [false, true] {
                let cpu = test
                    .iter()
                    .map(|t| scalar_cpu(t, format, corrected))
                    .collect::<Vec<_>>();
                let mut scratch = vec![[0; 16]; test.len()];
                let ms = benchmark(
                    || {
                        for (t, b) in test.iter().zip(&mut scratch) {
                            *b = scalar_cpu(black_box(t), format, corrected);
                        }
                        black_box(&scratch);
                        Ok(())
                    },
                    3,
                )?;
                rows.push(json!({"name":if corrected{"corrected_current_cpu_control"}else{"actual_current_cpu"},"quality":grouped(o,test,&cpu,format),"cpu_ms":ms}));
            }
        }
        let references = if format <= 3 { vec![3] } else { vec![4, 5] };
        for i in 0..rows.len() {
            let mut deltas = Vec::new();
            for &reference in &references {
                let base = &rows[reference]["quality"];
                let q = &rows[i]["quality"];
                let per = o
                    .corpus_groups
                    .iter()
                    .enumerate()
                    .map(|(j, g)| {
                        let a = q["textures"][j]["quality"]["psnr"].as_f64();
                        let b = base["textures"][j]["quality"]["psnr"].as_f64();
                        json!({"name":g.name,"delta_db":a.zip(b).map(|(a,b)|a-b)})
                    })
                    .collect::<Vec<_>>();
                let worst = per
                    .iter()
                    .filter_map(|v| v["delta_db"].as_f64().map(|d| (d, v)))
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|(_, v)| v.clone());
                deltas.push(json!({"reference":rows[reference]["name"],"aggregate_delta_db":q["aggregate"]["psnr"].as_f64().zip(base["aggregate"]["psnr"].as_f64()).map(|(a,b)|a-b),"per_texture":per,"worst":worst}));
            }
            rows[i]["comparison"] = json!(deltas);
        }
        formats.push(json!({"format":format,"rows":rows}));
    }
    if rgbbytes != fs::read(rgbpath.join("selected.json"))?
        || scalarbytes != fs::read(scalarpath.join("scalar-selected.json"))?
    {
        return Err("model artifact changed during frozen comparison".into());
    }
    fs::create_dir_all(&o.output)?;
    fs::write(
        o.output.join("bc2-5-report.json"),
        serde_json::to_vec_pretty(
            &json!({"gpu":gpu,"blocks":test.len(),"textures":o.corpus_groups.len(),"corpus":o.corpus_metadata,"rgb_model":rgbmeta,"scalar_model":scalarmeta,"frozen_artifacts_unchanged":true,"scalar_forward_max_deviation":scalar_forward_deviation,"scalar_parity":scalar_parity,"color_parity":color_parity,"repeats":o.repeats,"timing":"interleaved rotating methods; CPU wall-clock submission/wait, no GPU timestamps; host includes transfers; CPU baseline timing separate","scope":"RGB8 opaque color corpus; BC2/3 alpha constant 255 excluded from RGB PSNR; BC4 R and BC5 RG sampled quality, not natural alpha/normal-map corpus","current_baseline":"native GPU BC2 120 configs; BC3/4/5 experimental CPU algorithm ports, numerical parity qualified by reported checks, separate actual CPU rows; BC4/5 zero-index bug retained plus corrected controls","formats":formats}),
        )?,
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn perfect_bc4_quality() {
        let t = vec![[[0, 0, 0]; 16]];
        let b = [0; 16];
        assert_eq!(quality(&t, &[b], 4)["perfect"], true);
    }
    #[test]
    fn format_cli_separates_training_from_frozen_test() {
        let parse = |args: &[&str]| Options::parse_args(args.iter().map(|v| v.to_string()));
        assert!(parse(&["--formats-train", "--corpus-manifest", "manifest.json"]).is_ok());
        assert!(parse(&["--formats-compare", "--corpus-manifest", "manifest.json"]).is_err());
        assert!(
            parse(&[
                "--formats-train",
                "--formats-compare",
                "--corpus-manifest",
                "manifest.json"
            ])
            .is_err()
        );
        assert!(
            parse(&[
                "--formats-compare",
                "--corpus-manifest",
                "manifest.json",
                "--corpus-split",
                "test",
                "--load-models",
                "rgb",
                "--scalar-models",
                "scalar"
            ])
            .is_ok()
        );
        assert!(
            parse(&[
                "--formats-compare",
                "--corpus-manifest",
                "manifest.json",
                "--corpus-split",
                "test",
                "--load-models",
                "rgb",
                "--scalar-models",
                "scalar",
                "--steps",
                "1"
            ])
            .is_err()
        );
    }
}
