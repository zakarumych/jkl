//! Independent quality measurements; corpus provenance is in data/bc6-quality.md.
use jkl::{image::block::bc6::Block, math::Rgb32F};

const DATA: &[u8] = include_bytes!("data/bc6-quality.bin");
const RECORD_SIZE: usize = 228;
const NAMES: [&str; 9] = [
    "half-code gradient",
    "two groups",
    "correlated noise",
    "independent noise",
    "three clusters",
    "subnormal/sign",
    "narrow bright",
    "linear gradient",
    "known optimum",
];

struct Case {
    signed: bool,
    category: usize,
    ordinal: usize,
    input: [[(Rgb32F, bool); 4]; 4],
    codes: [i32; 48],
    reference: Block,
    reference_codes: [i32; 48],
    exact: Block,
}

fn half_float(half: u16) -> f32 {
    let sign = u32::from(half & 0x8000) << 16;
    let exponent = u32::from(half >> 10 & 31);
    let mantissa = u32::from(half & 1023);
    if exponent == 0 {
        let value = mantissa as f32 / 16777216.0;
        if sign != 0 { -value } else { value }
    } else {
        f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13))
    }
}

fn code(half: u16) -> i32 {
    let magnitude = i32::from(half & 0x7fff);
    if half & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

// Decoder outputs are exactly binary16, so inversion requires no rounding.
fn decoded_code(value: f32) -> i32 {
    let bits = value.to_bits();
    let magnitude = if value.abs() < 2.0f32.powi(-14) {
        (value.abs() * 16777216.0) as i32
    } else {
        (((bits >> 23 & 255) - 112) << 10 | ((bits & 0x7fffff) >> 13)) as i32
    };
    if bits >> 31 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn corpus() -> Vec<Case> {
    assert_eq!(DATA.len(), 184 * RECORD_SIZE);
    DATA.chunks_exact(RECORD_SIZE)
        .map(|record| {
            let halves = |offset: usize| {
                std::array::from_fn::<_, 48, _>(|i| {
                    u16::from_le_bytes(
                        record[offset + 2 * i..offset + 2 * i + 2]
                            .try_into()
                            .unwrap(),
                    )
                })
            };
            let input = halves(4);
            Case {
                signed: record[0] != 0,
                category: record[1] as usize,
                ordinal: u16::from_le_bytes(record[2..4].try_into().unwrap()) as usize,
                input: std::array::from_fn(|y| {
                    std::array::from_fn(|x| {
                        (
                            Rgb32F(std::array::from_fn(|c| {
                                half_float(input[(y * 4 + x) * 3 + c])
                            })),
                            true,
                        )
                    })
                }),
                codes: input.map(code),
                reference: Block::from_bytes(record[100..116].try_into().unwrap()),
                reference_codes: halves(116).map(code),
                exact: Block::from_bytes(record[212..228].try_into().unwrap()),
            }
        })
        .collect()
}

fn decoded(block: Block, signed: bool) -> [i32; 48] {
    let rgb = if signed {
        block.decode_signed()
    } else {
        block.decode()
    };
    std::array::from_fn(|i| decoded_code(rgb[i / 12][i / 3 % 4].0[i % 3]))
}

fn sse(input: &[i32; 48], output: &[i32; 48]) -> u64 {
    input
        .iter()
        .zip(output)
        .map(|(&a, &b)| i64::from(a - b).pow(2) as u64)
        .sum()
}

#[test]
fn independent_reference_decoding_and_known_optimum() {
    for case in corpus() {
        assert_eq!(
            decoded(case.reference, case.signed),
            case.reference_codes,
            "reference decode: signed={} category={} ordinal={}",
            case.signed,
            case.category,
            case.ordinal
        );
        if case.category == 8 {
            assert_eq!(
                decoded(case.exact, case.signed),
                case.codes,
                "known exact decode: signed={} ordinal={}",
                case.signed,
                case.ordinal
            );
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Stats {
    count: u64,
    production: u64,
    reference: u64,
    worse: u64,
    exact: u64,
    worst_ratio: f64,
    worst_sse: u64,
    worst_case: usize,
    variance: f64,
    worst_variance_ratio: f64,
}

/// Prints independent error measurements. Use release mode to evaluate the encoder.
#[test]
#[ignore = "expensive independent quality measurement"]
fn independent_quality_report() {
    let cases = corpus();
    let mut stats = [[Stats::default(); 9]; 2];
    for case in &cases {
        let block = if case.signed {
            Block::encode_signed(case.input)
        } else {
            Block::encode(case.input)
        };
        let production = sse(&case.codes, &decoded(block, case.signed));
        let reference = sse(&case.codes, &case.reference_codes);
        let stat = &mut stats[usize::from(case.signed)][case.category];
        stat.count += 1;
        stat.production += production;
        stat.reference += reference;
        stat.worse += u64::from(production > reference);
        stat.exact += u64::from(production == 0);
        let variance = (0..3)
            .map(|c| {
                let mean = (0..16)
                    .map(|i| f64::from(case.codes[i * 3 + c]))
                    .sum::<f64>()
                    / 16.0;
                (0..16)
                    .map(|i| (f64::from(case.codes[i * 3 + c]) - mean).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>();
        stat.variance += variance;
        let variance_ratio = if variance == 0.0 {
            if production == 0 { 0.0 } else { f64::INFINITY }
        } else {
            (production as f64 / variance).sqrt()
        };
        stat.worst_variance_ratio = stat.worst_variance_ratio.max(variance_ratio);
        if case.category == 8 || production > reference {
            let configuration = |block: Block| {
                let mode = if block.data[0] & 3 < 2 {
                    block.data[0] & 3
                } else {
                    block.data[0] & 31
                };
                let partition = if matches!(mode, 3 | 7 | 11 | 15) {
                    0
                } else {
                    (block.data[9] >> 5) | ((block.data[10] & 3) << 3)
                };
                (mode, partition)
            };
            eprintln!(
                "case {} category={} ordinal={} SSE={} ref={} RMSE={:.6} variance_RMSE={:.6} source={:?} output={:?}",
                if case.signed { "SF16" } else { "UF16" },
                case.category,
                case.ordinal,
                production,
                reference,
                (production as f64 / 48.0).sqrt(),
                variance_ratio,
                configuration(case.exact),
                configuration(block)
            );
        }
        let ratio = if reference == 0 {
            if production == 0 { 1.0 } else { f64::INFINITY }
        } else {
            production as f64 / reference as f64
        };
        stat.worst_ratio = stat.worst_ratio.max(ratio);
        if production > stat.worst_sse {
            stat.worst_sse = production;
            stat.worst_case = case.ordinal;
        }
    }
    for (signed, categories) in stats.iter().enumerate() {
        for (category, stat) in categories.iter().enumerate() {
            let rmse = (stat.production as f64 / (48 * stat.count) as f64).sqrt();
            eprintln!(
                "{} {} n={} SSE={} ref={} ratio={:.6} worse={} exact={} worst_ratio={:.6} worst_SSE={} ordinal={} RMSE={:.6} normalized_RMSE={:.9} variance_RMSE={:.6} worst_variance_RMSE={:.6}",
                if signed == 0 { "UF16" } else { "SF16" },
                NAMES[category],
                stat.count,
                stat.production,
                stat.reference,
                stat.production as f64 / stat.reference.max(1) as f64,
                stat.worse,
                stat.exact,
                stat.worst_ratio,
                stat.worst_sse,
                stat.worst_case,
                rmse,
                rmse / 31743.0,
                (stat.production as f64 / stat.variance).sqrt(),
                stat.worst_variance_ratio
            );
        }
    }
    // Independent reference quality is a regression floor for every cohort.
    // This comparison does not bound the distance from the global optimum.
    for categories in stats {
        for (category, stat) in categories.into_iter().enumerate() {
            if category == 8 {
                assert_eq!(stat.count, 28, "known-optimum corpus size");
                assert_eq!(
                    stat.production, 0,
                    "encoder fails to recover known global optimum"
                );
                assert_eq!(stat.exact, 28, "known-optimum blocks recovered exactly");
            }
            assert!(
                stat.production <= stat.reference,
                "{} cohort exceeds independent reference SSE: {} > {}",
                NAMES[category],
                stat.production,
                stat.reference
            );
        }
    }
}

/// Timing includes only calls to the production encoder.
#[test]
#[ignore = "manual release encoder timing"]
fn encoder_timing() {
    use std::{hint::black_box, time::Instant};
    let cases = corpus();
    let start = Instant::now();
    for case in &cases {
        let input = black_box(case.input);
        black_box(if case.signed {
            Block::encode_signed(input)
        } else {
            Block::encode(input)
        });
    }
    let elapsed = start.elapsed();
    eprintln!(
        "{} BC6H blocks in {:?}: {:.3} ms/block",
        cases.len(),
        elapsed,
        elapsed.as_secs_f64() * 1000.0 / cases.len() as f64
    );
}

/// Audits all independent representable blocks, including every partition.
#[test]
#[ignore = "expensive full representable input audit"]
fn full_reference_representable_audit() {
    const REFERENCE: &[u8] = include_bytes!("data/bc6-reference.bin");
    const MODES: [u8; 14] = [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15];
    assert_eq!(REFERENCE.len(), 476 * 208);
    let mut stats = [[Stats::default(); 14]; 2];
    let mut skipped = [0usize; 2];
    let mut residuals = Vec::new();
    for (ordinal, record) in REFERENCE.chunks_exact(208).enumerate() {
        if ordinal != 0 && ordinal % 50 == 0 {
            eprintln!(
                "full representable audit: {} / 952 source interpretations processed",
                ordinal * 2
            );
        }
        for signed in 0..2 {
            let halves = std::array::from_fn::<_, 48, _>(|i| {
                let offset = 16 + signed * 96 + i * 2;
                u16::from_le_bytes(record[offset..offset + 2].try_into().unwrap())
            });
            // Encoding clamps nonfinite values, so those inputs lack this zero-error witness.
            if halves.iter().any(|half| half & 0x7c00 == 0x7c00) {
                skipped[signed] += 1;
                continue;
            }
            let codes = halves.map(code);
            let source = Block::from_bytes(record[..16].try_into().unwrap());
            assert_eq!(decoded(source, signed != 0), codes);
            let input = std::array::from_fn(|y| {
                std::array::from_fn(|x| {
                    (
                        Rgb32F(std::array::from_fn(|c| {
                            half_float(halves[(y * 4 + x) * 3 + c])
                        })),
                        true,
                    )
                })
            });
            let output = if signed != 0 {
                Block::encode_signed(input)
            } else {
                Block::encode(input)
            };
            let error = sse(&codes, &decoded(output, signed != 0));
            let variance = (0..3)
                .map(|c| {
                    let mean = (0..16).map(|i| f64::from(codes[i * 3 + c])).sum::<f64>() / 16.0;
                    (0..16)
                        .map(|i| (f64::from(codes[i * 3 + c]) - mean).powi(2))
                        .sum::<f64>()
                })
                .sum::<f64>();
            let stat = &mut stats[signed][ordinal / 34];
            stat.count += 1;
            stat.exact += u64::from(error == 0);
            stat.production += error;
            stat.variance += variance;
            if error > stat.worst_sse {
                stat.worst_sse = error;
                stat.worst_case = ordinal;
            }
            if error != 0 {
                residuals.push((error, signed, ordinal, variance));
            }
        }
    }
    for (signed, modes) in stats.iter().enumerate() {
        let mut total = Stats::default();
        for (mode, stat) in modes.iter().enumerate() {
            total.count += stat.count;
            total.exact += stat.exact;
            total.production += stat.production;
            total.variance += stat.variance;
            eprintln!(
                "{} source_mode={} count={} exact={} SSE={} worst_SSE={} ordinal={}",
                if signed == 0 { "UF16" } else { "SF16" },
                MODES[mode],
                stat.count,
                stat.exact,
                stat.production,
                stat.worst_sse,
                stat.worst_case
            );
        }
        let rmse = (total.production as f64 / (48 * total.count) as f64).sqrt();
        eprintln!(
            "{} full reference count={} skipped_nonfinite={} exact={} SSE={} RMSE={:.6} normalized_RMSE={:.9} variance_RMSE={:.6}",
            if signed == 0 { "UF16" } else { "SF16" },
            total.count,
            skipped[signed],
            total.exact,
            total.production,
            rmse,
            rmse / 31743.0,
            (total.production as f64 / total.variance).sqrt()
        );
    }
    residuals.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    for (error, signed, ordinal, variance) in residuals.into_iter().take(10) {
        eprintln!(
            "residual {} ordinal={} source_mode={} partition_pattern={} SSE={} RMSE={:.6} variance_RMSE={:.6}",
            if signed == 0 { "UF16" } else { "SF16" },
            ordinal,
            MODES[ordinal / 34],
            ordinal % 34,
            error,
            (error as f64 / 48.0).sqrt(),
            (error as f64 / variance).sqrt()
        );
    }
    assert_eq!(skipped, [0, 0], "finite representable fixture coverage");
    for modes in stats {
        for (mode, stat) in modes.into_iter().enumerate() {
            assert_eq!(stat.count, 34, "source mode {} coverage", MODES[mode]);
            assert_eq!(stat.exact, 34, "source mode {} exact recovery", MODES[mode]);
            assert_eq!(
                stat.production, 0,
                "source mode {} global optimum",
                MODES[mode]
            );
        }
    }
}
