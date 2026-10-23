//! Explicit release quality/throughput audit; see data/bc7-reference.md.
use jkl::{image::block::bc7::Block, math::Rgba32F};
use std::{io::Write, time::Instant};

const NAMES: [&str; 8] = [
    "flat",
    "ramp",
    "two groups",
    "three groups",
    "independent alpha",
    "correlated",
    "random",
    "representable",
];

#[test]
#[ignore = "release encoder quality/throughput audit; writes differential records"]
fn independent_reference_quality_and_export() {
    let records: Vec<_> = include_bytes!("data/bc7-quality.bin")
        .chunks_exact(148)
        .collect();
    let inputs: Vec<_> = records
        .iter()
        .map(|r| {
            std::array::from_fn(|y| {
                std::array::from_fn(|x| {
                    (
                        Rgba32F(std::array::from_fn(|c| {
                            f32::from(r[4 + (y * 4 + x) * 4 + c]) / 255.0
                        })),
                        true,
                    )
                })
            })
        })
        .collect();
    let started = Instant::now();
    let encoded: Vec<_> = inputs.into_iter().map(Block::encode).collect();
    let elapsed = started.elapsed();
    let path = std::env::var_os("BC7_EXPORT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/bc7-encoded.bin")
        });
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut out = std::fs::File::create(&path).unwrap();
    let mut own = [0u64; 8];
    let mut reference = [0u64; 8];
    let mut wins = [0; 8];
    let mut exact = [0; 8];
    let mut modes = [0; 8];
    for (block, record) in encoded.iter().zip(&records) {
        let category = record[0] as usize;
        let decoded: Vec<u8> = block
            .decode()
            .iter()
            .flatten()
            .flat_map(|p| p.0.map(|c| (c * 255.0).round() as u8))
            .collect();
        let sse = |bytes: &[u8]| -> u64 {
            record[4..68]
                .iter()
                .zip(bytes)
                .map(|(&a, &b)| {
                    let delta = i64::from(a) - i64::from(b);
                    (delta * delta) as u64
                })
                .sum()
        };
        let actual = sse(&decoded);
        let baseline = sse(&record[84..]);
        if category == 7 {
            println!(
                "representable source mode {}: JKL={actual} reference={baseline}",
                record[1]
            );
        }
        own[category] += actual;
        reference[category] += baseline;
        wins[category] += usize::from(actual <= baseline);
        exact[category] += usize::from(actual == 0);
        modes[block.bytes()[0].trailing_zeros() as usize] += 1;
        out.write_all(&block.bytes()).unwrap();
        out.write_all(&decoded).unwrap();
    }
    for i in 0..8 {
        println!(
            "{}: JKL={} reference={} ratio={:.6} at_least_as_good={}/8 exact={}/8",
            NAMES[i],
            own[i],
            reference[i],
            own[i] as f64 / reference[i].max(1) as f64,
            wins[i],
            exact[i]
        );
    }
    println!(
        "total JKL={} reference={} ratio={:.6}",
        own.iter().sum::<u64>(),
        reference.iter().sum::<u64>(),
        own.iter().sum::<u64>() as f64 / reference.iter().sum::<u64>() as f64
    );
    println!(
        "encoder-only {:?}; {:.3} ms/block; modes {:?}; export {}",
        elapsed,
        elapsed.as_secs_f64() * 1000.0 / 64.0,
        modes,
        path.display()
    );
    // Partial masks stress independent anchor/end-point inversion choices.
    let fixtures = include_bytes!("data/bc7-reference.bin");
    for case in 0..32 {
        let source = &fixtures[(case * 137) * 80 + 16..(case * 137 + 1) * 80];
        let input = std::array::from_fn(|y| {
            std::array::from_fn(|x| {
                let i = y * 4 + x;
                (
                    Rgba32F(std::array::from_fn(|c| {
                        f32::from(source[i * 4 + c]) / 255.0
                    })),
                    (i * 13 + case * 7) % 17 < case % 16 + 1,
                )
            })
        });
        let block = Block::encode(input);
        let decoded: Vec<u8> = block
            .decode()
            .iter()
            .flatten()
            .flat_map(|p| p.0.map(|c| (c * 255.0).round() as u8))
            .collect();
        out.write_all(&block.bytes()).unwrap();
        out.write_all(&decoded).unwrap();
    }
    // Corpus-scoped measured ceilings, not a claim of optimal BC7 quality.
    let ceilings = [0, 206, 1054, 1533, 33440, 3232, 898033, 477];
    for i in 0..8 {
        assert!(
            own[i] <= ceilings[i],
            "{} SSE {} exceeds measured baseline {}",
            NAMES[i],
            own[i],
            ceilings[i]
        );
    }
}
