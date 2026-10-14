use jkl::image::block::{bc1, bc2, bc3, bc4, bc5};
use jkl::math::{R32F, Rg32F, Rgba32F};
use std::time::Instant;

fn fixture(size: [usize; 3], stride: [usize; 2]) -> Vec<u32> {
    let mut pixels = vec![0xff00ffff; stride[1] * size[2]];
    for z in 0..size[2] {
        for y in 0..size[1] {
            for x in 0..size[0] {
                let r = ((x * 5 + y * 3 + z * 37) % 256) as u8;
                let g = ((x * 2 + y * 7 + z * 53) % 256) as u8;
                let b = ((x * 3 + y * 4 + z * 19) % 256) as u8;
                let a = [0, 255, 51, 204][(x + 2 * y + z) % 4];
                pixels[z * stride[1] + y * stride[0] + x] = u32::from_le_bytes([r, g, b, a]);
            }
        }
    }
    pixels
}

fn decode(format: usize, bytes: &[u8]) -> [[[f32; 4]; 4]; 4] {
    match format {
        1 => bc1::Block::from_bytes(bytes.try_into().unwrap())
            .decode_with_alpha()
            .map(|r| r.map(|c| c.0)),
        2 => bc2::Block::from_bytes(bytes.try_into().unwrap())
            .decode_with_alpha()
            .map(|r| r.map(|c| c.0)),
        3 => bc3::Block::from_bytes(bytes.try_into().unwrap())
            .decode_with_alpha()
            .map(|r| r.map(|c| c.0)),
        4 => bc4::Block::from_bytes(bytes.try_into().unwrap())
            .decode()
            .map(|r| r.map(|c| [c.r(), 0., 0., 1.])),
        5 => bc5::Block::from_bytes(bytes.try_into().unwrap())
            .decode()
            .map(|r| r.map(|c| [c.r(), c.g(), 0., 1.])),
        _ => unreachable!(),
    }
}

fn cpu(format: usize, pixels: &[u32], size: [usize; 3], stride: [usize; 2]) -> Vec<u8> {
    let mut out = Vec::new();
    for z in 0..size[2] {
        for by in 0..size[1].div_ceil(4) {
            for bx in 0..size[0].div_ceil(4) {
                let c = std::array::from_fn(|y| {
                    std::array::from_fn(|x| {
                        let (x, y) = (bx * 4 + x, by * 4 + y);
                        if x >= size[0] || y >= size[1] {
                            (Rgba32F([0.; 4]), false)
                        } else {
                            (
                                Rgba32F(
                                    pixels[z * stride[1] + y * stride[0] + x]
                                        .to_le_bytes()
                                        .map(|v| v as f32 / 255.),
                                ),
                                true,
                            )
                        }
                    })
                });
                match format {
                    1 => out.extend(bc1::Block::encode_with_alpha(c, 0.5).bytes()),
                    2 => out.extend(bc2::Block::encode_with_alpha(c).bytes()),
                    3 => out.extend(bc3::Block::encode_with_alpha(c).bytes()),
                    4 => out.extend(
                        bc4::Block::encode(c.map(|r| r.map(|(c, v)| (R32F::new(c.r()), v))))
                            .bytes(),
                    ),
                    5 => out.extend(
                        bc5::Block::encode(
                            c.map(|r| r.map(|(c, v)| (Rg32F::new(c.r(), c.g()), v))),
                        )
                        .bytes(),
                    ),
                    _ => unreachable!(),
                }
            }
        }
    }
    out
}

fn error(
    format: usize,
    bytes: &[u8],
    pixels: &[u32],
    size: [usize; 3],
    stride: [usize; 2],
) -> (f64, usize) {
    let block_size = if format == 1 || format == 4 { 8 } else { 16 };
    let width = size[0].div_ceil(4);
    let height = size[1].div_ceil(4);
    let channels = if format == 4 {
        1
    } else if format == 5 {
        2
    } else {
        3
    };
    let mut sse = 0.;
    let mut count = 0;
    for z in 0..size[2] {
        for by in 0..height {
            for bx in 0..width {
                let offset = ((z * height + by) * width + bx) * block_size;
                let d = decode(format, &bytes[offset..offset + block_size]);
                for y in 0..4 {
                    for x in 0..4 {
                        let (px, py) = (bx * 4 + x, by * 4 + y);
                        if px >= size[0] || py >= size[1] {
                            continue;
                        }
                        let src = pixels[z * stride[1] + py * stride[0] + px].to_le_bytes();
                        if format == 1 {
                            assert_eq!(d[y][x][3], if src[3] >= 128 { 1. } else { 0. });
                        }
                        if format == 2 {
                            assert!((d[y][x][3] - src[3] as f32 / 255.).abs() <= 0.5 / 15. + 1e-6);
                        }
                        if format == 3 {
                            assert!((d[y][x][3] - src[3] as f32 / 255.).abs() <= 0.15);
                        }
                        if format == 1 && src[3] < 128 {
                            continue;
                        }
                        for c in 0..channels {
                            sse += (d[y][x][c] as f64 - src[c] as f64 / 255.).powi(2);
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    (sse, count)
}

fn suite(mut compress: impl FnMut(usize, &[u32], [usize; 3], [usize; 2], u32) -> Vec<u8>) {
    for pixel in [0u32, 0xff000000, 0xffffffff, 0xff773311, 0x66773311] {
        for format in 1..=5 {
            let out = compress(format, &[pixel], [1, 1, 1], [1, 1], 1);
            let (sse, n) = error(format, &out, &[pixel], [1, 1, 1], [1, 1]);
            assert!(sse < 0.004 * n as f64 + 1e-7, "flat BC{format}");
        }
    }
    let size = [263, 7, 2];
    let stride = [268, 268 * 9 + 3];
    let pixels = fixture(size, stride);
    for format in 1..=5 {
        let reference = compress(format, &pixels, size, stride, 10000);
        for batch in [1, 63, 65] {
            assert_eq!(
                compress(format, &pixels, size, stride, batch),
                reference,
                "format {format}, batch {batch}"
            );
        }
        let (sse, count) = error(format, &reference, &pixels, size, stride);
        assert!(
            sse / (count as f64) < 0.01,
            "BC{format} excessive error {sse}"
        );
    }
    let size = [512, 512, 1];
    let stride = [512, 512 * 512];
    let pixels = fixture(size, stride);
    for format in 1..=5 {
        let start = Instant::now();
        let reference = cpu(format, &pixels, size, stride);
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut times = Vec::new();
        let mut output = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            output = compress(format, &pixels, size, stride, 65536);
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
        times.sort_by(f64::total_cmp);
        let (cpu_sse, n) = error(format, &reference, &pixels, size, stride);
        let (gpu_sse, _) = error(format, &output, &pixels, size, stride);
        assert!(
            gpu_sse <= cpu_sse * 1.1 + 1e-4,
            "BC{format} GPU quality differs: CPU {cpu_sse}, GPU {gpu_sse}"
        );
        eprintln!(
            "BC{format}: CPU {cpu_ms:.3} ms, GPU upload+submit+wait+read median {:.3} ms, CPU {:.5} dB, GPU {:.5} dB",
            times[1],
            10. * (n as f64 / cpu_sse).log10(),
            10. * (n as f64 / gpu_sse).log10()
        );
    }
}
