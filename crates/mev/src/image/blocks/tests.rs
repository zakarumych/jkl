use super::*;
include!("../../../../wgpu/src/image/block_tests.rs");

#[test]
#[ignore = "requires a native GPU"]
fn public_routes_preserve_layout_alpha_and_quality() {
    let instance = mev::Instance::load().expect("native instance");
    let (idx, family) = instance
        .capabilities()
        .devices
        .iter()
        .enumerate()
        .find_map(|(idx, caps)| {
            caps.families
                .iter()
                .position(|f| f.queue_flags.contains(mev::QueueFlags::COMPUTE))
                .map(|f| (idx, f as u32))
        })
        .expect("native compute queue");
    eprintln!("mev GPU: {}", instance.capabilities().devices[idx].name);
    let features = instance.capabilities().devices[idx].features
        & (mev::Features::BUFFER_MAP | mev::Features::BUFFER_MAP_AT_CREATION);
    let (device, mut queues) = instance
        .new_device(mev::DeviceDesc {
            idx,
            queues: &[family],
            features,
        })
        .expect("native device");
    let mut queue = queues.pop().unwrap();
    let start = Instant::now();
    let compressor = BlockCompressor::new(&device);
    eprintln!(
        "pipeline construction {:.3} ms",
        start.elapsed().as_secs_f64() * 1000.
    );
    let mut resident = vec![Vec::<f64>::new(); 5];
    suite(|format, pixels, size, stride, batch| {
        let input = Image::with_stride(
            jkl::image::Dimensions::D2Array,
            size,
            stride,
            GpuPixels::new(
                mev::PixelFormat::Rgba8Unorm,
                device.new_buffer_init(mev::BufferInitDesc {
                    name: "test input",
                    data: pixels,
                    usage: mev::BufferUsage::STORAGE,
                }),
            ),
        );
        let usage = mev::BufferUsage::HOST_READ;
        let dispatch_start = Instant::now();
        let mut out = match format {
            1 => compressor.compress_rgba_to_bc1(input, 0.5, &device, &mut queue, batch, usage),
            2 => compressor.compress_rgba_to_bc2(input, &device, &mut queue, batch, usage),
            3 => compressor.compress_rgba_to_bc3(input, &device, &mut queue, batch, usage),
            4 => compressor.compress_rgba_to_bc4(input, &device, &mut queue, batch, usage),
            5 => compressor.compress_rgba_to_bc5(input, &device, &mut queue, batch, usage),
            _ => unreachable!(),
        };
        let checkpoint = queue.new_command_encoder().finish();
        queue
            .submit_checkpoint([checkpoint])
            .expect("close compression epoch");
        queue.wait_idle().expect("wait for compression");
        if size == [512, 512, 1] {
            resident[format - 1].push(dispatch_start.elapsed().as_secs_f64() * 1000.);
        }
        let block_size = if format == 1 || format == 4 { 8 } else { 16 };
        assert_eq!(
            out.extent().raw_size(),
            [size[0].div_ceil(4), size[1].div_ceil(4), size[2]]
        );
        assert_eq!(out.row_stride() * block_size % 256, 0);
        let mut raw = vec![0u8; out.data().buffer.size()];
        out.data_mut()
            .buffer
            .read(0, &mut raw)
            .expect("read compressed blocks");
        let mut tight = Vec::new();
        for z in 0..size[2] {
            for y in 0..size[1].div_ceil(4) {
                let offset = (z * out.plane_stride() + y * out.row_stride()) * block_size;
                tight.extend_from_slice(&raw[offset..offset + size[0].div_ceil(4) * block_size]);
            }
        }
        tight
    });
    for (i, times) in resident.iter_mut().enumerate() {
        times.sort_by(f64::total_cmp);
        eprintln!(
            "BC{} public route submit+wait median {:.3} ms",
            i + 1,
            times[1]
        );
    }
    queue.wait_idle().unwrap();
    let pipelines = [
        &compressor.bc1,
        &compressor.bc2,
        &compressor.bc3,
        &compressor.bc4,
        &compressor.bc5,
    ];
    for opaque in [false, true] {
        let mut pixels = fixture([512, 512, 1], [512, 512 * 512]);
        if opaque {
            for pixel in &mut pixels {
                *pixel |= 0xff000000;
            }
        }
        let input = device.new_buffer_init(mev::BufferInitDesc {
            name: "resident input",
            data: &pixels,
            usage: mev::BufferUsage::STORAGE,
        });
        let output = device.new_buffer(mev::BufferDesc {
            name: "resident output",
            size: 128 * 128 * 16,
            usage: mev::BufferUsage::STORAGE,
        });
        let group = BlockCompressorArguments {
            image_input: input,
            image_output: output,
        };
        let params = Params {
            in_width: 512,
            in_height: 512,
            layers: 1,
            in_row_stride: 512,
            in_plane_stride: 512 * 512,
            out_row_stride: 128,
            out_plane_stride: 128 * 128,
            alpha_threshold: 0.5,
            x_offset: 0,
            y_offset: 0,
            z_offset: 0,
            batch_width: 128,
        };
        let mut times = vec![Vec::<f64>::new(); 5];
        for round in 0..7 {
            for step in 0..5 {
                let i = (round + step) % 5;
                let start = Instant::now();
                let mut encoder = queue.new_command_encoder();
                {
                    let mut pass = encoder.compute();
                    pass.with_pipeline(pipelines[i]);
                    pass.with_arguments(0, &group);
                    pass.with_constants(&params);
                    pass.dispatch(mev::Extent3::new(2, 128, 1));
                }
                queue.submit_checkpoint([encoder.finish()]).unwrap();
                queue.wait_idle().unwrap();
                if round >= 2 {
                    times[i].push(start.elapsed().as_secs_f64() * 1000.);
                }
            }
        }
        for (i, time) in times.iter_mut().enumerate() {
            time.sort_by(f64::total_cmp);
            eprintln!(
                "BC{} reused resident variable_alpha={} median {:.3} ms (5 rotating rounds)",
                i + 1,
                !opaque,
                time[2]
            );
        }
    }
}
