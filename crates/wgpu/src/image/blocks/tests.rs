use super::*;
use wgpu::util::DeviceExt;
include!("../block_tests.rs");

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    struct Wake(std::thread::Thread);
    impl std::task::Wake for Wake {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(Wake(std::thread::current())));
    let mut cx = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

#[test]
#[ignore = "requires a native GPU with immediate constants"]
fn public_routes_preserve_layout_alpha_and_quality() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("native adapter");
    eprintln!("wgpu GPU: {}", adapter.get_info().name);
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 48,
            ..Default::default()
        },
        ..Default::default()
    }))
    .expect("native device");
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
                wgpu::TextureFormat::Rgba8Unorm,
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test input"),
                    contents: bytemuck::cast_slice(pixels),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            ),
        );
        let dispatch_start = Instant::now();
        let out = match format {
            1 => compressor.compress_rgba_to_bc1(input, 0.5, &device, &queue, batch),
            2 => compressor.compress_rgba_to_bc2(input, &device, &queue, batch),
            3 => compressor.compress_rgba_to_bc3(input, &device, &queue, batch),
            4 => compressor.compress_rgba_to_bc4(input, &device, &queue, batch),
            5 => compressor.compress_rgba_to_bc5(input, &device, &queue, batch),
            _ => unreachable!(),
        };
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        if size == [512, 512, 1] {
            resident[format - 1].push(dispatch_start.elapsed().as_secs_f64() * 1000.);
        }
        let block_size = if format == 1 || format == 4 { 8 } else { 16 };
        assert_eq!(
            out.extent().raw_size(),
            [size[0].div_ceil(4), size[1].div_ceil(4), size[2]]
        );
        assert_eq!(out.row_stride() * block_size % 256, 0);
        let bytes = out.data().buffer.size();
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&out.data().buffer, 0, &read, 0, bytes);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = read.get_mapped_range(..).unwrap();
        let mut tight = Vec::new();
        for z in 0..size[2] {
            for y in 0..size[1].div_ceil(4) {
                let offset = (z * out.plane_stride() + y * out.row_stride()) * block_size;
                tight.extend_from_slice(&mapped[offset..offset + size[0].div_ceil(4) * block_size]);
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
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resident input"),
            contents: bytemuck::cast_slice(&pixels),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident output"),
            size: 128 * 128 * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident group"),
            layout: &compressor.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
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
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("resident test"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipelines[i]);
                    pass.set_bind_group(0, &group, &[]);
                    pass.set_immediates(0, bytemuck::bytes_of(&params));
                    pass.dispatch_workgroups(2, 128, 1);
                }
                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
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
