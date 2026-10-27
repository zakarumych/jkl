//! Compare native GPU decompression against the production CPU decoder.
use jkl::jackal::mesh::JackalMeshReader;
use jkl_wgpu::mesh::Uploader;
use std::{error::Error, fs::File, future::Future};

fn block_on<F: Future>(future: F) -> F::Output {
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
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}
fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mesh validation readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap()
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = staging.get_mapped_range(..).unwrap();
    bytemuck::cast_slice(&mapped).to_vec()
}
#[derive(Default)]
struct Stats {
    sum: f64,
    max: f64,
    count: usize,
    worst: usize,
}
impl Stats {
    fn push(&mut self, index: usize, error: f64) {
        assert!(error.is_finite(), "nonfinite GPU difference at {index}");
        self.sum += error;
        self.count += 1;
        if error > self.max {
            self.max = error;
            self.worst = index;
        }
    }
    fn report(&self, label: &str) {
        println!(
            "{label} mean={:.12} max={:.12} worst_vertex={}",
            self.sum / self.count.max(1) as f64,
            self.max,
            self.worst
        );
    }
}
fn angular(a: [f32; 3], b: [f32; 3]) -> f64 {
    let a = a.map(f64::from);
    let b = b.map(f64::from);
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    cross
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        .atan2((0..3).map(|k| a[k] * b[k]).sum::<f64>())
        .to_degrees()
}
fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: mesh_validate INPUT.jklm")?;
    let mut reader = JackalMeshReader::open(File::open(path)?)?;
    let expected = reader.read_mesh()?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        backend_options: Default::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
    });
    let adapter = block_on(instance.request_adapter(&Default::default()))?;
    println!("adapter={:?}", adapter.get_info());
    let (device, queue) = block_on(adapter.request_device(&Default::default()))?;
    let uploader = Uploader::new(&device);
    let mut encoder = device.create_command_encoder(&Default::default());
    let mesh = uploader.upload_from_reader(&mut reader, &device, &mut encoder)?;
    queue.submit([encoder.finish()]);
    let vertices = read(&device, &queue, &mesh.vertices);
    let indices = read(&device, &queue, &mesh.indices);
    assert_eq!(
        &indices[..expected.triangles.len() * 3],
        bytemuck::cast_slice::<[u32; 3], u32>(&expected.triangles)
    );
    assert_eq!(
        &indices[expected.triangles.len() * 3..expected.triangles.len() * 4],
        &expected.primitive_ids
    );
    assert_eq!(mesh.materials, expected.materials);
    let mut pos = Stats::default();
    let mut normal = Stats::default();
    let mut tangent = Stats::default();
    let mut uv: Vec<_> = expected
        .tex_coords
        .iter()
        .map(|_| Stats::default())
        .collect();
    let mut n_component = Stats::default();
    let mut t_component = Stats::default();
    for i in 0..expected.positions.len() {
        let base = i * mesh.layout.stride as usize / 4;
        let get = |offset: u32, k: usize| f32::from_bits(vertices[base + offset as usize / 4 + k]);
        let p: [f32; 3] = std::array::from_fn(|k| get(mesh.layout.position, k));
        pos.push(
            i,
            (0..3)
                .map(|k| (p[k] as f64 - expected.positions[i][k] as f64).powi(2))
                .sum::<f64>()
                .sqrt(),
        );
        if let Some(values) = &expected.normals {
            let n = std::array::from_fn(|k| get(mesh.layout.normal.unwrap(), k));
            normal.push(i, angular(values[i], n));
            n_component.push(
                i,
                (0..3)
                    .map(|k| (n[k] - values[i][k]).abs() as f64)
                    .fold(0., f64::max),
            );
        }
        if let Some(values) = &expected.tangents {
            let t = std::array::from_fn(|k| get(mesh.layout.tangent.unwrap(), k));
            tangent.push(i, angular([values[i][0], values[i][1], values[i][2]], t));
            t_component.push(
                i,
                (0..3)
                    .map(|k| (t[k] - values[i][k]).abs() as f64)
                    .fold(0., f64::max),
            );
            assert_eq!(get(mesh.layout.tangent.unwrap(), 3), values[i][3]);
        }
        for (set, values) in expected.tex_coords.iter().enumerate() {
            uv[set].push(
                i,
                (0..2)
                    .map(|k| {
                        (get(mesh.layout.tex_coords[set], k) as f64 - values[i][k] as f64).powi(2)
                    })
                    .sum::<f64>()
                    .sqrt(),
            );
        }
        if let Some(values) = &expected.colors {
            assert_eq!(
                vertices[base + mesh.layout.color.unwrap() as usize / 4],
                u32::from_le_bytes(values[i])
            );
        }
    }
    println!(
        "vertices={} triangles={} exact_indices=true exact_primitive_ids=true exact_colors=true exact_handedness=true",
        expected.positions.len(),
        expected.triangles.len()
    );
    pos.report("position_difference");
    normal.report("normal_difference_degrees");
    tangent.report("tangent_difference_degrees");
    n_component.report("normal_component_difference");
    t_component.report("tangent_component_difference");
    for (i, s) in uv.iter().enumerate() {
        s.report(&format!("uv{i}_difference"));
    }
    assert!(pos.max < 2e-5, "position CPU/GPU mismatch");
    assert!(uv.iter().all(|s| s.max < 2e-5), "UV CPU/GPU mismatch");
    assert!(n_component.max < 0.002, "normal CPU/GPU mismatch");
    assert!(t_component.max < 0.002, "tangent CPU/GPU mismatch");
    println!("validation_passed=true");
    Ok(())
}
