//! Parallel patch decompression into renderer-ready vertex and index buffers.
use jkl::{
    jackal::mesh::JackalMeshReader,
    mesh::gpu::{self, VertexLayout},
};
use std::io::{self, Read, Seek};
use wgpu::util::DeviceExt;

/// Interleaved attributes and global u32 triangle indices, with retained primitive metadata.
pub struct GpuMesh {
    pub vertices: wgpu::Buffer,
    /// Triangle indices occupy `0..triangle_count*12`; primitive IDs follow them.
    pub indices: wgpu::Buffer,
    pub layout: VertexLayout,
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub materials: Vec<i32>,
}
impl GpuMesh {
    pub fn primitive_offset(&self) -> u64 {
        self.triangle_count as u64 * 12
    }
}
/// One GPU invocation decodes each patch. Payload content must be trusted or CPU validated.
pub struct Uploader {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}
impl Uploader {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-mesh-decoder"),
            source: wgpu::ShaderSource::Wgsl(gpu::DECODER_WGSL.into()),
        });
        let entries = std::array::from_fn::<_, 6, _>(|binding| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: binding < 3,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jkl-mesh-bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jkl-mesh-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-mesh-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("decode_mesh"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }
    /// Prepares file metadata, uploads compressed words, and records patch decoding.
    /// Outputs have STORAGE, COPY_SRC, and VERTEX/INDEX usage for direct consumers.
    pub fn upload_from_reader<R: Read + Seek>(
        &self,
        reader: &mut JackalMeshReader<R>,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> io::Result<GpuMesh> {
        let prepared = reader.prepare_gpu()?;
        let streams: Vec<_> = prepared
            .streams
            .iter()
            .map(|s| {
                [
                    s.payload_offset,
                    s.payload_len,
                    s.table_offset,
                    s.table_len,
                    s.symbol_count,
                ]
            })
            .collect();
        let (metadata, layout) = gpu::metadata(
            &prepared.schema,
            &prepared.patches,
            &streams,
            prepared.vertex_count,
            prepared.triangle_count,
        )
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut tables: Vec<_> = prepared
            .table
            .iter()
            .flat_map(|e| [e.freq, e.cumul, e.symbol])
            .collect();
        if tables.is_empty() {
            tables.push(0);
        }
        let mut payload = prepared.payload;
        if payload.is_empty() {
            payload.push(0);
        }
        let vertex_bytes = (prepared.vertex_count as u64 * layout.stride as u64).max(4);
        let index_bytes = (prepared.triangle_count as u64 * 16).max(4);
        let scratch_bytes = (prepared.vertex_count as u64 * 12).max(4);
        let groups = u32::try_from(prepared.patches.len().div_ceil(64))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "GPU patch count overflow"))?;
        let limits = device.limits();
        if groups > limits.max_compute_workgroups_per_dimension
            || [
                vertex_bytes,
                index_bytes,
                scratch_bytes,
                metadata.len() as u64 * 4,
                tables.len() as u64 * 4,
                payload.len() as u64 * 4,
            ]
            .into_iter()
            .any(|n| {
                n > limits.max_buffer_size
                    || n > limits.max_storage_buffer_binding_size as u64
                    || n / 4 > u32::MAX as u64
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mesh exceeds GPU device limits",
            ));
        }
        let output = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | usage,
                mapped_at_creation: false,
            })
        };
        let vertices = output(
            "jkl-mesh-vertices",
            vertex_bytes,
            wgpu::BufferUsages::VERTEX,
        );
        let indices = output("jkl-mesh-indices", index_bytes, wgpu::BufferUsages::INDEX);
        if !prepared.patches.is_empty() {
            let scratch = output(
                "jkl-mesh-scratch",
                scratch_bytes,
                wgpu::BufferUsages::empty(),
            );
            let input = |label, words: &[u32]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(words),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let payload = input("jkl-mesh-payload", &payload);
            let metadata = input("jkl-mesh-metadata", &metadata);
            let tables = input("jkl-mesh-tables", &tables);
            let buffers = [&payload, &metadata, &tables, &vertices, &indices, &scratch];
            let entries = std::array::from_fn::<_, 6, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffers[i].as_entire_binding(),
            });
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jkl-mesh-decode-bindings"),
                layout: &self.layout,
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jkl-mesh-decode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        Ok(GpuMesh {
            vertices,
            indices,
            layout,
            vertex_count: prepared.vertex_count,
            triangle_count: prepared.triangle_count,
            materials: prepared.materials,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jkl::{
        jackal::mesh::write_mesh,
        mesh::data::{EncodingOptions, MeshData},
    };
    use std::io::Cursor;
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
    fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<u32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh readback"),
            size: source.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = staging.get_mapped_range(..).unwrap();
        bytemuck::cast_slice(&mapped).to_vec()
    }
    fn fixture() -> MeshData {
        let mut mesh = MeshData {
            normals: Some(Vec::new()),
            tangents: Some(Vec::new()),
            tex_coords: vec![Vec::new(), Vec::new()],
            colors: Some(Vec::new()),
            materials: vec![4, 7],
            ..Default::default()
        };
        for y in 0..9 {
            for x in 0..9 {
                let a = x as f32 * 0.17;
                let b = y as f32 * 0.13;
                mesh.positions
                    .push([a, b, (a * 1.3).sin() * 0.3 + (b * 0.9).cos() * 0.2]);
                let n = [-0.39 * (a * 1.3).cos(), 0.18 * (b * 0.9).sin(), 1.];
                let len = n.iter().map(|v| v * v).sum::<f32>().sqrt();
                let n = n.map(|v| v / len);
                mesh.normals.as_mut().unwrap().push(n);
                let t = [n[2], 0., -n[0]];
                let len = t.iter().map(|v| v * v).sum::<f32>().sqrt();
                mesh.tangents.as_mut().unwrap().push([
                    t[0] / len,
                    t[1] / len,
                    t[2] / len,
                    if x % 2 == 0 { 1. } else { -1. },
                ]);
                mesh.tex_coords[0].push([0., 0.]);
                mesh.tex_coords[1].push([a, b]);
                mesh.colors
                    .as_mut()
                    .unwrap()
                    .push([(x * 23) as u8, (y * 29) as u8, 127, 255]);
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                let a = y * 9 + x;
                mesh.triangles
                    .extend([[a, a + 1, a + 9], [a + 1, a + 10, a + 9]]);
                mesh.primitive_ids.extend([x % 2, x % 2]);
            }
        }
        mesh.triangles.push([0, 0, 0]);
        mesh.primitive_ids.push(0);
        // Unreferenced vertices exercise the loose suffix and degenerate tangent fallback.
        for i in 0..3 {
            mesh.positions.push([i as f32 * 3., 3., 1.]);
            mesh.normals.as_mut().unwrap().push([0., 0., 1.]);
            mesh.tangents.as_mut().unwrap().push([0., 1., 0., -1.]);
            for uv in &mut mesh.tex_coords {
                uv.push([0., 0.]);
            }
            mesh.colors.as_mut().unwrap().push([17; 4]);
        }
        mesh
    }
    #[test]
    #[ignore = "requires a native Vulkan GPU"]
    fn native_mesh_roundtrip() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
        eprintln!("mesh GPU: {}", adapter.get_info().name);
        let (device, queue) = block_on(adapter.request_device(&Default::default())).unwrap();
        let uploader = Uploader::new(&device);
        for bits in [6, 10, 20] {
            let options = EncodingOptions {
                max_vertices: 16,
                position_bits: bits,
                uv_bits: bits,
                normal_bits: bits,
                tangent_bits: bits,
                tangent_uv_set: 1,
                weld_epsilon: None,
            };
            let mut bytes = Vec::new();
            write_mesh(&fixture(), &options, &mut bytes).unwrap();
            let mut reader = JackalMeshReader::open(Cursor::new(bytes)).unwrap();
            let expected = reader.read_mesh().unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            let gpu = uploader
                .upload_from_reader(&mut reader, &device, &mut encoder)
                .unwrap();
            queue.submit([encoder.finish()]);
            let vertices = read(&device, &queue, &gpu.vertices);
            let indices = read(&device, &queue, &gpu.indices);
            assert_eq!(
                &indices[..expected.triangles.len() * 3],
                bytemuck::cast_slice::<[u32; 3], u32>(&expected.triangles)
            );
            assert_eq!(
                &indices[expected.triangles.len() * 3..expected.triangles.len() * 4],
                &expected.primitive_ids
            );
            for i in 0..expected.positions.len() {
                let b = i * gpu.layout.stride as usize / 4;
                let get = |off: u32, k: usize| f32::from_bits(vertices[b + off as usize / 4 + k]);
                for k in 0..3 {
                    assert!(
                        (get(0, k) - expected.positions[i][k]).abs() < 2e-5,
                        "position bits={bits} vertex={i}"
                    );
                    assert!(
                        (get(gpu.layout.normal.unwrap(), k)
                            - expected.normals.as_ref().unwrap()[i][k])
                            .abs()
                            < 0.002,
                        "normal bits={bits} vertex={i}"
                    );
                    assert!(
                        (get(gpu.layout.tangent.unwrap(), k)
                            - expected.tangents.as_ref().unwrap()[i][k])
                            .abs()
                            < 0.002,
                        "tangent bits={bits} vertex={i}"
                    );
                }
                assert_eq!(
                    get(gpu.layout.tangent.unwrap(), 3),
                    expected.tangents.as_ref().unwrap()[i][3]
                );
                for set in 0..2 {
                    for k in 0..2 {
                        assert!(
                            (get(gpu.layout.tex_coords[set], k) - expected.tex_coords[set][i][k])
                                .abs()
                                < 2e-5
                        );
                    }
                }
                assert_eq!(
                    vertices[b + gpu.layout.color.unwrap() as usize / 4],
                    u32::from_le_bytes(expected.colors.as_ref().unwrap()[i])
                );
            }
            eprintln!(
                "mesh {bits}-bit all attributes, {} vertices, {} triangles passed",
                gpu.vertex_count, gpu.triangle_count
            );
        }
        for mesh in [
            MeshData::default(),
            MeshData {
                positions: vec![[1., 2., 3.], [4., 5., 6.]],
                ..Default::default()
            },
            MeshData {
                positions: vec![[0.; 3]; 3],
                triangles: vec![[0, 1, 2]],
                colors: Some(vec![[255; 4]; 3]),
                ..Default::default()
            },
        ] {
            let mut bytes = Vec::new();
            write_mesh(
                &mesh,
                &EncodingOptions {
                    max_vertices: 1,
                    weld_epsilon: None,
                    ..Default::default()
                },
                &mut bytes,
            )
            .unwrap();
            let mut reader = JackalMeshReader::open(Cursor::new(bytes)).unwrap();
            let expected = reader.read_mesh().unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            let gpu = uploader
                .upload_from_reader(&mut reader, &device, &mut encoder)
                .unwrap();
            queue.submit([encoder.finish()]);
            let values = read(&device, &queue, &gpu.vertices);
            let indices = read(&device, &queue, &gpu.indices);
            assert_eq!(
                &indices[..expected.triangles.len() * 3],
                bytemuck::cast_slice::<[u32; 3], u32>(&expected.triangles)
            );
            for (i, p) in expected.positions.iter().enumerate() {
                for k in 0..3 {
                    assert_eq!(
                        f32::from_bits(values[i * gpu.layout.stride as usize / 4 + k]),
                        p[k]
                    );
                }
                if let Some(colors) = &expected.colors {
                    assert_eq!(
                        values[i * gpu.layout.stride as usize / 4
                            + gpu.layout.color.unwrap() as usize / 4],
                        u32::from_le_bytes(colors[i])
                    );
                }
            }
        }
        // The loose UV anchor fixes both quantizer bounds at N, preserving det ±1.
        let n = ((1u32 << 20) - 1) as f32;
        for case in 0..5 {
            let mut uv = vec![[0., 0.], [n, n - 1.], [n - 1., n - 2.], [n, n]];
            if case == 1 {
                uv.swap(1, 2);
            }
            if case >= 2 {
                let off = if case == 2 { 0. } else { 8192. };
                uv = vec![
                    [off, off],
                    [off + 0.5, off + 0.5],
                    [off + 1., off + 1.],
                    [off + 1., off + 1.],
                ];
            }
            let normal = [1.0 / 3.0f32.sqrt(); 3];
            let mut mesh = MeshData {
                positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0.2, 0.3, 1.]],
                triangles: vec![[0, 1, 2]],
                normals: Some(vec![normal; 4]),
                tangents: Some(vec![[1.0 / 2.0f32.sqrt(), -1.0 / 2.0f32.sqrt(), 0., 1.]; 4]),
                tex_coords: vec![uv],
                ..Default::default()
            };
            if case == 4 {
                mesh.positions = vec![
                    [0., 0., 0.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [1., 0., 0.],
                    [0.2, 0.3, 1.],
                ];
                mesh.triangles = vec![[0, 1, 2], [0, 3, 2]];
                mesh.tex_coords = vec![vec![[1., 0.], [2., 0.], [1., 1.], [0., 0.], [n, n]]];
                mesh.normals.as_mut().unwrap().push(normal);
                mesh.tangents.as_mut().unwrap().push([
                    1.0 / 2.0f32.sqrt(),
                    -1.0 / 2.0f32.sqrt(),
                    0.,
                    1.,
                ]);
            }
            let options = EncodingOptions {
                position_bits: 20,
                uv_bits: 20,
                normal_bits: 20,
                tangent_bits: 20,
                weld_epsilon: None,
                ..Default::default()
            };
            let prepared = jkl::mesh::codec::prepare(&mesh, &options).unwrap();
            let patch = &prepared.patches[0];
            let (head, _) = jkl::mesh::codec::decode_indices(
                &patch.streams[0],
                patch.info.internal_triangle_count as usize,
                0,
            )
            .unwrap();
            let (q, _) =
                jkl::mesh::spatial::decode::<2>(&head, mesh.positions.len(), 20, &patch.streams[3])
                    .unwrap();
            let t = head[0];
            let a = q[t[0] as usize];
            let b = q[t[1] as usize];
            let c = q[t[2] as usize];
            let det = (b[0] as i64 - a[0] as i64) * (c[1] as i64 - a[1] as i64)
                - (b[1] as i64 - a[1] as i64) * (c[0] as i64 - a[0] as i64);
            assert_eq!(
                det,
                match case {
                    0 => -1,
                    1 | 4 => 1,
                    _ => 0,
                }
            );
            let mut bytes = Vec::new();
            jkl::jackal::mesh::write_prepared(&prepared, &mut bytes).unwrap();
            let mut reader = JackalMeshReader::open(Cursor::new(bytes)).unwrap();
            let expected = reader.read_mesh().unwrap();
            if case == 4 {
                let predicted = jkl::mesh::tangent::predictors_quantized(
                    &expected.positions,
                    &q,
                    expected.normals.as_ref().unwrap(),
                    &head,
                )
                .unwrap();
                // Opposite exact-grid derivatives cancel at the two shared vertices.
                for id in [0usize, 2] {
                    assert_eq!(
                        predicted[id],
                        jkl::mesh::normal::frame(expected.normals.as_ref().unwrap()[id]).1
                    );
                }
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            let gpu = uploader
                .upload_from_reader(&mut reader, &device, &mut encoder)
                .unwrap();
            queue.submit([encoder.finish()]);
            let values = read(&device, &queue, &gpu.vertices);
            for (i, t) in expected.tangents.unwrap().iter().enumerate() {
                for k in 0..3 {
                    let actual = f32::from_bits(
                        values[i * gpu.layout.stride as usize / 4
                            + gpu.layout.tangent.unwrap() as usize / 4
                            + k],
                    );
                    assert!(
                        (actual - t[k]).abs() < 0.002,
                        "adversarial UV case={case} det={det} vertex={i} component={k} actual={actual} expected={}",
                        t[k]
                    );
                }
            }
        }
    }
}
