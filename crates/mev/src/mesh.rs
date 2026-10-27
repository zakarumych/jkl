//! Parallel mesh patch decoding into renderer-ready native buffers.
//!
//! Uploading records compute writes. Before drawing from the returned buffers,
//! the caller records `encoder.barrier(mev::PipelineStages::COMPUTE_SHADER,
//! mev::PipelineStages::VERTEX_INPUT)`. Include shader stages when attributes or
//! primitive IDs are read through storage bindings. A subsequent buffer copy
//! instead needs a compute-to-transfer barrier. Submission and completion waits
//! remain the caller's responsibility; no cross-patch synchronization is needed.
use jkl::{
    jackal::mesh::JackalMeshReader,
    mesh::gpu::{self, VertexLayout},
};
use mev::Arguments;
use std::io::{self, Read, Seek};

const SHADER: mev::ShaderSource = mev::include_shader_source!(Wgsl concat!(env!("CARGO_MANIFEST_DIR"), "/../jkl/src/mesh/shaders/decode.wgsl"));

/// Conservative per-buffer bound because mev does not expose device storage limits.
pub const MAX_STORAGE_BUFFER_BYTES: u64 = 128 * 1024 * 1024;
/// Conservative dispatch bound because mev does not expose device dispatch limits.
pub const MAX_WORKGROUPS: u32 = 65535;

/// Interleaved attributes and global u32 triangle indices.
pub struct GpuMesh {
    pub vertices: mev::Buffer,
    /// Triangle indices occupy `0..triangle_count*12`; primitive IDs follow.
    pub indices: mev::Buffer,
    pub layout: VertexLayout,
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub materials: Vec<i32>,
}
impl GpuMesh {
    pub fn primitive_offset(&self) -> u64 {
        u64::from(self.triangle_count) * 12
    }
}
/// One invocation per patch. Compressed content must be trusted or CPU validated.
pub struct Uploader {
    pipeline: mev::ComputePipeline,
}

#[derive(mev::Arguments)]
struct MeshArguments {
    #[mev(compute)]
    #[mev(storage)]
    payload: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    metadata: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    tables: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    vertices: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    indices: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    scratch: mev::Buffer,
}
fn upload_words(device: &mev::Device, name: &str, words: &[u32]) -> io::Result<mev::Buffer> {
    let mut buffer = device.new_buffer_mapped(mev::BufferDesc {
        name,
        size: std::mem::size_of_val(words).max(4),
        usage: mev::BufferUsage::STORAGE | mev::BufferUsage::HOST_WRITE,
    });
    {
        let mut mapped = buffer.write_mapped_range(..).map_err(io::Error::other)?;
        if words.is_empty() {
            mapped.write(0, &[0; 4]);
        } else {
            mapped.write(0, bytemuck::cast_slice(words));
        }
    }
    buffer.unmap();
    Ok(buffer)
}
impl Uploader {
    pub fn new(device: &mev::Device) -> io::Result<Self> {
        let library = device
            .new_shader_library(mev::LibraryDesc {
                name: "jkl-mesh-decoder",
                input: SHADER.into(),
            })
            .map_err(io::Error::other)?;
        let pipeline = device
            .new_compute_pipeline(mev::ComputePipelineDesc {
                name: "jkl-mesh-decoder",
                shader: mev::Shader {
                    library,
                    entry: "decode_mesh".into(),
                },
                work_group_size: [64, 1, 1],
                constants: 0,
                arguments: &[MeshArguments::LAYOUT],
            })
            .map_err(io::Error::other)?;
        Ok(Self { pipeline })
    }
    /// Uploads compressed data and records decoding. Record a compute-to-consumer
    /// barrier before rendering or copying, then submit the command encoder.
    pub fn upload_from_reader<R: Read + Seek>(
        &self,
        reader: &mut JackalMeshReader<R>,
        device: &mev::Device,
        encoder: &mut mev::CommandEncoder,
    ) -> io::Result<GpuMesh> {
        self.upload_from_reader_with_usage(reader, device, encoder, mev::BufferUsage::empty())
    }
    /// Adds output usage flags, such as `HOST_READ` for validation readback.
    /// Host reads require completed submission; GPU consumers need the same
    /// compute-to-consumer barrier as [`Self::upload_from_reader`].
    pub fn upload_from_reader_with_usage<R: Read + Seek>(
        &self,
        reader: &mut JackalMeshReader<R>,
        device: &mev::Device,
        encoder: &mut mev::CommandEncoder,
        usage: mev::BufferUsage,
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
        let tables: Vec<_> = prepared
            .table
            .iter()
            .flat_map(|e| [e.freq, e.cumul, e.symbol])
            .collect();
        let vertex_bytes = (u64::from(prepared.vertex_count) * u64::from(layout.stride)).max(4);
        let index_bytes = (u64::from(prepared.triangle_count) * 16).max(4);
        let scratch_bytes = (u64::from(prepared.vertex_count) * 12).max(4);
        let groups = u32::try_from(prepared.patches.len().div_ceil(64)).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "mesh patch count overflow")
        })?;
        // The mev public API does not expose per-device limits. Stay within
        // portable storage-buffer and dispatch bounds before allocating.
        if groups > MAX_WORKGROUPS
            || [
                vertex_bytes,
                index_bytes,
                scratch_bytes,
                metadata.len() as u64 * 4,
                tables.len() as u64 * 4,
                prepared.payload.len() as u64 * 4,
            ]
            .into_iter()
            .any(|size| size > MAX_STORAGE_BUFFER_BYTES)
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "mesh exceeds conservative mev backend limits (128 MiB per buffer, 65535 workgroups)",
            ));
        }
        let output = |name, size, extra| {
            device.new_buffer(mev::BufferDesc {
                name,
                size: size as usize,
                usage: usage | mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_SRC | extra,
            })
        };
        let vertices = output("jkl-mesh-vertices", vertex_bytes, mev::BufferUsage::VERTEX);
        let indices = output("jkl-mesh-indices", index_bytes, mev::BufferUsage::INDEX);
        if groups != 0 {
            let scratch = device.new_buffer(mev::BufferDesc {
                name: "jkl-mesh-scratch",
                size: scratch_bytes as usize,
                usage: mev::BufferUsage::STORAGE,
            });
            let arguments = MeshArguments {
                payload: upload_words(device, "jkl-mesh-payload", &prepared.payload)?,
                metadata: upload_words(device, "jkl-mesh-metadata", &metadata)?,
                tables: upload_words(device, "jkl-mesh-tables", &tables)?,
                vertices: vertices.clone(),
                indices: indices.clone(),
                scratch,
            };
            let mut compute = encoder.compute();
            compute.with_pipeline(&self.pipeline);
            compute.with_arguments(0, &arguments);
            compute.dispatch(mev::Extent3::new(groups, 1, 1));
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use jkl::{
        jackal::mesh::write_mesh,
        mesh::data::{EncodingOptions, MeshData},
    };
    use std::io::Cursor;
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
    fn read(buffer: &mut mev::Buffer) -> Vec<u32> {
        let mut bytes = vec![0; buffer.size()];
        buffer.read(0, &mut bytes).unwrap();
        bytes
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    }
    #[test]
    #[ignore = "requires a native GPU device"]
    fn native_mesh_roundtrip() {
        let instance = mev::Instance::load().expect("GPU instance");
        let (idx, family) = instance
            .capabilities()
            .devices
            .iter()
            .enumerate()
            .find_map(|(i, c)| {
                c.families
                    .iter()
                    .position(|f| f.queue_flags.contains(mev::QueueFlags::COMPUTE))
                    .map(|f| (i, f as u32))
            })
            .expect("GPU compute queue");
        eprintln!(
            "mev mesh GPU: {}",
            instance.capabilities().devices[idx].name
        );
        let features = instance.capabilities().devices[idx].features
            & (mev::Features::BUFFER_MAP | mev::Features::BUFFER_MAP_AT_CREATION);
        let (device, mut queues) = instance
            .new_device(mev::DeviceDesc {
                idx,
                queues: &[family],
                features,
            })
            .expect("GPU device");
        let mut queue = queues.pop().unwrap();
        let uploader = Uploader::new(&device).expect("mesh uploader");
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
            let mut encoder = queue.new_command_encoder();
            let mut gpu = uploader
                .upload_from_reader_with_usage(
                    &mut reader,
                    &device,
                    &mut encoder,
                    mev::BufferUsage::HOST_READ,
                )
                .unwrap();
            queue.submit_checkpoint([encoder.finish()]).unwrap();
            queue.wait_idle().unwrap();
            let vertices = read(&mut gpu.vertices);
            let indices = read(&mut gpu.indices);
            assert_eq!(
                &indices[..expected.triangles.len() * 3],
                bytemuck::cast_slice::<[u32; 3], u32>(&expected.triangles)
            );
            assert_eq!(
                &indices[expected.triangles.len() * 3..expected.triangles.len() * 4],
                &expected.primitive_ids
            );
            for i in 0..expected.positions.len() {
                let base = i * gpu.layout.stride as usize / 4;
                let get =
                    |off: u32, k: usize| f32::from_bits(vertices[base + off as usize / 4 + k]);
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
                    vertices[base + gpu.layout.color.unwrap() as usize / 4],
                    u32::from_le_bytes(expected.colors.as_ref().unwrap()[i])
                );
            }
            assert_eq!(gpu.materials, expected.materials);
            eprintln!(
                "mev mesh {bits}-bit all attributes, {} vertices, {} triangles passed",
                gpu.vertex_count, gpu.triangle_count
            );
        }
    }
}
