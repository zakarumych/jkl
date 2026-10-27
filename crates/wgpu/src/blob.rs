//! GPU decompression of general-purpose Jackal blobs.
use std::io::{self, Read, Seek};

use jkl::jackal::blob::JackalBlobReader as Reader;
use wgpu::util::DeviceExt;

/// A GPU buffer with a logical byte length. Allocation is padded to four bytes.
pub struct GpuBlob {
    pub buffer: wgpu::Buffer,
    pub len: u64,
}

impl GpuBlob {
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// Decodes raw, ANS, and LZ77+ANS chunks on the GPU.
///
/// Unsupported codecs fail before recording commands. Compressed payloads must
/// come from a trusted encoder: GPU execution does not report malformed streams.
pub struct Uploader {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl Uploader {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-blob-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blob.wgsl").into()),
        });
        let entries = std::array::from_fn::<_, 4, _>(|binding| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: binding != 3,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jkl-blob-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jkl-blob-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-blob-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("decompress_blob"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    pub fn upload_from_reader<R: Read + Seek>(
        &self,
        reader: &mut Reader<R>,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> io::Result<GpuBlob> {
        self.upload_from_reader_with_usage(reader, device, encoder, wgpu::BufferUsages::COPY_SRC)
    }

    /// Adds consumer usages such as `VERTEX`, `INDEX`, or `INDIRECT` to the output.
    pub fn upload_from_reader_with_usage<R: Read + Seek>(
        &self,
        reader: &mut Reader<R>,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        usage: wgpu::BufferUsages,
    ) -> io::Result<GpuBlob> {
        if usage.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "map blob output through a staging buffer",
            ));
        }
        let prepared = reader.prepare_gpu()?;
        let len = u64::from(prepared.len);
        let size = len.div_ceil(4).max(1) * 4;
        let limits = device.limits();
        let groups = u32::try_from(prepared.chunks.len().div_ceil(64))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many blob chunks"))?;
        // Serialize fields explicitly so the core types need no backend dependency.
        let chunks: Vec<u32> = prepared
            .chunks
            .iter()
            .flat_map(|c| {
                [
                    c.compression,
                    c.output_offset,
                    c.output_len,
                    c.payload_offset,
                    c.payload_len,
                    c.table_offset,
                    c.table_len,
                ]
            })
            .collect();
        let mut table: Vec<u32> = prepared
            .table
            .iter()
            .flat_map(|e| [e.freq, e.cumul, e.symbol, e.length])
            .collect();
        if table.is_empty() {
            table.resize(4, 0);
        }
        let sizes = [
            size,
            prepared.payload.len() as u64 * 4,
            chunks.len() as u64 * 4,
            table.len() as u64 * 4,
        ];
        if groups > limits.max_compute_workgroups_per_dimension
            || sizes.iter().any(|&size| {
                size > limits.max_buffer_size
                    || size > u64::from(limits.max_storage_buffer_binding_size)
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "blob exceeds GPU device limits",
            ));
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-blob-output"),
            size,
            usage: usage | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if prepared.chunks.is_empty() {
            return Ok(GpuBlob { buffer, len });
        }
        let input = |label, words: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(words),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let payload = input("jkl-blob-payload", &prepared.payload);
        let chunks = input("jkl-blob-chunks", &chunks);
        let table = input("jkl-blob-table", &table);
        let entries = [&payload, &chunks, &table, &buffer].map(|buffer| buffer.as_entire_binding());
        let entries = std::array::from_fn::<_, 4, _>(|binding| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: entries[binding].clone(),
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jkl-blob-bindings"),
            layout: &self.layout,
            entries: &entries,
        });
        encoder.clear_buffer(&buffer, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jkl-blob-decompress"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        drop(pass);
        Ok(GpuBlob { buffer, len })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jkl::jackal::blob::{Compression, Options, write_blob};
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
                std::task::Poll::Ready(value) => return value,
                std::task::Poll::Pending => std::thread::park(),
            }
        }
    }

    #[test]
    #[ignore = "requires a native Vulkan GPU"]
    fn native_blob_roundtrips() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
        eprintln!("blob GPU: {}", adapter.get_info().name);
        let (device, queue) = block_on(adapter.request_device(&Default::default())).unwrap();
        let uploader = Uploader::new(&device);
        let mut random = 0x12345678u32;
        let data: Vec<u8> = (0..32779)
            .map(|i| {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                if i < 2048 {
                    0
                } else if i < 8192 {
                    (i % 19) as u8
                } else if i < 16384 {
                    (random & 15) as u8
                } else {
                    random as u8
                }
            })
            .collect();
        for compression in [
            vec![Compression::None],
            vec![Compression::Ans],
            vec![Compression::Lz77Ans],
            vec![Compression::None, Compression::Ans, Compression::Lz77Ans],
        ] {
            for chunk_size in [1, 3, 5, 1025, 32779] {
                let mut encoded = Vec::new();
                write_blob(
                    &data,
                    &Options {
                        chunk_size,
                        allowed_compressions: compression.clone(),
                    },
                    &mut encoded,
                )
                .unwrap();
                let mut reader = Reader::open(Cursor::new(encoded)).unwrap();
                let prepared = reader.prepare_gpu().unwrap();
                if compression.len() == 1 {
                    assert!(
                        prepared
                            .chunks
                            .iter()
                            .all(|c| c.compression == compression[0] as u32)
                    );
                }
                if compression.len() > 1 && chunk_size == 1025 {
                    for codec in &compression {
                        assert!(
                            prepared
                                .chunks
                                .iter()
                                .any(|c| c.compression == *codec as u32),
                            "missing {codec:?}"
                        );
                    }
                }
                let mut encoder = device.create_command_encoder(&Default::default());
                let blob = uploader
                    .upload_from_reader(&mut reader, &device, &mut encoder)
                    .unwrap();
                assert_eq!(blob.len, data.len() as u64);
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("blob-test-readback"),
                    size: blob.buffer.size(),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(&blob.buffer, 0, &readback, 0, blob.buffer.size());
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                readback.map_async(wgpu::MapMode::Read, .., move |result| {
                    tx.send(result).unwrap()
                });
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let mapped = readback.get_mapped_range(..).unwrap();
                assert_eq!(
                    &mapped[..data.len()],
                    &data,
                    "{compression:?}, chunk size {chunk_size}"
                );
                assert!(mapped[data.len()..].iter().all(|&byte| byte == 0));
            }
        }
        let mut encoded = Vec::new();
        write_blob(&[], &Options::default(), &mut encoded).unwrap();
        let mut reader = Reader::open(Cursor::new(encoded)).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        let blob = uploader
            .upload_from_reader(&mut reader, &device, &mut encoder)
            .unwrap();
        assert!(blob.is_empty());
        assert_eq!(blob.buffer.size(), 4);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        for compression in [
            Compression::Lz77,
            Compression::Rle,
            Compression::RleAns,
            Compression::Lz78,
        ] {
            let mut encoded = Vec::new();
            write_blob(
                b"unsupported codec",
                &Options {
                    chunk_size: 1024,
                    allowed_compressions: vec![compression],
                },
                &mut encoded,
            )
            .unwrap();
            let mut reader = Reader::open(Cursor::new(encoded)).unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            let error = uploader
                .upload_from_reader(&mut reader, &device, &mut encoder)
                .err()
                .unwrap();
            assert_eq!(error.kind(), io::ErrorKind::Unsupported);
            queue.submit([encoder.finish()]);
        }
    }
}
