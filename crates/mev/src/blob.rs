//! GPU decompression of general-purpose JKLB blobs.
//!
//! GPU decompression is intended for trusted compressed data. Container bounds
//! and codec support are checked before commands are recorded; malformed codec
//! streams cannot report decoding errors from the GPU.

use std::io::{self, Read, Seek};

use jkl::jackal::blob::JackalBlobReader;
use mev::Arguments;

const SHADER: mev::ShaderSource = mev::include_shader_source!(Wgsl concat!(env!("CARGO_MANIFEST_DIR"), "/../wgpu/shaders/blob.wgsl"));

/// Decompressed bytes in a storage buffer, padded to a multiple of four bytes.
pub struct GpuBlob {
    pub len: u64,
    pub buffer: mev::Buffer,
}

impl GpuBlob {
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn buffer(&self) -> &mev::Buffer {
        &self.buffer
    }
}

pub struct Uploader {
    pipeline: mev::ComputePipeline,
}

#[derive(mev::Arguments)]
struct BlobArguments {
    #[mev(compute)]
    #[mev(storage)]
    payload: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    chunks: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    table: mev::Buffer,
    #[mev(compute)]
    #[mev(storage)]
    output: mev::Buffer,
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
                name: "jkl-blob-decompress",
                input: SHADER.into(),
            })
            .map_err(io::Error::other)?;
        let pipeline = device
            .new_compute_pipeline(mev::ComputePipelineDesc {
                name: "jkl-blob-decompress",
                shader: mev::Shader {
                    library,
                    entry: "decompress_blob".into(),
                },
                work_group_size: [64, 1, 1],
                constants: 0,
                arguments: &[BlobArguments::LAYOUT],
            })
            .map_err(io::Error::other)?;
        Ok(Self { pipeline })
    }

    /// Record decompression commands. Submit and synchronize the encoder before
    /// consuming the result. Unsupported codecs return an error without dispatch.
    pub fn upload_from_reader<R: Read + Seek>(
        &self,
        reader: &mut JackalBlobReader<R>,
        device: &mev::Device,
        encoder: &mut mev::CommandEncoder,
    ) -> io::Result<GpuBlob> {
        self.upload_from_reader_with_usage(reader, device, encoder, mev::BufferUsage::empty())
    }

    /// Add usage flags, for example `HOST_READ` for readback or `TRANSFER_SRC`.
    pub fn upload_from_reader_with_usage<R: Read + Seek>(
        &self,
        reader: &mut JackalBlobReader<R>,
        device: &mev::Device,
        encoder: &mut mev::CommandEncoder,
        usage: mev::BufferUsage,
    ) -> io::Result<GpuBlob> {
        let prepared = reader.prepare_gpu()?;
        let groups = u32::try_from(prepared.chunks.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many blob chunks"))?
            .div_ceil(64);
        // The public mev API does not expose device limits. Use the portable
        // dispatch bound supported by its native and WebGPU backends.
        if groups > 65535 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "too many blob workgroups",
            ));
        }
        let size = usize::try_from(u64::from(prepared.len).div_ceil(4) * 4)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "blob too large"))?
            .max(4);
        let mut output = device.new_buffer_mapped(mev::BufferDesc {
            name: "jkl-blob-output",
            size,
            usage: usage | mev::BufferUsage::STORAGE | mev::BufferUsage::HOST_WRITE,
        });
        {
            let mut mapped = output.write_mapped_range(..).map_err(io::Error::other)?;
            let zeros = [0; 4096];
            for offset in (0..size).step_by(zeros.len()) {
                mapped.write(offset, &zeros[..(size - offset).min(zeros.len())]);
            }
        }
        output.unmap();
        if groups != 0 {
            let chunks: Vec<u32> = prepared
                .chunks
                .iter()
                .flat_map(|chunk| {
                    [
                        chunk.compression,
                        chunk.output_offset,
                        chunk.output_len,
                        chunk.payload_offset,
                        chunk.payload_len,
                        chunk.table_offset,
                        chunk.table_len,
                    ]
                })
                .collect();
            let table: Vec<u32> = prepared
                .table
                .iter()
                .flat_map(|entry| [entry.freq, entry.cumul, entry.symbol, entry.length])
                .collect();
            let arguments = BlobArguments {
                payload: upload_words(device, "jkl-blob-payload", &prepared.payload)?,
                chunks: upload_words(device, "jkl-blob-chunks", &chunks)?,
                table: upload_words(device, "jkl-blob-table", &table)?,
                output: output.clone(),
            };
            let mut compute = encoder.compute();
            compute.with_pipeline(&self.pipeline);
            compute.with_arguments(0, &arguments);
            compute.dispatch(mev::Extent3::new(groups, 1, 1));
        }
        Ok(GpuBlob {
            len: u64::from(prepared.len),
            buffer: output,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use jkl::jackal::blob::{Compression, Options, write_blob};
    use std::io::Cursor;

    #[test]
    #[ignore = "requires a native GPU device"]
    fn gpu_blob_roundtrip() {
        let instance = mev::Instance::load().expect("GPU instance");
        let (idx, family) = instance
            .capabilities()
            .devices
            .iter()
            .enumerate()
            .find_map(|(idx, caps)| {
                caps.families
                    .iter()
                    .position(|family| family.queue_flags.contains(mev::QueueFlags::COMPUTE))
                    .map(|family| (idx, family as u32))
            })
            .expect("GPU compute queue");
        eprintln!(
            "mev blob GPU: {}",
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
        let uploader = Uploader::new(&device).expect("blob uploader");
        let patterned: Vec<u8> = (0..1739).map(|i| (i % 13) as u8).collect();
        let mut mixed = vec![42; 1023];
        let mut random = 7u32;
        mixed.extend((0..1023).map(|_| {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random as u8
        }));
        mixed.extend_from_slice(b"tail!");
        let mut cases = Vec::new();
        for codec in [Compression::None, Compression::Ans, Compression::Lz77Ans] {
            cases.push((patterned.clone(), 67, vec![codec]));
        }
        cases.push((patterned.clone(), 7, vec![Compression::None]));
        cases.push((patterned, 1739, vec![Compression::Lz77Ans]));
        cases.push((
            mixed,
            1023,
            vec![Compression::None, Compression::Ans, Compression::Lz77Ans],
        ));
        cases.push((Vec::new(), 7, vec![Compression::None]));
        cases.push((vec![9], 1, vec![Compression::Lz77Ans]));
        let mut results = Vec::new();
        for (input, chunk_size, allowed_compressions) in cases {
            let mixed_codecs = allowed_compressions.len() > 1;
            let mut encoded = Vec::new();
            write_blob(
                &input,
                &Options {
                    chunk_size,
                    allowed_compressions,
                },
                &mut encoded,
            )
            .unwrap();
            let mut reader = JackalBlobReader::open(Cursor::new(encoded)).unwrap();
            if mixed_codecs {
                assert!(
                    reader
                        .chunks()
                        .windows(2)
                        .any(|pair| { pair[0].compression != pair[1].compression })
                );
            }
            let mut encoder = queue.new_command_encoder();
            let out = uploader
                .upload_from_reader_with_usage(
                    &mut reader,
                    &device,
                    &mut encoder,
                    mev::BufferUsage::HOST_READ,
                )
                .expect("record blob decompression");
            assert_eq!(out.len(), input.len() as u64);
            assert_eq!(out.is_empty(), input.is_empty());
            assert_eq!(out.buffer().size(), input.len().div_ceil(4).max(1) * 4);
            queue
                .submit_checkpoint([encoder.finish()])
                .expect("submit blob decompression");
            results.push((input, out));
        }
        let mut encoded = Vec::new();
        write_blob(
            b"unsupported",
            &Options {
                chunk_size: 7,
                allowed_compressions: vec![Compression::Lz77],
            },
            &mut encoded,
        )
        .unwrap();
        let mut reader = JackalBlobReader::open(Cursor::new(encoded)).unwrap();
        let mut encoder = queue.new_command_encoder();
        let error = uploader
            .upload_from_reader(&mut reader, &device, &mut encoder)
            .err()
            .unwrap();
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
        queue
            .submit_checkpoint([encoder.finish()])
            .expect("submit empty encoder");
        queue.wait_idle().expect("wait for blob decompression");
        drop(queue);
        for (input, mut out) in results {
            let mut actual = vec![0xff; out.buffer.size()];
            out.buffer
                .read(0, &mut actual)
                .expect("read decompressed blob");
            assert_eq!(&actual[..input.len()], input.as_slice());
            assert!(actual[input.len()..].iter().all(|&byte| byte == 0));
        }
    }
}
