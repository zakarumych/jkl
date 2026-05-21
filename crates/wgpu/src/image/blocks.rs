use jkl::image::Image;
use wgpu::util::DeviceExt;

use crate::image::WgpuPixels;

const BINOM_WGSL: &'static str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/binom.wgsl"));

const REDUCE_WGSL: &'static str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/reduce.wgsl"));

const CLUSTER_FIT_WGSL: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/cluster_fit.wgsl"
));

const BC1_WGSL: &'static str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/bc1.wgsl"));

const BC2_WGSL: &'static str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/bc2.wgsl"));

pub struct BlockCompressor {
    bc1: wgpu::ComputePipeline,
    bc2: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::NoUninit)]
struct Params {
    in_width: u32,
    in_height: u32,
    layers: u32,
    in_row_stride: u32,
    in_plane_stride: u32,
    out_row_stride: u32,
    out_plane_stride: u32,
    alpha_threshold: f32,
    x_offset: u32,
    y_offset: u32,
    z_offset: u32,
    _pad: u32,
}

impl BlockCompressor {
    pub fn new(device: &wgpu::Device) -> Self {
        let bc1_shader = format!("{BINOM_WGSL}\n{REDUCE_WGSL}\n{CLUSTER_FIT_WGSL}\n{BC1_WGSL}");
        let bc2_shader = format!("{BINOM_WGSL}\n{REDUCE_WGSL}\n{CLUSTER_FIT_WGSL}\n{BC2_WGSL}");

        let bc1_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-bc1-shader"),
            source: wgpu::ShaderSource::Wgsl(bc1_shader.into()),
        });

        let bc2_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-bc2-shader"),
            source: wgpu::ShaderSource::Wgsl(bc2_shader.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jkl-blocks-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jkl-blocks-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let bc1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-bc1-pipeline"),
            layout: Some(&pipeline_layout),
            module: &bc1_shader,
            entry_point: Some("compress_bc1"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bc2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jkl-bc2-pipeline"),
            layout: Some(&pipeline_layout),
            module: &bc2_shader,
            entry_point: Some("compress_bc2"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        BlockCompressor {
            bc1,
            bc2,
            bind_group_layout,
        }
    }

    pub fn compress_rgba_to_bc1(
        &self,
        image: Image<WgpuPixels>,
        alpha_threshold: f32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: u32,
    ) -> Image<WgpuPixels> {
        assert_ne!(batch_size, 0);

        let batch_size = usize::try_from(batch_size).unwrap_or(1 << 30);

        let extent = image.extent();
        let dim = extent.dimensions();

        let [in_width, in_height, layers] = extent.raw_size();

        assert_ne!(in_width, 0);
        assert_ne!(in_height, 0);
        assert_ne!(layers, 0);

        let out_width = in_width.div_ceil(4);
        let out_height = in_height.div_ceil(4);

        let out_row_align = 256 / 16;

        let out_row_stride = out_width.div_ceil(out_row_align) * out_row_align;
        let out_plane_stride = out_row_stride * out_height;

        let out_buf: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-blocks-out"),
            size: out_plane_stride as u64 * layers as u64 * 8, // Each BC1 block is 8 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dispatch in batches along all three axes to avoid the GPU timeout.
        // Each batch submits its own command buffer so the GPU processes them sequentially.
        let batch_layers = usize::min(layers, batch_size / out_width / out_height).max(1);
        let batch_height = usize::min(out_height, batch_size / out_width).max(1);
        let batch_width = usize::min(out_width, batch_size);

        for z_offset in (0..layers).step_by(batch_layers) {
            debug_assert!(u32::try_from(z_offset).is_ok());
            let this_batch_layers = usize::min(batch_layers, layers - z_offset);

            for y_offset in (0..out_height).step_by(batch_height) {
                debug_assert!(u32::try_from(y_offset).is_ok());
                let this_batch_height = usize::min(batch_height, out_height - y_offset);

                for x_offset in (0..out_width).step_by(batch_width) {
                    debug_assert!(u32::try_from(x_offset).is_ok());
                    let this_batch_width = usize::min(batch_width, out_width - x_offset);

                    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("jkl-blocks-params"),
                        contents: bytemuck::cast_slice(&[Params {
                            in_width: in_width as u32,
                            in_height: in_height as u32,
                            layers: layers as u32,
                            in_row_stride: image.row_stride() as u32,
                            in_plane_stride: image.plane_stride() as u32,
                            out_row_stride: out_row_stride as u32,
                            out_plane_stride: out_plane_stride as u32,
                            alpha_threshold,
                            x_offset: x_offset as u32,
                            y_offset: y_offset as u32,
                            z_offset: z_offset as u32,
                            _pad: 0,
                        }]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("jkl-wgpu-decode-bg"),
                        layout: &self.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: image.data().buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let mut batch_encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("jkl-bc1-batch-encoder"),
                        });
                    {
                        let mut cpass =
                            batch_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("jkl-bc1-compress-pass"),
                                timestamp_writes: None,
                            });
                        cpass.set_pipeline(&self.bc1);
                        cpass.set_bind_group(0, &bind_group, &[]);
                        cpass.dispatch_workgroups(
                            this_batch_width as u32,
                            this_batch_height as u32,
                            this_batch_layers as u32,
                        );
                    }
                    queue.submit([batch_encoder.finish()]);
                }
            }
        }

        Image::with_stride(
            dim,
            [out_width, out_height, layers],
            [out_row_stride, out_plane_stride],
            WgpuPixels::new(wgpu::TextureFormat::Bc1RgbaUnorm, out_buf),
        )
    }

    pub fn compress_rgba_to_bc2(
        &self,
        image: Image<WgpuPixels>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: u32,
    ) -> Image<WgpuPixels> {
        assert_ne!(batch_size, 0);

        let batch_size = usize::try_from(batch_size).unwrap_or(1 << 30);

        let extent = image.extent();
        let dim = extent.dimensions();

        let [in_width, in_height, layers] = extent.raw_size();

        assert_ne!(in_width, 0);
        assert_ne!(in_height, 0);
        assert_ne!(layers, 0);

        let out_width = in_width.div_ceil(4);
        let out_height = in_height.div_ceil(4);

        let out_row_align = 256 / 16; // BC2 blocks are 16 bytes

        let out_row_stride = out_width.div_ceil(out_row_align) * out_row_align;
        let out_plane_stride = out_row_stride * out_height;

        let out_buf: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-bc2-blocks-out"),
            size: out_plane_stride as u64 * layers as u64 * 16, // Each BC2 block is 16 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let batch_layers = usize::min(layers, batch_size / out_width / out_height).max(1);
        let batch_height = usize::min(out_height, batch_size / out_width).max(1);
        let batch_width = usize::min(out_width, batch_size);

        for z_offset in (0..layers).step_by(batch_layers) {
            debug_assert!(u32::try_from(z_offset).is_ok());
            let this_batch_layers = usize::min(batch_layers, layers - z_offset);

            for y_offset in (0..out_height).step_by(batch_height) {
                debug_assert!(u32::try_from(y_offset).is_ok());
                let this_batch_height = usize::min(batch_height, out_height - y_offset);

                for x_offset in (0..out_width).step_by(batch_width) {
                    debug_assert!(u32::try_from(x_offset).is_ok());
                    let this_batch_width = usize::min(batch_width, out_width - x_offset);

                    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("jkl-bc2-blocks-params"),
                        contents: bytemuck::cast_slice(&[Params {
                            in_width: in_width as u32,
                            in_height: in_height as u32,
                            layers: layers as u32,
                            in_row_stride: image.row_stride() as u32,
                            in_plane_stride: image.plane_stride() as u32,
                            out_row_stride: out_row_stride as u32,
                            out_plane_stride: out_plane_stride as u32,
                            alpha_threshold: 0.0,
                            x_offset: x_offset as u32,
                            y_offset: y_offset as u32,
                            z_offset: z_offset as u32,
                            _pad: 0,
                        }]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("jkl-bc2-wgpu-decode-bg"),
                        layout: &self.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: image.data().buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let mut batch_encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("jkl-bc2-batch-encoder"),
                        });
                    {
                        let mut cpass =
                            batch_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("jkl-bc2-compress-pass"),
                                timestamp_writes: None,
                            });
                        cpass.set_pipeline(&self.bc2);
                        cpass.set_bind_group(0, &bind_group, &[]);
                        cpass.dispatch_workgroups(
                            this_batch_width as u32,
                            this_batch_height as u32,
                            this_batch_layers as u32,
                        );
                    }
                    queue.submit([batch_encoder.finish()]);
                }
            }
        }

        Image::with_stride(
            dim,
            [out_width, out_height, layers],
            [out_row_stride, out_plane_stride],
            WgpuPixels::new(wgpu::TextureFormat::Bc2RgbaUnorm, out_buf),
        )
    }
}
