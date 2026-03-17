use jkl::image::Image;
use wgpu::util::DeviceExt;

use crate::image::PixelBuffer;

const CLUSTER_FIT_WGSL: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/cluster_fit.wgsl"
));

const BC1_WGSL: &'static str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/bc1.wgsl"));

pub struct BlockCompressor {
    bc1: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::NoUninit)]
struct Params {
    width: u32,
    height: u32,
    layers: u32,
    in_row_stride: u32,
    in_plane_stride: u32,
    out_row_stride: u32,
    out_plane_stride: u32,
    alpha_threshold: f32,
}

impl BlockCompressor {
    pub fn new(device: &wgpu::Device) -> Self {
        let bc1_shader = format!("{CLUSTER_FIT_WGSL}\n{BC1_WGSL}");

        let bc1_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-bc1-shader"),
            source: wgpu::ShaderSource::Wgsl(bc1_shader.into()),
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

        BlockCompressor {
            bc1,
            bind_group_layout,
        }
    }

    pub fn compress_rgba_to_bc1(
        &self,
        image: Image<PixelBuffer>,
        alpha_threshold: f32,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Image<PixelBuffer> {
        let extent = image.extent();
        let dim = extent.dimensions();

        let [in_width, in_height, layers] = extent.raw_size();

        let out_width = in_width.div_ceil(4);
        let out_height = in_height.div_ceil(4);

        let out_row_stride = (out_width * 16).div_ceil(256) * 256; // Each BC1 block is 16 bytes
        let out_plane_stride = out_row_stride * out_height;

        let out_buf: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-blocks-out"),
            size: (out_width as u64 * out_height as u64) * 8, // Each BC1 block is 8 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-blocks-params"),
            contents: bytemuck::cast_slice(&[Params {
                width: in_width as u32,
                height: in_height as u32,
                layers: layers as u32,
                in_row_stride: image.row_stride() as u32,
                in_plane_stride: image.plane_stride() as u32,
                out_row_stride: out_row_stride as u32,
                out_plane_stride: out_plane_stride as u32,
                alpha_threshold,
            }]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jkl-wgpu-decode-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image.data().buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jkl-bc1-compress-pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.bc1);
            cpass.set_bind_group(0, &bind_group, &[]);

            let group_x = out_width.div_ceil(8) as u32;
            let group_y = out_height.div_ceil(8) as u32;
            let group_z = layers as u32;

            if group_x > 0 && group_y > 0 && group_z > 0 {
                cpass.dispatch_workgroups(group_x, group_y, group_z);
            }
        }

        Image::with_stride(
            dim,
            [out_width, out_height, layers],
            [out_row_stride, out_plane_stride],
            PixelBuffer::new(wgpu::TextureFormat::Bc1RgbaUnorm, out_buf),
        )
    }
}
