use jkl::{
    image::Image2DRef,
    math::{Rgb8U, Rgba8U},
};
use wgpu::util::DeviceExt;

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
    alpha_threshold: f32,
}

impl BlockCompressor {
    pub fn new(device: &wgpu::Device) -> Self {
        let bc1_shader = format!("{CLUSTER_FIT_WGSL}\n{BC1_WGSL}");

        let bc1_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-bc1-rgb-shader"),
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

    pub fn compress_rgb_to_bc1(
        &self,
        image: Image2DRef<Rgb8U>,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert!(
            u32::try_from(image.width()).is_ok(),
            "Image width exceeds u32::MAX"
        );
        assert!(
            u32::try_from(image.height()).is_ok(),
            "Image height exceeds u32::MAX"
        );

        let input_image_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-image-input"),
            size: (image.width() * image.height() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });

        {
            let mut mapped_range = input_image_buffer.get_mapped_range_mut(..);
            for (i, pixel) in image.pixels().iter().enumerate() {
                let offset = i * 4;
                mapped_range[offset] = pixel.r();
                mapped_range[offset + 1] = pixel.g();
                mapped_range[offset + 2] = pixel.b();
                mapped_range[offset + 3] = 255; // Alpha channel, set to opaque
            }
        }
        input_image_buffer.unmap();

        self.compress_rgba_buffer_to_bc1(
            input_image_buffer,
            image.width() as u32,
            image.height() as u32,
            0.0,
            device,
            encoder,
        );
    }

    pub fn compress_rgba_to_bc1(
        &self,
        image: Image2DRef<Rgba8U>,
        alpha_threshold: f32,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert!(
            u32::try_from(image.width()).is_ok(),
            "Image width exceeds u32::MAX"
        );
        assert!(
            u32::try_from(image.height()).is_ok(),
            "Image height exceeds u32::MAX"
        );

        let input_image_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-image-input"),
            size: (image.width() * image.height() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });

        {
            let mut mapped_range = input_image_buffer.get_mapped_range_mut(..);
            for (i, pixel) in image.pixels().iter().enumerate() {
                let offset = i * 4;
                mapped_range[offset] = pixel.r();
                mapped_range[offset + 1] = pixel.g();
                mapped_range[offset + 2] = pixel.b();
                mapped_range[offset + 3] = pixel.a();
            }
        }
        input_image_buffer.unmap();

        self.compress_rgba_buffer_to_bc1(
            input_image_buffer,
            image.width() as u32,
            image.height() as u32,
            alpha_threshold,
            device,
            encoder,
        );
    }

    pub fn compress_rgba_buffer_to_bc1(
        &self,
        image: wgpu::Buffer,
        width: u32,
        height: u32,
        alpha_threshold: f32,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let out_buf: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-blocks-out"),
            size: (width as u64 * height as u64 / 16) * 8, // Each BC1 block is 16 pixels, 8 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jkl-blocks-params"),
            contents: bytemuck::cast_slice(&[Params {
                width,
                height,
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
                    resource: image.as_entire_binding(),
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

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jkl-bc1-compress-pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.bc1);
        cpass.set_bind_group(0, &bind_group, &[]);

        let group_x = width.div_ceil(8);
        let group_y = height.div_ceil(8);

        if group_x > 0 && group_y > 0 {
            cpass.dispatch_workgroups(group_x, group_y, 1);
        }

        cpass.map_buffer_on_submit(&out_buf, wgpu::MapMode::Read, .., |result| {});
    }
}
