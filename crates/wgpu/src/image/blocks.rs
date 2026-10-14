use crate::image::GpuPixels;
use jkl::image::Image;

const SHADER: &str = include_str!("../../shaders/pa_ls_ns.wgsl");

pub struct BlockCompressor {
    bc1: wgpu::ComputePipeline,
    bc2: wgpu::ComputePipeline,
    bc3: wgpu::ComputePipeline,
    bc4: wgpu::ComputePipeline,
    bc5: wgpu::ComputePipeline,
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
    batch_width: u32,
}

impl BlockCompressor {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jkl-pa-ls-ns-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jkl-blocks-bgl"),
            entries: &[true, false].map(|read_only| wgpu::BindGroupLayoutEntry {
                binding: if read_only { 0 } else { 1 },
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jkl-blocks-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: size_of::<Params>() as u32,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            bc1: pipeline("compress_bc1"),
            bc2: pipeline("compress_bc2"),
            bc3: pipeline("compress_bc3"),
            bc4: pipeline("compress_bc4"),
            bc5: pipeline("compress_bc5"),
            bind_group_layout,
        }
    }

    /// Encodes RGBA8 texels into BC1 with punch-through alpha below the threshold.
    /// Uses principal-axis extrema, up to four least-squares passes, and one endpoint neighbor round.
    pub fn compress_rgba_to_bc1(
        &self,
        image: Image<GpuPixels>,
        alpha_threshold: f32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: u32,
    ) -> Image<GpuPixels> {
        self.compress(
            image,
            alpha_threshold,
            device,
            queue,
            batch_size,
            &self.bc1,
            wgpu::TextureFormat::Bc1RgbaUnorm,
        )
    }

    /// Encodes RGBA8 texels into BC2 with explicit four-bit alpha.
    /// Uses principal-axis extrema, up to four least-squares passes, and one endpoint neighbor round.
    pub fn compress_rgba_to_bc2(
        &self,
        image: Image<GpuPixels>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: u32,
    ) -> Image<GpuPixels> {
        self.compress(
            image,
            0.0,
            device,
            queue,
            batch_size,
            &self.bc2,
            wgpu::TextureFormat::Bc2RgbaUnorm,
        )
    }

    /// Encodes RGBA8 texels into BC3 with interpolated alpha.
    /// Uses principal-axis extrema, up to four least-squares passes, and one endpoint neighbor round.
    pub fn compress_rgba_to_bc3(
        &self,
        image: Image<GpuPixels>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: u32,
    ) -> Image<GpuPixels> {
        self.compress(
            image,
            0.0,
            device,
            queue,
            batch_size,
            &self.bc3,
            wgpu::TextureFormat::Bc3RgbaUnorm,
        )
    }

    /// Encodes the red channel of RGBA8 texels into BC4 UNORM.
    /// Uses principal-axis extrema, up to four least-squares passes, and one endpoint neighbor round.
    pub fn compress_rgba_to_bc4(
        &self,
        image: Image<GpuPixels>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: u32,
    ) -> Image<GpuPixels> {
        self.compress(
            image,
            0.0,
            device,
            queue,
            batch_size,
            &self.bc4,
            wgpu::TextureFormat::Bc4RUnorm,
        )
    }

    /// Encodes the red and green channels of RGBA8 texels into BC5 UNORM.
    /// Uses principal-axis extrema, up to four least-squares passes, and one endpoint neighbor round.
    pub fn compress_rgba_to_bc5(
        &self,
        image: Image<GpuPixels>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: u32,
    ) -> Image<GpuPixels> {
        self.compress(
            image,
            0.0,
            device,
            queue,
            batch_size,
            &self.bc5,
            wgpu::TextureFormat::Bc5RgUnorm,
        )
    }

    fn compress(
        &self,
        image: Image<GpuPixels>,
        alpha_threshold: f32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: u32,
        pipeline: &wgpu::ComputePipeline,
        format: wgpu::TextureFormat,
    ) -> Image<GpuPixels> {
        assert_ne!(batch_size, 0);
        assert!(
            matches!(
                image.data().format,
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
            ),
            "compression requires RGBA8 UNORM or sRGB input"
        );
        let [in_width, in_height, layers] = image.extent().raw_size();
        assert!(in_width != 0 && in_height != 0 && layers != 0);
        let out_width = in_width.div_ceil(4);
        let out_height = in_height.div_ceil(4);
        let block_size = format.block_copy_size(None).unwrap() as usize;
        let align = 256 / block_size;
        let out_row_stride = out_width.div_ceil(align) * align;
        let out_plane_stride = out_row_stride * out_height;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-blocks-out"),
            size: (out_plane_stride * layers * block_size) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jkl-blocks-bg"),
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
            ],
        });
        let batch_size = batch_size as usize;
        let batch_width = out_width.min(batch_size).min(65535 * 64);
        let batch_height = out_height.min(batch_size / batch_width).clamp(1, 65535);
        let batch_layers = layers
            .min(batch_size / batch_width / batch_height)
            .clamp(1, 65535);
        let u32_size =
            |value| u32::try_from(value).expect("image dimension or stride exceeds shader range");
        for z in (0..layers).step_by(batch_layers) {
            for y in (0..out_height).step_by(batch_height) {
                for x in (0..out_width).step_by(batch_width) {
                    let width = batch_width.min(out_width - x);
                    let params = Params {
                        in_width: u32_size(in_width),
                        in_height: u32_size(in_height),
                        layers: u32_size(layers),
                        in_row_stride: u32_size(image.row_stride()),
                        in_plane_stride: u32_size(image.plane_stride()),
                        out_row_stride: u32_size(out_row_stride),
                        out_plane_stride: u32_size(out_plane_stride),
                        alpha_threshold,
                        x_offset: u32_size(x),
                        y_offset: u32_size(y),
                        z_offset: u32_size(z),
                        batch_width: u32_size(width),
                    };
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("jkl-blocks-batch"),
                        });
                    {
                        let mut compute =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("jkl-blocks-compress"),
                                timestamp_writes: None,
                            });
                        compute.set_pipeline(pipeline);
                        compute.set_bind_group(0, &bind_group, &[]);
                        compute.set_immediates(0, bytemuck::bytes_of(&params));
                        compute.dispatch_workgroups(
                            u32_size(width.div_ceil(64)),
                            u32_size(batch_height.min(out_height - y)),
                            u32_size(batch_layers.min(layers - z)),
                        );
                    }
                    queue.submit([encoder.finish()]);
                }
            }
        }
        Image::with_stride(
            image.dimensions(),
            [out_width, out_height, layers],
            [out_row_stride, out_plane_stride],
            GpuPixels::new(format, out_buf),
        )
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
