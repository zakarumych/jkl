use jkl::image::Image;
use mev::{Arguments, BufferUsage, DeviceRepr, PixelFormat};

use crate::image::GpuPixels;

const BINOM_WGSL: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../wgpu/shaders/binom.wgsl"
));

const REDUCE_WGSL: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../wgpu/shaders/reduce.wgsl"
));

const CLUSTER_FIT_WGSL: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../wgpu/shaders/cluster_fit.wgsl"
));

const BC1_WGSL: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../wgpu/shaders/bc1.wgsl"
));

const BC2_WGSL: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../wgpu/shaders/bc2.wgsl"
));

pub struct BlockCompressor {
    bc1: mev::ComputePipeline,
    bc2: mev::ComputePipeline,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod, mev::AutoDeviceRepr)]
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
}

#[derive(Arguments)]
struct BlockCompressorArguments {
    #[mev(compute)]
    #[mev(storage)]
    image_input: mev::Buffer,

    #[mev(compute)]
    #[mev(storage)]
    image_output: mev::Buffer,
}

impl BlockCompressor {
    pub fn new(device: &mev::Device) -> Self {
        let bc1_shader = format!("{BINOM_WGSL}\n{REDUCE_WGSL}\n{CLUSTER_FIT_WGSL}\n{BC1_WGSL}");
        let bc2_shader = format!("{BINOM_WGSL}\n{REDUCE_WGSL}\n{CLUSTER_FIT_WGSL}\n{BC2_WGSL}");

        let bc1_shader = device
            .new_shader_library(mev::LibraryDesc {
                name: "jkl-bc1-shader",
                input: mev::LibraryInput::wgsl(bc1_shader).into(),
            })
            .expect("Failed to create BC1 shader library");

        let bc2_shader = device
            .new_shader_library(mev::LibraryDesc {
                name: "jkl-bc2-shader",
                input: mev::LibraryInput::wgsl(bc2_shader).into(),
            })
            .expect("Failed to create BC2 shader library");

        let bc1 = device
            .new_compute_pipeline(mev::ComputePipelineDesc {
                name: "jkl-bc1-pipeline",
                shader: bc1_shader.entry("compress_bc1"),
                work_group_size: [1, 1, 64],
                constants: Params::SIZE,
                arguments: &[BlockCompressorArguments::LAYOUT],
            })
            .unwrap_or_else(|err| match err {
                mev::PipelineError::InvalidShaderEntry => {
                    unreachable!("BC1 shader entry point not found");
                }
                mev::PipelineError::Failure(message) => {
                    panic!("Failed to link BC1 shader: {message}");
                }
            });

        let bc2 = device
            .new_compute_pipeline(mev::ComputePipelineDesc {
                name: "jkl-bc2-pipeline",
                shader: bc2_shader.entry("compress_bc2"),
                work_group_size: [1, 1, 64],
                constants: size_of::<Params>(),
                arguments: &[BlockCompressorArguments::LAYOUT],
            })
            .unwrap_or_else(|err| match err {
                mev::PipelineError::InvalidShaderEntry => {
                    unreachable!("BC2 shader entry point not found");
                }
                mev::PipelineError::Failure(message) => {
                    panic!("Failed to link BC2 shader: {message}");
                }
            });

        BlockCompressor { bc1, bc2 }
    }

    pub fn compress_rgba_to_bc1(
        &self,
        image: Image<GpuPixels>,
        alpha_threshold: f32,
        device: &mev::Device,
        queue: &mut mev::Queue,
        batch_size: u32,
        usage: BufferUsage,
    ) -> Image<GpuPixels> {
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

        let out_buf: mev::Buffer = device.new_buffer(mev::BufferDesc {
            name: "jkl-blocks-out",
            size: out_plane_stride * layers * PixelFormat::Bc1RgbaSrgb.block_size(), // Each BC1 block is 8 bytes
            usage: mev::BufferUsage::STORAGE | usage,
        });

        // Dispatch in batches along all three axes to avoid the GPU timeout.
        // Each batch submits its own command buffer so the GPU processes them sequentially.
        let batch_layers = usize::min(layers, batch_size / out_width / out_height).max(1);
        let batch_height = usize::min(out_height, batch_size / out_width).max(1);
        let batch_width = usize::min(out_width, batch_size);

        let arguments = BlockCompressorArguments {
            image_input: image.data().buffer.clone(),
            image_output: out_buf.clone(),
        };

        for z_offset in (0..layers).step_by(batch_layers) {
            debug_assert!(u32::try_from(z_offset).is_ok());
            let this_batch_layers = usize::min(batch_layers, layers - z_offset);

            for y_offset in (0..out_height).step_by(batch_height) {
                debug_assert!(u32::try_from(y_offset).is_ok());
                let this_batch_height = usize::min(batch_height, out_height - y_offset);

                for x_offset in (0..out_width).step_by(batch_width) {
                    debug_assert!(u32::try_from(x_offset).is_ok());
                    let this_batch_width = usize::min(batch_width, out_width - x_offset);

                    let mut batch_encoder = queue.new_command_encoder();
                    {
                        let mut compute_encoder = batch_encoder.compute();
                        compute_encoder.with_pipeline(&self.bc1);
                        compute_encoder.with_arguments(0, &arguments);

                        let params = Params {
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
                        };

                        compute_encoder.with_constants(&params);

                        compute_encoder.dispatch(mev::Extent3::new(
                            this_batch_width as u32,
                            this_batch_height as u32,
                            this_batch_layers as u32,
                        ));
                    }
                    let _ = queue.submit([batch_encoder.finish()]);
                }
            }
        }

        Image::with_stride(
            dim,
            [out_width, out_height, layers],
            [out_row_stride, out_plane_stride],
            GpuPixels::new(mev::PixelFormat::Bc1RgbaUnorm, out_buf),
        )
    }

    pub fn compress_rgba_to_bc2(
        &self,
        image: Image<GpuPixels>,
        device: &mev::Device,
        queue: &mut mev::Queue,
        batch_size: u32,
        usage: BufferUsage,
    ) -> Image<GpuPixels> {
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

        let out_buf: mev::Buffer = device.new_buffer(mev::BufferDesc {
            name: "jkl-bc2-blocks-out",
            size: out_plane_stride * layers * 16, // Each BC2 block is 16 bytes
            usage: mev::BufferUsage::STORAGE | mev::BufferUsage::TRANSFER_SRC | usage,
        });

        let batch_layers = usize::min(layers, batch_size / out_width / out_height).max(1);
        let batch_height = usize::min(out_height, batch_size / out_width).max(1);
        let batch_width = usize::min(out_width, batch_size);

        let arguments = BlockCompressorArguments {
            image_input: image.data().buffer.clone(),
            image_output: out_buf.clone(),
        };

        for z_offset in (0..layers).step_by(batch_layers) {
            debug_assert!(u32::try_from(z_offset).is_ok());
            let this_batch_layers = usize::min(batch_layers, layers - z_offset);

            for y_offset in (0..out_height).step_by(batch_height) {
                debug_assert!(u32::try_from(y_offset).is_ok());
                let this_batch_height = usize::min(batch_height, out_height - y_offset);

                for x_offset in (0..out_width).step_by(batch_width) {
                    debug_assert!(u32::try_from(x_offset).is_ok());
                    let this_batch_width = usize::min(batch_width, out_width - x_offset);

                    let mut batch_encoder = queue.new_command_encoder();
                    {
                        let mut compute_encoder = batch_encoder.compute();
                        compute_encoder.with_pipeline(&self.bc2);
                        compute_encoder.with_arguments(0, &arguments);
                        compute_encoder.with_constants(&Params {
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
                        });
                        compute_encoder.dispatch(mev::Extent3::new(
                            this_batch_width as u32,
                            this_batch_height as u32,
                            this_batch_layers as u32,
                        ));
                    }
                    let _ = queue.submit([batch_encoder.finish()]);
                }
            }
        }

        Image::with_stride(
            dim,
            [out_width, out_height, layers],
            [out_row_stride, out_plane_stride],
            GpuPixels::new(mev::PixelFormat::Bc2Unorm, out_buf),
        )
    }
}
