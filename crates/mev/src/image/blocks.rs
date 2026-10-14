use crate::image::GpuPixels;
use jkl::image::Image;
use mev::{Arguments, BufferUsage, DeviceRepr};

const SHADER: &str = include_str!("../../../wgpu/shaders/pa_ls_ns.wgsl");

pub struct BlockCompressor {
    bc1: mev::ComputePipeline,
    bc2: mev::ComputePipeline,
    bc3: mev::ComputePipeline,
    bc4: mev::ComputePipeline,
    bc5: mev::ComputePipeline,
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
    batch_width: u32,
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
        let shader = device
            .new_shader_library(mev::LibraryDesc {
                name: "jkl-pa-ls-ns-shader",
                input: mev::LibraryInput::wgsl(SHADER),
            })
            .expect("create block compression shader");
        let pipeline = |entry| {
            device
                .new_compute_pipeline(mev::ComputePipelineDesc {
                    name: entry,
                    shader: shader.entry(entry),
                    work_group_size: [64, 1, 1],
                    constants: Params::SIZE,
                    arguments: &[BlockCompressorArguments::LAYOUT],
                })
                .expect("create block compression pipeline")
        };
        Self {
            bc1: pipeline("compress_bc1"),
            bc2: pipeline("compress_bc2"),
            bc3: pipeline("compress_bc3"),
            bc4: pipeline("compress_bc4"),
            bc5: pipeline("compress_bc5"),
        }
    }

    /// Encodes RGBA8 texels into BC1 with punch-through alpha below the threshold.
    /// Uses principal-axis extrema, up to four least-squares passes, and one endpoint neighbor round.
    pub fn compress_rgba_to_bc1(
        &self,
        image: Image<GpuPixels>,
        alpha_threshold: f32,
        device: &mev::Device,
        queue: &mut mev::Queue,
        batch_size: u32,
        usage: BufferUsage,
    ) -> Image<GpuPixels> {
        self.compress(
            image,
            alpha_threshold,
            device,
            queue,
            batch_size,
            usage,
            &self.bc1,
            mev::PixelFormat::Bc1RgbaUnorm,
        )
    }

    /// Encodes RGBA8 texels into BC2 with explicit four-bit alpha.
    /// Uses principal-axis extrema, up to four least-squares passes, and one endpoint neighbor round.
    pub fn compress_rgba_to_bc2(
        &self,
        image: Image<GpuPixels>,
        device: &mev::Device,
        queue: &mut mev::Queue,
        batch_size: u32,
        usage: BufferUsage,
    ) -> Image<GpuPixels> {
        self.compress(
            image,
            0.0,
            device,
            queue,
            batch_size,
            usage,
            &self.bc2,
            mev::PixelFormat::Bc2Unorm,
        )
    }

    /// Encodes RGBA8 texels into BC3 with interpolated alpha.
    /// Uses principal-axis extrema, up to four least-squares passes, and one endpoint neighbor round.
    pub fn compress_rgba_to_bc3(
        &self,
        image: Image<GpuPixels>,
        device: &mev::Device,
        queue: &mut mev::Queue,
        batch_size: u32,
        usage: BufferUsage,
    ) -> Image<GpuPixels> {
        self.compress(
            image,
            0.0,
            device,
            queue,
            batch_size,
            usage,
            &self.bc3,
            mev::PixelFormat::Bc3Unorm,
        )
    }

    /// Encodes the red channel of RGBA8 texels into BC4 UNORM.
    /// Uses principal-axis extrema, up to four least-squares passes, and one endpoint neighbor round.
    pub fn compress_rgba_to_bc4(
        &self,
        image: Image<GpuPixels>,
        device: &mev::Device,
        queue: &mut mev::Queue,
        batch_size: u32,
        usage: BufferUsage,
    ) -> Image<GpuPixels> {
        self.compress(
            image,
            0.0,
            device,
            queue,
            batch_size,
            usage,
            &self.bc4,
            mev::PixelFormat::Bc4Unorm,
        )
    }

    /// Encodes the red and green channels of RGBA8 texels into BC5 UNORM.
    /// Uses principal-axis extrema, up to four least-squares passes, and one endpoint neighbor round.
    pub fn compress_rgba_to_bc5(
        &self,
        image: Image<GpuPixels>,
        device: &mev::Device,
        queue: &mut mev::Queue,
        batch_size: u32,
        usage: BufferUsage,
    ) -> Image<GpuPixels> {
        self.compress(
            image,
            0.0,
            device,
            queue,
            batch_size,
            usage,
            &self.bc5,
            mev::PixelFormat::Bc5Unorm,
        )
    }

    fn compress(
        &self,
        image: Image<GpuPixels>,
        alpha_threshold: f32,
        device: &mev::Device,
        queue: &mut mev::Queue,
        batch_size: u32,
        usage: BufferUsage,
        pipeline: &mev::ComputePipeline,
        format: mev::PixelFormat,
    ) -> Image<GpuPixels> {
        assert_ne!(batch_size, 0);
        assert!(
            matches!(
                image.data().format,
                mev::PixelFormat::Rgba8Unorm | mev::PixelFormat::Rgba8Srgb
            ),
            "compression requires RGBA8 UNORM or sRGB input"
        );
        let [in_width, in_height, layers] = image.extent().raw_size();
        assert!(in_width != 0 && in_height != 0 && layers != 0);
        let out_width = in_width.div_ceil(4);
        let out_height = in_height.div_ceil(4);
        let block_size = format.block_size();
        let align = 256 / block_size;
        let out_row_stride = out_width.div_ceil(align) * align;
        let out_plane_stride = out_row_stride * out_height;
        let out_buf = device.new_buffer(mev::BufferDesc {
            name: "jkl-blocks-out",
            size: out_plane_stride * layers * block_size,
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | usage,
        });
        let arguments = BlockCompressorArguments {
            image_input: image.data().buffer.clone(),
            image_output: out_buf.clone(),
        };
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
                    let mut encoder = queue.new_command_encoder();
                    {
                        let mut compute = encoder.compute();
                        compute.with_pipeline(pipeline);
                        compute.with_arguments(0, &arguments);
                        compute.with_constants(&params);
                        compute.dispatch(mev::Extent3::new(
                            u32_size(width.div_ceil(64)),
                            u32_size(batch_height.min(out_height - y)),
                            u32_size(batch_layers.min(layers - z)),
                        ));
                    }
                    queue
                        .submit([encoder.finish()])
                        .expect("submit block compression");
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
