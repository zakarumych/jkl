use std::mem::size_of;

use bytemuck::{AnyBitPattern, NoUninit};
use jkl::image::{Extent, Image, ImageMut, ImageRef};

pub mod blocks;
pub mod uploader;

pub struct GpuPixels {
    format: mev::PixelFormat,
    buffer: mev::Buffer,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use jkl::image::Dimensions;

    fn gpu() -> (mev::Device, mev::Queue) {
        let instance = mev::Instance::load().expect("GPU instance");
        let (idx, family) = instance
            .capabilities()
            .devices
            .iter()
            .enumerate()
            .find_map(|(idx, caps)| {
                caps.families
                    .iter()
                    .position(|family| family.queue_flags.contains(mev::QueueFlags::TRANSFER))
                    .map(|family| (idx, family as u32))
            })
            .expect("GPU transfer queue");
        let features = instance.capabilities().devices[idx].features
            & (mev::Features::BUFFER_MAP
                | mev::Features::BUFFER_MAP_AT_CREATION
                | mev::Features::TEXTURE_COMPRESSION_BC);
        let (device, mut queues) = instance
            .new_device(mev::DeviceDesc {
                idx,
                queues: &[family],
                features,
            })
            .expect("GPU device");
        (device, queues.pop().expect("requested queue"))
    }

    #[test]
    #[ignore = "requires a native GPU device"]
    fn copy_to_preserves_pixels_with_different_strides() {
        let (device, mut queue) = gpu();
        let mut results = Vec::new();
        for (src_stride, dst_stride) in [([4, 12], [4, 12]), ([4, 12], [4, 16]), ([4, 12], [8, 24])]
        {
            let src_pixels: Vec<u32> = (1..=24).collect();
            let dst_pixels = vec![0u32; dst_stride[1] * 2];
            let src = Image::with_stride(
                Dimensions::D3,
                [3, 2, 2],
                src_stride,
                GpuPixels::new(
                    mev::PixelFormat::Rgba8Unorm,
                    device.new_buffer_init(mev::BufferInitDesc {
                        name: "copy source",
                        data: src_pixels.as_slice(),
                        usage: mev::BufferUsage::TRANSFER_SRC,
                    }),
                ),
            );
            let dst = Image::with_stride(
                Dimensions::D3,
                [3, 2, 2],
                dst_stride,
                GpuPixels::new(
                    mev::PixelFormat::Rgba8Unorm,
                    device.new_buffer_init(mev::BufferInitDesc {
                        name: "copy destination",
                        data: dst_pixels.as_slice(),
                        usage: mev::BufferUsage::TRANSFER_DST | mev::BufferUsage::HOST_READ,
                    }),
                ),
            );
            let mut encoder = queue.new_command_encoder();
            src.copy_to(&dst, &mut encoder);
            queue
                .submit_checkpoint([encoder.finish()])
                .expect("submit buffer copy");
            results.push((src_pixels, src_stride, dst_stride, dst));
        }
        queue.wait_idle().expect("wait for buffer copies");
        drop(queue);
        for (src_pixels, src_stride, dst_stride, mut dst) in results {
            let mut actual = vec![0u32; dst_stride[1] * 2];
            dst.data_mut()
                .buffer
                .read(0, bytemuck::cast_slice_mut(&mut actual))
                .expect("read copied pixels");
            for z in 0..2 {
                for y in 0..2 {
                    let src_offset = z * src_stride[1] + y * src_stride[0];
                    let dst_offset = z * dst_stride[1] + y * dst_stride[0];
                    assert_eq!(
                        &actual[dst_offset..dst_offset + 3],
                        &src_pixels[src_offset..src_offset + 3]
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a native GPU device"]
    fn upload_and_download_preserve_pixels() {
        let (device, _queue) = gpu();
        let pixels = vec![1u32, 2, 3, 4, 5, 6];
        let src = Image::new_2d(3, 2, pixels.clone().into_boxed_slice());
        let mut uploaded = Image::<GpuPixels>::upload(
            &device,
            mev::PixelFormat::Rgba8Unorm,
            mev::BufferUsage::HOST_READ | mev::BufferUsage::TRANSFER_SRC,
            src.as_ref(),
            |pixel| pixel,
        );
        assert!(
            uploaded
                .buffer()
                .usage()
                .contains(mev::BufferUsage::HOST_WRITE)
        );
        let mut dst = Image::new_2d(3, 2, vec![0u32; 6].into_boxed_slice());
        uploaded.download(dst.as_mut(), |pixel: u32| pixel);
        assert_eq!(dst.data().as_ref(), pixels.as_slice());
    }

    #[test]
    #[ignore = "requires a native GPU device"]
    fn make_image_handles_dimensions_and_array_layers() {
        let (device, mut queue) = gpu();
        for (dimensions, size, stride, expected_extent, layers) in [
            (
                Dimensions::D1,
                [3, 1, 1],
                [64, 64],
                mev::ImageExtent::D1(mev::Extent1::new(3)),
                1,
            ),
            (
                Dimensions::D1Array,
                [3, 2, 1],
                [64, 0],
                mev::ImageExtent::D1(mev::Extent1::new(3)),
                2,
            ),
            (
                Dimensions::D2,
                [3, 2, 1],
                [64, 192],
                mev::ImageExtent::D2(mev::Extent2::new(3, 2)),
                1,
            ),
            (
                Dimensions::D2Array,
                [3, 2, 2],
                [64, 192],
                mev::ImageExtent::D2(mev::Extent2::new(3, 2)),
                2,
            ),
            (
                Dimensions::D3,
                [3, 2, 2],
                [64, 192],
                mev::ImageExtent::D3(mev::Extent3::new(3, 2, 2)),
                1,
            ),
        ] {
            let pixels = vec![0u32; 384];
            let src = Image::with_stride(
                dimensions,
                size,
                stride,
                GpuPixels::new(
                    mev::PixelFormat::Rgba8Unorm,
                    device.new_buffer_init(mev::BufferInitDesc {
                        name: "image source",
                        data: pixels.as_slice(),
                        usage: mev::BufferUsage::TRANSFER_SRC,
                    }),
                ),
            );
            let mut encoder = queue.new_command_encoder();
            let image = src.make_image(&device, &mut encoder);
            assert_eq!(image.extent(), expected_extent);
            assert_eq!(image.layers(), layers);
            queue
                .submit_checkpoint([encoder.finish()])
                .expect("submit image copy");
        }
        queue.wait_idle().expect("wait for image copies");
    }
}

impl GpuPixels {
    pub fn new(format: mev::PixelFormat, buffer: mev::Buffer) -> Self {
        GpuPixels { format, buffer }
    }
}

pub trait MevImage {
    fn new(
        device: &mev::Device,
        format: mev::PixelFormat,
        usage: mev::BufferUsage,
        extent: Extent,
    ) -> Self;

    fn upload<T, U>(
        device: &mev::Device,
        format: mev::PixelFormat,
        usage: mev::BufferUsage,
        image: ImageRef<T>,
        map: impl Fn(T) -> U,
    ) -> Self
    where
        T: Copy,
        U: NoUninit;

    fn download<T, U>(&mut self, dst: ImageMut<T>, map: impl Fn(U) -> T)
    where
        U: AnyBitPattern;

    fn copy_to(&self, dst: &Self, encoder: &mut mev::CommandEncoder);

    fn make_image(&self, device: &mev::Device, encoder: &mut mev::CommandEncoder) -> mev::Image;

    fn format(&self) -> mev::PixelFormat;
    fn buffer(&self) -> &mev::Buffer;
}

impl MevImage for Image<GpuPixels> {
    fn new(
        device: &mev::Device,
        format: mev::PixelFormat,
        usage: mev::BufferUsage,
        extent: Extent,
    ) -> Image<GpuPixels> {
        let dimsionsions = extent.dimensions();
        let raw_size = extent.raw_size();

        let block_size = format.block_size();

        assert!(
            256usize.is_multiple_of(block_size),
            "Format size must divide 256 exactly"
        );

        let row_align = 256 / block_size as usize;

        let row_stride = raw_size[0].div_ceil(row_align) * row_align;
        let plane_stride = row_stride * raw_size[1];

        let buffer = device.new_buffer(mev::BufferDesc {
            name: "jkl-image-buffer",
            size: plane_stride * raw_size[2] * block_size,
            usage,
        });

        Image::with_stride(
            dimsionsions,
            raw_size,
            [row_stride, plane_stride],
            GpuPixels { format, buffer },
        )
    }

    fn upload<T, U>(
        device: &mev::Device,
        format: mev::PixelFormat,
        usage: mev::BufferUsage,
        image: ImageRef<T>,
        map: impl Fn(T) -> U,
    ) -> Image<GpuPixels>
    where
        T: Copy,
        U: NoUninit,
    {
        const {
            assert!(
                256usize.is_multiple_of(size_of::<U>()),
                "Format size must be divide 256 exactly"
            );
        }

        let extent = image.extent();
        let dimsionsions = extent.dimensions();
        let raw_size = extent.raw_size();

        let u_size = size_of::<U>();

        let block_size = format.block_size();

        assert_eq!(
            block_size, u_size,
            "Format block size does not match type size"
        );

        let row_align = 256 / u_size;

        let row_stride = raw_size[0].div_ceil(row_align) * row_align;
        let plane_stride = row_stride * raw_size[1];

        let mut buffer = device.new_buffer_mapped(mev::BufferDesc {
            name: "jkl-image-buffer",
            size: plane_stride * raw_size[2] * block_size,
            usage: usage | mev::BufferUsage::HOST_WRITE,
        });

        if let Ok(mut mapped) = buffer.write_mapped_range(..) {
            let image = image.reinterpret_as_3d();

            for z in 0..raw_size[2] {
                let plane_offset = z * plane_stride;
                for y in 0..raw_size[1] {
                    let row_offset = plane_offset + y * row_stride;
                    for x in 0..raw_size[0] {
                        let pixel = *image.get_pixel(x, y, z);
                        let mapped_pixel = map(pixel);
                        let bytes = bytemuck::bytes_of(&mapped_pixel);
                        mapped.write((row_offset + x) * u_size, bytes);
                    }
                }
            }
        }
        buffer.unmap();

        Image::with_stride(
            dimsionsions,
            raw_size,
            [row_stride, plane_stride],
            GpuPixels { format, buffer },
        )
    }

    fn copy_to(&self, dst: &Image<GpuPixels>, encoder: &mut mev::CommandEncoder) {
        assert_eq!(
            self.data().format,
            dst.data().format,
            "Source and destination formats must match"
        );

        let format = self.data().format;
        let block_size = format.block_size();

        let src_extent = self.extent();
        let dst_extent = dst.extent();

        assert_eq!(src_extent, dst_extent);

        let size = src_extent.raw_size();

        if size[0] == 0 || size[1] == 0 || size[2] == 0 {
            return;
        }

        let src_row_stride = self.row_stride();
        let src_plane_stride = self.plane_stride();
        let dst_row_stride = dst.row_stride();
        let dst_plane_stride = dst.plane_stride();

        let mut encoder = encoder.copy();

        match (
            src_row_stride == dst_row_stride,
            src_plane_stride == dst_plane_stride,
        ) {
            (true, true) => {
                // If the row and plane strides are the same, we can copy the entire buffer at once

                let copy_len =
                    src_plane_stride * (size[2] - 1) + src_row_stride * (size[1] - 1) + size[0];

                encoder.copy_buffer_to_buffer(
                    &self.data().buffer.slice(..copy_len * block_size),
                    &dst.data().buffer.slice(..copy_len * block_size),
                );
            }
            (true, false) => {
                // If only the row strides are the same, we can copy each plane at once

                for z in 0..size[2] {
                    let src_offset = z * src_plane_stride;
                    let dst_offset = z * dst_plane_stride;

                    let copy_len = src_row_stride * (size[1] - 1) + size[0];

                    encoder.copy_buffer_to_buffer(
                        &self
                            .data()
                            .buffer
                            .slice(src_offset * block_size..(src_offset + copy_len) * block_size),
                        &dst.data()
                            .buffer
                            .slice(dst_offset * block_size..(dst_offset + copy_len) * block_size),
                    );
                }
            }
            _ => {
                // If the row strides are different, we have to copy each row separately

                for z in 0..size[2] {
                    for y in 0..size[1] {
                        let src_offset = z * src_plane_stride + y * src_row_stride;
                        let dst_offset = z * dst_plane_stride + y * dst_row_stride;

                        let copy_len = size[0];

                        encoder.copy_buffer_to_buffer(
                            &self.data().buffer.slice(
                                src_offset * block_size..(src_offset + copy_len) * block_size,
                            ),
                            &dst.data().buffer.slice(
                                dst_offset * block_size..(dst_offset + copy_len) * block_size,
                            ),
                        );
                    }
                }
            }
        }
    }

    fn download<T, U>(&mut self, mut dst: ImageMut<T>, map: impl Fn(U) -> T)
    where
        U: AnyBitPattern,
    {
        const {
            assert!(
                256usize.is_multiple_of(size_of::<U>()),
                "Format size must be divide 256 exactly"
            );
        }

        assert_eq!(
            self.extent(),
            dst.extent(),
            "Source and destination extents must match"
        );

        let u_size = size_of::<U>();

        let block_size = self.data().format.block_size();

        assert_eq!(
            block_size, u_size,
            "Format block size does not match type size"
        );

        let extent = self.extent();
        let raw_size = extent.raw_size();

        let row_align = 256 / u_size;

        let row_stride = raw_size[0].div_ceil(row_align) * row_align;
        let plane_stride = row_stride * raw_size[1];

        if self.data_mut().buffer.map(..).is_err() {
            return;
        }

        if let Ok(mapped) = self.data_mut().buffer.read_mapped_range(..) {
            let src = mapped.as_ref();
            let mut dst = dst.as_mut_3d();

            for z in 0..raw_size[2] {
                let plane_offset = z * plane_stride;
                for y in 0..raw_size[1] {
                    let row_offset = plane_offset + y * row_stride;
                    let src_row = &src[row_offset * u_size..];

                    for x in 0..raw_size[0] {
                        let bytes = &src_row[x * u_size..][..u_size];
                        let pixel: U = *bytemuck::from_bytes(bytes);

                        let mapped_pixel = map(pixel);
                        dst.set(x, y, z, mapped_pixel);
                    }
                }
            }
        }
        self.data_mut().buffer.unmap();
    }

    fn format(&self) -> mev::PixelFormat {
        self.data().format
    }

    fn buffer(&self) -> &mev::Buffer {
        &self.data().buffer
    }

    fn make_image(&self, device: &mev::Device, encoder: &mut mev::CommandEncoder) -> mev::Image {
        let extent = self.extent();
        let format = self.data().format;

        let block_extent = format.block_extent();
        let block_size = format.block_size();

        let image_extent = match extent {
            Extent::D1 { width } | Extent::D1Array { width, .. } => {
                mev::ImageExtent::D1(mev::Extent1::new(width as u32 * block_extent.width()))
            }

            Extent::D2 { width, height } | Extent::D2Array { width, height, .. } => {
                mev::ImageExtent::D2(mev::Extent2::new(
                    width as u32 * block_extent.width(),
                    height as u32 * block_extent.height(),
                ))
            }
            Extent::D3 {
                width,
                height,
                depth,
            } => mev::ImageExtent::D3(mev::Extent3::new(
                width as u32 * block_extent.width(),
                height as u32 * block_extent.height(),
                depth as u32,
            )),
        };

        let layers = match extent {
            Extent::D1 { .. } | Extent::D2 { .. } | Extent::D3 { .. } => 1,
            Extent::D1Array { layers, .. } | Extent::D2Array { layers, .. } => layers as u32,
        };

        let image = device.new_image(mev::ImageDesc {
            name: "jkl-image",
            dims: (image_extent, layers),
            format,
            levels: 1,
            usage: mev::ImageUsage::TRANSFER_DST | mev::ImageUsage::SAMPLED,
        });

        encoder.init_image(
            mev::PipelineStages::empty(),
            mev::PipelineStages::TRANSFER,
            &image,
        );
        let mut copy = encoder.copy();
        match image_extent {
            mev::ImageExtent::D1(extent) => {
                for layer in 0..layers {
                    let offset = layer as usize * self.row_stride() * block_size;
                    copy.copy_buffer_to_image(
                        &self.data().buffer.slice(offset..),
                        mev::TexelShape::new(format, extent),
                        &image.slice((extent, (layer..layer + 1).into())),
                    );
                }
            }
            mev::ImageExtent::D2(extent) => {
                for layer in 0..layers {
                    let offset = layer as usize * self.plane_stride() * block_size;
                    copy.copy_buffer_to_image(
                        &self.data().buffer.slice(offset..),
                        mev::TexelShape::new(format, extent)
                            .with_row_stride(self.row_stride() * block_size),
                        &image.slice((extent, (layer..layer + 1).into())),
                    );
                }
            }
            mev::ImageExtent::D3(extent) => {
                copy.copy_buffer_to_image(
                    &self.data().buffer,
                    mev::TexelShape::new(format, extent)
                        .with_row_stride(self.row_stride() * block_size)
                        .with_plane_stride(self.plane_stride() * block_size),
                    &image,
                );
            }
        }

        image
    }
}
