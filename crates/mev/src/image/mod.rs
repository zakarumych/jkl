use std::mem::size_of;

use bytemuck::{AnyBitPattern, NoUninit};
use jkl::image::{Extent, Image, ImageMut, ImageRef};

pub mod blocks;
pub mod uploader;

pub struct GpuPixels {
    format: mev::PixelFormat,
    buffer: mev::Buffer,
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

        let mut buffer = device.new_buffer(mev::BufferDesc {
            name: "jkl-image-buffer",
            size: plane_stride * raw_size[2] * block_size,
            usage: usage | mev::BufferUsage::HOST_WRITE,
        });

        if let Ok(mut mapped) = buffer.write_mapped_range(..) {
            let dst = mapped.as_mut();
            let image = image.reinterpret_as_3d();

            for z in 0..raw_size[2] {
                let plane_offset = z * plane_stride;
                for y in 0..raw_size[1] {
                    let row_offset = plane_offset + y * row_stride;
                    let dst_row = &mut dst[row_offset * u_size..];

                    for x in 0..raw_size[0] {
                        let pixel = *image.get_pixel(x, y, z);
                        let mapped_pixel = map(pixel);
                        let bytes = bytemuck::bytes_of(&mapped_pixel);
                        dst_row[x * u_size..][..u_size].copy_from_slice(bytes);
                    }
                }
            }
        }

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
                    &self.data().buffer,
                    0,
                    &dst.data().buffer,
                    0,
                    copy_len * block_size,
                );
            }
            (true, false) => {
                // If only the row strides are the same, we can copy each plane at once

                for z in 0..size[2] {
                    let src_offset = z * src_plane_stride;
                    let dst_offset = z * dst_plane_stride;

                    let copy_len = src_row_stride * (size[1] - 1) + size[0] * block_size;

                    encoder.copy_buffer_to_buffer(
                        &self.data().buffer,
                        src_offset * block_size,
                        &dst.data().buffer,
                        dst_offset * block_size,
                        copy_len * block_size,
                    );
                }
            }
            _ => {
                // If the row strides are different, we have to copy each row separately

                for z in 0..size[2] {
                    for y in 0..size[1] {
                        let src_offset = z * src_plane_stride + y * src_row_stride;
                        let dst_offset = z * dst_plane_stride + y * dst_row_stride;

                        let copy_len = size[0] * block_size;

                        encoder.copy_buffer_to_buffer(
                            &self.data().buffer,
                            src_offset * block_size,
                            &dst.data().buffer,
                            dst_offset * block_size,
                            copy_len * block_size,
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
            extent: image_extent,
            format,
            levels: 1,
            layers,
            usage: mev::ImageUsage::TRANSFER_DST | mev::ImageUsage::SAMPLED,
        });

        encoder.copy().copy_buffer_to_image(
            &self.data().buffer,
            0,
            self.row_stride() * block_size,
            self.plane_stride() * block_size,
            &image,
            mev::Offset::ZERO,
            image_extent.into_3d(),
            0..layers,
            0,
        );

        image
    }
}
