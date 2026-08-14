use std::mem::size_of;

use bytemuck::{AnyBitPattern, NoUninit};
use jkl::image::{Dimensions, Extent, Image, ImageMut, ImageRef};

pub mod blocks;
pub mod uploader;

pub struct GpuPixels {
    format: wgpu::TextureFormat,
    buffer: wgpu::Buffer,
}

impl GpuPixels {
    pub fn new(format: wgpu::TextureFormat, buffer: wgpu::Buffer) -> Self {
        GpuPixels { format, buffer }
    }
}

pub trait WgpuImage {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        usage: wgpu::BufferUsages,
        extent: Extent,
    ) -> Self;

    fn map_on_submit(&self, mode: wgpu::MapMode, encoder: &mut wgpu::CommandEncoder);
    fn unmap(&self);

    fn upload<T, U>(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        usage: wgpu::BufferUsages,
        image: ImageRef<T>,
        map: impl Fn(T) -> U,
    ) -> Self
    where
        T: Copy,
        U: NoUninit;

    fn download<T, U>(&self, dst: ImageMut<T>, map: impl Fn(U) -> T)
    where
        U: AnyBitPattern;

    fn copy_to(&self, dst: &Self, encoder: &mut wgpu::CommandEncoder);

    fn make_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture;

    fn format(&self) -> wgpu::TextureFormat;
    fn buffer(&self) -> &wgpu::Buffer;
}

impl WgpuImage for Image<GpuPixels> {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        usage: wgpu::BufferUsages,
        extent: Extent,
    ) -> Image<GpuPixels> {
        let dimsionsions = extent.dimensions();
        let raw_size = extent.raw_size();

        let block_size = format
            .block_copy_size(None)
            .expect("Planar formats not supported");

        assert!(
            256u32.is_multiple_of(block_size),
            "Format size must divide 256 exactly"
        );

        let row_align = 256 / block_size as usize;

        let row_stride = raw_size[0].div_ceil(row_align) * row_align;
        let plane_stride = row_stride * raw_size[1];

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-image-buffer"),
            size: plane_stride as u64 * raw_size[2] as u64 * u64::from(block_size),
            usage,
            mapped_at_creation: false,
        });

        Image::with_stride(
            dimsionsions,
            raw_size,
            [row_stride, plane_stride],
            GpuPixels { format, buffer },
        )
    }

    fn map_on_submit(&self, mode: wgpu::MapMode, encoder: &mut wgpu::CommandEncoder) {
        encoder.map_buffer_on_submit(&self.data().buffer, mode, .., |_| {});
    }

    fn unmap(&self) {
        self.data().buffer.unmap();
    }

    fn upload<T, U>(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        usage: wgpu::BufferUsages,
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

        let block_size = format
            .block_copy_size(None)
            .expect("Planar formats not supported");

        assert_eq!(
            block_size, u_size as u32,
            "Format block size does not match type size"
        );

        let row_align = 256 / u_size;

        let row_stride = raw_size[0].div_ceil(row_align) * row_align;
        let plane_stride = row_stride * raw_size[1];

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-image-buffer"),
            size: plane_stride as u64 * raw_size[2] as u64 * u64::from(block_size),
            usage,
            mapped_at_creation: true,
        });

        {
            let mut mapped = buffer.get_mapped_range_mut(..).unwrap();

            let image = image.reinterpret_as_3d();

            for z in 0..raw_size[2] {
                let plane_offset = z * plane_stride;
                for y in 0..raw_size[1] {
                    let row_offset = plane_offset + y * row_stride;
                    let mut mapped_row = mapped.slice(row_offset * u_size..);

                    for x in 0..raw_size[0] {
                        let pixel = *image.get_pixel(x, y, z);
                        let mapped_pixel = map(pixel);
                        let bytes = bytemuck::bytes_of(&mapped_pixel);
                        mapped_row
                            .slice(x * u_size..x * u_size + u_size)
                            .copy_from_slice(bytes);
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

    fn copy_to(&self, dst: &Image<GpuPixels>, encoder: &mut wgpu::CommandEncoder) {
        assert_eq!(
            self.data().format,
            dst.data().format,
            "Source and destination formats must match"
        );

        let format = self.data().format;
        let block_size = format
            .block_copy_size(None)
            .expect("Planar formats not supported");

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
                    copy_len as u64 * u64::from(block_size),
                );
            }
            (true, false) => {
                // If only the row strides are the same, we can copy each plane at once

                for z in 0..size[2] {
                    let src_offset = z * src_plane_stride;
                    let dst_offset = z * dst_plane_stride;

                    let copy_len = src_row_stride * (size[1] - 1) + size[0] * block_size as usize;

                    encoder.copy_buffer_to_buffer(
                        &self.data().buffer,
                        src_offset as u64 * u64::from(block_size),
                        &dst.data().buffer,
                        dst_offset as u64 * u64::from(block_size),
                        copy_len as u64 * u64::from(block_size),
                    );
                }
            }
            _ => {
                // If the row strides are different, we have to copy each row separately

                for z in 0..size[2] {
                    for y in 0..size[1] {
                        let src_offset = z * src_plane_stride + y * src_row_stride;
                        let dst_offset = z * dst_plane_stride + y * dst_row_stride;

                        let copy_len = size[0] * block_size as usize;

                        encoder.copy_buffer_to_buffer(
                            &self.data().buffer,
                            src_offset as u64 * u64::from(block_size),
                            &dst.data().buffer,
                            dst_offset as u64 * u64::from(block_size),
                            copy_len as u64 * u64::from(block_size),
                        );
                    }
                }
            }
        }
    }

    fn download<T, U>(&self, mut dst: ImageMut<T>, map: impl Fn(U) -> T)
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

        let block_size = self
            .data()
            .format
            .block_copy_size(None)
            .expect("Planar formats not supported");

        assert_eq!(
            block_size, u_size as u32,
            "Format block size does not match type size"
        );

        let extent = self.extent();
        let raw_size = extent.raw_size();

        let row_align = 256 / u_size;

        let row_stride = raw_size[0].div_ceil(row_align) * row_align;
        let plane_stride = row_stride * raw_size[1];

        {
            let src = self.data().buffer.get_mapped_range(..).unwrap();

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

    fn format(&self) -> wgpu::TextureFormat {
        self.data().format
    }

    fn buffer(&self) -> &wgpu::Buffer {
        &self.data().buffer
    }

    fn make_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
        let extent = self.extent();
        let raw_size = extent.raw_size();
        let format = self.data().format;

        let (bw, bh) = format.block_dimensions();
        let block_size = format
            .block_copy_size(None)
            .expect("Planar formats not supported");

        let extent = wgpu::Extent3d {
            width: raw_size[0] as u32 * bw,
            height: raw_size[1] as u32 * bh,
            depth_or_array_layers: raw_size[2] as u32,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jkl-image-texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: match self.dimensions() {
                Dimensions::D1 | Dimensions::D1Array => wgpu::TextureDimension::D1,
                Dimensions::D2 | Dimensions::D2Array => wgpu::TextureDimension::D2,
                Dimensions::D3 => wgpu::TextureDimension::D3,
            },
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.data().buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.row_stride() as u32 * block_size),
                    rows_per_image: Some((self.plane_stride() / self.row_stride()) as u32),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            extent,
        );

        texture
    }
}
