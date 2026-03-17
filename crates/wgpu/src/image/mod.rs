use std::mem::size_of;

use bytemuck::NoUninit;
use jkl::image::{Dimensions, Image, ImageRef};

pub mod blocks;
pub mod uploader;

pub struct PixelBuffer {
    format: wgpu::TextureFormat,
    buffer: wgpu::Buffer,
}

impl PixelBuffer {
    pub fn new(format: wgpu::TextureFormat, buffer: wgpu::Buffer) -> Self {
        PixelBuffer { format, buffer }
    }

    pub fn upload<T, U>(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        usage: wgpu::BufferUsages,
        image: ImageRef<T>,
        map: impl Fn(T) -> U,
    ) -> Image<PixelBuffer>
    where
        T: Copy,
        U: NoUninit,
    {
        let extent = image.extent();
        let dimsionsions = extent.dimensions();
        let raw_size = extent.raw_size();

        let row_stride = (raw_size[0] * size_of::<U>()).div_ceil(256) * 256;
        let plane_stride = row_stride * raw_size[1];

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jkl-image-buffer"),
            size: plane_stride as u64 * raw_size[2] as u64,
            usage,
            mapped_at_creation: true,
        });

        {
            let mut mapped = buffer.get_mapped_range_mut(..);

            let image = image.as_ref_3d();

            for z in 0..raw_size[2] {
                for y in 0..raw_size[1] {
                    let offset = (z * plane_stride + y * row_stride) as usize;
                    let mapped_row = &mut mapped[offset..];

                    for x in 0..raw_size[0] {
                        let pixel = *image.get(x, y, z);
                        let mapped_pixel = map(pixel);
                        let bytes = bytemuck::bytes_of(&mapped_pixel);
                        mapped_row[x as usize * size_of::<U>()..][..bytes.len()]
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
            PixelBuffer { format, buffer },
        )
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn copy_to_texture(
        image: &Image<PixelBuffer>,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
        let extent = image.extent();
        let raw_size = extent.raw_size();

        let (bw, bh) = image.data().format.block_dimensions();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jkl-image-texture"),
            size: wgpu::Extent3d {
                width: raw_size[0] as u32 * bw,
                height: raw_size[1] as u32 * bh,
                depth_or_array_layers: raw_size[2] as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: match extent.dimensions() {
                Dimensions::D1 | Dimensions::D1Array => wgpu::TextureDimension::D1,
                Dimensions::D2 | Dimensions::D2Array => wgpu::TextureDimension::D2,
                Dimensions::D3 => wgpu::TextureDimension::D3,
            },
            format: image.data().format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &image.data().buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(image.row_stride() as u32),
                    rows_per_image: Some((image.plane_stride() / image.row_stride()) as u32),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: raw_size[0] as u32 * bw,
                height: raw_size[1] as u32 * bh,
                depth_or_array_layers: raw_size[2] as u32,
            },
        );

        texture
    }
}
