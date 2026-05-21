//! Lossless image format conversions.
//!
//! Provides traits for converting an image from one element
//! type to another where the conversion is obvious.
//!

use crate::{
    image::{
        Dimensions, Image, Image1DArrayRef, Image1DRef, Image2DArrayRef, Image2DRef, Image3DRef,
        ImageRef, OwnedImage, OwnedImage1D, OwnedImage1DArray, OwnedImage2D, OwnedImage2DArray,
        OwnedImage3D,
        block::{bc1, bc2},
    },
    math::{Rgb8U, Rgba8U},
};

/// Converts an image to an equivalent representation in a different element type.
///
/// The conversion may be lossless or lossy, but should be obvious and not require user-specified parameters.
/// For example, converting from RGB8 to RGBA8 is lossless and obvious (just add an alpha channel with value 255),
/// but converting from RGBA8 to BC1 is lossy and not obvious (it requires choosing a quantization method and error metric).
pub trait IntoFormat<T>: Sized {
    /// Produces an owned image with element type `T`.
    fn into_format_1d(image: Image1DRef<'_, Self>) -> OwnedImage1D<T>;

    /// Produces an owned image with element type `T`.
    fn into_format_1d_array(image: Image1DArrayRef<'_, Self>) -> OwnedImage1DArray<T>;

    /// Produces an owned image with element type `T`.
    fn into_format_2d(image: Image2DRef<'_, Self>) -> OwnedImage2D<T>;

    /// Produces an owned image with element type `T`.
    fn into_format_2d_array(image: Image2DArrayRef<'_, Self>) -> OwnedImage2DArray<T>;

    /// Produces an owned image with element type `T`.
    fn into_format_3d(image: Image3DRef<'_, Self>) -> OwnedImage3D<T>;

    /// Produces an owned image with element type `T`.
    fn into_format(image: ImageRef<'_, Self>) -> OwnedImage<T> {
        match image.dimensions() {
            Dimensions::D1 => Self::into_format_1d(image.as_1d().unwrap()).into(),
            Dimensions::D1Array => Self::into_format_1d_array(image.as_1d_array().unwrap()).into(),
            Dimensions::D2 => Self::into_format_2d(image.as_2d().unwrap()).into(),
            Dimensions::D2Array => Self::into_format_2d_array(image.as_2d_array().unwrap()).into(),
            Dimensions::D3 => Self::into_format_3d(image.as_3d().unwrap()).into(),
        }
    }
}

impl<T> IntoFormat<T> for T
where
    T: Copy,
{
    #[inline]
    fn into_format_1d(image: Image1DRef<'_, T>) -> OwnedImage1D<T> {
        OwnedImage1D::new(image.width(), image.iter_pixels().copied().collect())
    }

    #[inline]
    fn into_format_1d_array(image: Image1DArrayRef<'_, Self>) -> OwnedImage1DArray<T> {
        OwnedImage1DArray::new(
            image.width(),
            image.layers(),
            image.iter_pixels().copied().collect(),
        )
    }

    #[inline]
    fn into_format_2d(image: Image2DRef<'_, T>) -> OwnedImage2D<T> {
        OwnedImage2D::new(
            image.width(),
            image.height(),
            image.iter_pixels().copied().collect(),
        )
    }

    #[inline]
    fn into_format_2d_array(image: Image2DArrayRef<'_, T>) -> OwnedImage2DArray<T> {
        OwnedImage2DArray::new(
            image.width(),
            image.height(),
            image.layers(),
            image.iter_pixels().copied().collect(),
        )
    }

    #[inline]
    fn into_format_3d(image: Image3DRef<'_, T>) -> OwnedImage3D<T> {
        OwnedImage3D::new(
            image.width(),
            image.height(),
            image.depth(),
            image.iter_pixels().copied().collect(),
        )
    }
}

impl IntoFormat<Rgba8U> for Rgb8U {
    #[inline]
    fn into_format_1d(image: Image1DRef<'_, Rgb8U>) -> OwnedImage1D<Rgba8U> {
        OwnedImage1D::new(
            image.width(),
            image
                .iter_pixels()
                .map(|&pixel| pixel.into_opaque())
                .collect(),
        )
    }

    #[inline]
    fn into_format_1d_array(image: Image1DArrayRef<'_, Rgb8U>) -> OwnedImage1DArray<Rgba8U> {
        OwnedImage1DArray::new(
            image.width(),
            image.layers(),
            image
                .iter_pixels()
                .map(|&pixel| pixel.into_opaque())
                .collect(),
        )
    }

    #[inline]
    fn into_format_2d(image: Image2DRef<'_, Rgb8U>) -> OwnedImage2D<Rgba8U> {
        OwnedImage2D::new(
            image.width(),
            image.height(),
            image
                .iter_pixels()
                .map(|&pixel| pixel.into_opaque())
                .collect(),
        )
    }

    #[inline]
    fn into_format_2d_array(image: Image2DArrayRef<'_, Rgb8U>) -> OwnedImage2DArray<Rgba8U> {
        OwnedImage2DArray::new(
            image.width(),
            image.height(),
            image.layers(),
            image
                .iter_pixels()
                .map(|&pixel| pixel.into_opaque())
                .collect(),
        )
    }

    #[inline]
    fn into_format_3d(image: Image3DRef<'_, Rgb8U>) -> OwnedImage3D<Rgba8U> {
        OwnedImage3D::new(
            image.width(),
            image.height(),
            image.depth(),
            image
                .iter_pixels()
                .map(|&pixel| pixel.into_opaque())
                .collect(),
        )
    }
}

impl IntoFormat<Rgb8U> for Rgba8U {
    #[inline]
    fn into_format_1d(image: Image1DRef<'_, Rgba8U>) -> OwnedImage1D<Rgb8U> {
        OwnedImage1D::new(
            image.width(),
            image.iter_pixels().map(|&pixel| pixel.rgb()).collect(),
        )
    }

    #[inline]
    fn into_format_1d_array(image: Image1DArrayRef<'_, Rgba8U>) -> OwnedImage1DArray<Rgb8U> {
        OwnedImage1DArray::new(
            image.width(),
            image.layers(),
            image.iter_pixels().map(|&pixel| pixel.rgb()).collect(),
        )
    }

    #[inline]
    fn into_format_2d(image: Image2DRef<'_, Rgba8U>) -> OwnedImage2D<Rgb8U> {
        OwnedImage2D::new(
            image.width(),
            image.height(),
            image.iter_pixels().map(|&pixel| pixel.rgb()).collect(),
        )
    }

    #[inline]
    fn into_format_2d_array(image: Image2DArrayRef<'_, Rgba8U>) -> OwnedImage2DArray<Rgb8U> {
        OwnedImage2DArray::new(
            image.width(),
            image.height(),
            image.layers(),
            image.iter_pixels().map(|&pixel| pixel.rgb()).collect(),
        )
    }

    #[inline]
    fn into_format_3d(image: Image3DRef<'_, Rgba8U>) -> OwnedImage3D<Rgb8U> {
        OwnedImage3D::new(
            image.width(),
            image.height(),
            image.depth(),
            image.iter_pixels().map(|&pixel| pixel.rgb()).collect(),
        )
    }
}

impl IntoFormat<Rgb8U> for bc1::Block {
    #[inline]
    fn into_format_1d(_: Image1DRef<'_, bc1::Block>) -> OwnedImage1D<Rgb8U> {
        unreachable!("BC1 image cannot be 1D");
    }

    #[inline]
    fn into_format_1d_array(_: Image1DArrayRef<'_, bc1::Block>) -> OwnedImage1DArray<Rgb8U> {
        unreachable!("BC1 image cannot be 1D array");
    }

    #[inline]
    fn into_format_2d(image: Image2DRef<'_, bc1::Block>) -> OwnedImage2D<Rgb8U> {
        let width = image.width() * 4;
        let height = image.height() * 4;
        let mut out = OwnedImage2D::new(
            width,
            height,
            vec![Rgb8U::BLACK; width * height].into_boxed_slice(),
        );
        bc1::decode_image(
            Image::from(image),
            |rgba| Rgb8U::from_f32(rgba.rgb()),
            Image::from(out.as_mut()),
        );
        out
    }

    #[inline]
    fn into_format_2d_array(image: Image2DArrayRef<'_, bc1::Block>) -> OwnedImage2DArray<Rgb8U> {
        let width = image.width() * 4;
        let height = image.height() * 4;
        let layers = image.layers();
        let mut out = OwnedImage2DArray::new(
            width,
            height,
            layers,
            vec![Rgb8U::BLACK; width * height * layers].into_boxed_slice(),
        );
        bc1::decode_image(
            Image::from(image),
            |rgba| Rgb8U::from_f32(rgba.rgb()),
            Image::from(out.as_mut()),
        );
        out
    }

    #[inline]
    fn into_format_3d(image: Image3DRef<'_, bc1::Block>) -> OwnedImage3D<Rgb8U> {
        let width = image.width() * 4;
        let height = image.height() * 4;
        let depth = image.depth();
        let mut out = OwnedImage3D::new(
            width,
            height,
            depth,
            vec![Rgb8U::BLACK; width * height * depth].into_boxed_slice(),
        );
        bc1::decode_image(
            Image::from(image),
            |rgba| Rgb8U::from_f32(rgba.rgb()),
            Image::from(out.as_mut()),
        );
        out
    }
}

impl IntoFormat<Rgba8U> for bc1::Block {
    #[inline]
    fn into_format_1d(_: Image1DRef<'_, bc1::Block>) -> OwnedImage1D<Rgba8U> {
        unreachable!("BC1 image cannot be 1D");
    }

    #[inline]
    fn into_format_1d_array(_: Image1DArrayRef<'_, bc1::Block>) -> OwnedImage1DArray<Rgba8U> {
        unreachable!("BC1 image cannot be 1D array");
    }

    #[inline]
    fn into_format_2d(image: Image2DRef<'_, bc1::Block>) -> OwnedImage2D<Rgba8U> {
        let width = image.width() * 4;
        let height = image.height() * 4;
        let mut out = OwnedImage2D::new(
            width,
            height,
            vec![Rgba8U::TRANSPARENT; width * height].into_boxed_slice(),
        );
        bc1::decode_image(
            Image::from(image),
            |rgba| Rgba8U::from_f32(rgba),
            Image::from(out.as_mut()),
        );
        out
    }

    #[inline]
    fn into_format_2d_array(image: Image2DArrayRef<'_, bc1::Block>) -> OwnedImage2DArray<Rgba8U> {
        let width = image.width() * 4;
        let height = image.height() * 4;
        let layers = image.layers();
        let mut out = OwnedImage2DArray::new(
            width,
            height,
            layers,
            vec![Rgba8U::TRANSPARENT; width * height * layers].into_boxed_slice(),
        );
        bc1::decode_image(
            Image::from(image),
            |rgba| Rgba8U::from_f32(rgba),
            Image::from(out.as_mut()),
        );
        out
    }

    #[inline]
    fn into_format_3d(image: Image3DRef<'_, bc1::Block>) -> OwnedImage3D<Rgba8U> {
        let width = image.width() * 4;
        let height = image.height() * 4;
        let depth = image.depth();
        let mut out = OwnedImage3D::new(
            width,
            height,
            depth,
            vec![Rgba8U::TRANSPARENT; width * height * depth].into_boxed_slice(),
        );
        bc1::decode_image(
            Image::from(image),
            |rgba| Rgba8U::from_f32(rgba),
            Image::from(out.as_mut()),
        );
        out
    }
}

impl IntoFormat<Rgba8U> for bc2::Block {
    #[inline]
    fn into_format_1d(_: Image1DRef<'_, bc2::Block>) -> OwnedImage1D<Rgba8U> {
        unreachable!("BC2 image cannot be 1D");
    }

    #[inline]
    fn into_format_1d_array(_: Image1DArrayRef<'_, bc2::Block>) -> OwnedImage1DArray<Rgba8U> {
        unreachable!("BC2 image cannot be 1D array");
    }

    #[inline]
    fn into_format_2d(image: Image2DRef<'_, bc2::Block>) -> OwnedImage2D<Rgba8U> {
        let width = image.width() * 4;
        let height = image.height() * 4;
        let mut out = OwnedImage2D::new(
            width,
            height,
            vec![Rgba8U::TRANSPARENT; width * height].into_boxed_slice(),
        );
        bc2::decode_image(
            Image::from(image),
            |rgba| Rgba8U::from_f32(rgba),
            Image::from(out.as_mut()),
        );
        out
    }

    #[inline]
    fn into_format_2d_array(image: Image2DArrayRef<'_, bc2::Block>) -> OwnedImage2DArray<Rgba8U> {
        let width = image.width() * 4;
        let height = image.height() * 4;
        let layers = image.layers();
        let mut out = OwnedImage2DArray::new(
            width,
            height,
            layers,
            vec![Rgba8U::TRANSPARENT; width * height * layers].into_boxed_slice(),
        );
        bc2::decode_image(
            Image::from(image),
            |rgba| Rgba8U::from_f32(rgba),
            Image::from(out.as_mut()),
        );
        out
    }

    #[inline]
    fn into_format_3d(image: Image3DRef<'_, bc2::Block>) -> OwnedImage3D<Rgba8U> {
        let width = image.width() * 4;
        let height = image.height() * 4;
        let depth = image.depth();
        let mut out = OwnedImage3D::new(
            width,
            height,
            depth,
            vec![Rgba8U::TRANSPARENT; width * height * depth].into_boxed_slice(),
        );
        bc2::decode_image(
            Image::from(image),
            |rgba| Rgba8U::from_f32(rgba),
            Image::from(out.as_mut()),
        );
        out
    }
}
