use std::{hash::Hash, io};

use crate::{
    algos::vle::Vle,
    bits::{ReadBits, write_bits_scope},
    encode::{FixedCode, VarCode},
    image::{
        Image2DMut, Image2DRef,
        block::{bc1, bc2},
        compress::Compressor,
        format::Format,
    },
    jackal::image::WriteOffsets,
    math::{Rgb8U, Rgb565, Rgba8U},
};

/// This trait is an interface for compression images.
pub trait Pixel: Copy + Eq + Hash + FixedCode + VarCode + 'static {
    type Context<C: Compressor>: VarCode;

    const FORMAT: Format;

    /// Compress the input images of data.
    ///
    /// `input` is an iterator over images, where each image is made of pixels of type `Self`.
    /// `compressor` is the compression algorithm to use.
    /// `write` is the output stream to write compressed data to.
    ///
    /// Function writes a context of type `Self::Context<C>` to the output stream,
    /// followed by offsets for each image tile in `input` and finally the compressed data for each tile
    /// at the corresponding offsets.
    fn compress_images<'a, C>(
        input: impl Iterator<Item = Image2DRef<'a, Self>> + Clone,
        compressor: C,
        write: impl io::Write + io::Seek,
    ) -> io::Result<()>
    where
        C: Compressor;

    #[inline]
    fn read_context<C>(read: impl io::Read) -> io::Result<Self::Context<C>>
    where
        C: Compressor,
    {
        let mut read_bits = ReadBits::new(read);
        Self::Context::<C>::var_read(&mut read_bits)
    }

    /// Decompress the input compressed data into output images.
    /// `compressor` is the compression algorithm to use.
    /// `context` is the decompression context produced during compression.
    /// `read` is the input stream to read compressed data from.
    /// `output` is an image buffer to write decompressed pixels to.
    fn decompress_image<'a, C>(
        compressor: C,
        context: &Self::Context<C>,
        read: impl io::Read,
        image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor;
}

impl Pixel for Rgb8U {
    type Context<C: Compressor> = C::Context<Vle<u32>>;

    const FORMAT: Format = Format::RGB8;

    fn compress_images<'a, C>(
        input: impl Iterator<Item = Image2DRef<'a, Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut tokens = Vec::new();

        let context = compressor.compress_symbols(
            input.map(|image| image.iter_pixels().map(|rgb| Vle(rgb.bits_interleaved()))),
            &mut tokens,
        )?;

        let mut offsets = WriteOffsets::new(tokens.len(), &mut write)?;

        write_bits_scope(&mut write, |write_bits| context.var_write(write_bits))?;

        for token_group in &tokens {
            offsets.push_next(&mut write)?;

            write_bits_scope(&mut write, |write| {
                for token in token_group {
                    token.var_write(write)?;
                }
                Ok(())
            })?;
        }

        offsets.write(&mut write)?;
        Ok(())
    }

    fn decompress_image<'a, C>(
        compressor: C,
        context: &Self::Context<C>,
        read: impl io::Read,
        mut image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut read_bits = crate::bits::ReadBits::new(read);

        let mut symbols = compressor.decompress_tokens2(context, read_tokens(&mut read_bits));

        let height = image.height();
        for y in 0..height {
            let mut row = image.get_row_mut(y);
            for pixel in row.iter_pixels_mut() {
                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                *pixel = Rgb8U::from_bits_interleaved(bits);
            }
        }

        Ok(())
    }
}

impl Pixel for Rgba8U {
    type Context<C: Compressor> = C::Context<Vle<u32>>;

    const FORMAT: Format = Format::RGBA8;

    fn compress_images<'a, C>(
        input: impl Iterator<Item = Image2DRef<'a, Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut tokens = Vec::new();

        let context = compressor.compress_symbols(
            input.map(|image| image.iter_pixels().map(|rgb| Vle(rgb.bits_interleaved()))),
            &mut tokens,
        )?;

        let mut offsets = WriteOffsets::new(tokens.len(), &mut write)?;

        write_bits_scope(&mut write, |write_bits| context.var_write(write_bits))?;

        for token_group in &tokens {
            offsets.push_next(&mut write)?;

            write_bits_scope(&mut write, |write| {
                for token in token_group {
                    token.var_write(write)?;
                }
                Ok(())
            })?;
        }

        offsets.write(&mut write)?;
        Ok(())
    }

    fn decompress_image<'a, C>(
        compressor: C,
        context: &Self::Context<C>,
        read: impl io::Read,
        mut image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut read_bits = crate::bits::ReadBits::new(read);

        let mut symbols = compressor.decompress_tokens2(context, read_tokens(&mut read_bits));

        let height = image.height();
        for y in 0..height {
            let mut row = image.get_row_mut(y);
            for pixel in row.iter_pixels_mut() {
                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                *pixel = Rgba8U::from_bits_interleaved(bits);
            }
        }

        Ok(())
    }
}

impl Pixel for bc1::Block {
    type Context<C: Compressor> = (
        C::Context<Vle<u16>>, // colors
        C::Context<u8>,       // indices
    );

    const FORMAT: Format = Format::BC1;

    fn compress_images<'a, C>(
        input: impl Iterator<Item = Image2DRef<'a, Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut colors_tokens = Vec::new();

        let colors_cx = compressor.compress_symbols(
            input.clone().map(|image| {
                image.iter_pixels().flat_map(|b| {
                    [
                        Vle(b.color0.bits_interleaved()),
                        Vle(b.color1.bits_interleaved()),
                    ]
                })
            }),
            &mut colors_tokens,
        )?;

        let total_colors_tokens: usize = colors_tokens.iter().map(|tile| tile.len()).sum();

        let mut indices_tokens = Vec::new();
        let texel_cx = compressor.compress_symbols(
            input.map(|image| image.iter_pixels().flat_map(|b| b.indices)),
            &mut indices_tokens,
        )?;

        let total_indices_tokens: usize = indices_tokens.iter().map(|tile| tile.len()).sum();

        dbg!(total_colors_tokens * 4);
        dbg!(total_indices_tokens * 4);

        assert_eq!(
            colors_tokens.len(),
            indices_tokens.len(),
            "Tile count mismatch"
        );

        let mut offsets = WriteOffsets::new(colors_tokens.len(), &mut write)?;

        write_bits_scope(&mut write, |write| {
            (colors_cx, texel_cx).var_write(write)?;
            Ok(())
        })?;

        for idx in 0..colors_tokens.len() {
            offsets.push_next(&mut write)?;

            let color_tile = &colors_tokens[idx];
            let texel_tile = &indices_tokens[idx];

            write_bits_scope(&mut write, |write| {
                for token in color_tile {
                    token.var_write(write)?;
                }

                for token in texel_tile {
                    token.var_write(write)?;
                }

                Ok(())
            })?;
        }

        offsets.write(&mut write)?;

        Ok(())
    }

    fn decompress_image<'a, C>(
        compressor: C,
        context: &Self::Context<C>,
        read: impl io::Read,
        mut image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let (colors_cx, texel_cx) = context;

        let mut read_bits = crate::bits::ReadBits::new(read);

        let mut symbols = compressor.decompress_tokens2(colors_cx, read_tokens(&mut read_bits));

        let height = image.height();

        for y in 0..height {
            let mut row = image.get_row_mut(y);
            for pixel in row.iter_pixels_mut() {
                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                pixel.color0 = Rgb565::from_bits_interleaved(bits);

                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                pixel.color1 = Rgb565::from_bits_interleaved(bits);
            }
        }

        drop(symbols);

        let mut symbols = compressor.decompress_tokens2(texel_cx, read_tokens(&mut read_bits));

        for y in 0..height {
            let mut row = image.get_row_mut(y);
            for pixel in row.iter_pixels_mut() {
                for i in 0..4 {
                    let bits = symbols.next().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "Not enough data to fill the image",
                        )
                    })??;

                    pixel.indices[i] = bits;
                }
            }
        }

        Ok(())
    }
}

impl Pixel for bc2::Block {
    type Context<C: Compressor> = (
        C::Context<u8>,       // alpha bytes
        C::Context<Vle<u16>>, // colors
        C::Context<u8>,       // indices
    );

    const FORMAT: Format = Format::BC2;

    fn compress_images<'a, C>(
        input: impl Iterator<Item = Image2DRef<'a, Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut alpha_tokens = Vec::new();

        let alpha_cx = compressor.compress_symbols(
            input
                .clone()
                .map(|image| image.iter_pixels().flat_map(|b| b.alpha)),
            &mut alpha_tokens,
        )?;

        let mut colors_tokens = Vec::new();

        let colors_cx = compressor.compress_symbols(
            input.clone().map(|image| {
                image.iter_pixels().flat_map(|b| {
                    [
                        Vle(b.color0.bits_interleaved()),
                        Vle(b.color1.bits_interleaved()),
                    ]
                })
            }),
            &mut colors_tokens,
        )?;

        let mut indices_tokens = Vec::new();
        let texel_cx = compressor.compress_symbols(
            input.map(|image| image.iter_pixels().flat_map(|b| b.indices)),
            &mut indices_tokens,
        )?;

        assert_eq!(
            alpha_tokens.len(),
            colors_tokens.len(),
            "Tile count mismatch"
        );
        assert_eq!(
            colors_tokens.len(),
            indices_tokens.len(),
            "Tile count mismatch"
        );

        let mut offsets = WriteOffsets::new(colors_tokens.len(), &mut write)?;

        write_bits_scope(&mut write, |write| {
            (alpha_cx, colors_cx, texel_cx).var_write(write)?;
            Ok(())
        })?;

        for idx in 0..colors_tokens.len() {
            offsets.push_next(&mut write)?;

            let alpha_tile = &alpha_tokens[idx];
            let color_tile = &colors_tokens[idx];
            let texel_tile = &indices_tokens[idx];

            write_bits_scope(&mut write, |write| {
                for token in alpha_tile {
                    token.var_write(write)?;
                }

                for token in color_tile {
                    token.var_write(write)?;
                }

                for token in texel_tile {
                    token.var_write(write)?;
                }

                Ok(())
            })?;
        }

        offsets.write(&mut write)?;

        Ok(())
    }

    fn decompress_image<'a, C>(
        compressor: C,
        context: &Self::Context<C>,
        read: impl io::Read,
        mut image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let (alpha_cx, colors_cx, texel_cx) = context;

        let mut read_bits = crate::bits::ReadBits::new(read);

        let mut symbols = compressor.decompress_tokens2(alpha_cx, read_tokens(&mut read_bits));

        let height = image.height();

        for y in 0..height {
            let mut row = image.get_row_mut(y);
            for pixel in row.iter_pixels_mut() {
                for i in 0..8 {
                    let byte = symbols.next().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "Not enough data to fill the image",
                        )
                    })??;

                    pixel.alpha[i] = byte;
                }
            }
        }

        drop(symbols);

        let mut symbols = compressor.decompress_tokens2(colors_cx, read_tokens(&mut read_bits));

        for y in 0..height {
            let mut row = image.get_row_mut(y);
            for pixel in row.iter_pixels_mut() {
                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                pixel.color0 = Rgb565::from_bits_interleaved(bits);

                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                pixel.color1 = Rgb565::from_bits_interleaved(bits);
            }
        }

        drop(symbols);

        let mut symbols = compressor.decompress_tokens2(texel_cx, read_tokens(&mut read_bits));

        for y in 0..height {
            let mut row = image.get_row_mut(y);
            for pixel in row.iter_pixels_mut() {
                for i in 0..4 {
                    let bits = symbols.next().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "Not enough data to fill the image",
                        )
                    })??;

                    pixel.indices[i] = bits;
                }
            }
        }

        Ok(())
    }
}

fn read_tokens<T>(read: &mut ReadBits<impl io::Read>) -> impl Iterator<Item = io::Result<T>> + '_
where
    T: VarCode,
{
    std::iter::from_fn(move || match T::var_read(read) {
        Ok(token) => Some(Ok(token)),
        Err(err) => Some(Err(err)),
    })
}
