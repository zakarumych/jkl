use std::io;

use smallvec::{smallvec, SmallVec};

use crate::{
    bc1,
    bits::{ReadBits, WriteBits},
    encode::{FixedCode, VarCode},
    image::{Image2DMut, Image2DRef},
    math::{Rgb565, Rgb8U},
    vle::{self, Vle},
};

use super::{compress::Compressor, Format};

/// This trait is an interface for compression images.
pub(super) trait AnyFormat: Sized + 'static {
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

    #[inline]
    fn read_offsets(read: impl io::Read, offsets: &mut [u64]) -> io::Result<()> {
        let mut read_bits = ReadBits::new(read);
        for offset in offsets {
            *offset = vle::decode(&mut read_bits)?;
        }
        Ok(())
    }

    /// Decompress the input compressed data into output images.
    /// `compressor` is the compression algorithm to use.
    /// `cx` is the decompression context produced during compression.
    /// `read` is the input stream to read compressed data from.
    /// `output` is an image buffer to write decompressed pixels to.
    fn decompress_image<'a, C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: impl io::Read,
        image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor;
}

impl AnyFormat for Rgb8U {
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

        let len = u64::try_from(tokens.len()).expect("Too many tiles to compress");

        let cx = compressor.compress_symbols(
            input.map(|image| image.iter_pixels().map(|rgb| Vle(rgb.bits_interleaved()))),
            &mut tokens,
        )?;

        let mut write_bits = WriteBits::new(&mut write);
        cx.var_write(&mut write_bits)?;
        write_bits.finish()?;

        let offsets_start = write.stream_position()?;
        let offsets_len = 8 * len; // 8 is the size of u64 in bytes
        let offsets_end = offsets_start + offsets_len;

        write.seek(io::SeekFrom::Start(offsets_end))?;

        let mut offsets: SmallVec<[u64; 64]> = smallvec![0; tokens.len()];

        for idx in 0..tokens.len() {
            offsets[idx] = write.stream_position()?;

            let mut write_bits = WriteBits::new(&mut write);
            for token in &tokens[idx] {
                token.var_write(&mut write_bits)?;
            }
            write_bits.finish()?;
        }

        write.seek(io::SeekFrom::Start(offsets_start))?;

        for offset in offsets {
            offset.fix_write(&mut write)?;
        }

        Ok(())
    }

    fn decompress_image<'a, C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: impl io::Read,
        mut image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut read_bits = crate::bits::ReadBits::new(read);

        let mut symbols = compressor.decompress_tokens2(cx, read_tokens(&mut read_bits));

        let width = image.width();
        let height = image.height();
        for y in 0..height {
            let row = image.row_mut(y);
            for x in 0..width {
                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                row[x] = Rgb8U::from_bits_interleaved(bits);
            }
        }

        Ok(())
    }
}

impl AnyFormat for bc1::Block {
    type Context<C: Compressor> = (
        C::Context<Vle<u16>>, // colors
        C::Context<u8>,       // texels
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
        let mut color_tokens = Vec::new();

        let color_cx = compressor.compress_symbols(
            input.clone().map(|image| {
                image.iter_pixels().flat_map(|b| {
                    [
                        Vle(b.color0.bits_interleaved()),
                        Vle(b.color1.bits_interleaved()),
                    ]
                })
            }),
            &mut color_tokens,
        )?;

        let mut texel_tokens = Vec::new();
        let texel_cx = compressor.compress_symbols(
            input.map(|image| image.iter_pixels().flat_map(|b| b.texels)),
            &mut texel_tokens,
        )?;

        assert_eq!(
            color_tokens.len(),
            texel_tokens.len(),
            "Tile count mismatch"
        );

        let len = u64::try_from(color_tokens.len()).expect("Too many tiles to compress");

        let mut write_bits = WriteBits::new(&mut write);
        color_cx.var_write(&mut write_bits)?;
        texel_cx.var_write(&mut write_bits)?;
        write_bits.finish()?;

        let offsets_start = write.stream_position()?;
        let offsets_len = 8 * len; // 8 is the size of u64 in bytes
        let offsets_end = offsets_start + offsets_len;

        write.seek(io::SeekFrom::Start(offsets_end))?;

        let mut offsets: SmallVec<[u64; 64]> = smallvec![0; color_tokens.len()];

        for idx in 0..color_tokens.len() {
            offsets[idx] = write.stream_position()?;

            let color_tile = &color_tokens[idx];
            let texel_tile = &texel_tokens[idx];

            let mut write_bits = WriteBits::new(&mut write);

            for token in color_tile {
                token.var_write(&mut write_bits)?;
            }

            for token in texel_tile {
                token.var_write(&mut write_bits)?;
            }

            write_bits.finish()?;
        }

        Ok(())
    }

    fn decompress_image<'a, C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: impl io::Read,
        mut image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let (color_cx, texel_cx) = cx;

        let mut read_bits = crate::bits::ReadBits::new(read);

        // let finished = Cell::new(false);
        // let mut result = Ok(());
        // let (input, mut output) = iter_fill(
        //     &finished,
        //     &mut result,
        //     &mut read_bits,
        //     image
        //         .iter_mut()
        //         .flat_map(|b| [&mut b.color0, &mut b.color1]),
        //     |Vle(bits)| Rgb565::from_bits_interleaved(bits),
        // );

        // compressor.decompress_tokens2(color_cx, input, &mut { output })?;
        // result?;

        // let finished = Cell::new(false);
        // let mut result = Ok(());
        // let (input, mut output) = iter_fill(
        //     &finished,
        //     &mut result,
        //     &mut read_bits,
        //     image.iter_mut().flat_map(|b| b.texels.iter_mut()),
        //     |byte: u8| byte,
        // );

        // compressor.decompress_tokens(texel_cx, input, &mut { output })?;
        // result

        let mut symbols = compressor.decompress_tokens2(color_cx, read_tokens(&mut read_bits));

        let width = image.width();
        let height = image.height();

        for y in 0..height {
            let row = image.row_mut(y);
            for x in 0..width {
                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                row[x].color0 = Rgb565::from_bits_interleaved(bits);

                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                row[x].color1 = Rgb565::from_bits_interleaved(bits);
            }
        }

        drop(symbols);

        let mut symbols = compressor.decompress_tokens2(texel_cx, read_tokens(&mut read_bits));

        for y in 0..height {
            let row = image.row_mut(y);
            for x in 0..width {
                for i in 0..4 {
                    let bits = symbols.next().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "Not enough data to fill the image",
                        )
                    })??;

                    row[x].texels[i] = bits;
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
