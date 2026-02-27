use std::io;

use crate::{
    bc1, bc3, bc4, bc5,
    bits::ReadBits,
    bits::WriteBits,
    encode::Encode,
    jackal::compress::Compressor,
    math::{R8U, Rgb565, Rgb8U, Rgba8U},
    vle::{self, Vle},
};

/// This trait is an interface for compression chunks of data in any format.
pub(super) trait AnyFormat: Sized + 'static {
    type Context<C: Compressor>: Encode;

    /// Compress the input chunks of data.
    ///
    /// `input` is an iterator over chunks, where each chunk is an iterator over items of type `Self`.
    /// `compressor` is the compression algorithm to use.
    /// `write` is the output stream to write compressed data to.
    /// `offsets` is a vector to store the starting offset of each compressed chunk in the output stream.
    fn compress_chunks<C>(
        input: impl DoubleEndedIterator<Item = impl DoubleEndedIterator<Item = Self>> + Clone,
        compressor: C,
        write: impl io::Write + io::Seek,
        offsets: &mut Vec<u64>,
    ) -> io::Result<Self::Context<C>>
    where
        C: Compressor;

    fn decompress_chunk<C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: &[u8],
        output: &mut Vec<Self>,
    ) -> io::Result<()>
    where
        C: Compressor;
}

fn read_all_tokens<T: Encode>(read: &[u8]) -> io::Result<Vec<T>> {
    let mut read_bits = ReadBits::new(read);
    let mut tokens = Vec::new();
    loop {
        match T::read(&mut read_bits) {
            Ok(token) => tokens.push(token),
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err),
        }
    }
    Ok(tokens)
}

fn read_counted_tokens<T: Encode, R: io::Read>(read: &mut ReadBits<R>) -> io::Result<Vec<T>> {
    let len = vle::decode::<u64, _>(read)? as usize;
    let mut tokens = Vec::with_capacity(len);
    for _ in 0..len {
        tokens.push(T::read(read)?);
    }
    Ok(tokens)
}

impl AnyFormat for R8U {
    type Context<C: Compressor> = C::Context<Vle<u8>>;

    fn compress_chunks<C>(
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
        offsets: &mut Vec<u64>,
    ) -> io::Result<C::Context<Vle<u8>>>
    where
        C: Compressor,
    {
        let mut tokens = Vec::new();
        let cx = compressor.compress_symbols(
            input.map(|chunk| chunk.map(|value| Vle(value.bits()))),
            &mut tokens,
        )?;

        for chunk in tokens {
            offsets.push(write.stream_position()?);

            let mut write_bits = WriteBits::new(&mut write);
            for token in chunk {
                token.write(&mut write_bits)?;
            }
            write_bits.finish()?;
        }

        Ok(cx)
    }

    fn decompress_chunk<C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: &[u8],
        output: &mut Vec<Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let tokens = read_all_tokens::<C::Token<Vle<u8>>>(read)?;
        let mut values = Vec::new();
        compressor.decompress_tokens(cx, tokens.into_iter(), &mut values)?;
        output.extend(values.into_iter().map(|value| R8U::from_bits(value.0)));
        Ok(())
    }
}

impl AnyFormat for Rgb8U {
    type Context<C: Compressor> = C::Context<Vle<u32>>;

    fn compress_chunks<C>(
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
        offsets: &mut Vec<u64>,
    ) -> io::Result<C::Context<Vle<u32>>>
    where
        C: Compressor,
    {
        let mut tokens = Vec::new();
        let cx = compressor.compress_symbols(
            input.map(|chunk| chunk.map(|rgb| Vle(rgb.bits_interleaved()))),
            &mut tokens,
        )?;

        for chunk in tokens {
            offsets.push(write.stream_position()?);

            let mut write_bits = WriteBits::new(&mut write);
            for token in chunk {
                token.write(&mut write_bits)?;
            }
            write_bits.finish()?;
        }

        Ok(cx)
    }

    fn decompress_chunk<C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: &[u8],
        output: &mut Vec<Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let tokens = read_all_tokens::<C::Token<Vle<u32>>>(read)?;
        let mut values = Vec::new();
        compressor.decompress_tokens(cx, tokens.into_iter(), &mut values)?;
        output.extend(
            values
                .into_iter()
                .map(|value| Rgb8U::from_bits_interleaved(value.0)),
        );
        Ok(())
    }
}

impl AnyFormat for Rgba8U {
    type Context<C: Compressor> = C::Context<Vle<u32>>;

    fn compress_chunks<C>(
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
        offsets: &mut Vec<u64>,
    ) -> io::Result<C::Context<Vle<u32>>>
    where
        C: Compressor,
    {
        let mut tokens = Vec::new();
        let cx = compressor.compress_symbols(
            input.map(|chunk| chunk.map(|rgba| Vle(rgba.bits_interleaved()))),
            &mut tokens,
        )?;

        for chunk in tokens {
            offsets.push(write.stream_position()?);

            let mut write_bits = WriteBits::new(&mut write);
            for token in chunk {
                token.write(&mut write_bits)?;
            }
            write_bits.finish()?;
        }

        Ok(cx)
    }

    fn decompress_chunk<C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: &[u8],
        output: &mut Vec<Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let tokens = read_all_tokens::<C::Token<Vle<u32>>>(read)?;
        let mut values = Vec::new();
        compressor.decompress_tokens(cx, tokens.into_iter(), &mut values)?;
        output.extend(
            values
                .into_iter()
                .map(|value| Rgba8U::from_bits_interleaved(value.0)),
        );
        Ok(())
    }
}

impl AnyFormat for bc1::Block {
    type Context<C: Compressor> = (
        C::Context<Vle<u16>>, // colors
        C::Context<u8>,       // texels
    );

    fn compress_chunks<C>(
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
        offsets: &mut Vec<u64>,
    ) -> io::Result<(C::Context<Vle<u16>>, C::Context<u8>)>
    where
        C: Compressor,
    {
        let mut color_tokens = Vec::new();

        let color_cx = compressor.compress_symbols(
            input.clone().map(|chunk| {
                chunk.flat_map(|b| {
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
            input.map(|chunk| chunk.flat_map(|b| b.texels)),
            &mut texel_tokens,
        )?;

        assert_eq!(
            color_tokens.len(),
            texel_tokens.len(),
            "Chunk count mismatch"
        );

        for (color_chunk, texel_chunk) in
            Iterator::zip(color_tokens.into_iter(), texel_tokens.into_iter())
        {
            offsets.push(write.stream_position()?);

            let mut write_bits = WriteBits::new(&mut write);

            vle::encode(color_chunk.len(), &mut write_bits)?;

            for token in color_chunk {
                token.write(&mut write_bits)?;
            }

            vle::encode(texel_chunk.len(), &mut write_bits)?;

            for token in texel_chunk {
                token.write(&mut write_bits)?;
            }

            write_bits.finish()?;
        }

        Ok((color_cx, texel_cx))
    }

    fn decompress_chunk<C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: &[u8],
        output: &mut Vec<Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut read_bits = ReadBits::new(read);
        let color_tokens = read_counted_tokens::<C::Token<Vle<u16>>, _>(&mut read_bits)?;
        let texel_tokens = read_counted_tokens::<C::Token<u8>, _>(&mut read_bits)?;

        let (color_cx, texel_cx) = cx;
        let mut colors = Vec::new();
        compressor.decompress_tokens(color_cx, color_tokens.into_iter(), &mut colors)?;

        let mut texels = Vec::new();
        compressor.decompress_tokens(texel_cx, texel_tokens.into_iter(), &mut texels)?;

        if colors.len() % 2 != 0 || texels.len() % 4 != 0 || colors.len() / 2 != texels.len() / 4 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Corrupt BC1 chunk"));
        }

        for (color_pair, texel_quad) in colors.chunks_exact(2).zip(texels.chunks_exact(4)) {
            output.push(bc1::Block {
                color0: Rgb565::from_bits_interleaved(color_pair[0].0),
                color1: Rgb565::from_bits_interleaved(color_pair[1].0),
                texels: [texel_quad[0], texel_quad[1], texel_quad[2], texel_quad[3]],
            });
        }

        Ok(())
    }
}

impl AnyFormat for bc4::Block {
    type Context<C: Compressor> = (C::Context<Vle<u8>>, C::Context<u8>);

    fn compress_chunks<C>(
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
        offsets: &mut Vec<u64>,
    ) -> io::Result<Self::Context<C>>
    where
        C: Compressor,
    {
        let mut color_tokens = Vec::new();
        let color_cx = compressor.compress_symbols(
            input
                .clone()
                .map(|chunk| chunk.flat_map(|b| [Vle(b.color0.bits()), Vle(b.color1.bits())])),
            &mut color_tokens,
        )?;

        let mut texel_tokens = Vec::new();
        let texel_cx = compressor.compress_symbols(
            input.map(|chunk| chunk.flat_map(|b| b.texels)),
            &mut texel_tokens,
        )?;

        assert_eq!(
            color_tokens.len(),
            texel_tokens.len(),
            "Chunk count mismatch"
        );

        for (color_chunk, texel_chunk) in
            Iterator::zip(color_tokens.into_iter(), texel_tokens.into_iter())
        {
            offsets.push(write.stream_position()?);

            let mut write_bits = WriteBits::new(&mut write);
            vle::encode(color_chunk.len(), &mut write_bits)?;
            for token in color_chunk {
                token.write(&mut write_bits)?;
            }

            vle::encode(texel_chunk.len(), &mut write_bits)?;
            for token in texel_chunk {
                token.write(&mut write_bits)?;
            }
            write_bits.finish()?;
        }

        Ok((color_cx, texel_cx))
    }

    fn decompress_chunk<C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: &[u8],
        output: &mut Vec<Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut read_bits = ReadBits::new(read);
        let color_tokens = read_counted_tokens::<C::Token<Vle<u8>>, _>(&mut read_bits)?;
        let texel_tokens = read_counted_tokens::<C::Token<u8>, _>(&mut read_bits)?;

        let (color_cx, texel_cx) = cx;
        let mut colors = Vec::new();
        compressor.decompress_tokens(color_cx, color_tokens.into_iter(), &mut colors)?;

        let mut texels = Vec::new();
        compressor.decompress_tokens(texel_cx, texel_tokens.into_iter(), &mut texels)?;

        if colors.len() % 2 != 0 || texels.len() % 6 != 0 || colors.len() / 2 != texels.len() / 6 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Corrupt BC4 chunk"));
        }

        for (color_pair, texel_chunk) in colors.chunks_exact(2).zip(texels.chunks_exact(6)) {
            output.push(bc4::Block {
                color0: R8U::from_bits(color_pair[0].0),
                color1: R8U::from_bits(color_pair[1].0),
                texels: [
                    texel_chunk[0],
                    texel_chunk[1],
                    texel_chunk[2],
                    texel_chunk[3],
                    texel_chunk[4],
                    texel_chunk[5],
                ],
            });
        }

        Ok(())
    }
}

impl AnyFormat for bc5::Block {
    type Context<C: Compressor> = (C::Context<Vle<u8>>, C::Context<u8>);

    fn compress_chunks<C>(
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
        offsets: &mut Vec<u64>,
    ) -> io::Result<Self::Context<C>>
    where
        C: Compressor,
    {
        let mut color_tokens = Vec::new();
        let color_cx = compressor.compress_symbols(
            input.clone().map(|chunk| {
                chunk.flat_map(|b| {
                    [
                        Vle(b.red.color0.bits()),
                        Vle(b.red.color1.bits()),
                        Vle(b.green.color0.bits()),
                        Vle(b.green.color1.bits()),
                    ]
                })
            }),
            &mut color_tokens,
        )?;

        let mut texel_tokens = Vec::new();
        let texel_cx = compressor.compress_symbols(
            input.map(|chunk| chunk.flat_map(|b| b.red.texels.into_iter().chain(b.green.texels))),
            &mut texel_tokens,
        )?;

        assert_eq!(
            color_tokens.len(),
            texel_tokens.len(),
            "Chunk count mismatch"
        );

        for (color_chunk, texel_chunk) in
            Iterator::zip(color_tokens.into_iter(), texel_tokens.into_iter())
        {
            offsets.push(write.stream_position()?);

            let mut write_bits = WriteBits::new(&mut write);
            vle::encode(color_chunk.len(), &mut write_bits)?;
            for token in color_chunk {
                token.write(&mut write_bits)?;
            }

            vle::encode(texel_chunk.len(), &mut write_bits)?;
            for token in texel_chunk {
                token.write(&mut write_bits)?;
            }
            write_bits.finish()?;
        }

        Ok((color_cx, texel_cx))
    }

    fn decompress_chunk<C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: &[u8],
        output: &mut Vec<Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut read_bits = ReadBits::new(read);
        let color_tokens = read_counted_tokens::<C::Token<Vle<u8>>, _>(&mut read_bits)?;
        let texel_tokens = read_counted_tokens::<C::Token<u8>, _>(&mut read_bits)?;

        let (color_cx, texel_cx) = cx;
        let mut colors = Vec::new();
        compressor.decompress_tokens(color_cx, color_tokens.into_iter(), &mut colors)?;

        let mut texels = Vec::new();
        compressor.decompress_tokens(texel_cx, texel_tokens.into_iter(), &mut texels)?;

        if colors.len() % 4 != 0 || texels.len() % 12 != 0 || colors.len() / 4 != texels.len() / 12
        {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Corrupt BC5 chunk"));
        }

        for (color_chunk, texel_chunk) in colors.chunks_exact(4).zip(texels.chunks_exact(12)) {
            output.push(bc5::Block {
                red: bc4::Block {
                    color0: R8U::from_bits(color_chunk[0].0),
                    color1: R8U::from_bits(color_chunk[1].0),
                    texels: [
                        texel_chunk[0],
                        texel_chunk[1],
                        texel_chunk[2],
                        texel_chunk[3],
                        texel_chunk[4],
                        texel_chunk[5],
                    ],
                },
                green: bc4::Block {
                    color0: R8U::from_bits(color_chunk[2].0),
                    color1: R8U::from_bits(color_chunk[3].0),
                    texels: [
                        texel_chunk[6],
                        texel_chunk[7],
                        texel_chunk[8],
                        texel_chunk[9],
                        texel_chunk[10],
                        texel_chunk[11],
                    ],
                },
            });
        }

        Ok(())
    }
}

impl AnyFormat for bc3::Block {
    type Context<C: Compressor> = (
        C::Context<Vle<u8>>,
        C::Context<u8>,
        C::Context<Vle<u16>>,
        C::Context<u8>,
    );

    fn compress_chunks<C>(
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
        offsets: &mut Vec<u64>,
    ) -> io::Result<Self::Context<C>>
    where
        C: Compressor,
    {
        let mut alpha_color_tokens = Vec::new();
        let alpha_color_cx = compressor.compress_symbols(
            input.clone().map(|chunk| {
                chunk.flat_map(|b| [Vle(b.alpha.color0.bits()), Vle(b.alpha.color1.bits())])
            }),
            &mut alpha_color_tokens,
        )?;

        let mut alpha_texel_tokens = Vec::new();
        let alpha_texel_cx = compressor.compress_symbols(
            input.clone().map(|chunk| chunk.flat_map(|b| b.alpha.texels)),
            &mut alpha_texel_tokens,
        )?;

        let mut rgb_color_tokens = Vec::new();
        let rgb_color_cx = compressor.compress_symbols(
            input.clone().map(|chunk| {
                chunk.flat_map(|b| {
                    [
                        Vle(b.rgb.color0.bits_interleaved()),
                        Vle(b.rgb.color1.bits_interleaved()),
                    ]
                })
            }),
            &mut rgb_color_tokens,
        )?;

        let mut rgb_texel_tokens = Vec::new();
        let rgb_texel_cx = compressor.compress_symbols(
            input.map(|chunk| chunk.flat_map(|b| b.rgb.texels)),
            &mut rgb_texel_tokens,
        )?;

        assert_eq!(
            alpha_color_tokens.len(),
            alpha_texel_tokens.len(),
            "Chunk count mismatch"
        );
        assert_eq!(
            alpha_color_tokens.len(),
            rgb_color_tokens.len(),
            "Chunk count mismatch"
        );
        assert_eq!(
            alpha_color_tokens.len(),
            rgb_texel_tokens.len(),
            "Chunk count mismatch"
        );

        for (((alpha_color_chunk, alpha_texel_chunk), rgb_color_chunk), rgb_texel_chunk) in
            alpha_color_tokens
                .into_iter()
                .zip(alpha_texel_tokens.into_iter())
                .zip(rgb_color_tokens.into_iter())
                .zip(rgb_texel_tokens.into_iter())
        {
            offsets.push(write.stream_position()?);

            let mut write_bits = WriteBits::new(&mut write);
            vle::encode(alpha_color_chunk.len(), &mut write_bits)?;
            for token in alpha_color_chunk {
                token.write(&mut write_bits)?;
            }

            vle::encode(alpha_texel_chunk.len(), &mut write_bits)?;
            for token in alpha_texel_chunk {
                token.write(&mut write_bits)?;
            }

            vle::encode(rgb_color_chunk.len(), &mut write_bits)?;
            for token in rgb_color_chunk {
                token.write(&mut write_bits)?;
            }

            vle::encode(rgb_texel_chunk.len(), &mut write_bits)?;
            for token in rgb_texel_chunk {
                token.write(&mut write_bits)?;
            }

            write_bits.finish()?;
        }

        Ok((alpha_color_cx, alpha_texel_cx, rgb_color_cx, rgb_texel_cx))
    }

    fn decompress_chunk<C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: &[u8],
        output: &mut Vec<Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut read_bits = ReadBits::new(read);
        let alpha_color_tokens = read_counted_tokens::<C::Token<Vle<u8>>, _>(&mut read_bits)?;
        let alpha_texel_tokens = read_counted_tokens::<C::Token<u8>, _>(&mut read_bits)?;
        let rgb_color_tokens = read_counted_tokens::<C::Token<Vle<u16>>, _>(&mut read_bits)?;
        let rgb_texel_tokens = read_counted_tokens::<C::Token<u8>, _>(&mut read_bits)?;

        let (alpha_color_cx, alpha_texel_cx, rgb_color_cx, rgb_texel_cx) = cx;

        let mut alpha_colors = Vec::new();
        compressor.decompress_tokens(alpha_color_cx, alpha_color_tokens.into_iter(), &mut alpha_colors)?;
        let mut alpha_texels = Vec::new();
        compressor.decompress_tokens(alpha_texel_cx, alpha_texel_tokens.into_iter(), &mut alpha_texels)?;
        let mut rgb_colors = Vec::new();
        compressor.decompress_tokens(rgb_color_cx, rgb_color_tokens.into_iter(), &mut rgb_colors)?;
        let mut rgb_texels = Vec::new();
        compressor.decompress_tokens(rgb_texel_cx, rgb_texel_tokens.into_iter(), &mut rgb_texels)?;

        if alpha_colors.len() % 2 != 0
            || alpha_texels.len() % 6 != 0
            || rgb_colors.len() % 2 != 0
            || rgb_texels.len() % 4 != 0
            || alpha_colors.len() / 2 != alpha_texels.len() / 6
            || alpha_colors.len() / 2 != rgb_colors.len() / 2
            || alpha_colors.len() / 2 != rgb_texels.len() / 4
        {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Corrupt BC3 chunk"));
        }

        for (((alpha_color_chunk, alpha_texel_chunk), rgb_color_chunk), rgb_texel_chunk) in
            alpha_colors
                .chunks_exact(2)
                .zip(alpha_texels.chunks_exact(6))
                .zip(rgb_colors.chunks_exact(2))
                .zip(rgb_texels.chunks_exact(4))
        {
            output.push(bc3::Block {
                alpha: bc4::Block {
                    color0: R8U::from_bits(alpha_color_chunk[0].0),
                    color1: R8U::from_bits(alpha_color_chunk[1].0),
                    texels: [
                        alpha_texel_chunk[0],
                        alpha_texel_chunk[1],
                        alpha_texel_chunk[2],
                        alpha_texel_chunk[3],
                        alpha_texel_chunk[4],
                        alpha_texel_chunk[5],
                    ],
                },
                rgb: bc1::Block {
                    color0: Rgb565::from_bits_interleaved(rgb_color_chunk[0].0),
                    color1: Rgb565::from_bits_interleaved(rgb_color_chunk[1].0),
                    texels: [
                        rgb_texel_chunk[0],
                        rgb_texel_chunk[1],
                        rgb_texel_chunk[2],
                        rgb_texel_chunk[3],
                    ],
                },
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_any_format<T: AnyFormat>() {}

    #[test]
    fn any_format_impls_for_available_types() {
        assert_any_format::<R8U>();
        assert_any_format::<Rgb8U>();
        assert_any_format::<Rgba8U>();
        assert_any_format::<bc1::Block>();
        assert_any_format::<bc3::Block>();
        assert_any_format::<bc4::Block>();
        assert_any_format::<bc5::Block>();
    }
}
