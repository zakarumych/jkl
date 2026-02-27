use std::io;

use crate::{
    bc1,
    bits::WriteBits,
    encode::Encode,
    jackal::compress::Compressor,
    math::Rgb8U,
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
}
