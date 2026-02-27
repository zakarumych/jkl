use std::{cell::Cell, io, marker::PhantomData};

use crate::{
    bc1,
    bits::{ReadBits, WriteBits},
    encode::Encode,
    image::{ImageMut, ImageRef},
    jackal::compress::Compressor,
    math::{Rgb565, Rgb8U},
    vle::Vle,
};

/// This trait is an interface for compression images.
pub(super) trait AnyFormat: Sized + 'static {
    type Context<C: Compressor>: Encode;

    /// Compress the input images of data.
    ///
    /// `input` is an iterator over images, where each image is made of pixels of type `Self`.
    /// `compressor` is the compression algorithm to use.
    /// `write` is the output stream to write compressed data to.
    /// `offsets` is a output slice where the function will write the file offsets of each compressed image.
    fn compress_images<'a, C>(
        input: impl Iterator<Item = ImageRef<'a, Self>> + Clone,
        compressor: C,
        write: impl io::Write + io::Seek,
        offsets: &mut [u64],
    ) -> io::Result<Self::Context<C>>
    where
        C: Compressor;

    /// Decompress the input compressed data into output images.
    /// `compressor` is the compression algorithm to use.
    /// `cx` is the decompression context produced during compression.
    /// `read` is the input stream to read compressed data from.
    /// `output` is an image buffer to write decompressed pixels to.
    fn decompress_image<'a, C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: impl io::Read,
        image: ImageMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor;
}

/// Makes and iterator and an extend adapter for filling an iterator while reading compressed data.
/// Reading will stop only when the image is fully filled or an error occurs.
/// The error will be stored in `error` and checked after adapter usage.
fn iter_fill<'b, R, I, T, F, S, P>(
    finished: &'b Cell<bool>,
    result: &'b mut io::Result<()>,
    read: &'b mut ReadBits<R>,
    image: I,
    map: F,
) -> (impl Iterator<Item = T> + 'b, impl Extend<S> + 'b)
where
    R: io::Read,
    I: Iterator<Item = &'b mut P> + 'b,
    T: Encode + 'static,
    F: Fn(S) -> P + 'static,
    P: 'static,
{
    struct IterToken<'a, T, R> {
        read: &'a mut ReadBits<R>,
        result: &'a mut io::Result<()>,
        _marker: std::marker::PhantomData<T>,
        finished: &'a Cell<bool>,
    }

    impl<'a, T, R> Iterator for IterToken<'a, T, R>
    where
        T: Encode,
        R: io::Read,
    {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.finished.get() {
                return None;
            }

            match T::read(self.read) {
                Ok(token) => Some(token),
                Err(err) => {
                    *self.result = Err(err);
                    self.finished.set(true);
                    None
                }
            }
        }
    }

    struct ExtendImage<'a, I, F, P> {
        image: I,
        map: F,
        finished: &'a Cell<bool>,
        marker: PhantomData<fn(&mut I) -> P>,
    }

    impl<'a, S, I, F, P: 'a> Extend<S> for ExtendImage<'a, I, F, P>
    where
        F: Fn(S) -> P + 'static,
        I: Iterator<Item = &'a mut P>,
    {
        fn extend<E: IntoIterator<Item = S>>(&mut self, iter: E) {
            if self.finished.get() {
                return;
            }
            let mut iter = iter.into_iter();
            while let Some(s) = iter.next() {
                match self.image.next() {
                    Some(p) => *p = (self.map)(s),
                    None => {
                        self.finished.set(true);
                        return;
                    }
                }
            }
        }
    }

    let iter = IterToken {
        read,
        result,
        _marker: std::marker::PhantomData,
        finished: &finished,
    };

    let extend = ExtendImage {
        image: image,
        marker: PhantomData,
        map,
        finished: &finished,
    };

    (iter, extend)
}

impl AnyFormat for Rgb8U {
    type Context<C: Compressor> = C::Context<Vle<u32>>;

    fn compress_images<'a, C>(
        input: impl Iterator<Item = ImageRef<'a, Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
        offsets: &mut [u64],
    ) -> io::Result<C::Context<Vle<u32>>>
    where
        C: Compressor,
    {
        let mut tokens = Vec::new();
        let cx = compressor.compress_symbols(
            input.map(|image| image.iter().map(|rgb| Vle(rgb.bits_interleaved()))),
            &mut tokens,
        )?;

        for (i, chunk) in tokens.into_iter().enumerate() {
            offsets[i] = write.stream_position()?;

            let mut write_bits = WriteBits::new(&mut write);
            for token in chunk {
                token.write(&mut write_bits)?;
            }
            write_bits.finish()?;
        }

        Ok(cx)
    }

    fn decompress_image<'a, C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: impl io::Read,
        mut image: ImageMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut read_bits = crate::bits::ReadBits::new(read);
        let finished = Cell::new(false);
        let mut result = Ok(());
        let (input, mut output) = iter_fill(
            &finished,
            &mut result,
            &mut read_bits,
            image.iter_mut(),
            |Vle(bits)| Rgb8U::from_bits_interleaved(bits),
        );

        compressor.decompress_tokens(cx, input, &mut { output })?;
        result
    }
}

impl AnyFormat for bc1::Block {
    type Context<C: Compressor> = (
        C::Context<Vle<u16>>, // colors
        C::Context<u8>,       // texels
    );

    fn compress_images<'a, C>(
        input: impl Iterator<Item = ImageRef<'a, Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
        offsets: &mut [u64],
    ) -> io::Result<(C::Context<Vle<u16>>, C::Context<u8>)>
    where
        C: Compressor,
    {
        let mut color_tokens = Vec::new();

        let color_cx = compressor.compress_symbols(
            input.clone().map(|image| {
                image.iter().flat_map(|b| {
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
            input.map(|image| image.iter().flat_map(|b| b.texels)),
            &mut texel_tokens,
        )?;

        assert_eq!(
            color_tokens.len(),
            texel_tokens.len(),
            "Chunk count mismatch"
        );

        assert_eq!(
            color_tokens.len(),
            offsets.len(),
            "Offsets count must match chunk count"
        );

        for idx in 0..offsets.len() {
            offsets[idx] = write.stream_position()?;

            let color_chunk = &color_tokens[idx];
            let texel_chunk = &texel_tokens[idx];

            let mut write_bits = WriteBits::new(&mut write);

            for token in color_chunk {
                token.write(&mut write_bits)?;
            }

            for token in texel_chunk {
                token.write(&mut write_bits)?;
            }

            write_bits.finish()?;
        }

        Ok((color_cx, texel_cx))
    }

    fn decompress_image<'a, C>(
        compressor: C,
        cx: &Self::Context<C>,
        read: impl io::Read,
        mut image: ImageMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let (color_cx, texel_cx) = cx;

        let mut read_bits = crate::bits::ReadBits::new(read);

        let finished = Cell::new(false);
        let mut result = Ok(());
        let (input, mut output) = iter_fill(
            &finished,
            &mut result,
            &mut read_bits,
            image
                .iter_mut()
                .flat_map(|b| [&mut b.color0, &mut b.color1]),
            |Vle(bits)| Rgb565::from_bits_interleaved(bits),
        );

        compressor.decompress_tokens(color_cx, input, &mut { output })?;
        result?;

        let finished = Cell::new(false);
        let mut result = Ok(());
        let (input, mut output) = iter_fill(
            &finished,
            &mut result,
            &mut read_bits,
            image.iter_mut().flat_map(|b| b.texels.iter_mut()),
            |byte: u8| byte,
        );

        compressor.decompress_tokens(texel_cx, input, &mut { output })?;
        result
    }
}
