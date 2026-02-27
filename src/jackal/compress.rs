use std::{hash::Hash, io};

use crate::{ans, encode::Encode, lz77, math::Delta};

/// Trait for types viable as symbols for compression.
pub trait Symbol: Copy + Default + Ord + Hash + Delta + Encode + 'static {}

/// Blanket implementation for all types that satisfy the trait bounds.
impl<T> Symbol for T where T: Default + Copy + Ord + Hash + Encode + Delta + Sized + 'static {}

/// Compression algorithm interface.
pub trait Compressor {
    /// Token type produced by compression for symbols of type `T`.
    type Token<T: Symbol>: Symbol;

    /// Context is produced during compression and is used during decompression.
    type Context<T: Symbol>: Encode;

    /// Compress the input symbols of type `T`.
    ///
    /// This function gets called with all chunks of symbols.
    ///
    /// Produces decompression context and writes output tokens of type `Self::Token<T>` to `output`.
    fn compress_symbols<T: Symbol>(
        &self,
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = T>> + Clone,
        output: &mut Vec<Vec<Self::Token<T>>>,
    ) -> io::Result<Self::Context<T>>;

    /// Decompress the input tokens of type `Self::Token<T>`.
    ///
    /// Called for each chunk independently.
    fn decompress_tokens<T: Symbol>(
        &self,
        cx: &Self::Context<T>,
        input: impl Iterator<Item = Self::Token<T>>,
        output: &mut Vec<T>,
    ) -> io::Result<()>;
}

#[derive(Clone, Copy, Debug)]
pub struct LZ77Compressor {
    pub window_size: u32,
}

pub struct LZ77Context;

impl Encode for LZ77Context {
    #[inline]
    fn bit_len(&self) -> usize {
        0
    }

    #[inline]
    fn write(&self, _write: &mut crate::bits::WriteBits<impl io::Write>) -> io::Result<()> {
        Ok(())
    }

    #[inline]
    fn read(_read: &mut crate::bits::ReadBits<impl io::Read>) -> io::Result<Self> {
        Ok(LZ77Context)
    }
}

impl Compressor for LZ77Compressor {
    type Token<T: Symbol> = lz77::Token<T>;
    type Context<T: Symbol> = LZ77Context;

    fn compress_symbols<T: Symbol>(
        &self,
        input: impl Iterator<Item = impl Iterator<Item = T>>,
        output: &mut Vec<Vec<lz77::Token<T>>>,
    ) -> io::Result<LZ77Context> {
        for (i, chunk) in input.enumerate() {
            let mut encoder = lz77::Encoder::new(T::default(), self.window_size);

            if output.len() == i {
                output.push(Vec::new());
            }
            let out_slice = &mut output[i];
            out_slice.clear();

            for symbol in chunk {
                encoder.encode(symbol, out_slice);
            }

            encoder.finish(out_slice);
        }

        Ok(LZ77Context)
    }

    fn decompress_tokens<T: Symbol>(
        &self,
        _cx: &LZ77Context,
        mut input: impl Iterator<Item = lz77::Token<T>>,
        output: &mut Vec<T>,
    ) -> io::Result<()> {
        let mut decoder = lz77::Decoder::new(T::default(), self.window_size);

        loop {
            match decoder.decode(input.by_ref()) {
                Ok(Some(symbol)) => output.push(symbol),
                Ok(None) => break,
                Err(err) => return Err(io::Error::new(io::ErrorKind::InvalidData, err)),
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AnsCompressor;

#[derive(Clone, Copy, Debug)]
pub struct AnsConfig;

impl Compressor for AnsCompressor {
    type Token<T: Symbol> = u32;
    type Context<T: Symbol> = ans::Context<T>;

    fn compress_symbols<T: Symbol>(
        &self,
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = T>> + Clone,
        output: &mut Vec<Vec<u32>>,
    ) -> io::Result<ans::Context<T>> {
        let cx = ans::Context::from_input(input.clone().flatten());

        for (i, chunk) in input.enumerate() {
            if output.len() == i {
                output.push(Vec::new());
            }

            let out_slice = &mut output[i];
            out_slice.clear();

            let mut encoder = ans::Encoder::new(&cx);

            for symbol in chunk.rev() {
                if let Some(code) = encoder.encode(symbol) {
                    out_slice.push(code);
                }
            }

            let [x, y] = encoder.finish();
            out_slice.push(x);
            out_slice.push(y);

            out_slice.reverse();
        }

        Ok(cx)
    }

    fn decompress_tokens<T: Symbol>(
        &self,
        cx: &ans::Context<T>,
        mut input: impl Iterator<Item = u32>,
        output: &mut Vec<T>,
    ) -> io::Result<()> {
        let mut decoder = ans::Decoder::new(&cx);

        loop {
            match decoder.decode(input.by_ref()) {
                Some(symbol) => output.push(symbol),
                None => break,
            }
        }

        if let Err(err) = decoder.finish() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, err));
        }

        Ok(())
    }
}

impl<A, B> Compressor for (A, B)
where
    A: Compressor,
    B: Compressor,
{
    type Token<T: Symbol> = B::Token<A::Token<T>>;
    type Context<T: Symbol> = (A::Context<T>, B::Context<A::Token<T>>);

    fn compress_symbols<T: Symbol>(
        &self,
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = T>> + Clone,
        output: &mut Vec<Vec<B::Token<A::Token<T>>>>,
    ) -> io::Result<Self::Context<T>> {
        let (a, b) = self;

        let mut a_tokens = Vec::new();

        let a_cx = a.compress_symbols(input, &mut a_tokens)?;
        let b_cx = b.compress_symbols(a_tokens.iter().map(|v| v.iter().copied()), output)?;
        Ok((a_cx, b_cx))
    }

    fn decompress_tokens<T: Symbol>(
        &self,
        cx: &Self::Context<T>,
        input: impl Iterator<Item = B::Token<A::Token<T>>>,
        output: &mut Vec<T>,
    ) -> io::Result<()> {
        let (a, b) = self;

        let (a_cx, b_cx) = cx;
        let mut a_tokens = Vec::new();

        b.decompress_tokens(b_cx, input, &mut a_tokens)?;
        a.decompress_tokens(a_cx, a_tokens.into_iter(), output)?;
        Ok(())
    }
}

// fn take_error_or_eof(read_error: &mut Option<io::Error>) -> io::Error {
//     match read_error.take() {
//         Some(err) => err,
//         None => io::Error::new(io::ErrorKind::UnexpectedEof, "Unexpected EOF"),
//     }
// }

// fn codes_read_iter<'a>(
//     read: &'a mut ReadBits<impl io::Read>,
//     read_error: &'a mut Option<io::Error>,
// ) -> impl Iterator<Item = u32> + 'a {
//     std::iter::from_fn(|| match u32::read(read) {
//         Ok(code) => Some(code),
//         Err(err) => {
//             *read_error = Some(err);
//             None
//         }
//     })
// }
