use std::{hash::Hash, io};

use crate::{ans, encode::Encode, lz77, math::Delta, rle};

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
        output: &mut impl Extend<T>,
    ) -> io::Result<()>;

    fn decompress_tokens2<T: Symbol>(
        &self,
        cx: &Self::Context<T>,
        input: impl Iterator<Item = io::Result<Self::Token<T>>>,
    ) -> impl Iterator<Item = io::Result<T>>;
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

    fn compress_symbols<'a, T: Symbol>(
        &self,
        input: impl Iterator<Item = impl Iterator<Item = T>>,
        output: &mut Vec<Vec<lz77::Token<T>>>,
    ) -> io::Result<Self::Context<T>> {
        for (i, chunk) in input.enumerate() {
            let mut encoder = lz77::Encoder::new(T::default(), self.window_size);
            if output.len() == i {
                output.push(Vec::new());
            }

            let stream = &mut output[i];

            for symbol in chunk {
                encoder.encode(symbol, stream);
            }

            encoder.finish(stream);
        }
        Ok(LZ77Context)
    }

    fn decompress_tokens<T: Symbol>(
        &self,
        _cx: &LZ77Context,
        mut input: impl Iterator<Item = lz77::Token<T>>,
        output: &mut impl Extend<T>,
    ) -> io::Result<()> {
        let mut decoder = lz77::Decoder::new(T::default(), self.window_size);

        loop {
            match decoder.decode(input.by_ref()) {
                Ok(Some(symbol)) => output.extend(std::iter::once(symbol)),
                Ok(None) => break,
                Err(err) => return Err(io::Error::new(io::ErrorKind::InvalidData, err)),
            }
        }

        Ok(())
    }

    fn decompress_tokens2<T: Symbol>(
        &self,
        _cx: &LZ77Context,
        input: impl Iterator<Item = io::Result<lz77::Token<T>>>,
    ) -> impl Iterator<Item = io::Result<T>> {
        let mut decoder = lz77::Decoder::new(T::default(), self.window_size);
        let mut extact_error = ExtractError::new(input);

        std::iter::from_fn(move || match decoder.decode(extact_error.by_ref()) {
            Ok(Some(symbol)) => Some(Ok(symbol)),
            Ok(None) => match extact_error.result() {
                Ok(()) => match decoder.finish() {
                    Ok(()) => None,
                    Err(err) => Some(Err(io::Error::new(io::ErrorKind::InvalidData, err))),
                },
                Err(err) => Some(Err(err)),
            },
            Err(err) => Some(Err(io::Error::new(io::ErrorKind::InvalidData, err))),
        })
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
    ) -> io::Result<Self::Context<T>> {
        let cx = ans::Context::from_input(input.clone().flatten());

        for (i, chunk) in input.enumerate() {
            let mut encoder = ans::Encoder::new(&cx);

            if output.len() == i {
                output.push(Vec::new());
            }
            let stream = &mut output[i];

            for symbol in chunk.rev() {
                stream.extend(encoder.encode(symbol));
            }

            stream.extend(encoder.finish());
        }

        Ok(cx)
    }

    fn decompress_tokens<T: Symbol>(
        &self,
        cx: &ans::Context<T>,
        mut input: impl Iterator<Item = u32>,
        output: &mut impl Extend<T>,
    ) -> io::Result<()> {
        let mut decoder = ans::Decoder::new(&cx);

        loop {
            match decoder.decode(input.by_ref()) {
                Some(symbol) => output.extend(std::iter::once(symbol)),
                None => break,
            }
        }

        match decoder.finish() {
            Ok(()) => Ok(()),
            Err(err) => Err(io::Error::new(io::ErrorKind::InvalidData, err)),
        }
    }

    fn decompress_tokens2<T: Symbol>(
        &self,
        cx: &ans::Context<T>,
        input: impl Iterator<Item = io::Result<u32>>,
    ) -> impl Iterator<Item = io::Result<T>> {
        let mut decoder = ans::Decoder::new(&cx);
        let mut extact_error = ExtractError::new(input);

        std::iter::from_fn(move || match decoder.decode(extact_error.by_ref()) {
            Some(symbol) => Some(Ok(symbol)),
            None => match extact_error.result() {
                Ok(()) => match decoder.finish() {
                    Ok(()) => None,
                    Err(err) => Some(Err(io::Error::new(io::ErrorKind::InvalidData, err))),
                },
                Err(err) => Some(Err(err)),
            },
        })
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
        output: &mut impl Extend<T>,
    ) -> io::Result<()> {
        let (a, b) = self;

        let (a_cx, b_cx) = cx;
        let mut a_tokens = Vec::new();

        b.decompress_tokens(b_cx, input, &mut a_tokens)?;
        a.decompress_tokens(a_cx, a_tokens.into_iter(), output)?;
        Ok(())
    }

    fn decompress_tokens2<T: Symbol>(
        &self,
        cx: &Self::Context<T>,
        input: impl Iterator<Item = io::Result<Self::Token<T>>>,
    ) -> impl Iterator<Item = io::Result<T>> {
        let (a, b) = self;

        let (a_cx, b_cx) = cx;

        let a_tokens = b.decompress_tokens2(b_cx, input);
        let symbols = a.decompress_tokens2(a_cx, a_tokens);

        symbols
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

/// Context for RLE compression. RLE is stateless, so this holds no data.
#[derive(Clone, Copy, Debug)]
pub struct RleContext;

/// Iterator that expands a single RLE token into its individual symbols,
/// or yields one error and then stops. Used by `RleCompressor::decompress_tokens2`.
///
/// The two cases are mutually exclusive and represented as enum variants.
enum RleExpand<T> {
    Repeat { value: T, remaining: usize },
    Error(Option<io::Error>),
}

impl<T: Copy> Iterator for RleExpand<T> {
    type Item = io::Result<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RleExpand::Repeat { value, remaining } => {
                if *remaining == 0 {
                    None
                } else {
                    *remaining -= 1;
                    Some(Ok(*value))
                }
            }
            RleExpand::Error(e) => e.take().map(Err),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match self {
            RleExpand::Repeat { remaining, .. } => *remaining,
            RleExpand::Error(e) => e.is_some() as usize,
        };
        (n, Some(n))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self {
            RleExpand::Repeat { value, remaining } => {
                if n >= *remaining {
                    *remaining = 0;
                    None
                } else {
                    *remaining -= n + 1;
                    Some(Ok(*value))
                }
            }
            RleExpand::Error(e) => {
                if n == 0 {
                    e.take().map(Err)
                } else {
                    *e = None;
                    None
                }
            }
        }
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self {
            RleExpand::Repeat { value, remaining } => {
                let mut acc = init;
                for _ in 0..remaining {
                    acc = f(acc, Ok(value));
                }
                acc
            }
            RleExpand::Error(Some(e)) => f(init, Err(e)),
            RleExpand::Error(None) => init,
        }
    }
}

impl Encode for RleContext {
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
        Ok(RleContext)
    }
}

/// Compressor that uses run-length encoding (RLE).
///
/// Consecutive equal symbols are compressed into a single `Rle` token
/// carrying the symbol value and its repetition count.
#[derive(Clone, Copy, Debug)]
pub struct RleCompressor;

impl Compressor for RleCompressor {
    type Token<T: Symbol> = rle::Rle<T>;
    type Context<T: Symbol> = RleContext;

    fn compress_symbols<T: Symbol>(
        &self,
        input: impl Iterator<Item = impl Iterator<Item = T>>,
        output: &mut Vec<Vec<rle::Rle<T>>>,
    ) -> io::Result<RleContext> {
        for (i, chunk) in input.enumerate() {
            if output.len() == i {
                output.push(Vec::new());
            }
            output[i].extend(rle::rle(chunk));
        }
        Ok(RleContext)
    }

    #[inline]
    fn decompress_tokens<T: Symbol>(
        &self,
        _cx: &RleContext,
        input: impl Iterator<Item = rle::Rle<T>>,
        output: &mut impl Extend<T>,
    ) -> io::Result<()> {
        for rle::Rle { value, count } in input {
            output.extend(std::iter::repeat(value).take(count));
        }
        Ok(())
    }

    #[inline]
    fn decompress_tokens2<T: Symbol>(
        &self,
        _cx: &RleContext,
        input: impl Iterator<Item = io::Result<rle::Rle<T>>>,
    ) -> impl Iterator<Item = io::Result<T>> {
        input.flat_map(|result| match result {
            Ok(rle::Rle { value, count }) => RleExpand::Repeat { value, remaining: count },
            Err(e) => RleExpand::Error(Some(e)),
        })
    }
}

struct ExtractError<I> {
    iter: I,
    error: Option<io::Error>,
}

impl<I> ExtractError<I> {
    fn new(iter: I) -> Self {
        ExtractError { iter, error: None }
    }

    fn result(&mut self) -> io::Result<()> {
        match self.error.take() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl<T, I> Iterator for ExtractError<I>
where
    I: Iterator<Item = io::Result<T>>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next()? {
            Ok(token) => Some(token),
            Err(err) => {
                self.error = Some(err);
                None
            }
        }
    }
}
