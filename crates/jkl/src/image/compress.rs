use std::{hash::Hash, io, num::NonZero};

use crate::{
    ans,
    encode::VarCode,
    lz77,
    math::Delta,
    rle::{Rle, rle},
};

/// Trait for types viable as symbols for compression.
pub trait Symbol: Copy + Default + Ord + Hash + Delta + VarCode + 'static {}

/// Blanket implementation for all types that satisfy the trait bounds.
impl<T> Symbol for T where T: Default + Copy + Ord + Hash + VarCode + Delta + Sized + 'static {}

/// Compression algorithm interface.
pub trait Compressor {
    /// Token type produced by compression for symbols of type `T`.
    type Token<T: Symbol>: Symbol;

    /// Context is produced during compression and is used during decompression.
    type Context<T: Symbol>: VarCode;

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
        context: &Self::Context<T>,
        input: impl Iterator<Item = Self::Token<T>>,
        output: &mut impl Extend<T>,
    ) -> io::Result<()>;

    fn decompress_tokens2<T: Symbol>(
        &self,
        context: &Self::Context<T>,
        input: impl Iterator<Item = io::Result<Self::Token<T>>>,
    ) -> impl Iterator<Item = io::Result<T>>;
}

#[derive(Clone, Copy, Debug)]
pub struct LZ77Compressor {
    pub window_size: u16,
}

impl LZ77Compressor {
    pub fn new() -> Self {
        LZ77Compressor { window_size: 1024 }
    }
}

pub struct LZ77Context;

impl_fixedcode_zero!(LZ77Context);

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
            stream.clear();

            for symbol in chunk {
                encoder.encode(symbol, stream);
            }

            encoder.finish(stream);
        }
        Ok(LZ77Context)
    }

    #[inline]
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

    #[inline]
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

impl Compressor for AnsCompressor {
    type Token<T: Symbol> = u32;
    type Context<T: Symbol> = ans::Context<T>;

    fn compress_symbols<T: Symbol>(
        &self,
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = T>> + Clone,
        output: &mut Vec<Vec<u32>>,
    ) -> io::Result<Self::Context<T>> {
        let mut context = ans::Context::from_input(input.clone().flatten());
        context.build_encoder_index();

        // context.print_frequencies();

        for (i, chunk) in input.enumerate() {
            let mut encoder = ans::Encoder::new(&context);

            if output.len() == i {
                output.push(Vec::new());
            }
            let stream = &mut output[i];
            stream.clear();

            for symbol in chunk.rev() {
                stream.extend(encoder.encode(symbol));
            }

            stream.extend(encoder.finish());
            stream.reverse();
        }

        Ok(context)
    }

    #[inline]
    fn decompress_tokens<T: Symbol>(
        &self,
        context: &ans::Context<T>,
        mut input: impl Iterator<Item = u32>,
        output: &mut impl Extend<T>,
    ) -> io::Result<()> {
        let mut decoder = ans::Decoder::new(context);

        while let Some(symbol) = decoder.decode(input.by_ref()) {
            output.extend(std::iter::once(symbol));
        }

        match decoder.finish() {
            Ok(()) => Ok(()),
            Err(err) => Err(io::Error::new(io::ErrorKind::InvalidData, err)),
        }
    }

    #[inline]
    fn decompress_tokens2<T: Symbol>(
        &self,
        context: &ans::Context<T>,
        input: impl Iterator<Item = io::Result<u32>>,
    ) -> impl Iterator<Item = io::Result<T>> {
        let mut decoder = ans::Decoder::new(context);
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

impl Compressor for () {
    type Token<T: Symbol> = T;
    type Context<T: Symbol> = ();

    fn compress_symbols<T: Symbol>(
        &self,
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = T>> + Clone,
        output: &mut Vec<Vec<T>>,
    ) -> io::Result<()> {
        for (i, chunk) in input.enumerate() {
            if output.len() == i {
                output.push(Vec::new());
            }
            let stream = &mut output[i];
            stream.clear();

            for symbol in chunk {
                stream.push(symbol);
            }
        }
        Ok(())
    }

    #[inline]
    fn decompress_tokens<T: Symbol>(
        &self,
        _cx: &(),
        input: impl Iterator<Item = T>,
        output: &mut impl Extend<T>,
    ) -> io::Result<()> {
        output.extend(input);
        Ok(())
    }

    #[inline]
    fn decompress_tokens2<T: Symbol>(
        &self,
        _cx: &(),
        input: impl Iterator<Item = io::Result<T>>,
    ) -> impl Iterator<Item = io::Result<T>> {
        input
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

    #[inline]
    fn decompress_tokens<T: Symbol>(
        &self,
        context: &Self::Context<T>,
        input: impl Iterator<Item = B::Token<A::Token<T>>>,
        output: &mut impl Extend<T>,
    ) -> io::Result<()> {
        let (a, b) = self;

        let (a_cx, b_cx) = context;
        let mut a_tokens = Vec::new();

        b.decompress_tokens(b_cx, input, &mut a_tokens)?;
        a.decompress_tokens(a_cx, a_tokens.into_iter(), output)?;
        Ok(())
    }

    #[inline]
    fn decompress_tokens2<T: Symbol>(
        &self,
        context: &Self::Context<T>,
        input: impl Iterator<Item = io::Result<Self::Token<T>>>,
    ) -> impl Iterator<Item = io::Result<T>> {
        let (a, b) = self;

        let (a_cx, b_cx) = context;

        let a_tokens = b.decompress_tokens2(b_cx, input);
        let symbols = a.decompress_tokens2(a_cx, a_tokens);

        symbols
    }
}

struct ExtractError<I> {
    iter: I,
    error: Option<io::Error>,
}

impl<I> ExtractError<I> {
    #[inline]
    fn new(iter: I) -> Self {
        ExtractError { iter, error: None }
    }

    #[inline]
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

    #[inline]
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

pub struct RleIoExpand<T>(RleIoExpandVariant<T>);

impl<T> RleIoExpand<T> {
    pub fn new(rle: Rle<T>) -> Self {
        RleIoExpand(RleIoExpandVariant::Repeat {
            value: rle.value,
            count: rle.count,
        })
    }

    pub fn error(err: io::Error) -> Self {
        RleIoExpand(RleIoExpandVariant::NoRepeat { error: Some(err) })
    }
}

enum RleIoExpandVariant<T> {
    Repeat { value: T, count: NonZero<usize> },
    NoRepeat { error: Option<io::Error> },
}

impl<T> Iterator for RleIoExpand<T>
where
    T: Copy,
{
    type Item = io::Result<T>;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            RleIoExpandVariant::Repeat { value, count } => {
                let result = *value;
                match NonZero::new(count.get() - 1) {
                    Some(new_count) => *count = new_count,
                    None => self.0 = RleIoExpandVariant::NoRepeat { error: None },
                }
                Some(Ok(result))
            }
            RleIoExpandVariant::NoRepeat { error } => error.take().map(Err),
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match &mut self.0 {
            RleIoExpandVariant::Repeat { value, count } => {
                let result = *value;

                match count.get() {
                    c if c >= n => {
                        match NonZero::new(c - n) {
                            Some(new_count) => *count = new_count,
                            None => self.0 = RleIoExpandVariant::NoRepeat { error: None },
                        }
                        Some(Ok(result))
                    }
                    _ => {
                        self.0 = RleIoExpandVariant::NoRepeat { error: None };
                        None
                    }
                }
            }
            RleIoExpandVariant::NoRepeat { error } => error.take().map(Err),
        }
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, io::Result<T>) -> B,
    {
        match self.0 {
            RleIoExpandVariant::Repeat { value, count } => {
                let mut acc = init;
                for _ in 0..count.get() {
                    acc = f(acc, Ok(value));
                }
                acc
            }
            RleIoExpandVariant::NoRepeat { error } => match error {
                Some(err) => f(init, Err(err)),
                None => init,
            },
        }
    }
}

impl<T> ExactSizeIterator for RleIoExpand<T>
where
    T: Copy,
{
    #[inline]
    fn len(&self) -> usize {
        match &self.0 {
            RleIoExpandVariant::Repeat { count, .. } => count.get(),
            RleIoExpandVariant::NoRepeat { error } => error.is_some() as usize,
        }
    }
}

impl<T> DoubleEndedIterator for RleIoExpand<T>
where
    T: Copy,
{
    #[inline]
    fn next_back(&mut self) -> Option<io::Result<T>> {
        // In case of repeat, it yields same `Ok(T)` values.
        // In case of error it yields just one `Err(io::Error)`.
        // Thus iteration direction doesn't matter, and we can just call `next()`.
        self.next()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // Similar to `next_back()`, we can just call `nth(n)`.
        self.nth(n)
    }

    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, io::Result<T>) -> B,
    {
        // Similar to `next_back()`, we can just call `fold(init, f)`.
        self.fold(init, f)
    }

    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        // Similar to `next_back()`, we can just call `find(predicate)`.
        self.find(predicate)
    }
}

pub struct RleCompressor;

pub struct RleContext;

impl_fixedcode_zero!(RleContext);

impl Compressor for RleCompressor {
    type Token<T: Symbol> = Rle<T>;
    type Context<T: Symbol> = RleContext;

    fn compress_symbols<T: Symbol>(
        &self,
        input: impl Iterator<Item = impl DoubleEndedIterator<Item = T>> + Clone,
        output: &mut Vec<Vec<Rle<T>>>,
    ) -> io::Result<Self::Context<T>> {
        for (i, chunk) in input.enumerate() {
            if output.len() == i {
                output.push(Vec::new());
            }

            let stream = &mut output[i];
            stream.clear();
            stream.extend(rle(chunk));
        }
        Ok(RleContext)
    }

    fn decompress_tokens<T: Symbol>(
        &self,
        _cx: &RleContext,
        input: impl Iterator<Item = Rle<T>>,
        output: &mut impl Extend<T>,
    ) -> io::Result<()> {
        for rle in input {
            output.extend(rle);
        }
        Ok(())
    }

    fn decompress_tokens2<T: Symbol>(
        &self,
        _cx: &RleContext,
        input: impl Iterator<Item = io::Result<Rle<T>>>,
    ) -> impl Iterator<Item = io::Result<T>> {
        input.flat_map(|rle| match rle {
            Ok(rle) => RleIoExpand::new(rle),
            Err(err) => RleIoExpand::error(err),
        })
    }
}
