use std::{hash::Hash, io};

use crate::{
    ans,
    bits::{ReadBits, WriteBits},
    encode::Encode,
    lz77,
};

/// Trait for types viable as symbols for compression.
pub trait Symbol: Default + Copy + Ord + Hash + Encode + Sized + 'static {}

/// Blanket implementation for all types that satisfy the trait bounds.
impl<T> Symbol for T where T: Default + Copy + Ord + Hash + Encode + Sized + 'static {}

/// Compression algorithm interface.
pub trait Compressor {
    /// Token type produced by compression for symbols of type `T`.
    type Token<T: Symbol>: Encode;

    /// Context for compression for symbols of type `T`.
    type Context<T: Symbol>;

    /// Error type for decompression operation.
    type Error;

    /// Build a context for compression for symbols of type `T`.
    /// Can be no-op.
    ///
    /// Should be called with all symbols that will be later passed to [`Compressor::compress_symbols`].
    fn build_context<T: Symbol>(&self, input: impl Iterator<Item = T>) -> Self::Context<T>;

    /// Write context to bit stream.
    fn write_context<T: Symbol>(
        &self,
        context: &Self::Context<T>,
        writer: &mut WriteBits<impl io::Write>,
    ) -> io::Result<()>;

    /// Read context from bit stream.
    fn read_context<T: Symbol>(
        &self,
        reader: &mut ReadBits<impl io::Read>,
    ) -> io::Result<Self::Context<T>>;

    /// Compress the input symbols of type `T`.
    fn compress_symbols<T: Symbol>(
        &self,
        context: &Self::Context<T>,
        input: impl Iterator<Item = T>,
        output: &mut impl Extend<Self::Token<T>>,
    );

    /// Decompress the input tokens of type `Self::Token<T>`.
    fn decompress_tokens<T: Symbol>(
        &self,
        context: &Self::Context<T>,
        input: impl Iterator<Item = Self::Token<T>>,
        count: usize,
        output: &mut impl Extend<T>,
    ) -> Result<usize, Self::Error>;
}

#[derive(Clone, Copy, Debug)]
pub struct LZ77Compressor {
    pub window_size: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct LZ77Context;

impl Compressor for LZ77Compressor {
    type Token<T: Symbol> = lz77::Token<T>;
    type Context<T: Symbol> = LZ77Context;
    type Error = lz77::DecodeError;

    #[inline]
    fn build_context<T: Symbol>(&self, _input: impl Iterator<Item = T>) -> LZ77Context {
        LZ77Context
    }

    #[inline]
    fn write_context<T: Symbol>(
        &self,
        _context: &LZ77Context,
        _writer: &mut WriteBits<impl io::Write>,
    ) -> io::Result<()> {
        // No context to write.
        Ok(())
    }

    #[inline]
    fn read_context<T: Symbol>(
        &self,
        _reader: &mut ReadBits<impl io::Read>,
    ) -> io::Result<LZ77Context> {
        // No context to read.
        Ok(LZ77Context)
    }

    fn compress_symbols<T: Symbol>(
        &self,
        _context: &LZ77Context,
        input: impl Iterator<Item = T>,
        output: &mut impl Extend<lz77::Token<T>>,
    ) {
        let mut encoder = lz77::Encoder::new(T::default(), self.window_size);

        let mut buffer = Vec::new();
        buffer.reserve(4);

        for symbol in input {
            encoder.encode(symbol, &mut buffer);

            for token in &buffer {
                output.extend(Some(*token));
            }

            buffer.clear();
        }

        encoder.finish(&mut buffer);

        for token in &buffer {
            output.extend(Some(*token));
        }
    }

    fn decompress_tokens<T: Symbol>(
        &self,
        _context: &LZ77Context,
        mut input: impl Iterator<Item = lz77::Token<T>>,
        count: usize,
        output: &mut impl Extend<T>,
    ) -> Result<usize, lz77::DecodeError> {
        let mut decoder = lz77::Decoder::new(T::default(), self.window_size);

        for i in 0..count {
            match decoder.decode(input.by_ref())? {
                None => return Ok(i),
                Some(symbol) => output.extend(Some(symbol)),
            }
        }

        Ok(count)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AnsCompressor;

#[derive(Clone, Copy, Debug)]
pub struct AnsConfig;

impl Compressor for AnsCompressor {
    type Token<T: Symbol> = u32;
    type Context<T: Symbol> = ans::Context<T>;
    type Error = ans::DecodeError;

    fn build_context<T: Symbol>(&self, input: impl Iterator<Item = T>) -> ans::Context<T> {
        ans::Context::from_input(input)
    }

    fn write_context<T: Symbol>(
        &self,
        context: &ans::Context<T>,
        writer: &mut WriteBits<impl io::Write>,
    ) -> io::Result<()> {
        context.write(writer)
    }

    fn read_context<T: Symbol>(
        &self,
        reader: &mut ReadBits<impl io::Read>,
    ) -> io::Result<ans::Context<T>> {
        ans::Context::read(reader)
    }

    fn compress_symbols<T: Symbol>(
        &self,
        context: &ans::Context<T>,
        input: impl Iterator<Item = T>,
        output: &mut impl Extend<u32>,
    ) {
        let mut encoder = ans::Encoder::new(context);

        for symbol in input {
            if let Some(code) = encoder.encode(symbol) {
                output.extend(Some(code));
            }
        }

        let [x, y] = encoder.finish();
        output.extend([x, y]);
    }

    fn decompress_tokens<T: Symbol>(
        &self,
        context: &ans::Context<T>,
        mut input: impl Iterator<Item = u32>,
        count: usize,
        output: &mut impl Extend<T>,
    ) -> Result<usize, ans::DecodeError> {
        let mut decoder = ans::Decoder::new(context);

        for i in 0..count {
            match decoder.decode(input.by_ref()) {
                None => return Ok(i),
                Some(symbol) => output.extend(Some(symbol)),
            }
        }

        decoder.finish()?;
        Ok(count)
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
