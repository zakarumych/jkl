//! LZ78 dictionary compression.
//!
//! Symbols are compressed into [`Token`]s - a prefix index into the dictionary paired with a trailing literal.
//! The [`Encoder`] builds tokens from an input stream and
//! the [`Decoder`] reconstructs the original data.

use std::{
    io,
    iter::{Fuse, FusedIterator},
};

use crate::{
    bits::{ReadBits, WriteBits},
    encode::VarCode,
    vle,
};

/// An LZ78 token: a dictionary prefix index paired with a trailing literal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token<T> {
    /// Zero-based index into the dictionary. `0` means no prefix (standalone literal).
    pub prefix: usize,
    /// The literal symbol that follows the prefix string.
    pub literal: T,
}

impl<T> VarCode for Token<T>
where
    T: VarCode,
{
    fn var_bit_len(&self) -> usize {
        vle::encode_bit_len(self.prefix) + self.literal.var_bit_len()
    }

    fn var_write(&self, writer: &mut WriteBits<impl io::Write>) -> io::Result<()> {
        vle::encode(self.prefix, writer)?;
        self.literal.var_write(writer)
    }

    fn var_read(reader: &mut ReadBits<impl io::Read>) -> io::Result<Self> {
        let prefix = vle::decode::<usize, _>(reader)?;
        let literal = T::var_read(reader)?;

        Ok(Token { prefix, literal })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Entry<T> {
    prefix: usize,
    literal: T,
}

/// LZ78 encoder that builds a dictionary on the fly and emits [`Token`]s.
pub struct Encoder<T> {
    entires: Vec<Entry<T>>,
    prefix: usize,
}

impl<T> Default for Encoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Encoder<T> {
    /// Creates a new, empty LZ78 encoder.
    pub fn new() -> Self {
        Encoder {
            entires: Vec::new(),
            prefix: 0,
        }
    }
}

impl<T> Encoder<T>
where
    T: Copy + Eq,
{
    fn lookup(&self, entry: Entry<T>) -> Option<usize> {
        for i in entry.prefix..self.entires.len() {
            if self.entires[i] == entry {
                return Some(i + 1);
            }
        }

        None
    }

    /// Feeds one symbol to the encoder; returns a token when a dictionary
    /// entry is completed.
    pub fn encode(&mut self, symbol: T) -> Option<Token<T>> {
        let entry = Entry {
            prefix: self.prefix,
            literal: symbol,
        };
        let index = self.lookup(entry);

        match index {
            None => {
                debug_assert!(u32::try_from(self.prefix).is_ok());

                let token = Token {
                    prefix: self.prefix,
                    literal: symbol,
                };
                self.entires.push(entry);
                self.prefix = 0;
                Some(token)
            }
            Some(index) => {
                self.prefix = index;
                None
            }
        }
    }

    /// Flushes the encoder, returning any final pending token.
    pub fn finish(self) -> Option<Token<T>> {
        { self }.finish_mut()
    }

    fn finish_mut(&mut self) -> Option<Token<T>> {
        if self.prefix == 0 {
            return None;
        }

        let last = &self.entires[self.prefix - 1];
        self.prefix = 0;

        Some(Token {
            prefix: last.prefix,
            literal: last.literal,
        })
    }

    /// Wraps this encoder with an input iterator, producing a lazy token
    /// stream.
    pub fn stream<I>(self, input: I) -> EncodeStream<T, I::IntoIter>
    where
        I: IntoIterator,
    {
        EncodeStream {
            encoder: self,
            input: input.into_iter().fuse(),
        }
    }
}

/// A lazy iterator adapter that feeds an input iterator through an
/// [`Encoder`] and yields [`Token`]s.
pub struct EncodeStream<T, I> {
    encoder: Encoder<T>,
    input: Fuse<I>,
}

impl<T, I> Iterator for EncodeStream<T, I>
where
    I: Iterator<Item = T>,
    T: Eq + Copy,
{
    type Item = Token<T>;

    fn next(&mut self) -> Option<Token<T>> {
        for input in self.input.by_ref() {
            if let Some(token) = self.encoder.encode(input) {
                return Some(token);
            }
        }

        self.encoder.finish_mut()
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.input
            .fold(init, |state, input| match self.encoder.encode(input) {
                None => state,
                Some(token) => f(state, token),
            })
    }
}

impl<T, I> FusedIterator for EncodeStream<T, I>
where
    I: Iterator<Item = T>,
    T: Eq + Copy,
{
}

/// LZ78 decoder that rebuilds the dictionary and expands tokens back into symbols.
pub struct Decoder<T> {
    scratch: Vec<T>,
    entires: Vec<(usize, usize)>,
}

impl<T> Default for Decoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Decoder<T> {
    /// Creates a new, empty LZ78 decoder.
    pub fn new() -> Self {
        Decoder {
            scratch: Vec::new(),
            entires: Vec::new(),
        }
    }
}

/// Errors that can occur while decoding an LZ78 stream.
#[derive(Debug)]
pub enum DecodeError {
    /// A token referenced a dictionary index that does not exist.
    InvalidIndex,
}

impl<T> Decoder<T>
where
    T: Copy + Eq,
{
    fn decode_next_range(&mut self, token: Token<T>) -> Result<(usize, usize), DecodeError> {
        // Add the new substring to the cache.
        let (prefix_start, prefix_end) = if token.prefix > 0 {
            if token.prefix > self.entires.len() {
                return Err(DecodeError::InvalidIndex);
            }
            self.entires[token.prefix - 1]
        } else {
            (0, 0)
        };

        debug_assert!(prefix_end >= prefix_start);
        let prefix_len = prefix_end - prefix_start;

        let element = token.literal;

        let end = if self.entires.is_empty() {
            0
        } else {
            let (_start, end) = *self.entires.last().unwrap();
            end
        };

        debug_assert_eq!(end, self.scratch.len());

        self.scratch.extend_from_within(prefix_start..prefix_end);
        self.scratch.push(element);

        let new_start = end;
        let new_end = new_start + prefix_len + 1;

        self.entires.push((new_start, new_end));

        Ok((new_start, new_end))
    }

    /// Decodes a single token, returning the expanded symbol slice.
    pub fn decode_next_slice(&mut self, token: Token<T>) -> Result<&[T], DecodeError> {
        let output = self.decode_next_range(token)?;

        let slice = &self.scratch[output.0..output.1];
        Ok(slice)
    }
}

#[test]
fn test_u16() {
    let mut encoder = Encoder::<u16>::new();
    let mut compressed = Vec::new();

    let data = [
        1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 1, 2,
        1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 1, 2,
        1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 1, 2,
        1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 1, 2,
        1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 1, 2,
        1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 1, 3, 1, 1, 1, 2, 2,
        1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 1, 3, 1,
        1, 1, 2, 2, 1, 1, 3, 3, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 3,
        3, 1, 1, 1, 2, 1, 3, 1, 1, 1, 2, 2, 1, 1, 3, 3,
    ];

    for symbol in data {
        if let Some(token) = encoder.encode(symbol) {
            compressed.push(token);
        }
    }

    if let Some(token) = encoder.finish() {
        compressed.push(token);
    }

    let mut decoder = Decoder::<u16>::new();
    let mut decoded = 0;

    for token in compressed {
        let slice = decoder.decode_next_slice(token).unwrap();
        assert_eq!(data[decoded..][..slice.len()], *slice);
        decoded += slice.len();
    }
}
