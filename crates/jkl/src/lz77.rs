//! LZ77 sliding-window compression.
//!
//! Symbols are compressed into [`Token`]s - either literals or back-references
//! into a sliding window. The [`Encoder`] builds tokens from an input stream and
//! the [`Decoder`] reconstructs the original data.

use std::{error::Error, fmt, io, num::NonZero};

use crate::{
    bits::{ReadBits, WriteBits},
    encode::VarCode,
    math::Delta,
    vle,
};

/// An LZ77 token: either a literal symbol or a back-reference into the sliding window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Token<T> {
    /// A single uncompressed symbol.
    Literal { symbol: T },
    /// A back-reference copying `length` symbols from `distance` positions back in the window.
    Reference { length_minus_2: u16, distance: u16 },
}

impl<T> Default for Token<T>
where
    T: Default,
{
    fn default() -> Self {
        Token::Literal {
            symbol: T::default(),
        }
    }
}

impl<T> Delta for Token<T>
where
    T: Delta,
{
    #[inline]
    fn delta(self, base: Self) -> Token<T> {
        match (self, base) {
            (Token::Literal { symbol: me }, Token::Literal { symbol: base }) => Token::Literal {
                symbol: me.delta(base),
            },
            (
                Token::Reference {
                    length_minus_2: length_minus_2_me,
                    distance: distance_me,
                },
                Token::Reference {
                    length_minus_2: length_minus_2_base,
                    distance: distance_base,
                },
            ) => {
                if length_minus_2_me == length_minus_2_base {
                    Token::Reference {
                        length_minus_2: 0,
                        distance: distance_me - distance_base,
                    }
                } else {
                    Token::Reference {
                        length_minus_2: length_minus_2_me - length_minus_2_base,
                        distance: distance_me,
                    }
                }
            }
            (me, Token::Reference { .. }) => me,
            (me, Token::Literal { .. }) => me,
        }
    }

    #[inline]
    fn from_delta(base: Self, delta: Token<T>) -> Token<T> {
        match (base, delta) {
            (Token::Literal { symbol: base }, Token::Literal { symbol: delta }) => Token::Literal {
                symbol: T::from_delta(base, delta),
            },
            (
                Token::Literal { .. },
                Token::Reference {
                    length_minus_2,
                    distance,
                },
            ) => Token::Reference {
                length_minus_2,
                distance,
            },
            (Token::Reference { .. }, Token::Literal { symbol }) => Token::Literal { symbol },
            (
                Token::Reference {
                    length_minus_2: length_minus_2_base,
                    distance: distance_base,
                },
                Token::Reference {
                    length_minus_2: length_minus_2_delta,
                    distance: distance_delta,
                },
            ) => {
                if length_minus_2_delta == 0 {
                    Token::Reference {
                        // Inverse of `delta`: `length_delta` stores `(me - base + 2)`.
                        length_minus_2: length_minus_2_base,
                        distance: distance_base + distance_delta,
                    }
                } else {
                    Token::Reference {
                        // Inverse of `delta`: `length_delta` stores `(me - base + 2)`.
                        length_minus_2: length_minus_2_base + length_minus_2_delta,
                        distance: distance_delta,
                    }
                }
            }
        }
    }
}

impl<T> VarCode for Token<T>
where
    T: VarCode,
{
    #[inline]
    fn var_bit_len(&self) -> usize {
        match *self {
            Token::Reference {
                length_minus_2,
                distance,
            } => {
                let length = NonZero::new(u32::from(length_minus_2) + 2).unwrap();
                let length_bits = vle::encode_non_zero_bit_len::<u32>(length);
                let distance_bits = vle::encode_bit_len(distance);
                length_bits + distance_bits
            }
            Token::Literal { ref symbol } => 1 + symbol.var_bit_len(),
        }
    }

    #[inline]
    fn var_write(&self, writer: &mut WriteBits<impl io::Write>) -> io::Result<()> {
        match *self {
            Token::Reference {
                length_minus_2,
                distance,
            } => {
                let length = const { NonZero::new(2u16).unwrap() }
                    .checked_add(length_minus_2)
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "length overflow")
                    })?;

                vle::encode_non_zero::<u16, _>(length, writer)?;
                vle::encode(distance, writer)?;
            }
            Token::Literal { ref symbol } => {
                vle::encode_non_zero::<u8, _>(const { NonZero::new(1).unwrap() }, writer)?;
                symbol.var_write(&mut *writer)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn var_read(reader: &mut ReadBits<impl io::Read>) -> io::Result<Self> {
        let length = vle::decode_non_zero::<u16, _>(reader)?;

        match length.get() {
            0 => unreachable!(),
            1 => {
                let symbol = T::var_read(&mut *reader)?;
                Ok(Token::Literal { symbol })
            }
            2.. => {
                let distance = vle::decode(reader)?;
                Ok(Token::Reference {
                    length_minus_2: length.get() - 2,
                    distance,
                })
            }
        }
    }
}

struct Window<T> {
    buffer: Vec<T>,
    head: u16,
}

impl<T> Window<T> {
    fn new(init: T, length: u16) -> Self
    where
        T: Copy,
    {
        Window {
            buffer: vec![init; usize::from(length)],
            head: 0,
        }
    }

    #[inline]
    fn len(&self) -> u16 {
        // buffer.len() is known to fit in u16
        self.buffer.len() as u16
    }

    #[inline]
    fn idx(&self, index: u16) -> u16 {
        assert!(index < self.len());
        let idx = (u32::from(self.head) + u32::from(self.len()) - 1 - u32::from(index))
            % u32::from(self.len());

        // Reminder of division by `u16` fits in `u16`.
        idx as u16
    }

    #[inline(always)]
    fn from_idx(&self, idx: u16) -> u16 {
        self.idx(idx)
    }

    #[inline]
    fn push(&mut self, value: T) {
        self.buffer[usize::from(self.head)] = value;
        self.head = (self.head + 1) % self.len();
    }

    #[inline]
    fn get(&self, index: u16) -> &T {
        let idx = self.idx(index);
        &self.buffer[usize::from(idx)]
    }

    #[inline]
    fn find_elem(&self, offset: u16, elem: &T) -> Option<u16>
    where
        T: PartialEq,
    {
        let offset = self.idx(offset);

        if offset < self.head {
            for i in (0..offset + 1).rev() {
                if self.buffer[usize::from(i)] == *elem {
                    return Some(self.from_idx(i));
                }
            }
            for i in (self.head..self.len()).rev() {
                if self.buffer[usize::from(i)] == *elem {
                    return Some(self.from_idx(i));
                }
            }
        } else {
            for i in (self.head..offset + 1).rev() {
                if self.buffer[usize::from(i)] == *elem {
                    return Some(self.from_idx(i));
                }
            }
        }

        None
    }

    // Searches window for a match of sequence specified by `distance`-`length` pair,
    // and followed by `next` symbol.
    // Only tries offsets larget than `distance`,
    // since it should be impossible to find a match of required length at smaller offset.
    #[inline]
    fn find_extension(&self, distance: u16, length: u16, next: &T) -> Option<u16>
    where
        T: PartialEq,
    {
        let d = self.idx(distance);

        // m is the length after which match spills into negative window index,
        // i.e. when it starts repeating until at least `length` symbols are matched.
        let m = if d < self.head {
            self.head - d
        } else {
            (self.len() - d) + self.head
        };

        if d < self.head {
            // Consider offsets before current one
            // In left part of the window.
            'a: for p in (0..d).rev() {
                // How many symbols there are until the end of the window,
                // i.e. when it starts repeating until at least `length` symbols are matched.
                let n = self.head - p;

                // How many symbols must match before it is guaranteed that `length` symbols are matched
                // Two repeating sequences match endlessly if they match on the sum of their repeating lengths.
                let l = length.min(n + m);

                // Check if there's match of required length.
                for j in 0..l {
                    let pj = p + (j % n);
                    let dj = d + (j % m);
                    if self.buffer[usize::from(pj)] != self.buffer[usize::from(dj)] {
                        // Not a match, try next offset.
                        continue 'a;
                    }
                }

                let pl = p + (length % n);
                if self.buffer[usize::from(pl)] != *next {
                    // Not a match, try next offset.
                    continue 'a;
                }

                // Found a match of required length, return offset.
                return Some(self.from_idx(p));
            }

            // Consider offsets after current one
            // In right part of the window.
            'a: for p in (self.head..self.len()).rev() {
                // How many symbols there are until the end of the window,
                // i.e. when it starts repeating until at least `length` symbols are matched.
                let n = (self.len() - p) + self.head;

                // How many symbols must match before it is guaranteed that `length` symbols are matched
                // Two repeating sequences match endlessly if they match on the sum of their repeating lengths.
                let l = length.min(n + m);

                // Check if there's match of required length.
                for j in 0..l {
                    let pj = (p + (j % n)) % self.len();
                    let dj = (d + (j % m)) % self.len();
                    if self.buffer[usize::from(pj)] != self.buffer[usize::from(dj)] {
                        // Not a match, try next offset.
                        continue 'a;
                    }
                }

                let pl = (p + (length % n)) % self.len();
                if self.buffer[usize::from(pl)] != *next {
                    // Not a match, try next offset.
                    continue 'a;
                }

                // Found a match of required length, return offset.
                return Some(self.idx(p));
            }

            None
        } else {
            // Consider offsets before current one
            // In right part of the window.
            'a: for p in (self.head..d).rev() {
                // How many symbols there are until the end of the window,
                // i.e. when it starts repeating until at least `length` symbols are matched.
                let n = (self.len() - p) + self.head;

                // How many symbols must match before it is guaranteed that `length` symbols are matched
                // Two repeating sequences match endlessly if they match on the sum of their repeating lengths.
                let l = length.min(n + m);

                // Check if there's match of required length.
                for j in 0..l {
                    let pj = (p + (j % n)) % self.len();
                    let dj = (d + (j % m)) % self.len();
                    if self.buffer[usize::from(pj)] != self.buffer[usize::from(dj)] {
                        // Not a match, try next offset.
                        continue 'a;
                    }
                }

                let pl = (p + (length % n)) % self.len();
                if self.buffer[usize::from(pl)] != *next {
                    // Not a match, try next offset.
                    continue 'a;
                }

                // Found a match of required length, return offset.
                return Some(self.from_idx(p));
            }

            None
        }
    }
}

#[inline]
fn distance_index(distance: u16, index: u16) -> u16 {
    distance - (index % (distance + 1))
}

/// LZ77 encoder that compresses a symbol stream into [`Token`]s using a sliding window.
pub struct Encoder<T> {
    window: Window<T>,
    distance: u16,
    length: u16,
}

impl<T> Encoder<T>
where
    T: Copy + Eq,
{
    /// Creates a new encoder with a sliding window of `length` entries, initialized to `init`.
    #[inline]
    pub fn new(init: T, length: u16) -> Self {
        Encoder {
            window: Window::new(init, length),
            distance: 0,
            length: 0,
        }
    }

    /// Feeds one symbol into the encoder, emitting tokens to `output` as matches are resolved.
    #[inline]
    pub fn encode(&mut self, symbol: T, output: &mut impl Extend<Token<T>>) {
        if self.length > 0 {
            if self.length < u16::MAX {
                // If longer match is possible to represent.

                if *self.window.get(distance_index(self.distance, self.length)) == symbol {
                    // Input continues the current match.
                    self.length += 1;
                    return;
                }

                // If input does not continues current match, try find extension deeper in the window.
                if let Some(pos) = self
                    .window
                    .find_extension(self.distance, self.length, &symbol)
                {
                    // Extension match found, update match to it and continue.
                    self.distance = pos;
                    self.length += 1;
                    return;
                }
            }

            // Failed to continue current match, emit it and start anew.
            let should_emit_reference = self.length >= 2;

            if should_emit_reference {
                debug_assert!(u16::try_from(self.distance).is_ok());
                debug_assert!(u16::try_from(self.length).is_ok());

                output.extend(Some(Token::Reference {
                    distance: self.distance,
                    length_minus_2: self.length - 2,
                }));
            }

            for i in 0..self.length {
                // Rotate window and emit literals if current match is not long enough.
                let symbol = *self.window.get(distance_index(self.distance, i));
                self.window.push(symbol);
                self.distance += 1;

                if !should_emit_reference {
                    output.extend(Some(Token::Literal { symbol }));
                }
            }

            self.distance = 0;
            self.length = 0;
        }

        match self.window.find_elem(0, &symbol) {
            None => {
                // Symbol is not in the window, emit it as literal and add to the window.
                self.window.push(symbol);
                self.distance = 0;
                self.length = 0;
                output.extend(Some(Token::Literal { symbol }));
            }
            Some(pos) => {
                // Symbol is in the window, start new match.

                self.distance = pos;
                self.length = 1;
            }
        }
    }

    /// Flushes any pending match, emitting final tokens to `output`.
    #[inline]
    pub fn finish(&mut self, output: &mut impl Extend<Token<T>>) {
        let should_emit_reference = self.length >= 2;

        if should_emit_reference {
            debug_assert!(u16::try_from(self.distance).is_ok());
            debug_assert!(u16::try_from(self.length).is_ok());

            output.extend(Some(Token::Reference {
                distance: self.distance,
                length_minus_2: self.length - 2,
            }));
        } else {
            for i in 0..self.length {
                let symbol = *self.window.get(distance_index(self.distance, i));
                self.distance += 1;
                output.extend(Some(Token::Literal { symbol }));
            }
        }
    }
}

struct Entry {
    distance: u16,
    length: u16,
}

/// LZ77 decoder that reconstructs the original symbol stream from [`Token`]s.
pub struct Decoder<T> {
    window: Window<T>,
    entry: Option<Entry>,
}

/// Errors that can occur while decoding an LZ77 stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeError {
    /// Signals that decoder did not end emitting tokens,
    /// which may mean that compressed data is corrupted.
    Incomplete,

    /// Signals that a reference token has invalid distance, i.e. distance is greater than or equal to the window size.
    InvalidDistance { distance: u16, window: u16 },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            DecodeError::Incomplete => write!(f, "incomplete decoding"),

            DecodeError::InvalidDistance { distance, window } => write!(
                f,
                "invalid distance {} for window of size {}",
                distance, window
            ),
        }
    }
}

impl Error for DecodeError {}

impl<T> Decoder<T>
where
    T: Copy + Eq,
{
    /// Creates a new decoder with a sliding window of `length` entries, initialized to `init`.
    #[inline]
    pub fn new(init: T, length: u16) -> Self {
        Decoder {
            window: Window::new(init, length),
            entry: None,
        }
    }

    /// Decodes one symbol from `tokens`, returning `Ok(None)` when the stream is exhausted.
    #[inline]
    pub fn decode(
        &mut self,
        mut tokens: impl Iterator<Item = Token<T>>,
    ) -> Result<Option<T>, DecodeError> {
        match &mut self.entry {
            None => {
                let Some(token) = tokens.next() else {
                    return Ok(None);
                };

                match token {
                    Token::Reference {
                        length_minus_2,
                        distance,
                    } => {
                        if distance >= self.window.len() {
                            return Err(DecodeError::InvalidDistance {
                                distance,
                                window: self.window.len(),
                            });
                        }

                        let length = length_minus_2 + 2;

                        let first = *self.window.get(distance);
                        self.window.push(first);

                        self.entry = Some(Entry {
                            distance,
                            length: length - 1,
                        });

                        Ok(Some(first))
                    }
                    Token::Literal { symbol } => {
                        self.window.push(symbol);
                        Ok(Some(symbol))
                    }
                }
            }
            Some(entry) => {
                debug_assert!(entry.length > 0);
                let first = *self.window.get(entry.distance);
                self.window.push(first);
                entry.length -= 1;
                if entry.length == 0 {
                    self.entry = None;
                }
                Ok(Some(first))
            }
        }
    }

    /// Decodes all remaining symbols from `tokens` and appends them to `extend`.
    #[inline]
    pub fn decode_all(
        &mut self,
        tokens: impl Iterator<Item = Token<T>>,
        extend: &mut impl Extend<T>,
    ) -> Result<(), DecodeError> {
        if let Some(mut entry) = self.entry.take() {
            while entry.length > 0 {
                debug_assert!(entry.length > 0);
                let first = *self.window.get(entry.distance);
                self.window.push(first);
                entry.length -= 1;
                extend.extend(Some(first));
            }
        }

        for token in tokens {
            match token {
                Token::Reference {
                    distance,
                    length_minus_2,
                } => {
                    if distance >= self.window.len() {
                        return Err(DecodeError::InvalidDistance {
                            distance,
                            window: self.window.len(),
                        });
                    }

                    let mut length = length_minus_2 + 2;
                    while length > 0 {
                        debug_assert!(length > 0);
                        let first = *self.window.get(distance);
                        self.window.push(first);
                        length -= 1;
                        extend.extend(Some(first));
                    }
                }
                Token::Literal { symbol } => {
                    self.window.push(symbol);
                    extend.extend(Some(symbol));
                }
            }
        }

        Ok(())
    }

    /// Verifies that decoding finished cleanly with no pending entries.
    pub fn finish(&self) -> Result<(), DecodeError> {
        if self.entry.is_some() {
            Err(DecodeError::Incomplete)
        } else {
            Ok(())
        }
    }
}

#[test]
fn test_u16() {
    let mut encoder = Encoder::<u16>::new(0, 256);
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

    for byte in data {
        encoder.encode(byte, &mut compressed);
    }

    encoder.finish(&mut compressed);

    let mut decoder = Decoder::<u16>::new(0, 256);

    let mut input = compressed.iter().copied();
    let mut decoded = Vec::new();

    while decoded.len() < data.len() {
        let elem = decoder.decode(&mut input).unwrap().unwrap();
        decoded.push(elem);
    }

    assert_eq!(data[..], decoded[..]);

    decoded.clear();
    decoder
        .decode_all(compressed.into_iter(), &mut decoded)
        .unwrap();

    assert_eq!(data[..], decoded[..]);
}
