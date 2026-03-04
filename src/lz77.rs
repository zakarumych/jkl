//! LZ77 sliding-window compression.
//!
//! Symbols are compressed into [`Token`]s - either literals or back-references
//! into a sliding window. The [`Encoder`] builds tokens from an input stream and
//! the [`Decoder`] reconstructs the original data.

use std::{error::Error, fmt, io};

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
    Reference { length: u32, distance: u32 },
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
                    length: me,
                    distance: me_d,
                },
                Token::Reference {
                    length: base,
                    distance: base_d,
                },
            ) => Token::Reference {
                // Keep reference-length deltas in valid reference domain (>= 2),
                // because length == 1 is reserved for literal tag in `VarCode`.
                length: me - base + 2,
                distance: if me == base { me_d - base_d } else { me_d },
            },
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
            (Token::Literal { .. }, Token::Reference { length, distance }) => {
                Token::Reference { length, distance }
            }
            (Token::Reference { .. }, Token::Literal { symbol }) => Token::Literal { symbol },
            (
                Token::Reference {
                    length: length_a,
                    distance: distance_a,
                },
                Token::Reference {
                    length: length_b,
                    distance: distance_b,
                },
            ) => Token::Reference {
                // Inverse of `delta`: `length_b` stores `(me - base + 2)`.
                length: length_a + (length_b - 2),
                distance: if length_b == 2 {
                    distance_a + distance_b
                } else {
                    distance_b
                },
            },
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
            Token::Reference { length, distance } => {
                debug_assert!(length >= 2);
                let length_bits = vle::encode_non_zero_bit_len(length);
                let distance_bits = vle::encode_bit_len(distance);
                length_bits + distance_bits
            }
            Token::Literal { ref symbol } => 1 + symbol.var_bit_len(),
        }
    }

    #[inline]
    fn var_write(&self, writer: &mut WriteBits<impl io::Write>) -> io::Result<()> {
        match *self {
            Token::Reference { length, distance } => {
                debug_assert!(length >= 2);
                vle::encode_non_zero(length, writer)?;
                vle::encode(distance, writer)?;
            }
            Token::Literal { ref symbol } => {
                vle::encode_non_zero(1u8, writer)?;
                symbol.var_write(&mut *writer)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn var_read(reader: &mut ReadBits<impl io::Read>) -> io::Result<Self> {
        let length = vle::decode_non_zero::<u32, _>(reader)?;

        match length {
            0 => unreachable!("decode_non_zero must never return 0"),
            1 => {
                let symbol = T::var_read(&mut *reader)?;
                Ok(Token::Literal { symbol })
            }
            _ => {
                let distance = vle::decode::<u32, _>(reader)?;
                Ok(Token::Reference { length, distance })
            }
        }
    }
}

struct Window<T> {
    buffer: Vec<T>,
    head: usize,
}

impl<T> Window<T> {
    fn new(init: T, length: usize) -> Self
    where
        T: Copy,
    {
        Window {
            buffer: vec![init; length],
            head: 0,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    fn idx(&self, index: usize) -> usize {
        assert!(index < self.buffer.len());
        (self.head + self.buffer.len() - 1 - index) % self.buffer.len()
    }

    #[inline]
    fn push(&mut self, value: T) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.buffer.len();
    }

    #[inline]
    fn get(&self, index: usize) -> &T {
        let idx = self.idx(index);
        &self.buffer[idx]
    }

    #[inline]
    fn find_elem(&self, offset: usize, elem: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        let offset = self.idx(offset);

        if offset < self.head {
            for i in (0..offset + 1).rev() {
                if self.buffer[i] == *elem {
                    return Some(self.idx(i));
                }
            }
            for i in (self.head..self.buffer.len()).rev() {
                if self.buffer[i] == *elem {
                    return Some(self.idx(i));
                }
            }
        } else {
            for i in (self.head..offset + 1).rev() {
                if self.buffer[i] == *elem {
                    return Some(self.idx(i));
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
    fn find_extension(&self, distance: usize, length: usize, next: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        let d = self.idx(distance);

        // m is the length after which match spills into negative window index,
        // i.e. when it starts repeating until at least `length` symbols are matched.
        let m = if d < self.head {
            self.head - d
        } else {
            (self.buffer.len() - d) + self.head
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
                    if self.buffer[pj] != self.buffer[dj] {
                        // Not a match, try next offset.
                        continue 'a;
                    }
                }

                let pl = p + (length % n);
                if self.buffer[pl] != *next {
                    // Not a match, try next offset.
                    continue 'a;
                }

                // Found a match of required length, return offset.
                return Some(self.idx(p));
            }

            // Consider offsets after current one
            // In right part of the window.
            'a: for p in (self.head..self.buffer.len()).rev() {
                // How many symbols there are until the end of the window,
                // i.e. when it starts repeating until at least `length` symbols are matched.
                let n = (self.buffer.len() - p) + self.head;

                // How many symbols must match before it is guaranteed that `length` symbols are matched
                // Two repeating sequences match endlessly if they match on the sum of their repeating lengths.
                let l = length.min(n + m);

                // Check if there's match of required length.
                for j in 0..l {
                    let pj = (p + (j % n)) % self.buffer.len();
                    let dj = (d + (j % m)) % self.buffer.len();
                    if self.buffer[pj] != self.buffer[dj] {
                        // Not a match, try next offset.
                        continue 'a;
                    }
                }

                let pl = (p + (length % n)) % self.buffer.len();
                if self.buffer[pl] != *next {
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
                let n = (self.buffer.len() - p) + self.head;

                // How many symbols must match before it is guaranteed that `length` symbols are matched
                // Two repeating sequences match endlessly if they match on the sum of their repeating lengths.
                let l = length.min(n + m);

                // Check if there's match of required length.
                for j in 0..l {
                    let pj: usize = (p + (j % n)) % self.buffer.len();
                    let dj = (d + (j % m)) % self.buffer.len();
                    if self.buffer[pj] != self.buffer[dj] {
                        // Not a match, try next offset.
                        continue 'a;
                    }
                }

                let pl = (p + (length % n)) % self.buffer.len();
                if self.buffer[pl] != *next {
                    // Not a match, try next offset.
                    continue 'a;
                }

                // Found a match of required length, return offset.
                return Some(self.idx(p));
            }

            None
        }
    }
}

#[inline]
fn distance_index(distance: usize, index: usize) -> usize {
    distance - (index % (distance + 1))
}

/// LZ77 encoder that compresses a symbol stream into [`Token`]s using a sliding window.
pub struct Encoder<T> {
    window: Window<T>,
    distance: usize,
    length: usize,
}

impl<T> Encoder<T>
where
    T: Copy + Eq,
{
    /// Creates a new encoder with a sliding window of `length` entries, initialized to `init`.
    #[inline]
    pub fn new(init: T, length: u32) -> Self {
        debug_assert!(usize::try_from(length).is_ok());

        Encoder {
            window: Window::new(init, length as usize),
            distance: 0,
            length: 0,
        }
    }

    /// Feeds one symbol into the encoder, emitting tokens to `output` as matches are resolved.
    #[inline]
    pub fn encode(&mut self, symbol: T, output: &mut impl Extend<Token<T>>) {
        if self.length > 0 {
            let max = usize::try_from(u32::MAX).unwrap_or(usize::MAX);

            if self.length < max {
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
                debug_assert!(u32::try_from(self.distance).is_ok());
                debug_assert!(u32::try_from(self.length).is_ok());

                output.extend(Some(Token::Reference {
                    distance: self.distance as u32,
                    length: self.length as u32,
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
            debug_assert!(u32::try_from(self.distance).is_ok());
            debug_assert!(u32::try_from(self.length).is_ok());

            output.extend(Some(Token::Reference {
                distance: self.distance as u32,
                length: self.length as u32,
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
    distance: usize,
    length: usize,
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
    InvalidDistance { distance: usize, window: usize },
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
    pub fn new(init: T, length: u32) -> Self {
        debug_assert!(usize::try_from(length).is_ok());

        Decoder {
            window: Window::new(init, length as usize),
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
                    Token::Reference { length, distance } => {
                        let distance = usize::try_from(distance).map_err(|_| {
                            DecodeError::InvalidDistance {
                                distance: usize::MAX,
                                window: self.window.len(),
                            }
                        })?;

                        if distance >= self.window.len() {
                            return Err(DecodeError::InvalidDistance {
                                distance,
                                window: self.window.len(),
                            });
                        }

                        debug_assert!(length > 0);
                        let first = *self.window.get(distance);
                        self.window.push(first);

                        if length > 1 {
                            self.entry = Some(Entry {
                                distance,
                                length: length as usize - 1,
                            });
                        }

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
                    mut length,
                } => {
                    let distance =
                        usize::try_from(distance).map_err(|_| DecodeError::InvalidDistance {
                            distance: usize::MAX,
                            window: self.window.len(),
                        })?;

                    if distance >= self.window.len() {
                        return Err(DecodeError::InvalidDistance {
                            distance,
                            window: self.window.len(),
                        });
                    }

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
