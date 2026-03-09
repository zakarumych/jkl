//! Asymmetric Numeral Systems (ANS) entropy coder.
//!
//! Provides a [`Context`] that holds per-symbol frequency tables, an [`Encoder`]
//! that compresses symbols into a stream of `u32` tokens, and a [`Decoder`] that
//! reconstructs the original symbols from that token stream in reverse order.

use std::{cmp, error::Error, fmt, hash::Hash, io, num::NonZero};

use hashbrown::{HashMap, hash_map::Entry};

use crate::{
    bits::{ReadBits, WriteBits},
    encode::VarCode,
    math::Delta,
    vle,
};

/// Shared frequency table used by [`Encoder`] and [`Decoder`].
///
/// A `Context` stores per-symbol frequencies, cumulative frequencies, and a
/// lookup table for fast symbol resolution during decoding. It is built once
/// from input data or a frequency map and then shared by reference with
/// encoders and decoders.
#[derive(Clone, Debug)]
pub struct Context<T> {
    freqs: HashMap<T, NonZero<u32>>,
    cumul: HashMap<T, u32>,
    map: Vec<(u32, T)>,
}

impl<T> Context<T> {
    /// Builds a context from sorted (symbol, frequency) pairs and an optional
    /// pre-built frequency map.
    fn build(
        freqs_sorted: impl IntoIterator<Item = (T, NonZero<u32>)>,
        freqs: Option<HashMap<T, NonZero<u32>>>,
    ) -> Self
    where
        T: Eq + Hash + Copy,
    {
        let mut cumul = HashMap::<T, u32>::new();
        let mut accum = 0u32;
        let build_freqs = freqs.is_none();
        let mut freqs = freqs.unwrap_or_default();

        for (symbol, count) in freqs_sorted {
            if build_freqs {
                freqs.insert(symbol, count);
            } else {
                debug_assert_eq!(freqs[&symbol], count);
            }

            cumul.insert(symbol, accum);

            if 0xFFFF_FFFF - accum <= count.get() {
                panic!("Total frequency overflow");
            }

            accum += count.get();
        }

        if freqs.len() == 1 {
            // Fix degenerate case.
            let (_, count) = freqs.iter_mut().next().unwrap();
            *count = const { NonZero::new(0xFFFF_FFFF).unwrap() };
        }

        let mut map = cumul.iter().map(|(s, c)| (*c, *s)).collect::<Vec<_>>();
        map.sort_unstable_by_key(|(c, _)| *c);

        Context { freqs, cumul, map }
    }

    /// Build context from sorted frequencies.
    ///
    /// Frequency sequence is sorted by symbols.
    pub fn from_sorted_frequencies(
        freqs_sorted: impl IntoIterator<Item = (T, NonZero<u32>)>,
    ) -> Self
    where
        T: Eq + Hash + Copy,
    {
        Self::build(freqs_sorted, None)
    }

    /// Build context from frequency map.
    pub fn from_frequency_map(freqs: HashMap<T, NonZero<u32>>) -> Self
    where
        T: Ord + Hash + Copy,
    {
        Self::from_frequency_map_ord_by(freqs, |a, b| a.cmp(&b))
    }

    /// Build context from frequency map.
    /// Uses provided order for symbols.
    pub fn from_frequency_map_ord_by(
        freqs: HashMap<T, NonZero<u32>>,
        ord: impl Fn(T, T) -> cmp::Ordering,
    ) -> Self
    where
        T: Eq + Hash + Copy,
    {
        let mut freqs_sorted = freqs.iter().map(|(s, c)| (*s, *c)).collect::<Vec<_>>();
        freqs_sorted.sort_unstable_by(|(a, _), (b, _)| ord(*a, *b));
        Self::build(freqs_sorted, Some(freqs))
    }

    /// Build context from input data.
    pub fn from_input(input: impl IntoIterator<Item = T>) -> Self
    where
        T: Ord + Hash + Copy,
    {
        Self::from_input_ord_by(input, |a, b| a.cmp(&b))
    }

    /// Build context from input data.
    ///
    /// Uses provided order for symbols.
    pub fn from_input_ord_by(
        input: impl IntoIterator<Item = T>,
        ord: impl Fn(T, T) -> cmp::Ordering,
    ) -> Self
    where
        T: Eq + Hash + Copy,
    {
        let mut total = 0u32;
        let mut freqs = HashMap::<T, NonZero<u32>>::new();

        input.into_iter().for_each(|symbol| {
            total += 1;
            match freqs.entry(symbol) {
                Entry::Occupied(mut entry) => {
                    let freq = entry.get_mut();
                    *freq = freq.checked_add(1).expect("frequency overflow");
                }
                Entry::Vacant(entry) => {
                    entry.insert(const { NonZero::new(1).unwrap() });
                }
            }
        });

        assert!(total > 0, "Context cannot be built from empty input");

        let ratio = 0xFFFF_FFFF / total;

        for freq in freqs.values_mut() {
            // ratio > 0
            // freq < total
            // total * ratio < 0xFFFF_FFFF
            // freq * ratio < 0xFFFF_FFFF
            *freq = NonZero::new(freq.get() * ratio).unwrap();
        }

        Self::from_frequency_map_ord_by(freqs, ord)
    }

    /// Returns an iterator over all `(symbol, frequency)` pairs in this context.
    pub fn freqs(&self) -> impl ExactSizeIterator<Item = (T, NonZero<u32>)> + '_
    where
        T: Copy,
    {
        self.freqs.iter().map(|(s, c)| (*s, *c))
    }

    fn bit_len(&self) -> usize
    where
        T: Copy + Default + Ord + Delta + VarCode,
    {
        let mut bit_len = 0;

        let mut freqs = self.freqs().collect::<Vec<_>>();
        freqs.sort_unstable_by_key(|(symbol, _)| *symbol);

        {
            // Write number of symbols.
            bit_len += vle::encode_bit_len(freqs.len());

            // Encode frequencies.
            let mut last = T::default();
            for (symbol, count) in &freqs {
                bit_len += vle::encode_non_zero_bit_len::<u32>(*count);

                let d = symbol.delta(last);
                last = *symbol;
                bit_len += d.var_bit_len();
            }
        }

        bit_len
    }

    /// Write minimal header for Ans encoding.
    pub fn write(&self, writer: &mut WriteBits<impl io::Write>) -> io::Result<()>
    where
        T: Copy + Default + Ord + Delta + VarCode,
    {
        let mut freqs = self.freqs().collect::<Vec<_>>();
        freqs.sort_unstable_by_key(|(symbol, _)| *symbol);

        {
            // Write number of symbols.
            vle::encode(freqs.len(), writer)?;

            // Encode frequencies.
            let mut last = T::default();
            for (symbol, count) in &freqs {
                vle::encode_non_zero::<u32, _>(*count, writer)?;

                let d = symbol.delta(last);
                last = *symbol;
                d.var_write(writer)?;
            }
        }

        Ok(())
    }

    /// Write minimal header for Ans encoding.
    /// Uses provided order and delta function for symbols.
    /// Encodes deltas between symbols, which can be more efficient.
    pub fn write_with_delta<U>(
        &self,
        writer: &mut WriteBits<impl io::Write>,
        init: T,
        ord: impl Fn(T, T) -> cmp::Ordering,
        delta: impl Fn(T, T) -> U,
    ) -> io::Result<()>
    where
        T: Copy,
        U: VarCode,
    {
        let mut freqs = self.freqs().collect::<Vec<_>>();
        freqs.sort_unstable_by(|(a, _), (b, _)| ord(*a, *b));

        {
            // Write number of symbols.
            vle::encode(freqs.len(), writer)?;

            // Encode frequencies.
            let mut last = init;
            for (symbol, count) in &freqs {
                vle::encode_non_zero::<u32, _>(*count, writer)?;

                let d = delta(last, *symbol);
                last = *symbol;
                d.var_write(writer)?;
            }
        }

        Ok(())
    }

    /// Read minimal header for ANS encoding.
    ///
    /// Should be used if context was written without delta encoding.
    pub fn read(reader: &mut ReadBits<impl io::Read>) -> io::Result<Self>
    where
        T: Copy + Default + Eq + Hash + Delta + VarCode,
    {
        // Read number of symbols.
        let len = { vle::decode::<usize, _>(reader)? };

        // Read symbols and build frequency map.
        let mut freqs_sorted = Vec::<(T, NonZero<u32>)>::with_capacity(len);

        let mut last = T::default();
        for _ in 0..len {
            let count = vle::decode_non_zero::<u32, _>(reader)?;

            let d = T::var_read(reader)?;
            let symbol = T::from_delta(last, d);
            last = symbol;
            freqs_sorted.push((symbol, count));
        }

        Ok(Self::from_sorted_frequencies(freqs_sorted))
    }

    /// Read minimal header for ANS encoding.
    ///
    /// Uses provided order and delta function for symbols.
    /// Decodes deltas between symbols, which can be more efficient.
    ///
    /// Should be used if context was written with delta encoding.
    /// Using same order and delta function as for writing is required for correct decoding.
    pub fn read_with_delta<U>(
        reader: &mut ReadBits<impl io::Read>,
        init: T,
        from_delta: impl Fn(T, U) -> T,
    ) -> io::Result<Self>
    where
        T: Copy + Eq + Hash,
        U: VarCode,
    {
        // Read number of symbols.
        let len = { vle::decode::<usize, _>(reader)? };

        // Read symbols and build frequency map.
        let mut freqs_sorted = Vec::<(T, NonZero<u32>)>::with_capacity(len);

        let mut last = init;
        for _ in 0..len {
            let count = vle::decode_non_zero::<u32, _>(reader)?;

            let d = U::var_read(reader)?;
            let symbol = from_delta(last, d);
            last = symbol;
            freqs_sorted.push((symbol, count));
        }

        Ok(Self::from_sorted_frequencies(freqs_sorted))
    }
}

impl<T> VarCode for Context<T>
where
    T: Copy + Default + Ord + Hash + Delta + VarCode,
{
    fn var_bit_len(&self) -> usize {
        Context::bit_len(self)
    }

    fn var_write(&self, writer: &mut WriteBits<impl io::Write>) -> io::Result<()> {
        Context::write(self, writer)
    }

    fn var_read(read: &mut ReadBits<impl io::Read>) -> io::Result<Self> {
        Context::read(read)
    }
}

/// ANS encoder that compresses symbols into a stream of `u32` tokens.
///
/// Symbols are fed one at a time via [`encode`](Self::encode). Each call may
/// return a `u32` token that must be stored. After all symbols have been
/// encoded, call [`finish`](Self::finish) to retrieve the final state pair.
///
/// The token stream must be provided to the [`Decoder`] in **reverse** order.
pub struct Encoder<'a, T> {
    state: u64,
    ctx: &'a Context<T>,
}

impl<'a, T> Encoder<'a, T>
where
    T: Eq + Hash + Copy,
{
    /// Prepare Ans encoder.
    pub fn new(ctx: &'a Context<T>) -> Self {
        Encoder {
            state: 0xFFFF_FFFF,
            ctx,
        }
    }

    /// Encodes a single symbol, returning a `u32` token if state bits need to be emitted.
    pub fn encode(&mut self, symbol: T) -> Option<u32> {
        let freq = self.ctx.freqs[&symbol];
        let cumul = self.ctx.cumul[&symbol];

        let mut emit = None;

        fn make_new_state(state: u64, freq: NonZero<u32>, cumul: u32) -> u64 {
            let freq = NonZero::from(freq);
            let cumul = u64::from(cumul);

            (state / freq) * 0x1_0000_0000 + state % freq + cumul
        }

        if self.state >= u64::from(freq.get()) * 0x1_0000_0000 {
            let lo_state = self.state & 0xFFFF_FFFF;
            let hi_state = self.state >> 32;

            emit = Some(lo_state as u32);

            let new_state = make_new_state(hi_state, freq, cumul);

            debug_assert!(new_state >= 0x1_0000_0000);
            self.state = new_state;
        } else {
            let new_state = make_new_state(self.state, freq, cumul);

            debug_assert!(new_state >= 0x1_0000_0000);
            self.state = new_state;
        }

        emit
    }

    /// Returns the current internal encoder state.
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Consumes the encoder and returns the two-word final state that the
    /// decoder needs to begin decoding.
    pub fn finish(self) -> [u32; 2] {
        debug_assert!(self.state >= 0x0000_0000_8000_0000);
        debug_assert!(self.state < 0xFFFF_FFE0_0000_0000);

        let hi_state = self.state >> 32;
        let lo_state = self.state & 0xFFFF_FFFF;

        [lo_state as u32, hi_state as u32]
    }
}

/// Error type for ANS decoding failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeError {
    /// Signals that decoder did not end in final state,
    /// which may mean that compressed data is corrupted.
    Incomplete,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Incomplete => write!(f, "Decoding did not finish in final state"),
        }
    }
}

impl Error for DecodeError {}

/// ANS decoder that reconstructs symbols from a token stream.
///
/// Tokens must be provided in the reverse of the order they were emitted by
/// the [`Encoder`]. After all symbols have been decoded, call
/// [`finish`](Self::finish) to verify the decoder returned to its initial state.
pub struct Decoder<'a, T> {
    state: u64,
    ctx: &'a Context<T>,
}

impl<'a, T> Decoder<'a, T>
where
    T: Eq + Hash + Copy,
{
    /// Creates a new decoder bound to the given context.
    pub fn new(ctx: &'a Context<T>) -> Self {
        Self { state: 0, ctx }
    }

    /// Decodes one symbol, pulling additional tokens from `tokens` as needed.
    ///
    /// Returns `None` when the token stream is exhausted.
    pub fn decode(&mut self, mut tokens: impl Iterator<Item = u32>) -> Option<T> {
        if self.state <= 0xFFFF_FFFF {
            let token = tokens.next()?;
            self.state = (self.state << 32) | u64::from(token);
        }

        if unlikely(self.state <= 0xFFFF_FFFF) {
            // Only occurs on first symbol.
            let token = tokens.next()?;
            self.state = (self.state << 32) | u64::from(token);
        }

        let c = (self.state & 0xFFFF_FFFF) as u32;

        let index = match self.ctx.map.binary_search_by_key(&c, |(start, _)| *start) {
            Ok(index) => index,
            Err(next) => next - 1,
        };

        let symbol = self.ctx.map[index].1;

        let new_state = (self.state >> 32) * u64::from(self.ctx.freqs[&symbol].get())
            + (self.state & 0xFFFF_FFFF)
            - u64::from(self.ctx.cumul[&symbol]);

        self.state = new_state;

        Some(symbol)
    }

    /// Decodes all remaining symbols from `tokens` and appends them to `extend`.
    pub fn decode_all(
        &mut self,
        mut tokens: impl Iterator<Item = u32>,
        extend: &mut impl Extend<T>,
    ) {
        if self.state <= 0xFFFF_FFFF {
            let Some(token) = tokens.next() else {
                return;
            };
            self.state = (self.state << 32) | u64::from(token);
        }

        loop {
            if self.state <= 0xFFFF_FFFF {
                let Some(token) = tokens.next() else {
                    return;
                };
                self.state = (self.state << 32) | u64::from(token);
            }

            let c = (self.state & 0xFFFF_FFFF) as u32;

            let index = match self.ctx.map.binary_search_by_key(&c, |(start, _)| *start) {
                Ok(index) => index,
                Err(next) => next - 1,
            };

            let symbol = self.ctx.map[index].1;

            let new_state = (self.state >> 32) * u64::from(self.ctx.freqs[&symbol].get())
                + (self.state & 0xFFFF_FFFF) as u64
                - u64::from(self.ctx.cumul[&symbol]);

            self.state = new_state;

            extend.extend(Some(symbol));
        }
    }

    /// Verifies that the decoder ended in the expected final state.
    ///
    /// Returns `Err(DecodeError::Incomplete)` if the internal state does not
    /// match the encoder's initial state, which typically indicates corrupted data.
    pub fn finish(&self) -> Result<(), DecodeError> {
        if self.state == 0xFFFF_FFFF {
            Ok(())
        } else {
            Err(DecodeError::Incomplete)
        }
    }
}

#[test]
fn test_u16() {
    use crate::bits::{read_bits_scope, write_bits_scope};

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

    let ctx = Context::from_input(data);

    let mut encoder = Encoder::<u16>::new(&ctx);
    let mut compressed = Vec::new();

    for symbol in data {
        compressed.extend(encoder.encode(symbol));
    }

    compressed.extend(encoder.finish());

    let mut ctx_buf = Vec::new();

    write_bits_scope(&mut ctx_buf, |writer| ctx.write(writer)).unwrap();

    let ctx2 = read_bits_scope(&ctx_buf[..], |reader| Context::read(reader)).unwrap();

    let mut decoder = Decoder::new(&ctx2);

    let mut decoded = Vec::new();
    let mut iter = compressed.iter().copied().rev();

    while decoded.len() < data.len() {
        match decoder.decode(&mut iter) {
            None => panic!(),
            Some(symbol) => {
                decoded.push(symbol);
            }
        }
    }

    decoded.reverse();

    assert_eq!(data[..], decoded[..]);

    assert!(decoder.finish().is_ok());

    decoded.clear();
    let mut decoder = Decoder::new(&ctx2);
    decoder.decode_all(compressed.into_iter().rev(), &mut decoded);
    decoded.reverse();

    assert_eq!(data[..], decoded[..]);

    assert!(decoder.finish().is_ok());
}

#[inline(always)]
fn unlikely(condition: bool) -> bool {
    if condition {
        cold_path();
        true
    } else {
        false
    }
}

#[cold]
fn cold_path() {}
