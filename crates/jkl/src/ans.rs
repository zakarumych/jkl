//! Asymmetric Numeral Systems (ANS) entropy coder.
//!
//! Provides a [`Context`] that holds per-symbol frequency tables, an [`Encoder`]
//! that compresses symbols into a stream of `u32` tokens, and a [`Decoder`] that
//! reconstructs the original symbols from that token stream in reverse order.

use std::{cmp, error::Error, fmt, hash::Hash, io, num::NonZero};

use hashbrown::HashMap;

use crate::{
    bits::{ReadBits, WriteBits},
    encode::VarCode,
    math::Delta,
    vle,
};

/// Symbol table entry containing the symbol, its frequency, and its cumulative frequency.
#[derive(Clone, Copy, Debug)]
pub struct Entry<T> {
    pub symbol: T,
    pub freq: NonZero<u32>,
    pub cumul: u32,
}

/// Shared frequency table used by [`Encoder`] and [`Decoder`].
///
/// A `Context` stores per-symbol frequencies, cumulative frequencies, and a
/// lookup table for fast symbol resolution during decoding. It is built once
/// from input data or a frequency map and then shared by reference with
/// encoders and decoders.
#[derive(Clone, Debug)]
pub struct Context<T> {
    /// Symbol table, sorted by symbol.
    /// Cumulative frequency is sum of frequencies of all preceding symbols.
    /// Consequently the array is sorted by cumulative frequency as well.
    table: Vec<Entry<T>>,

    /// Same, indexed by symbol.
    /// Used to speed-up encoding.
    map: Option<HashMap<T, (NonZero<u32>, u32)>>,
}

fn normalize_table<T>(table: &mut [Entry<T>], total: u32)
where
    T: Copy,
{
    if table.len() == 1 {
        // Fix degenerate case.
        let symbol = &mut table[0];
        *symbol = Entry {
            symbol: symbol.symbol,
            freq: const { NonZero::new(0xFFFF_FFFF).unwrap() },
            cumul: 0,
        };

        return;
    }

    if total.is_power_of_two() {
        let mut largest_freq = 1;
        let mut largest_freq_index = table.len();

        for (i, entry) in table.iter_mut().enumerate() {
            let new_freq = (u64::from(entry.freq.get()) << 32) / u64::from(total);

            // If new_freq is 1, total must be at least 0x8000_0001,
            // but it's a power of two and the largest power of two possible for u32 is 0x8000_0000.
            debug_assert!(new_freq > 1);

            // Given that freq is non-zero and less than total
            // following must hold for normalized frequency.
            // new_freq is larger than 0, because it was multiplied by 0x1_0000_0000 before division which is larger than total.
            // new_freq is not greater than 0xFFFF_FFFF, because freq is less than total, so result is less than 0x1_0000_0000, i.e. not greater than 0xFFFF_FFFF.
            debug_assert!(new_freq > 0);
            debug_assert!(new_freq <= 0xFFFF_FFFF);

            if new_freq > largest_freq {
                largest_freq = new_freq;
                largest_freq_index = i;
            }

            *entry = Entry {
                symbol: entry.symbol,
                freq: NonZero::new(new_freq as u32).unwrap(),
                cumul: 0,
            };
        }

        // All new_freq are at least 2,
        // so it will be assigned at the very first entry
        // and possibly reassigned later.
        debug_assert!(largest_freq_index < table.len());

        let mut accum = 0;
        for (i, entry) in table.iter_mut().enumerate() {
            if i == largest_freq_index {
                // All freq's are > 1 when total is power of two.
                debug_assert!(entry.freq.get() > 1);

                entry.freq = NonZero::new(entry.freq.get() - 1).unwrap();
            }

            // all freqs could add up to 0x1_0000_0000, but we reduced largest one by 1,
            // so now they add up to 0xFFFF_FFFF
            debug_assert!(0xFFFF_FFFF - accum >= entry.freq.get());

            entry.cumul = accum;
            accum += entry.freq.get();
        }
    } else {
        let mut accum = 0;
        for entry in table.iter_mut() {
            let new_freq = (u64::from(entry.freq.get()) << 32) / u64::from(total);

            // Given that freq is non-zero and less than total
            // following must hold for normalized frequency.
            // new_freq is larger than 0, because it was multiplied by 0x1_0000_0000 before division which is larger than total.
            // new_freq is not greater than 0xFFFF_FFFF, because freq is less than total, so result is less than 0x1_0000_0000, i.e. not greater than 0xFFFF_FFFF.
            debug_assert!(new_freq > 0);
            debug_assert!(new_freq <= 0xFFFF_FFFF);

            debug_assert!(0xFFFF_FFFF - accum >= entry.freq.get());

            *entry = Entry {
                symbol: entry.symbol,
                freq: NonZero::new(new_freq as u32).unwrap(),
                cumul: accum,
            };

            accum += entry.freq.get();
        }
    }
}

impl<T> Context<T> {
    /// Builds a context from sorted (symbol, frequency) pairs.
    pub fn from_frequencies(freqs_sorted: impl IntoIterator<Item = (T, NonZero<u32>)>) -> Self
    where
        T: Eq + Hash + Copy,
    {
        let mut table = Vec::new();
        let mut accum = 0u32;

        for (symbol, count) in freqs_sorted {
            table.push(Entry {
                symbol,
                freq: count,
                cumul: accum,
            });

            if 0xFFFF_FFFF - accum < count.get() {
                panic!("Total frequency overflow");
            }

            accum += count.get();
        }

        normalize_table(&mut table, accum);

        Context { table, map: None }
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
        let mut accum = 0u32;

        let mut map = HashMap::<T, (NonZero<u32>, u32)>::new();

        input.into_iter().for_each(|symbol| {
            if accum == 0xFFFF_FFFF {
                panic!("Total frequency overflow");
            }

            accum += 1;
            match map.entry(symbol) {
                hashbrown::hash_map::Entry::Occupied(mut entry) => {
                    let freq = &mut entry.get_mut().0;
                    *freq = freq.checked_add(1).expect("frequency overflow");
                }
                hashbrown::hash_map::Entry::Vacant(entry) => {
                    entry.insert(const { (NonZero::new(1).unwrap(), 0) });
                }
            }
        });

        let mut table = Vec::with_capacity(map.len());
        for (symbol, (freq, _)) in map {
            table.push(Entry {
                symbol,
                freq,
                cumul: 0,
            });
        }
        table.sort_by(|a, b| ord(a.symbol, b.symbol));

        let mut accum = 0;
        for entry in &mut table {
            debug_assert!(0xFFFF_FFFF - accum >= entry.freq.get());

            entry.cumul = accum;
            accum += entry.freq.get();
        }

        normalize_table(&mut table, accum);

        Context { table, map: None }
    }

    fn bit_len(&self) -> usize
    where
        T: Copy + Default + Ord + Delta + VarCode,
    {
        let mut bit_len = 0;

        {
            // Write number of symbols.
            bit_len += vle::encode_bit_len(self.table.len());

            // Encode frequencies.
            let mut last = T::default();
            for entry in &self.table {
                bit_len += vle::encode_non_zero_bit_len::<u32>(entry.freq);

                let d = entry.symbol.delta(last);
                last = entry.symbol;
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
        {
            // Write number of symbols.
            vle::encode(self.table.len(), writer)?;

            // Encode frequencies.
            let mut last = T::default();
            for entry in &self.table {
                vle::encode_non_zero::<u32, _>(entry.freq, writer)?;

                let d = entry.symbol.delta(last);
                last = entry.symbol;
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
        delta: impl Fn(T, T) -> U,
    ) -> io::Result<()>
    where
        T: Copy,
        U: VarCode,
    {
        {
            // Write number of symbols.
            vle::encode(self.table.len(), writer)?;

            // Encode frequencies.
            let mut last = init;
            for entry in &self.table {
                vle::encode_non_zero::<u32, _>(entry.freq, writer)?;

                let d = delta(last, entry.symbol);
                last = entry.symbol;
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
        T: Ord + Hash + Copy + Default + Delta + VarCode,
    {
        // Read number of symbols.
        let len = { vle::decode::<usize, _>(reader)? };

        // Read symbols and build frequency map.
        let mut table = Vec::<Entry<T>>::with_capacity(len);

        let mut last = T::default();
        let mut accum = 0;
        for _ in 0..len {
            let count = vle::decode_non_zero::<u32, _>(reader)?;

            let d = T::var_read(reader)?;
            let symbol = T::from_delta(last, d);
            last = symbol;
            table.push(Entry {
                symbol,
                freq: count,
                cumul: accum,
            });

            if 0xFFFF_FFFF - accum < count.get() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Total frequency overflow",
                ));
            }

            accum += count.get();
        }

        Ok(Context { table, map: None })
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
        T: Eq + Hash + Copy,
        U: VarCode,
    {
        // Read number of symbols.
        let len = { vle::decode::<usize, _>(reader)? };

        // Read symbols and build frequency map.
        let mut table = Vec::<Entry<T>>::with_capacity(len);

        let mut last = init;
        let mut accum = 0;
        for _ in 0..len {
            let count = vle::decode_non_zero::<u32, _>(reader)?;

            let d = U::var_read(reader)?;
            let symbol = from_delta(last, d);
            last = symbol;
            table.push(Entry {
                symbol,
                freq: count,
                cumul: accum,
            });

            if 0xFFFF_FFFF - accum < count.get() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Total frequency overflow",
                ));
            }

            accum += count.get();
        }

        Ok(Context { table, map: None })
    }

    pub fn table(&self) -> &[Entry<T>] {
        &self.table
    }

    fn find_by_bucket(&self, c: u32) -> (T, NonZero<u32>, u32)
    where
        T: Copy,
    {
        assert!(!self.table.is_empty());
        debug_assert_eq!(self.table[0].cumul, 0);

        let index = match self.table.binary_search_by_key(&c, |e| e.cumul) {
            Ok(index) => index,
            Err(next) => next - 1,
        };
        let e = &self.table[index];
        (e.symbol, e.freq, e.cumul)
    }

    fn find_by_symbol(&self, symbol: T) -> Option<(NonZero<u32>, u32)>
    where
        T: Ord + Hash + Copy,
    {
        if let Some(map) = &self.map {
            return map.get(&symbol).copied();
        }

        let index = self
            .table
            .binary_search_by_key(&symbol, |e| e.symbol)
            .ok()?;

        let e = &self.table[index];
        Some((e.freq, e.cumul))
    }

    /// Initialize by-symbol look-up hashmap.
    /// If not initialized, by-symbol look-up will use binary search over table.
    /// by-symbol lookup is only used during encoding, so calling this function is wasteful when context will be only used for decoding.
    pub fn build_encoder_index(&mut self)
    where
        T: Hash + Eq + Copy,
    {
        // Build a map from symbol to cumulative frequency for faster lookup during encoding.
        let mut map = HashMap::new();

        for entry in &self.table {
            map.insert(entry.symbol, (entry.freq, entry.cumul));
        }
        self.map = Some(map);
    }

    pub fn print_frequencies(&self) {
        for (i, entry) in self.table.iter().enumerate() {
            println!("{}: {}", i, entry.freq);
        }
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
    T: Ord + Hash + Copy,
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
        let (freq, cumul) = self
            .ctx
            .find_by_symbol(symbol)
            .expect("Symbol not found in context");

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

        let (symbol, freq, cumul) = self.ctx.find_by_bucket(c);

        let new_state = (self.state >> 32) * u64::from(freq.get()) + (self.state & 0xFFFF_FFFF)
            - u64::from(cumul);

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

            let (symbol, freq, cumul) = self.ctx.find_by_bucket(c);

            let new_state = (self.state >> 32) * u64::from(freq.get())
                + (self.state & 0xFFFF_FFFF) as u64
                - u64::from(cumul);

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
