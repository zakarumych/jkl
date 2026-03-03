use std::{cmp, error::Error, fmt, hash::Hash, io};

use hashbrown::HashMap;

use crate::{
    bits::{ReadBits, WriteBits},
    encode::VarCode,
    math::Delta,
    vle,
};

#[derive(Clone, Debug)]
pub struct Context<T> {
    freqs: HashMap<T, u64>,
    cumul: HashMap<T, u64>,
    map: Vec<(u64, T)>,
    total: u64,
}

impl<T> Context<T> {
    // Build context from sorted frequencies and frequency map.
    ///
    /// Frequency sequence is sorted by symbols.
    /// Frequency map is provided if it was already built.
    fn build(
        freqs_sorted: impl IntoIterator<Item = (T, u64)>,
        freqs: Option<HashMap<T, u64>>,
    ) -> Self
    where
        T: Eq + Hash + Copy,
    {
        let mut cumul = HashMap::<T, u64>::new();
        let mut accum = 0u64;
        let build_freqs = freqs.is_none();
        let mut freqs = freqs.unwrap_or_default();

        for (symbol, count) in freqs_sorted {
            if build_freqs {
                freqs.insert(symbol, count);
            } else {
                debug_assert_eq!(freqs[&symbol], count);
            }

            cumul.insert(symbol, accum as u64);
            accum += u64::from(count);
        }

        if freqs.len() == 1 {
            // Fix degenerate case.
            let (_, count) = freqs.iter_mut().next().unwrap();
            *count = 1;
            accum = 2;
        }

        if accum >= 0x8000_0000 {
            panic!("Too many symbols");
        }

        assert!(accum < 0x8000_0000);

        let mut map = cumul.iter().map(|(s, c)| (*c, *s)).collect::<Vec<_>>();
        map.sort_unstable_by_key(|(c, _)| *c);

        Context {
            freqs,
            cumul,
            map,
            total: accum,
        }
    }

    /// Build context from sorted frequencies.
    ///
    /// Frequency sequence is sorted by symbols.
    pub fn from_sorted_frequencies(freqs_sorted: impl IntoIterator<Item = (T, u64)>) -> Self
    where
        T: Eq + Hash + Copy,
    {
        Self::build(freqs_sorted, None)
    }

    /// Build context from frequency map.
    pub fn from_frequency_map(freqs: HashMap<T, u64>) -> Self
    where
        T: Ord + Hash + Copy,
    {
        Self::from_frequency_map_ord_by(freqs, |a, b| a.cmp(&b))
    }

    /// Build context from frequency map.
    /// Uses provided order for symbols.
    pub fn from_frequency_map_ord_by(
        freqs: HashMap<T, u64>,
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
        T: Ord + Hash + Copy,
    {
        let mut freqs = HashMap::<T, u64>::new();

        input.into_iter().for_each(|symbol| {
            *freqs.entry(symbol).or_default() += 1;
        });

        Self::from_frequency_map_ord_by(freqs, ord)
    }

    pub fn freqs(&self) -> impl ExactSizeIterator<Item = (T, u64)> + '_
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
                bit_len += vle::encode_bit_len(*count);

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
                vle::encode(*count, writer)?;

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
                vle::encode(*count, writer)?;

                let d = delta(last, *symbol);
                last = *symbol;
                d.var_write(writer)?;
            }
        }

        Ok(())
    }

    /// Write minimal header for Ans encoding.
    ///
    /// Should be used if context was writtent without delta encoding.
    pub fn read(reader: &mut ReadBits<impl io::Read>) -> io::Result<Self>
    where
        T: Copy + Default + Eq + Hash + Delta + VarCode,
    {
        // Read number of symbols.
        let len = { vle::decode::<usize, _>(reader)? };

        // Read symbols and build frequency map.
        let mut freqs_sorted = Vec::<(T, u64)>::new();
        freqs_sorted.reserve(len);

        let mut last = T::default();
        for _ in 0..len {
            let count = vle::decode::<u64, _>(reader)?;

            let d = T::var_read(reader)?;
            let symbol = T::from_delta(last, d);
            last = symbol;
            freqs_sorted.push((symbol, count));
        }

        Ok(Self::from_sorted_frequencies(freqs_sorted))
    }

    /// Write minimal header for Ans encoding.
    ///
    /// Uses provided order and delta function for symbols.
    /// Decodes deltas between symbols, which can be more efficient.
    ///
    /// Should be used if context was writtent with delta encoding.
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
        let mut freqs_sorted = Vec::<(T, u64)>::new();
        freqs_sorted.reserve(len);

        let mut last = init;
        for _ in 0..len {
            let count = vle::decode::<u64, _>(reader)?;

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
            // Starting state value is as large as maximum ctx.total.
            // It guarantees that any symbol will grow the state.
            // After first symbol state is guaranteed to be at least 0x8000_0000
            // This works in favor of decoder,
            // where it knows that state below 0x8000_0000 means that more bits need to be read.
            // And if there are none, decoding finished.
            state: 0x7FFF_FFFF,
            ctx,
        }
    }

    pub fn encode(&mut self, symbol: T) -> Option<u32> {
        let freq = self.ctx.freqs[&symbol];
        let cumul = self.ctx.cumul[&symbol];

        let mut emit = None;

        // Checks that new state is guaranteed to be flushable.
        // And guards against overflow in new state calculation.
        if 0x8000_0000_0000_0000 / self.ctx.total <= self.state / freq {
            let lo_state = self.state & 0xFFFF_FFFF;
            let hi_state = self.state >> 32;

            emit = Some(lo_state as u32);

            // Calculate new state.
            let new_state = (hi_state / freq) * self.ctx.total + hi_state % freq + cumul;

            debug_assert!(new_state >= 0x8000_0000);
            debug_assert!(new_state < 0x8000_0000_0000_0000);

            self.state = new_state;
        } else {
            // Calculate new state.
            let mut new_state = (self.state / freq) * self.ctx.total + self.state % freq + cumul;

            debug_assert!(freq < self.ctx.total);
            debug_assert!(new_state > self.state);

            // If it's too big, emit some bits and reduce it.
            if new_state >= 0x8000_0000_0000_0000 {
                let lo_state = self.state & 0xFFFF_FFFF;
                let hi_state = self.state >> 32;

                emit = Some(lo_state as u32);

                new_state = (hi_state / freq) * self.ctx.total + hi_state % freq + cumul;

                debug_assert!(new_state >= 0x8000_0000);
                debug_assert!(new_state < 0x8000_0000_0000_0000);
            }

            self.state = new_state;
        }

        emit
    }

    pub fn state(&self) -> u64 {
        self.state
    }

    pub fn finish(self) -> [u32; 2] {
        debug_assert!(self.state >= 0x0000_0000_8000_0000);
        debug_assert!(self.state < 0x8000_0000_0000_0000);

        let hi_state = self.state >> 32;
        let lo_state = self.state & 0xFFFF_FFFF;

        [lo_state as u32, hi_state as u32]
    }
}

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

pub struct Decoder<'a, T> {
    state: u64,
    ctx: &'a Context<T>,
}

impl<'a, T> Decoder<'a, T>
where
    T: Eq + Hash + Copy,
{
    pub fn new(ctx: &'a Context<T>) -> Self {
        Self { state: 0, ctx }
    }

    pub fn decode(&mut self, mut tokens: impl Iterator<Item = u32>) -> Option<T> {
        if self.state < 0x8000_0000 {
            let token = tokens.next()?;
            self.state = (self.state << 32) | u64::from(token);
        }

        if unlikely(self.state < 0x8000_0000) {
            // Only occurs on first symbol.
            let token = tokens.next()?;
            self.state = (self.state << 32) | u64::from(token);
        }

        let c = self.state % self.ctx.total;

        let index = match self.ctx.map.binary_search_by_key(&c, |(start, _)| *start) {
            Ok(index) => index,
            Err(next) => next - 1,
        };

        let symbol = self.ctx.map[index].1;

        let new_state = (self.state / self.ctx.total) * self.ctx.freqs[&symbol]
            + (self.state % self.ctx.total)
            - self.ctx.cumul[&symbol];

        self.state = new_state;

        Some(symbol)
    }

    pub fn decode_all(
        &mut self,
        mut tokens: impl Iterator<Item = u32>,
        extend: &mut impl Extend<T>,
    ) {
        if self.state < 0x8000_0000 {
            let Some(token) = tokens.next() else {
                return;
            };
            self.state = (self.state << 32) | u64::from(token);
        }

        loop {
            if self.state < 0x8000_0000 {
                let Some(token) = tokens.next() else {
                    return;
                };
                self.state = (self.state << 32) | u64::from(token);
            }

            let c = self.state % self.ctx.total;

            let index = match self.ctx.map.binary_search_by_key(&c, |(start, _)| *start) {
                Ok(index) => index,
                Err(next) => next - 1,
            };

            let symbol = self.ctx.map[index].1;

            let new_state = (self.state / self.ctx.total) * self.ctx.freqs[&symbol]
                + (self.state % self.ctx.total)
                - self.ctx.cumul[&symbol];

            self.state = new_state;

            extend.extend(Some(symbol));
        }
    }

    // If decoding finished correctly, state should be exactly 0x7FFF_FFFF, which is the initial state of encoder.
    pub fn finish(&self) -> Result<(), DecodeError> {
        if self.state == 0x7FFF_FFFF {
            Ok(())
        } else {
            Err(DecodeError::Incomplete)
        }
    }
}

#[test]
fn test_u16() {
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
    let mut bit_writer = WriteBits::new(&mut ctx_buf);
    ctx.write(&mut bit_writer).unwrap();
    bit_writer.finish().unwrap();
    let mut bit_reader = ReadBits::new(&ctx_buf[..]);
    let ctx2 = Context::read(&mut bit_reader).unwrap();

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
