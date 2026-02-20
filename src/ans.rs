use std::{hash::Hash, io};

use hashbrown::{HashMap, HashSet};

use crate::{
    bits::{ReadBits, WriteBits},
    vle, FixedCode,
};

#[derive(Clone, Debug)]
pub struct Context<T> {
    freqs: HashMap<T, u64>,
    cumul: HashMap<T, u64>,
    map: Vec<(u64, T)>,
    total: u64,
}

impl<T> Context<T>
where
    T: Eq + Hash + Copy,
{
    // Build context from sorted frequencies and frequency map.
    fn build(
        freqs_sorted: impl IntoIterator<Item = (T, u64)>,
        freqs: Option<HashMap<T, u64>>,
    ) -> Self {
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

        if accum > (1 << 31) {
            panic!("Too many symbols");
        }

        assert!(accum <= (1 << 31));

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
    pub fn from_sorted_frequencies(freqs_sorted: impl IntoIterator<Item = (T, u64)>) -> Self {
        Self::build(freqs_sorted, None)
    }

    /// Build context from frequency map.
    pub fn from_frequency_map(freqs: HashMap<T, u64>) -> Self {
        let mut freqs_sorted = freqs.iter().map(|(s, c)| (*s, *c)).collect::<Vec<_>>();
        freqs_sorted.sort_unstable_by_key(|(_, count)| *count);
        Self::build(freqs_sorted, Some(freqs))
    }

    /// Build context from input data.
    pub fn from_input(input: impl IntoIterator<Item = T>) -> Self {
        let mut freqs = HashMap::<T, u64>::new();

        input.into_iter().for_each(|symbol| {
            *freqs.entry(symbol).or_default() += 1;
        });

        Self::from_frequency_map(freqs)
    }

    pub fn freqs(&self) -> impl Iterator<Item = (T, u64)> + '_ {
        self.freqs.iter().map(|(s, c)| (*s, *c))
    }

    /// Write minimal header for Ans encoding.
    pub fn write(&self, mut writer: impl std::io::Write) -> std::io::Result<()>
    where
        T: FixedCode,
    {
        let mut freqs = self.freqs().collect::<Vec<_>>();
        freqs.sort_unstable_by_key(|(_, count)| *count);

        // Delta-code frequencies.
        let mut last = 0;
        for (_, slot) in freqs.iter_mut() {
            let count = *slot;
            let delta = count - last;
            *slot = delta;
            last = count;
        }

        {
            // Write number of symbols.
            let mut bit_writer = WriteBits::new(&mut writer);
            vle::encode(freqs.len(), &mut bit_writer)?;

            // Encode frequency deltas.
            for (_, delta) in &freqs {
                vle::encode(*delta, &mut bit_writer)?;
            }
            bit_writer.finish()?;
        }

        // Write symbols.
        for (symbol, _) in &freqs {
            let mut bytes = T::Array::default();
            symbol.encode(&mut bytes);
            writer.write_all(bytes.as_ref())?;
        }

        Ok(())
    }

    /// Write minimal header for Ans encoding.
    pub fn read(mut reader: impl std::io::Read) -> std::io::Result<Self>
    where
        T: FixedCode,
    {
        let mut bit_reader = ReadBits::new(&mut reader);

        // Read number of symbols.
        let count = { vle::decode::<usize, _>(&mut bit_reader)? };

        let deltas = {
            // Read frequency deltas.
            (0..count)
                .map(|_| vle::decode::<u64, _>(&mut bit_reader))
                .collect::<Result<Vec<u64>, _>>()?
        };

        // Read symbols and build frequency map.
        let mut freqs = Vec::<(T, u64)>::new();

        let mut last = 0;
        for delta in deltas {
            let count = last + delta;
            last = count;

            let mut bytes = T::Array::default();
            reader.read_exact(bytes.as_mut())?;
            let symbol = T::decode(&bytes)
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;

            freqs.push((symbol, count));
        }

        Ok(Self::from_sorted_frequencies(freqs))
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
            state: 1 << 31,
            ctx,
        }
    }

    pub fn encode(&mut self, symbol: T) -> Option<u32> {
        let mut emit = None;

        let freq = self.ctx.freqs[&symbol];

        if (1 << 63) / self.ctx.total < self.state / freq {
            emit = Some((self.state & 0xFFFF_FFFF) as u32);
            self.state = self.state >> 32;
        }

        let mut new_state =
            (self.state / freq) * self.ctx.total + self.state % freq + self.ctx.cumul[&symbol];

        if new_state >= (1 << 63) {
            debug_assert!(emit.is_none());
            emit = Some((self.state & 0xFFFF_FFFF) as u32);
            self.state = self.state >> 32;

            new_state =
                (self.state / freq) * self.ctx.total + self.state % freq + self.ctx.cumul[&symbol];
        }

        self.state = new_state;

        emit
    }

    pub fn state(&self) -> u64 {
        self.state
    }
}

pub struct Decoder<'a, T> {
    state: u64,
    ctx: &'a Context<T>,
}

impl<'a, T> Decoder<'a, T>
where
    T: Ord + Hash + Copy,
{
    pub fn new(state: u64, ctx: &'a Context<T>) -> Self {
        Self { state, ctx }
    }

    pub fn decode(&mut self, mut next_token: impl FnMut() -> Option<u32>) -> Option<T> {
        if self.state < (1 << 31) {
            return None;
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

        if self.state < (1 << 31) {
            if let Some(token) = next_token() {
                self.state = (self.state << 32) | u64::from(token);
            }
        }

        Some(symbol)
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
        if let Some(token) = encoder.encode(symbol) {
            compressed.push(token);
        }
    }

    let mut ctx_buf = Vec::new();
    ctx.write(&mut ctx_buf).unwrap();
    let ctx2 = Context::read(&ctx_buf[..]).unwrap();

    let mut decoder = Decoder::new(encoder.state(), &ctx2);

    let mut decoded = Vec::new();
    let mut iter = compressed.iter().copied().rev();

    while decoded.len() < data.len() {
        match decoder.decode(|| iter.next()) {
            None => break,
            Some(symbol) => {
                decoded.push(symbol);
            }
        }
    }

    decoded.reverse();

    assert_eq!(data[..], decoded[..]);
}
