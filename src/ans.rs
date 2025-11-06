use std::{collections::BTreeSet, hash::Hash};

use hashbrown::HashMap;

#[derive(Clone, Debug)]
pub struct Context<T> {
    freqs: HashMap<T, u64>,
    cumul: HashMap<T, u64>,
    map: Vec<(u64, T)>,
    total: u64,
}

impl<T> Context<T>
where
    T: Ord + Hash + Copy,
{
    pub fn new(input: impl IntoIterator<Item = T>) -> Self {
        let mut freqs = HashMap::<T, u64>::new();
        let mut set = BTreeSet::new();

        for symbol in input {
            *freqs.entry(symbol).or_default() += 1;
            set.insert(symbol);
        }

        let mut cumul = HashMap::<T, u64>::new();
        let mut accum = 0u64;

        for &symbol in &set {
            cumul.insert(symbol, accum as u64);
            accum += u64::from(freqs[&symbol]);
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

    pub fn map(&self) -> &[(u64, T)] {
        &self.map
    }
}

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

    let ctx = Context::new(data);

    let mut encoder = Encoder::<u16>::new(&ctx);
    let mut compressed = Vec::new();

    for symbol in data {
        if let Some(token) = encoder.encode(symbol) {
            compressed.push(token);
        }
    }

    let mut decoder = Decoder::new(encoder.state(), &ctx);

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
