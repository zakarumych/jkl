use std::io;

use crate::{
    bits::{ReadBits, WriteBits},
    encode::Encode,
    vle,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Token<T> {
    Reference { distance: usize, length: usize },
    Literal { literal: T },
}

impl<T> Encode for Token<T>
where
    T: Encode,
{
    fn bit_len(&self) -> usize {
        match *self {
            Token::Reference { distance, length } => {
                let length_bits = vle::encode_bit_len(length);
                let distance_bits = vle::encode_bit_len(distance);
                length_bits + distance_bits
            }
            Token::Literal { ref literal } => 1 + literal.bit_len(),
        }
    }

    fn write(&self, writer: &mut WriteBits<impl io::Write>) -> io::Result<()> {
        match *self {
            Token::Reference { distance, length } => {
                vle::encode(length, writer)?;
                vle::encode(distance, writer)?;
            }
            Token::Literal { ref literal } => {
                vle::encode(0u8, writer)?;
                literal.write(&mut *writer)?;
            }
        }
        Ok(())
    }

    fn read(reader: &mut ReadBits<impl io::Read>) -> io::Result<Self> {
        let length = vle::decode::<usize, _>(reader)?;

        match length {
            // 0 => unreachable!(), // `decode_non_zero` can't return 0.
            0 => {
                let literal = T::read(&mut *reader)?;
                Ok(Token::Literal { literal })
            }
            _ => {
                let distance = vle::decode::<usize, _>(reader)?;
                Ok(Token::Reference { distance, length })
            }
        }
    }
}

struct Window<T> {
    buffer: Vec<T>,
    head: usize,
}

impl<T> Window<T> {
    fn new(init: T, length: u32) -> Self
    where
        T: Copy,
    {
        Window {
            buffer: vec![init; length as usize],
            head: 0,
        }
    }

    fn idx(&self, index: usize) -> usize {
        (self.head + self.buffer.len() - 1 - index) % self.buffer.len()
    }

    fn push(&mut self, value: T) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.buffer.len();
    }

    fn get(&self, index: usize) -> &T {
        let idx = self.idx(index);
        &self.buffer[idx]
    }

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

fn distance_index(distance: usize, index: usize) -> usize {
    distance - (index % (distance + 1))
}

pub struct Encoder<T> {
    window: Window<T>,
    distance: usize,
    length: usize,

    // How many bits the match in form of literals would take.
    match_bit_len: usize,
}

impl<T> Encoder<T>
where
    T: Copy + Eq + Encode,
{
    pub fn new(init: T, length: u32) -> Self {
        Encoder {
            window: Window::new(init, length),
            distance: 0,
            length: 0,
            match_bit_len: 0,
        }
    }

    fn should_emit_reference(&self) -> bool {
        self.length >= 1
        // let reference = Token::<T>::Reference {
        //     distance: self.distance,
        //     length: self.length,
        // };
        // reference.bit_len() < self.match_bit_len
    }

    pub fn encode(&mut self, input: T, output: &mut impl Extend<Token<T>>) {
        if self.length > 0 {
            if self.length < usize::MAX {
                // If longer match is possible to represent.

                if *self.window.get(distance_index(self.distance, self.length)) == input {
                    // Input continues the current match.
                    self.length += 1;
                    self.match_bit_len += Token::Literal { literal: input }.bit_len();
                    return;
                }

                // If input does not continues current match, try find extension deeper in the window.
                if let Some(pos) = self
                    .window
                    .find_extension(self.distance, self.length, &input)
                {
                    // Extension match found, update match to it and continue.
                    self.distance = pos;
                    self.length += 1;
                    self.match_bit_len += Token::Literal { literal: input }.bit_len();
                    return;
                }
            }

            // Failed to continue current match, emit it and start anew.
            let should_emit_reference = self.should_emit_reference();

            if should_emit_reference {
                output.extend(Some(Token::Reference {
                    distance: self.distance,
                    length: self.length,
                }));
            }

            for i in 0..self.length {
                // Rotate window and emit literals if current match is not long enough.
                let elem = *self.window.get(distance_index(self.distance, i));
                self.window.push(elem);
                self.distance += 1;

                if !should_emit_reference {
                    output.extend(Some(Token::Literal { literal: elem }));
                }
            }

            self.distance = 0;
            self.length = 0;
        }

        match self.window.find_elem(0, &input) {
            None => {
                // Symbol is not in the window, emit it as literal and add to the window.

                let emit = Token::Literal { literal: input };
                self.window.push(input);
                self.distance = 0;
                self.length = 0;
                output.extend(Some(emit));
                return;
            }
            Some(pos) => {
                // Symbol is in the window, start new match.

                self.distance = pos;
                self.length = 1;
                return;
            }
        }
    }

    pub fn finish(mut self, output: &mut impl Extend<Token<T>>) {
        let should_emit_reference = self.should_emit_reference();

        if should_emit_reference {
            output.extend(Some(Token::Reference {
                distance: self.distance,
                length: self.length,
            }));
        } else {
            for i in 0..self.length {
                let elem = *self.window.get(distance_index(self.distance, i));
                self.distance += 1;
                output.extend(Some(Token::Literal { literal: elem }));
            }
        }
    }
}

struct Entry {
    distance: usize,
    length: usize,
}

pub struct Decoder<T> {
    window: Window<T>,
    entry: Option<Entry>,
}

impl<T> Decoder<T>
where
    T: Copy + Eq,
{
    pub fn new(init: T, length: u32) -> Self {
        Decoder {
            window: Window::new(init, length),
            entry: None,
        }
    }

    pub fn decode(&mut self, tokens: &mut impl Iterator<Item = Token<T>>) -> Option<T> {
        match &mut self.entry {
            None => {
                let token = tokens.next()?;

                match token {
                    Token::Reference { distance, length } => {
                        let first = *self.window.get(distance as usize);
                        self.window.push(first);

                        self.entry = Some(Entry {
                            distance: distance as usize,
                            length: length as usize - 1,
                        });

                        Some(first)
                    }
                    Token::Literal { literal } => {
                        self.window.push(literal);
                        return Some(literal);
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
                Some(first)
            }
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
        let elem = decoder.decode(&mut input).unwrap();
        decoded.push(elem);
    }

    assert_eq!(data[..], decoded[..]);
}
