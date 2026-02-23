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
    fn write(&self, writer: &mut WriteBits<impl io::Write>) -> io::Result<()> {
        match self {
            Token::Reference { distance, length } => {
                writer.write_bit(false)?;
                vle::encode(*distance, writer)?;
                vle::encode(*length, writer)?;
            }
            Token::Literal { literal } => {
                writer.write_bit(true)?;
                literal.write(&mut *writer)?;
            }
        }
        Ok(())
    }

    fn read(reader: &mut ReadBits<impl io::Read>) -> io::Result<Self> {
        let token_type = reader.read_bit()?;

        match token_type {
            false => {
                let distance = vle::decode::<usize, _>(reader)?;
                let length = vle::decode::<usize, _>(reader)?;
                Ok(Token::Reference { distance, length })
            }
            true => {
                let literal = T::read(&mut *reader)?;
                Ok(Token::Literal { literal })
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
            for i in (0..usize::min(offset + 1, self.head)).rev() {
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
            for i in (self.head..usize::min(offset + 1, self.buffer.len())).rev() {
                if self.buffer[i] == *elem {
                    return Some(self.idx(i));
                }
            }
        }

        None
    }

    fn find_range(&self, offset: usize, distance: usize, length: usize) -> Option<usize>
    where
        T: PartialEq,
    {
        let offset = self.idx(offset);
        let d = self.idx(distance);
        let m = if d < self.head {
            self.head - d
        } else {
            self.buffer.len() + self.head - d
        };

        if offset < self.head {
            'a: for i in (0..usize::min(offset, self.head)).rev() {
                let n = self.head - i;
                for j in 0..length {
                    let idx1 = i + (j % n);
                    let idx2 = (d + (j % m)) % self.buffer.len();
                    if self.buffer[idx1] != self.buffer[idx2] {
                        continue 'a;
                    }
                }

                return Some(self.idx(i));
            }

            'a: for i in (self.head..self.buffer.len()).rev() {
                let n = self.buffer.len() + self.head - i;
                for j in 0..length {
                    let idx1 = (i + (j % n)) % self.buffer.len();
                    let idx2 = (d + (j % m)) % self.buffer.len();
                    if self.buffer[idx1] != self.buffer[idx2] {
                        continue 'a;
                    }
                }

                return Some(self.idx(i));
            }

            None
        } else {
            'a: for i in (self.head..usize::min(offset, self.buffer.len())).rev() {
                let n = self.buffer.len() + self.head - i;

                for j in 0..length {
                    let idx1 = (i + (j % n)) % self.buffer.len();
                    let idx2 = (d + (j % m)) % self.buffer.len();
                    if self.buffer[idx1] != self.buffer[idx2] {
                        continue 'a;
                    }
                }

                return Some(self.idx(i));
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
}

impl<T> Encoder<T>
where
    T: Copy + Eq,
{
    pub fn new(init: T, length: u32) -> Self {
        Encoder {
            window: Window::new(init, length),
            distance: 0,
            length: 0,
        }
    }

    pub fn encode(&mut self, input: T, output: &mut impl Extend<Token<T>>) {
        if self.length > 0 {
            if self.length < usize::MAX {
                // If longer match is possible to represent.

                if *self.window.get(distance_index(self.distance, self.length)) == input {
                    // Input continues the current match.
                    self.length += 1;
                    return;
                }

                // If input does not continues current match, try find another one and test if input continues it.
                let mut offset = self.distance;
                loop {
                    match self.window.find_range(offset, self.distance, self.length) {
                        None => break, // No more matches.
                        Some(pos) => {
                            if *self.window.get(distance_index(pos, self.length)) == input {
                                self.distance = pos;
                                self.length += 1;
                                return;
                            }

                            offset = pos;
                        }
                    }
                }
            }

            // Failed to continue current match, emit it and start anew.
            let emit_reference = self.length >= 2;

            if emit_reference {
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

                if !emit_reference {
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
        let emit_reference = self.length >= 2;

        if emit_reference {
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
