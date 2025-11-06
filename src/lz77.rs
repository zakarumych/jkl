use std::io::{self, Read, Write};

use crate::bytes::LeBytes;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token<T> {
    pub distance: usize,
    pub length: usize,
    pub literal: T,
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

    fn len(&self) -> usize {
        self.buffer.len()
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
    T: Copy + Eq + LeBytes,
{
    pub fn new(init: T, length: usize) -> Self {
        Encoder {
            window: Window::new(init, length),
            distance: 0,
            length: 0,
        }
    }

    pub fn encode(&mut self, input: T) -> Option<Token<T>> {
        if self.length == 0 {
            match self.window.find_elem(0, &input) {
                None => {
                    let emit = Token {
                        distance: 0,
                        length: 0,
                        literal: input,
                    };
                    self.window.push(input);
                    self.distance = 0;
                    self.length = 0;
                    return Some(emit);
                }
                Some(pos) => {
                    self.distance = pos;
                    self.length = 1;
                    return None;
                }
            }
        } else {
            if *self.window.get(distance_index(self.distance, self.length)) == input {
                self.length += 1;
                return None;
            }

            let mut offset = self.distance;
            loop {
                match self.window.find_range(offset, self.distance, self.length) {
                    None => {
                        let emit = Token {
                            distance: self.distance,
                            length: self.length,
                            literal: input,
                        };

                        for i in 0..self.length {
                            let elem = *self.window.get(distance_index(self.distance, i));
                            self.window.push(elem);
                            self.distance += 1;
                        }

                        self.window.push(input);
                        self.distance = 0;
                        self.length = 0;

                        return Some(emit);
                    }
                    Some(pos) => {
                        if *self.window.get(distance_index(pos, self.length)) == input {
                            self.distance = pos;
                            self.length += 1;
                            return None;
                        }

                        offset = pos;
                    }
                }
            }
        }
    }

    pub fn finish(self) -> Option<Token<T>> {
        if self.length != 0 {
            let elem = self
                .window
                .get(distance_index(self.distance, self.length - 1));

            Some(Token {
                distance: self.distance,
                length: self.length - 1,
                literal: *elem,
            })
        } else {
            None
        }
    }
}

struct Entry<T> {
    distance: usize,
    length: usize,
    literal: T,
}

pub struct Decoder<T> {
    window: Window<T>,
    entry: Option<Entry<T>>,
}

impl<T> Decoder<T>
where
    T: Copy + Eq + LeBytes,
{
    pub fn new(init: T, length: usize) -> Self {
        Decoder {
            window: Window::new(init, length),
            entry: None,
        }
    }

    fn decode_next(&mut self, mut next_token: impl FnMut() -> Option<Token<T>>) -> Option<T> {
        match &mut self.entry {
            None => {
                let token = next_token()?;

                if token.length == 0 {
                    self.window.push(token.literal);
                    return Some(token.literal);
                }

                let first = *self.window.get(token.distance);
                self.window.push(first);

                self.entry = Some(Entry {
                    distance: token.distance,
                    length: token.length - 1,
                    literal: token.literal,
                });

                Some(first)
            }
            Some(entry) => {
                if entry.length == 0 {
                    let literal = entry.literal;
                    self.entry = None;
                    self.window.push(literal);
                    Some(literal)
                } else {
                    let first = *self.window.get(entry.distance);
                    entry.length -= 1;
                    self.window.push(first);
                    Some(first)
                }
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
        compressed.extend(encoder.encode(byte));
    }

    compressed.extend(encoder.finish());

    let mut decoder = Decoder::<u16>::new(0, 256);

    let mut input = compressed.iter().copied();
    let mut decoded = 0;

    while decoded < data.len() {
        let elem = decoder.decode_next(|| input.next()).unwrap();
        assert_eq!(data[decoded], elem);
        decoded += 1;
    }
}
