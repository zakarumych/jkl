use std::{
    io::{self, Read, Write},
    ops::Range,
};

use brotli::enc::histogram::HistogramDistanceScratch;
use rand::rand_core::le;

use crate::bytes::LeBytes;

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

#[derive(Clone, Copy)]
struct Entry<T> {
    distance: usize,
    length: usize,
    value: T,
}

impl<T> Entry<T> {
    fn write_to(self, mut writer: impl Write) -> io::Result<()>
    where
        T: LeBytes,
    {
        let dl = (usize::min(self.distance, 15) << 4 | usize::min(self.length, 15)) as u8;
        dl.write_to(&mut writer)?;

        if self.distance >= 15 {
            (usize::min(self.distance - 15, 255) as u8).write_to(&mut writer)?;

            if self.distance >= 270 {
                (usize::min(self.distance - 270, 65535) as u16).write_to(&mut writer)?;
            }
        }

        if self.length >= 15 {
            (usize::min(self.length - 15, 255) as u8).write_to(&mut writer)?;

            if self.length >= 270 {
                (usize::min(self.length - 270, 65535) as u16).write_to(&mut writer)?;
            }
        }

        self.value.write_to(&mut writer)?;

        Ok(())
    }

    fn read_from(mut reader: impl Read) -> io::Result<Self>
    where
        T: LeBytes,
    {
        let dl = u8::read_from(&mut reader)?;

        let mut distance = (dl >> 4) as usize;
        let mut length = (dl & 0x0F) as usize;

        if distance == 15 {
            let d = u8::read_from(&mut reader)?;
            distance += d as usize;
            if d == 255 {
                let d = u16::read_from(&mut reader)?;
                distance += d as usize;
            }
        }

        if length == 15 {
            let l = u8::read_from(&mut reader)?;
            length += l as usize;
            while l == 255 {
                let l = u16::read_from(&mut reader)?;
                length += l as usize;
            }
        }

        let value = T::read_from(&mut reader)?;

        Ok(Entry {
            distance,
            length,
            value,
        })
    }
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

    pub fn encode(&mut self, input: T, writer: impl Write) -> std::io::Result<()> {
        if self.length == 0 {
            match self.window.find_elem(0, &input) {
                None => {
                    Entry {
                        distance: 0,
                        length: 0,
                        value: input,
                    }
                    .write_to(writer)?;
                    self.window.push(input);
                    self.distance = 0;
                    self.length = 0;
                    return Ok(());
                }
                Some(pos) => {
                    self.distance = pos;
                    self.length = 1;
                    return Ok(());
                }
            }
        } else {
            if *self.window.get(distance_index(self.distance, self.length)) == input {
                self.length += 1;
                return Ok(());
            }

            let mut offset = self.distance;
            loop {
                match self.window.find_range(offset, self.distance, self.length) {
                    None => {
                        Entry {
                            distance: self.distance,
                            length: self.length,
                            value: input,
                        }
                        .write_to(writer)?;

                        for i in 0..self.length {
                            let elem = *self.window.get(distance_index(self.distance, i));
                            self.window.push(elem);
                            self.distance += 1;
                        }

                        self.window.push(input);
                        self.distance = 0;
                        self.length = 0;

                        return Ok(());
                    }
                    Some(pos) => {
                        if *self.window.get(distance_index(pos, self.length)) == input {
                            self.distance = pos;
                            self.length += 1;
                            return Ok(());
                        }

                        offset = pos;
                    }
                }
            }
        }
    }

    pub fn finish(self, writer: impl Write) -> std::io::Result<()> {
        if self.length != 0 {
            let elem = self
                .window
                .get(distance_index(self.distance, self.length - 1));

            Entry {
                value: *elem,
                distance: self.distance,
                length: self.length - 1,
            }
            .write_to(writer)?;
        }

        Ok(())
    }
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

    fn decode_next(&mut self, mut reader: impl Read) -> io::Result<T> {
        match &mut self.entry {
            None => {
                let entry = Entry::read_from(&mut reader)?;

                if entry.length == 0 {
                    self.window.push(entry.value);
                    return Ok(entry.value);
                }

                let first = *self.window.get(entry.distance);
                self.window.push(first);

                self.entry = Some(Entry {
                    distance: entry.distance,
                    length: entry.length - 1,
                    value: entry.value,
                });

                Ok(first)
            }
            Some(entry) => {
                if entry.length == 0 {
                    let value = entry.value;
                    self.entry = None;
                    self.window.push(value);
                    Ok(value)
                } else {
                    let first = *self.window.get(entry.distance);
                    entry.length -= 1;
                    self.window.push(first);
                    Ok(first)
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
        1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2, 2, 1, 1, 3, 3, 1, 1, 1, 2,
    ];

    for byte in data {
        encoder.encode(byte, &mut compressed).unwrap();
    }

    encoder.finish(&mut compressed).unwrap();

    let mut decoder = Decoder::<u16>::new(0, 256);

    let mut input = &compressed[..];
    let mut decoded = 0;

    while decoded < data.len() {
        let elem = decoder.decode_next(&mut input).unwrap();
        assert_eq!(data[decoded], elem);
        decoded += 1;
    }
}
