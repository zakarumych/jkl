//! This module contains implementation of LZW based compression for Jackal format.
//!
//! This variation is adapted to allow efficient decoding on GPUs.
//!
//! Uses 8 and 16-bit alphabets.
//! Indices in code grow by 1 bit.
//!

use std::io::{Read, Write};

use crate::bits::{ReadBits, WriteBits};

pub(crate) trait Element: Copy + Eq {
    // This parameter is related to the maximum size of the input data.
    // With maximum superblock size of 256x256 there's 65536 entries per block field.
    //
    // Now consider worst case scenario where dictionary grows as fast as possible.
    // This requires all symbols from the alphabet appear once,
    // then all possible pairs, then all possible triples, etc.
    const MAX_INPUT_SIZE: usize;

    const MAX_VALUE: u32;

    fn into_u32(self) -> u32;

    fn from_u32(value: u32) -> Self;
}

impl Element for u8 {
    // With 8-bit fields and maximum input size 256x256x16,
    // There can be all symbols, pairs, triples, quadruples, quintuples and some sextuples.
    // This means that the dictionary upper bound size is 2^8 * 6;
    const MAX_INPUT_SIZE: usize = 6 * 1 << 8;

    const MAX_VALUE: u32 = u8::MAX as u32;

    #[inline(always)]
    fn into_u32(self) -> u32 {
        self as u32
    }

    #[inline(always)]
    fn from_u32(value: u32) -> Self {
        debug_assert!(value <= u8::MAX as u32);
        value as u8
    }
}

impl Element for u16 {
    // With 16-bit fields and maximum input size 256x256x8,
    // there can be all symbols, pairs, triples and some quadruples at the end.
    // This means that the dictionary upper bound size is 2^16 * 4;
    const MAX_INPUT_SIZE: usize = 4 * 1 << 16;

    #[inline(always)]
    fn into_u32(self) -> u32 {
        self as u32
    }

    const MAX_VALUE: u32 = u16::MAX as u32;

    #[inline(always)]
    fn from_u32(value: u32) -> Self {
        debug_assert!(value <= u16::MAX as u32);
        value as u16
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Entry<T> {
    prefix: u32,
    element: T,
}

pub struct Encoder<T> {
    entries: Vec<Entry<T>>,
    prefix: Option<u32>,
}

impl<T> Encoder<T> {
    pub fn new() -> Self {
        Encoder {
            entries: Vec::new(),
            prefix: None,
        }
    }
}

impl<T> Encoder<T>
where
    T: Element,
{
    const T_ENTRIES: u32 = T::MAX_VALUE + 1;

    fn lookup(&self, entry: Entry<T>) -> Option<u32> {
        for i in entry.prefix.saturating_sub(Self::T_ENTRIES)..self.entries.len() as u32 {
            if self.entries[i as usize] == entry {
                return Some(i + Self::T_ENTRIES);
            }
        }

        None
    }

    fn insert(&mut self, entry: Entry<T>) {
        self.entries.push(entry);
    }

    fn write(&mut self, index: u32, writer: &mut WriteBits<impl Write>) -> std::io::Result<()> {
        let bits = (self.entries.len() + Self::T_ENTRIES as usize)
            .next_power_of_two()
            .trailing_zeros();

        debug_assert!(1 << bits > index);

        let index_bytes = index.to_le_bytes();
        writer.write_bits(&index_bytes, 0, bits as usize)?;

        // eprintln!("{} - {}", bits, index);

        Ok(())
    }

    pub fn encode(&mut self, input: T, writer: &mut WriteBits<impl Write>) -> std::io::Result<()> {
        // eprintln!("{}", input.into_u32());

        let Some(prefix) = self.prefix else {
            self.prefix = Some(input.into_u32());
            return Ok(());
        };

        let entry = Entry {
            prefix,
            element: input,
        };
        let index = self.lookup(entry);

        match index {
            None => {
                self.write(prefix, writer)?;
                self.insert(entry);
                self.prefix = Some(input.into_u32());
            }
            Some(index) => {
                self.prefix = Some(index);
            }
        }

        Ok(())
    }

    pub fn finish(mut self, writer: &mut WriteBits<impl Write>) -> std::io::Result<()> {
        let Some(prefix) = self.prefix else {
            return Ok(());
        };
        self.write(prefix, writer)?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum DecodeError {
    Io(std::io::Error),
    InvalidIndex,
}

impl From<std::io::Error> for DecodeError {
    #[inline(always)]
    fn from(err: std::io::Error) -> Self {
        DecodeError::Io(err)
    }
}

enum Output<T> {
    Element(T),
    Range(u32, u32),
}

pub struct Decoder<T> {
    scratch: Vec<T>,
    entries: Vec<(u32, u32)>,
    output: Output<T>,
    last: Option<Output<T>>,
}

impl<T> Decoder<T> {
    pub fn new() -> Self {
        Decoder {
            scratch: Vec::new(),
            entries: Vec::new(),
            output: Output::Range(0, 0),
            last: None,
        }
    }

    pub fn finish(&self) {
        match self.output {
            Output::Range(start, end) if start == end => {}
            _ => {
                panic!("Decoder output was not consumed.");
            }
        }
    }
}

impl<T> Decoder<T>
where
    T: Element,
{
    const T_ENTRIES: u32 = T::MAX_VALUE + 1;

    fn read_index(&mut self, reader: &mut ReadBits<impl Read>) -> std::io::Result<u32> {
        let bits = (self.entries.len() as u32 + Self::T_ENTRIES + self.last.is_some() as u32)
            .next_power_of_two()
            .trailing_zeros();

        let mut index_bytes = [0; 4];
        reader.read_bits(&mut index_bytes, 0, bits as usize)?;

        let index = u32::from_le_bytes(index_bytes);

        // eprintln!("{} - {}", bits, index);

        Ok(index)
    }

    fn decode_next_range<'a>(
        &'a mut self,
        reader: &mut ReadBits<impl Read>,
    ) -> Result<(), DecodeError> {
        let index = self.read_index(reader)?;

        if index < Self::T_ENTRIES {
            let element = T::from_u32(index);
            // One element.

            match self.last {
                None => {}
                Some(Output::Element(last_element)) => {
                    let new_start = self.scratch.len() as u32;

                    self.scratch.push(last_element);
                    self.scratch.push(element);

                    let new_end = self.scratch.len() as u32;
                    self.entries.push((new_start, new_end));
                }
                Some(Output::Range(last_start, last_end)) => {
                    let new_start = self.scratch.len() as u32;

                    self.scratch
                        .extend_from_within(last_start as usize..last_end as usize);
                    self.scratch.push(element);

                    let new_end = self.scratch.len() as u32;
                    self.entries.push((new_start, new_end));
                }
            }

            self.last = Some(Output::Element(element));
            self.output = Output::Element(element);
        } else if index - Self::T_ENTRIES < self.entries.len() as u32 {
            let (start, end) = self.entries[(index - Self::T_ENTRIES) as usize];
            let element = self.scratch[start as usize];

            match self.last {
                None => {}
                Some(Output::Element(last_element)) => {
                    let new_start = self.scratch.len() as u32;

                    self.scratch.push(last_element);
                    self.scratch.push(element);

                    let new_end = self.scratch.len() as u32;
                    self.entries.push((new_start, new_end));
                }
                Some(Output::Range(last_start, last_end)) => {
                    let new_start = self.scratch.len() as u32;

                    self.scratch
                        .extend_from_within(last_start as usize..last_end as usize);
                    self.scratch.push(element);

                    let new_end = self.scratch.len() as u32;
                    self.entries.push((new_start, new_end));
                }
            }

            self.last = Some(Output::Range(start, end));
            self.output = Output::Range(start, end);
        } else {
            match self.last {
                None => return Err(DecodeError::InvalidIndex),
                Some(Output::Element(last_element)) => {
                    let new_start = self.scratch.len() as u32;

                    self.scratch.push(last_element);
                    self.scratch.push(last_element);

                    let new_end = self.scratch.len() as u32;
                    self.entries.push((new_start, new_end));

                    self.last = Some(Output::Range(new_start, new_end));
                    self.output = Output::Range(new_start, new_end);
                }
                Some(Output::Range(last_start, last_end)) => {
                    let new_start = self.scratch.len() as u32;

                    self.scratch
                        .extend_from_within(last_start as usize..last_end as usize);
                    self.scratch.push(self.scratch[last_start as usize]);

                    let new_end = self.scratch.len() as u32;
                    self.entries.push((new_start, new_end));

                    self.last = Some(Output::Range(new_start, new_end));
                    self.output = Output::Range(new_start, new_end);
                }
            }
        }

        Ok(())
    }

    pub fn decode_next(&mut self, reader: &mut ReadBits<impl Read>) -> Result<T, DecodeError> {
        match self.output {
            Output::Element(element) => {
                self.output = Output::Range(0, 0);
                Ok(element)
            }
            Output::Range(start, end) if start < end => {
                let element = self.scratch[start as usize];
                self.output = Output::Range(start + 1, end);
                Ok(element)
            }
            _ => {
                self.decode_next_range(reader)?;
                self.decode_next(reader)
            }
        }
    }
}

#[test]
fn test_u16() {
    let mut encoder = Encoder::<u16>::new();
    let mut compressed = Vec::new();

    let data = [
        1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2,
    ];

    let mut writer = WriteBits::new(&mut compressed);

    for byte in data {
        encoder.encode(byte, &mut writer).unwrap();
    }

    encoder.finish(&mut writer).unwrap();
    writer.finish().unwrap();

    let mut decoder = Decoder::<u16>::new();

    let mut input = ReadBits::new(&compressed[..]);
    let mut decoded = 0;

    while decoded < data.len() {
        let word = decoder.decode_next(&mut input).unwrap();
        assert_eq!(data[decoded], word);
        decoded += 1;
    }
}
