//! This module contains implementation of LZ78 based compression for Jackal format.
//!
//! This variation is adapted to allow efficient decoding on GPUs.
//!
//! Uses 8 and 16-bit alphabets.
//! Indices in code grow by 8 bits, with first sequence added to the dictionary, the index becomes 8 bits,
//! when 256th entry is added, the index becomes 16bits.
//!

use std::io::{Read, Write};

use crate::bytes::LeBytes;

pub(crate) trait Element {
    // This parameter is related to the maximum size of the input data.
    // With maximum superblock size of 256x256 there's 65536 entries per block field.
    //
    // Now consider worst case scenario where dictionary grows as fast as possible.
    // This requires all symbols from the alphabet appear once,
    // then all possible pairs, then all possible triples, etc.
    const MAX_INPUT_SIZE: usize;
}

impl Element for u8 {
    // With 8-bit fields and maximum input size 256x256x16,
    // There can be all symbols, pairs, triples, quadruples, quintuples and some sextuples.
    // This means that the dictionary upper bound size is 2^8 * 6;
    const MAX_INPUT_SIZE: usize = 6 * 1 << 8;
}

impl Element for u16 {
    // With 16-bit fields and maximum input size 256x256x8,
    // there can be all symbols, pairs, triples and some quadruples at the end.
    // This means that the dictionary upper bound size is 2^16 * 4;
    const MAX_INPUT_SIZE: usize = 4 * 1 << 16;
}

impl Element for u32 {
    // With 32-bit fields and maximum input size 256x256x4,
    // there can be all some symbols.
    const MAX_INPUT_SIZE: usize = 4 * 1 << 16;
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Entry<T> {
    prefix: u32,
    element: T,
}

#[derive(Clone, Copy, Debug)]
enum Index {
    Zero,
    Short(u8),
    Long(u16),
    VeryLong { high: u8, low: u16 },
}

impl Index {
    fn write_to(&self, output: &mut impl Write) -> std::io::Result<()> {
        match self {
            Index::Zero => {}
            Index::Short(index) => {
                output.write_all(&index.to_le_bytes())?;
            }
            Index::Long(index) => {
                output.write_all(&index.to_le_bytes())?;
            }
            Index::VeryLong { high, low } => {
                let mut bytes = [0; 3];
                bytes[0..2].copy_from_slice(&low.to_le_bytes());
                bytes[2] = *high;
                output.write_all(&bytes)?;
            }
        }

        Ok(())
    }
}

pub struct Encoder<T> {
    entires: Vec<Entry<T>>,
    prefix: u32,
}

impl<T> Encoder<T> {
    pub fn new() -> Self {
        Encoder {
            entires: Vec::new(),
            prefix: 0,
        }
    }
}

impl<T> Encoder<T>
where
    T: Copy + Eq + LeBytes,
{
    fn lookup(&self, entry: Entry<T>) -> Option<u32> {
        for i in entry.prefix..self.entires.len() as u32 {
            if self.entires[i as usize] == entry {
                return Some(i + 1);
            }
        }

        None
    }

    fn insert(&mut self, entry: Entry<T>) {
        self.entires.push(entry);
    }

    fn index_code(&self, index: u32) -> Index {
        if self.entires.len() == 0 {
            Index::Zero
        } else if self.entires.len() < 256 {
            Index::Short(index as u8)
        } else if self.entires.len() < 65536 {
            Index::Long(index as u16)
        } else {
            Index::VeryLong {
                high: (index >> 16) as u8,
                low: index as u16,
            }
        }
    }

    pub fn encode(&mut self, input: T, output: &mut impl Write) -> std::io::Result<()> {
        let entry = Entry {
            prefix: self.prefix,
            element: input,
        };
        let index = self.lookup(entry);

        match index {
            None => {
                let index = self.index_code(self.prefix);
                index.write_to(output)?;
                <T as LeBytes>::write_to(&input, output)?;
                self.insert(entry);
                self.prefix = 0;
            }
            Some(index) => {
                self.prefix = index;
            }
        }

        Ok(())
    }

    pub fn finish(self, output: &mut impl Write) -> std::io::Result<()> {
        let index = self.index_code(self.prefix);
        index.write_to(output)?;
        Ok(())
    }
}

pub struct Decoder<T> {
    scratch: Vec<T>,
    substrings: Vec<(u32, u32)>,
    output_range: (u32, u32),
}

impl<T> Decoder<T> {
    pub fn new() -> Self {
        Decoder {
            scratch: Vec::new(),
            substrings: Vec::new(),
            output_range: (0, 0),
        }
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

impl<T> Decoder<T>
where
    T: Copy + Eq + LeBytes,
{
    fn read_index(&self, read: &mut impl Read) -> std::io::Result<u32> {
        if self.substrings.len() == 0 {
            Ok(0)
        } else if self.substrings.len() < 256 {
            let mut bytes = [0u8];
            read.read_exact(&mut bytes)?;
            Ok(u8::from_le_bytes(bytes) as u32)
        } else if self.substrings.len() < 65536 {
            let mut bytes = [0; 2];
            read.read_exact(&mut bytes)?;
            Ok(u16::from_le_bytes(bytes) as u32)
        } else {
            let mut bytes = [0; 4];
            read.read_exact(&mut bytes[0..3])?;
            Ok(u32::from_le_bytes(bytes))
        }
    }

    fn decode_next_range<'a>(
        &'a mut self,
        input: &mut impl Read,
    ) -> Result<(u32, u32), DecodeError> {
        let index = self.read_index(input)?;

        // Add the new substring to the cache.
        let (prefix_start, prefix_end) = if index > 0 {
            if index as usize > self.substrings.len() {
                return Err(DecodeError::InvalidIndex);
            }
            self.substrings[(index - 1) as usize]
        } else {
            (0, 0)
        };

        debug_assert!(prefix_end >= prefix_start);
        let prefix_len = prefix_end - prefix_start;

        let element = match <T as LeBytes>::read_from(input) {
            Ok(element) => element,
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof && prefix_len != 0 => {
                // Last piece.
                return Ok((prefix_start, prefix_end));
            }
            Err(err) => return Err(DecodeError::Io(err)),
        };

        let end = if self.substrings.is_empty() {
            0
        } else {
            let (_start, end) = *self.substrings.last().unwrap();
            end
        };

        debug_assert_eq!(end as usize, self.scratch.len());

        self.scratch
            .extend_from_within(prefix_start as usize..prefix_end as usize);
        self.scratch.push(element);

        let new_start = end;
        let new_end = new_start + prefix_len + 1;

        self.substrings.push((new_start, new_end));

        Ok((new_start, new_end))
    }

    pub fn decode_next_slice<'a>(
        &'a mut self,
        input: &mut impl Read,
    ) -> Result<&'a [T], DecodeError> {
        if self.output_range.0 >= self.output_range.1 {
            self.output_range = self.decode_next_range(input)?;
        }

        let slice = &self.scratch[self.output_range.0 as usize..self.output_range.1 as usize];
        self.output_range.0 = self.output_range.1;
        Ok(slice)
    }

    pub fn decode_next(&mut self, input: &mut impl Read) -> Result<&T, DecodeError> {
        if self.output_range.0 >= self.output_range.1 {
            self.output_range = self.decode_next_range(input)?;
        }

        let element = &self.scratch[self.output_range.0 as usize];
        self.output_range.0 += 1;
        Ok(element)
    }
}

#[test]
fn test_u16() {
    let mut encoder = Encoder::<u16>::new();
    let mut compressed = Vec::new();

    let data = [
        1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 1, 2,
    ];

    for byte in data {
        encoder.encode(byte, &mut compressed).unwrap();
    }

    encoder.finish(&mut compressed).unwrap();

    let mut decoder = Decoder::<u16>::new();

    let mut input = &compressed[..];
    let mut decoded = 0;

    while decoded < data.len() {
        let slice = decoder.decode_next_slice(&mut input).unwrap();
        assert_eq!(data[decoded..][..slice.len()], *slice);
        decoded += slice.len();
    }
}
