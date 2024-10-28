//! This module contains implementation of LZ78 based compression for Jackal format.
//!
//! This variation is adapted to allow efficient decoding on GPUs.
//!
//! Uses 8 and 16-bit alphabets.
//! Indices in code grow by 8 bits, with first sequence added to the dictionary, the index becomes 8 bits,
//! when 256th entry is added, the index becomes 16bits.
//!

use std::io::{Read, Write};

use crate::{
    bits::{ReadBits, WriteBits},
    bytes::LeBytes,
};

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

    fn write(
        &self,
        index: u32,
        input: T,
        writer: &mut WriteBits<impl Write>,
    ) -> std::io::Result<()> {
        let bits = (self.entires.len() + 1)
            .next_power_of_two()
            .trailing_zeros();

        debug_assert!(1 << bits > index);

        let index_bytes = index.to_le_bytes();
        writer.write_bits(&index_bytes, 0, bits as usize)?;
        <T as LeBytes>::write_to(&input, writer)?;

        Ok(())
    }

    fn write_last_index(
        &self,
        index: u32,
        writer: &mut WriteBits<impl Write>,
    ) -> std::io::Result<()> {
        let bits = (self.entires.len() + 1)
            .next_power_of_two()
            .trailing_zeros();

        debug_assert!(1 << bits > index);

        let index_bytes = index.to_le_bytes();
        writer.write_bits(&index_bytes, 0, bits as usize)?;

        Ok(())
    }

    pub fn encode(&mut self, input: T, writer: &mut WriteBits<impl Write>) -> std::io::Result<()> {
        let entry = Entry {
            prefix: self.prefix,
            element: input,
        };
        let index = self.lookup(entry);

        match index {
            None => {
                self.write(self.prefix, input, writer)?;
                self.insert(entry);
                self.prefix = 0;
            }
            Some(index) => {
                self.prefix = index;
            }
        }

        Ok(())
    }

    pub fn finish(self, writer: &mut WriteBits<impl Write>) -> std::io::Result<()> {
        self.write_last_index(self.prefix, writer)?;
        Ok(())
    }
}

pub struct Decoder<T> {
    scratch: Vec<T>,
    entires: Vec<(u32, u32)>,
    output: (u32, u32),
}

impl<T> Decoder<T> {
    pub fn new() -> Self {
        Decoder {
            scratch: Vec::new(),
            entires: Vec::new(),
            output: (0, 0),
        }
    }

    pub fn finish(&self) {
        if self.output.0 != self.output.1 {
            panic!("Decoder output was not consumed.");
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
    fn read(&self, reader: &mut ReadBits<impl Read>) -> std::io::Result<(u32, Option<T>)> {
        let bits = (self.entires.len() + 1)
            .next_power_of_two()
            .trailing_zeros();

        let mut index_bytes = [0; 4];
        reader.read_bits(&mut index_bytes, 0, bits as usize)?;
        let index = u32::from_le_bytes(index_bytes);

        let element = match <T as LeBytes>::read_from(reader) {
            Ok(element) => Some(element),
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                // Last piece.
                None
            }
            Err(err) => return Err(err),
        };

        Ok((index, element))
    }

    fn decode_next_range<'a>(
        &'a mut self,
        reader: &mut ReadBits<impl Read>,
    ) -> Result<(u32, u32), DecodeError> {
        let (index, element) = self.read(reader)?;

        // Add the new substring to the cache.
        let (prefix_start, prefix_end) = if index > 0 {
            if index as usize > self.entires.len() {
                return Err(DecodeError::InvalidIndex);
            }
            self.entires[(index - 1) as usize]
        } else {
            (0, 0)
        };

        debug_assert!(prefix_end >= prefix_start);
        let prefix_len = prefix_end - prefix_start;

        let element = match element {
            Some(element) => element,
            None => {
                // Last piece.
                return Ok((prefix_start, prefix_end));
            }
        };

        let end = if self.entires.is_empty() {
            0
        } else {
            let (_start, end) = *self.entires.last().unwrap();
            end
        };

        debug_assert_eq!(end as usize, self.scratch.len());

        self.scratch
            .extend_from_within(prefix_start as usize..prefix_end as usize);
        self.scratch.push(element);

        let new_start = end;
        let new_end = new_start + prefix_len + 1;

        self.entires.push((new_start, new_end));

        Ok((new_start, new_end))
    }

    pub fn decode_next_slice<'a>(
        &'a mut self,
        input: &mut ReadBits<impl Read>,
    ) -> Result<&'a [T], DecodeError> {
        if self.output.0 >= self.output.1 {
            self.output = self.decode_next_range(input)?;
        }

        let slice = &self.scratch[self.output.0 as usize..self.output.1 as usize];
        self.output.0 = self.output.1;
        Ok(slice)
    }

    pub fn decode_next(&mut self, input: &mut ReadBits<impl Read>) -> Result<T, DecodeError> {
        if self.output.0 >= self.output.1 {
            self.output = self.decode_next_range(input)?;
        }

        let element = self.scratch[self.output.0 as usize];
        self.output.0 += 1;
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
        let slice = decoder.decode_next_slice(&mut input).unwrap();
        assert_eq!(data[decoded..][..slice.len()], *slice);
        decoded += slice.len();
    }
}
