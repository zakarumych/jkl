//! Types and functions to work with individual bits.

use std::io::{Read, Write};

/// Wrapper around writer to write bits.
pub struct WriteBits<W> {
    writer: W,
    buffer: u128,
    buffer_len: u8,
}

impl<W> WriteBits<W> {
    pub fn new(write: W) -> Self {
        WriteBits {
            writer: write,
            buffer: 0,
            buffer_len: 0,
        }
    }
}

impl<W> WriteBits<W>
where
    W: Write,
{
    /// Write bits from the slice.
    /// `bit_offset` specifies the bit offset in the buffer.
    /// `bit_len` specifies number of bits to write.
    ///
    /// Returns number of bits written.
    /// It would be the `bit_len`.
    /// Unless writer is exhausted.
    pub fn write_bits(
        &mut self,
        buffer: &[u8],
        bit_offset: usize,
        bit_len: usize,
    ) -> std::io::Result<usize> {
        let mut buffer = buffer;
        let mut bit_offset = bit_offset;
        let mut bit_len = bit_len;

        let mut total_bits_written = 0;

        loop {
            let (new_buffer, new_bit_offset, new_bit_len) =
                self.copy_from_buffer(buffer, bit_offset, bit_len);

            total_bits_written += bit_len - new_bit_len;
            buffer = new_buffer;
            bit_offset = new_bit_offset;
            bit_len = new_bit_len;

            if bit_len == 0 {
                return Ok(total_bits_written);
            }

            if !self.flush()? {
                return Ok(total_bits_written);
            }
        }
    }

    fn flush(&mut self) -> std::io::Result<bool> {
        if self.buffer_len > 8 {
            let write_bytes = self.buffer_len / 8;

            let bytes_written = loop {
                let r = self
                    .writer
                    .write(&self.buffer.to_le_bytes()[..write_bytes as usize]);

                match r {
                    Ok(0) => return Ok(false),
                    Ok(n) => break n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
            };

            self.buffer_len -= (bytes_written * 8) as u8;
            if self.buffer_len > 0 {
                self.buffer >>= bytes_written * 8;
            } else {
                self.buffer = 0;
            }
        }

        Ok(true)
    }

    pub fn finish(&mut self) -> std::io::Result<usize> {
        if self.buffer_len > 0 {
            let write_bytes = (self.buffer_len + 7) / 8;
            self.writer
                .write_all(&self.buffer.to_le_bytes()[..write_bytes as usize])?;

            let written_bits = write_bytes * 8;
            self.buffer_len = 0;

            Ok(written_bits as usize)
        } else {
            Ok(0)
        }
    }

    fn copy_from_buffer<'a>(
        &mut self,
        mut buffer: &'a [u8],
        mut bit_offset: usize,
        mut bit_len: usize,
    ) -> (&'a [u8], usize, usize) {
        if bit_offset >= 8 {
            let byte_offset = bit_offset / 8;
            buffer = &buffer[byte_offset..];
            bit_offset %= 8;
        }

        let mut buffer_free = 128 - self.buffer_len;

        if buffer_free > 0 && bit_len > 0 && bit_offset > 0 {
            let copy_len = (buffer_free as usize).min(8 - bit_offset).min(bit_len);

            let mut copy_bits = buffer[0];

            copy_bits >>= bit_offset;
            copy_bits &= (1 << copy_len) - 1;

            self.buffer |= (copy_bits as u128) << self.buffer_len;
            self.buffer_len += copy_len as u8;
            buffer_free -= copy_len as u8;

            if bit_offset + copy_len >= 8 {
                debug_assert_eq!(bit_offset + copy_len, 8);
                bit_offset = 0;
                bit_len -= copy_len;
                buffer = &buffer[1..];
            } else {
                bit_offset += copy_len;
                bit_len -= copy_len;

                return (buffer, bit_offset, bit_len);
            }
        }

        if buffer_free > 0 && bit_len > 0 {
            debug_assert_eq!(bit_offset, 0);
            let copy_len = (buffer_free as usize).min(bit_len);

            let mut copy_bytes = [0; 16];

            let copy_bytes_len = (copy_len + 7) / 8;
            copy_bytes[..copy_bytes_len].copy_from_slice(&buffer[..copy_bytes_len]);

            let mut copy_bits = u128::from_le_bytes(copy_bytes);
            if copy_len < 128 {
                copy_bits &= (1u128 << copy_len) - 1;
            }

            self.buffer |= copy_bits << self.buffer_len;
            self.buffer_len += copy_len as u8;

            bit_offset += copy_len as usize;
            bit_len -= copy_len;
        }

        (buffer, bit_offset, bit_len)
    }
}

impl<W> Write for WriteBits<W>
where
    W: Write,
{
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_bits(buf, 0, buf.len() * 8).map(|n| n / 8)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flush()?;
        self.writer.flush()
    }
}

/// Wrapper around reader to read bits.
pub struct ReadBits<R> {
    reader: R,
    buffer: u128,
    buffer_len: u8,
}

impl<R> ReadBits<R> {
    pub fn new(reader: R) -> Self {
        ReadBits {
            reader,
            buffer: 0,
            buffer_len: 0,
        }
    }
}

impl<R> ReadBits<R>
where
    R: Read,
{
    /// Read bits into the buffer.
    /// `bit_offset` specifies the bit offset in the buffer to read into.
    /// All bits before [`bit_offset`] will be preserved.
    /// `bit_len` specifies number of bits to read.
    ///
    /// Returns number of bits read.
    /// It would be the `bit_len`.
    /// Unless reader is exhausted.
    ///
    /// # Panics
    ///
    /// The function will panic if `buffer` doesn't fit bits in range `bit_offset..bit_offset+bit_len`.
    /// This means that `buffer.len()` must be equal or greater than `(bit_offset + bit_len + 7) / 8`.
    ///
    /// Function can also panic if internal reader panics on read.
    pub fn read_bits(
        &mut self,
        buffer: &mut [u8],
        bit_offset: usize,
        bit_len: usize,
    ) -> std::io::Result<usize> {
        assert!(buffer.len() >= (bit_offset + bit_len + 7) / 8);

        if bit_len == 0 {
            return Ok(0);
        }

        let mut total_bits_read = 0;
        let mut buffer = buffer;
        let mut bit_offset = bit_offset;
        let mut bit_len = bit_len;

        loop {
            let (new_buffer, new_bit_offset, new_bit_len) =
                self.copy_from_buffer(buffer, bit_offset, bit_len);

            total_bits_read += bit_len - new_bit_len;
            buffer = new_buffer;
            bit_offset = new_bit_offset;
            bit_len = new_bit_len;

            if bit_len == 0 {
                return Ok(total_bits_read);
            }

            debug_assert_eq!(self.buffer_len, 0);
            if !self.fill_buffer(bit_len)? {
                return Ok(total_bits_read);
            }
        }
    }

    fn fill_buffer(&mut self, bit_len: usize) -> std::io::Result<bool> {
        debug_assert_eq!(self.buffer_len, 0);

        let mut buffer = [0; 16];
        let bytes_read = loop {
            let r = self.reader.read(&mut buffer[..((bit_len + 7) / 8).min(16)]);

            match r {
                Ok(0) => return Ok(false),
                Ok(n) => break n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        };

        self.buffer = u128::from_le_bytes(buffer);
        self.buffer_len = bytes_read as u8 * 8;
        Ok(true)
    }

    fn copy_from_buffer<'a>(
        &mut self,
        mut buffer: &'a mut [u8],
        mut bit_offset: usize,
        mut bit_len: usize,
    ) -> (&'a mut [u8], usize, usize) {
        if bit_offset >= 8 {
            let byte_offset = bit_offset / 8;
            buffer = &mut buffer[byte_offset..];
            bit_offset %= 8;
        }

        if self.buffer_len > 0 && bit_len > 0 && bit_offset > 0 {
            let mut copy_bits = self.buffer.to_le_bytes()[0];

            let copy_len = (self.buffer_len as usize).min(8 - bit_offset).min(bit_len);

            copy_bits &= (1 << copy_len) - 1;
            copy_bits <<= bit_offset;
            buffer[0] |= copy_bits;

            self.buffer >>= copy_len;
            self.buffer_len -= copy_len as u8;

            if bit_offset + copy_len >= 8 {
                debug_assert_eq!(bit_offset + copy_len, 8);
                bit_offset = 0;
                bit_len -= copy_len;
                buffer = &mut buffer[1..];
            } else {
                bit_offset += copy_len;
                bit_len -= copy_len;

                return (buffer, bit_offset, bit_len);
            }
        }

        if self.buffer_len > 0 && bit_len > 0 {
            debug_assert_eq!(bit_offset, 0);
            let copy_len = (self.buffer_len as usize).min(bit_len);

            let mut copy_bits = self.buffer;

            if copy_len < 128 {
                copy_bits &= (1u128 << copy_len) - 1;
            }

            let copy_bytes = copy_bits.to_le_bytes();

            let copy_bytes_len = (copy_len + 7) / 8;
            buffer[..copy_bytes_len].copy_from_slice(&copy_bytes[..copy_bytes_len]);

            self.buffer >>= copy_len;
            self.buffer_len -= copy_len as u8;

            bit_offset += copy_len as usize;
            bit_len -= copy_len;
        }

        (buffer, bit_offset, bit_len)
    }
}

impl<R> Read for ReadBits<R>
where
    R: Read,
{
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.read_bits(buf, 0, buf.len() * 8).map(|n| n / 8)
    }
}

#[test]
fn test_writer() {
    let writes = [
        (&[1u8, 2, 3, 4][..], 27),
        (&[5], 3),
        (&[6, 7], 16),
        (&[8, 9, 10], 22),
        (&[11, 12, 13, 14], 30),
    ];

    let mut buffer = Vec::new();

    let mut write = WriteBits::new(&mut buffer);

    for (data, bit_len) in writes.iter() {
        write.write_bits(data, 0, *bit_len).unwrap();
    }

    write.finish().unwrap();

    assert_eq!(
        buffer[..],
        [0x01, 0x02, 0x03, 0xAC, 0xC1, 0x01, 0x42, 0x82, 0xB2, 0xC0, 0xD0, 0xE0, 0]
    );
}

#[test]
fn test_reader() {
    let reads = [
        (&[1u8, 2, 3, 4][..], 27),
        (&[5], 3),
        (&[6, 7], 16),
        (&[8, 9, 10], 22),
        (&[11, 12, 13, 14], 30),
    ];

    let buffer = vec![
        0x01, 0x02, 0x03, 0xAC, 0xC1, 0x01, 0x42, 0x82, 0xB2, 0xC0, 0xD0, 0xE0, 0,
    ];

    let mut read = ReadBits::new(&buffer[..]);

    for &(data, bit_len) in reads.iter() {
        let mut buffer = [0; 4];
        read.read_bits(&mut buffer, 0, bit_len).unwrap();
        assert_eq!(buffer[..data.len()], data[..]);
    }
}

#[test]
fn test_test() {
    let writes = [(0, &[0][..], &[0]), (1, &[1], &[0]), (2, &[0], &[0x1F])];

    let mut buffer = Vec::new();
    let mut write = WriteBits::new(&mut buffer);

    for (bit_len, index, value) in writes {
        write.write_bits(index, 0, bit_len).unwrap();
        write.write_bits(value, 0, 8).unwrap();
    }

    write.finish().unwrap();

    drop(write);
    let mut read = ReadBits::new(&buffer[..]);

    for (bit_len, index, value) in writes {
        let mut buffer = [0u8];
        read.read_bits(&mut buffer, 0, bit_len).unwrap();
        assert_eq!(buffer[0], index[0]);

        read.read_bits(&mut buffer, 0, 8).unwrap();
        assert_eq!(buffer[0], value[0]);
    }
}
