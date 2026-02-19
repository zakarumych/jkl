use core::fmt;



// Simple variable-length encoding (VLE) for integer streams.
//
// Design goals:
// - Works for any primitive integer type via a trait (`VleInt`).
// - Very small values, especially `0`, encode to very few bits (`0` -> 1 bit).
// - Encodes a stream into a compact bitstream (bit-packed into bytes).
//
// Encoding (for an unsigned integer `v`):
// - Emit a 1-bit tag:
//   - If `v == 0`: emit single bit `0`.
//   - Else: emit bit `1` followed by Elias-gamma coding of `v`.
// - Elias-gamma coding (for `v >= 1`):
//   - Let `n = floor(log2(v))` (0..=127 for `u128`).
//   - Emit `n` zero bits.
//   - Emit `v` in binary using exactly `n + 1` bits (MSB first).
//
// Signed integers are mapped to unsigned via zigzag encoding.


/// Encoded bitstream (bit-packed into bytes) plus the exact number of valid bits.
///
/// The last byte may contain unused padding bits (zeros). `bit_len` prevents
/// the decoder from interpreting padding as extra values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    pub bit_len: usize,
}

impl Encoded {
    pub fn new(bytes: Vec<u8>, bit_len: usize) -> Self {
        Self { bytes, bit_len }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VleError {
    UnexpectedEof,
    InvalidLengthPrefix,
    ValueOutOfRange,
}

impl fmt::Display for VleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VleError::UnexpectedEof => write!(f, "unexpected end of input"),
            VleError::InvalidLengthPrefix => write!(f, "invalid length prefix"),
            VleError::ValueOutOfRange => write!(f, "decoded value does not fit target type"),
        }
    }
}

impl std::error::Error for VleError {}

/// Conversion between an integer type and the unsigned domain used by this VLE.
pub trait VleInt: Copy + Sized {
    fn to_vle_u128(self) -> u128;
    fn from_vle_u128(v: u128) -> Option<Self>;
}

macro_rules! impl_vle_unsigned {
    ($($t:ty),* $(,)?) => {
        $(
            impl VleInt for $t {
                #[inline]
                fn to_vle_u128(self) -> u128 {
                    self as u128
                }
                #[inline]
                fn from_vle_u128(v: u128) -> Option<Self> {
                    <$t>::try_from(v).ok()
                }
            }
        )*
    };
}

macro_rules! impl_vle_signed {
    ($($t:ty),* $(,)?) => {
        $(
            impl VleInt for $t {
                #[inline]
                fn to_vle_u128(self) -> u128 {
                    zigzag_encode(self as i128, (<$t>::BITS as u32)) // width of original type
                }
                #[inline]
                fn from_vle_u128(v: u128) -> Option<Self> {
                    let decoded = zigzag_decode(v);
                    <$t>::try_from(decoded).ok()
                }
            }
        )*
    };
}

impl_vle_unsigned!(u8, u16, u32, u64, u128, usize);
impl_vle_signed!(i8, i16, i32, i64, i128, isize);

#[inline]
fn zigzag_encode(x: i128, bits: u32) -> u128 {
    // (x << 1) ^ (x >> (bits - 1))
    // Using i128 operations; overflow wraps in two's complement (fine for zigzag).
    let shift = bits.saturating_sub(1);
    let v = (x << 1) ^ (x >> shift);
    v as u128
}

#[inline]
fn zigzag_decode(u: u128) -> i128 {
    // (u >> 1) ^ -(u & 1)
    let a = (u >> 1) as i128;
    let b = -((u & 1) as i128);
    a ^ b
}

struct BitWriter {
    bytes: Vec<u8>,
    bit_len: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_len: 0,
        }
    }

    #[inline]
    fn push_bit(&mut self, bit: bool) {
        let byte_index = self.bit_len / 8;
        let bit_index = self.bit_len % 8; // 0..7, MSB-first
        if byte_index == self.bytes.len() {
            self.bytes.push(0);
        }
        if bit {
            let mask = 1u8 << (7 - bit_index);
            self.bytes[byte_index] |= mask;
        }
        self.bit_len += 1;
    }

    #[inline]
    fn push_bits_msb(&mut self, value: u128, bits: usize) {
        for i in (0..bits).rev() {
            self.push_bit(((value >> i) & 1) != 0);
        }
    }

    fn finish(self) -> Encoded {
        Encoded {
            bytes: self.bytes,
            bit_len: self.bit_len,
        }
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit_len: usize,
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8], bit_len: usize) -> Self {
        Self { bytes, bit_len, pos: 0 }
    }

    #[inline]
    fn remaining_bits(&self) -> usize {
        self.bit_len.saturating_sub(self.pos)
    }

    #[inline]
    fn read_bit(&mut self) -> Result<bool, VleError> {
        if self.pos >= self.bit_len {
            return Err(VleError::UnexpectedEof);
        }
        let byte_index = self.pos / 8;
        let bit_index = self.pos % 8;
        let b = *self.bytes.get(byte_index).ok_or(VleError::UnexpectedEof)?;
        let bit = ((b >> (7 - bit_index)) & 1) != 0;
        self.pos += 1;
        Ok(bit)
    }

    #[inline]
    fn read_bits_msb(&mut self, bits: usize) -> Result<u128, VleError> {
        if bits > 128 {
            return Err(VleError::InvalidLengthPrefix);
        }
        if self.remaining_bits() < bits {
            return Err(VleError::UnexpectedEof);
        }
        let mut v: u128 = 0;
        for _ in 0..bits {
            v <<= 1;
            v |= self.read_bit()? as u128;
        }
        Ok(v)
    }
}

/// Encoder for a stream of integers.
pub struct VleWriter {
    bw: BitWriter,
}

impl VleWriter {
    pub fn new() -> Self {
        Self { bw: BitWriter::new() }
    }

    /// Append one value.
    pub fn push<T: VleInt>(&mut self, value: T) {
        let v = value.to_vle_u128();
        encode_u128(&mut self.bw, v);
    }

    /// Finish and return the encoded bytes plus exact bit length.
    pub fn finish(self) -> Encoded {
        self.bw.finish()
    }
}

/// Decoder for a stream of integers.
pub struct VleReader<'a> {
    br: BitReader<'a>,
}

impl<'a> VleReader<'a> {
    pub fn new(bytes: &'a [u8], bit_len: usize) -> Self {
        Self { br: BitReader::new(bytes, bit_len) }
    }

    /// Read one value; returns `Ok(None)` if no bits remain.
    pub fn read<T: VleInt>(&mut self) -> Result<Option<T>, VleError> {
        if self.br.remaining_bits() == 0 {
            return Ok(None);
        }
        let u = decode_u128(&mut self.br)?;
        T::from_vle_u128(u).ok_or(VleError::ValueOutOfRange).map(Some)
    }
}

/// Encode a slice of integers into an `Encoded` bitstream.
pub fn encode_slice<T: VleInt>(values: &[T]) -> Encoded {
    let mut w = VleWriter::new();
    for &v in values {
        w.push(v);
    }
    w.finish()
}

/// Decode exactly `count` integers from an `Encoded` bitstream.
pub fn decode_n<T: VleInt>(bytes: &[u8], bit_len: usize, count: usize) -> Result<Vec<T>, VleError> {
    let mut r = VleReader::new(bytes, bit_len);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        match r.read::<T>()? {
            Some(v) => out.push(v),
            None => return Err(VleError::UnexpectedEof),
        }
    }
    Ok(out)
}

/// Encode one unsigned value (u128-domain) into the bitstream.
fn encode_u128(bw: &mut BitWriter, v: u128) {
    if v == 0 {
        bw.push_bit(false);
        return;
    }

    // Tag bit: 1 means "non-zero follows"
    bw.push_bit(true);
    encode_gamma_u128(bw, v);
}

#[inline]
fn encode_gamma_u128(bw: &mut BitWriter, v: u128) {
    debug_assert!(v != 0);

    // n = floor(log2(v)) in 0..=127
    let n = 127 - v.leading_zeros() as usize;

    // n leading zeros
    for _ in 0..n {
        bw.push_bit(false);
    }

    // then v in (n + 1) bits, MSB first
    bw.push_bits_msb(v, n + 1);
}

/// Decode one unsigned value (u128-domain) from the bitstream.
fn decode_u128(br: &mut BitReader<'_>) -> Result<u128, VleError> {
    let tag = br.read_bit()?;
    if !tag {
        return Ok(0);
    }

    decode_gamma_u128(br)
}

#[inline]
fn decode_gamma_u128(br: &mut BitReader<'_>) -> Result<u128, VleError> {
    // Count leading zeros until the first '1' (which is the MSB of the value).
    let mut zeros: usize = 0;
    loop {
        let bit = br.read_bit()?;
        if bit {
            break;
        }
        zeros += 1;
        if zeros > 127 {
            return Err(VleError::InvalidLengthPrefix);
        }
    }

    // Read the remaining `zeros` bits (the low bits after the leading 1).
    let tail = br.read_bits_msb(zeros)?;
    Ok((1u128 << zeros) | tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_one_bit() {
        let enc = encode_slice(&[0u32]);
        assert_eq!(enc.bit_len, 1);
        assert_eq!(enc.bytes.len(), 1);
        assert_eq!(enc.bytes[0] & 0b1000_0000, 0); // first bit is 0
        let dec: Vec<u32> = decode_n(&enc.bytes, enc.bit_len, 1).unwrap();
        assert_eq!(dec, vec![0]);
    }

    #[test]
    fn roundtrip_unsigned() {
        let vals: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 255, 256, 1024, 1 << 40];
        let enc = encode_slice(&vals);
        let dec: Vec<u64> = decode_n(&enc.bytes, enc.bit_len, vals.len()).unwrap();
        assert_eq!(dec, vals);
    }

    #[test]
    fn roundtrip_signed() {
        let vals: Vec<i32> = vec![0, -1, 1, -2, 2, -1000, 1000, i32::MIN + 1, i32::MAX];
        let enc = encode_slice(&vals);
        let dec: Vec<i32> = decode_n(&enc.bytes, enc.bit_len, vals.len()).unwrap();
        assert_eq!(dec, vals);
    }
}