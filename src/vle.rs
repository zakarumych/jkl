use std::{error::Error, fmt, io, ops, process::Output};

use crate::{
    bits::{ReadBits, WriteBits},
    encode::Encode,
    math::Delta,
};

pub trait Unsigned:
    fmt::Debug + Ord + ops::Add<Output = Self> + ops::Sub<Output = Self> + Eq + Copy + 'static
{
    const BITS: u32;
    const MAX: Self;
    const ZERO: Self;
    const ONE: Self;

    fn next(self) -> Self;

    fn leading_zeros(self) -> u32;
    fn reverse_bits(self) -> Self;
    fn pow2(v: u32) -> Self;

    // Returns bytes in little-endian order.
    fn to_le_bytes(self) -> [u8; 16];
    fn from_le_bytes(bytes: [u8; 16]) -> Self;
}

macro_rules! impl_unsigned {
    ($($t:ty),* $(,)?) => {
        $(
            impl Unsigned for $t {
                const BITS: u32 = Self::BITS;
                const MAX: Self = Self::MAX;
                const ZERO: Self = 0;
                const ONE: Self = 1;

                fn next(self) -> Self {
                    self + 1
                }

                #[inline]
                fn leading_zeros(self) -> u32 {
                    self.leading_zeros()
                }

                #[inline]
                fn reverse_bits(self) -> Self {
                    self.reverse_bits()
                }

                #[inline]
                fn pow2(v: u32) -> Self {
                    1 << v
                }

                #[inline]
                fn to_le_bytes(self) -> [u8; 16] {
                    let le_bytes = self.to_le_bytes();
                    let mut buffer = [0u8; 16];
                    buffer[..le_bytes.len()].copy_from_slice(&le_bytes);
                    buffer
                }

                #[inline]
                fn from_le_bytes(bytes: [u8; 16]) -> Self {
                    let mut le_bytes = [0u8; size_of::<Self>()];
                    le_bytes.copy_from_slice(&bytes[..size_of::<Self>()]);
                    Self::from_le_bytes(le_bytes)
                }
            }
        )*
    };
}

impl_unsigned!(u8, u16, u32, u64, u128, usize);

fn gamma_bit_len<T>(v: T) -> u32
where
    T: Unsigned,
{
    debug_assert_ne!(v, T::ZERO);

    let msb = T::BITS - v.leading_zeros() - 1;

    // Unary bits for `msb` and 1 bit for the first '1'.
    let unary_bits = msb + 1;

    // Tail bits in little-endian order.
    let tail_bits = msb;

    unary_bits + tail_bits
}

#[inline]
fn encode_gamma<T, W>(v: T, writer: &mut WriteBits<W>) -> io::Result<()>
where
    T: Unsigned,
    W: io::Write,
{
    debug_assert_ne!(v, T::ZERO);

    let msb = T::BITS - v.leading_zeros() - 1;

    // Unary encode `msb` as n zeros followed by a one.
    for _ in 0..msb {
        writer.write_bit(false)?;
    }
    writer.write_bit(true)?;

    // tail = v - 2^msb;
    let tail: T = v - T::pow2(msb);

    if msb > 0 {
        // write remaining bits in LE.
        writer.write_all_bits(&tail.to_le_bytes(), 0, msb as usize)?;
    }

    Ok(())
}

pub fn encode_bit_len<T>(v: T) -> usize
where
    T: Unsigned,
{
    let msb = if v < T::MAX {
        // Can safely compute v+1 without overflow since v < MAX.
        let v = v.next();

        // n = floor(log2(v))
        let msb = T::BITS - v.leading_zeros() - 1;

        msb
    } else {
        // v+1 is 2^BITS, so pos of MSB is BITS and the rest bits are 0.
        T::BITS
    };

    // gamma encode `msb + 1`.
    let gamma_bits = gamma_bit_len(msb + 1);

    let tail_bits = msb;

    (gamma_bits + tail_bits) as usize
}

pub fn encode_non_zero_bit_len<T>(v: T) -> usize
where
    T: Unsigned,
{
    // n = floor(log2(v))
    let msb = T::BITS - v.leading_zeros() - 1;

    // gamma encode `msb + 1`.
    let gamma_bits = gamma_bit_len(msb + 1);

    let tail_bits = msb;

    (gamma_bits + tail_bits) as usize
}

/// Encode unsigned value `v`.
///
/// Encodes `v+1` using Elias delta code.
pub fn encode<T, W>(v: T, writer: &mut WriteBits<W>) -> io::Result<()>
where
    T: Unsigned,
    W: io::Write,
{
    let (msb, tail) = if v < T::MAX {
        // Can safely compute v+1 without overflow since v < MAX.
        let v = v.next();

        // n = floor(log2(v))
        let msb = T::BITS - v.leading_zeros() - 1;

        let tail = v - T::pow2(msb);

        (msb, tail)
    } else {
        // v+1 is 2^BITS, so pos of MSB is BITS and the rest bits are 0.
        (T::BITS, T::ZERO)
    };

    // gamma encode `msb + 1`.
    encode_gamma(msb + 1, &mut *writer)?;

    if msb > 0 {
        // write the remainig bits in LE
        writer.write_all_bits(&tail.to_le_bytes(), 0, msb as usize)?;
    }

    Ok(())
}

/// Encode unsigned value `v`.
///
/// Encodes `v` using Elias delta code.
pub fn encode_non_zero<T, W>(v: T, writer: &mut WriteBits<W>) -> io::Result<()>
where
    T: Unsigned,
    W: io::Write,
{
    assert_ne!(v, T::ZERO);

    // n = floor(log2(v))
    let msb = T::BITS - v.leading_zeros() - 1;

    let tail = v - T::pow2(msb);

    // gamma encode `msb + 1`.
    encode_gamma(msb + 1, &mut *writer)?;

    if msb > 0 {
        // write the remainig bits in LE
        writer.write_all_bits(&tail.to_le_bytes(), 0, msb as usize)?;
    }

    Ok(())
}

/// Error specifies that the decoded value is too large to fit in the target type.
#[derive(Clone, Copy, Debug)]
pub struct TooLarge;

impl fmt::Display for TooLarge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value is too large to decode")
    }
}

impl Error for TooLarge {}

#[inline]
fn decode_gamma<T, R>(reader: &mut ReadBits<R>) -> io::Result<T>
where
    T: Unsigned,
    R: io::Read,
{
    // Count leading zeros until the first '1' (which is the MSB of the value).
    let mut msb: u32 = 0;
    loop {
        let bit = reader.read_bit()?;
        if bit {
            break;
        }

        msb += 1;

        if msb == T::BITS {
            // if msb == BITS, the value is 2^BITS, which is out of range for T.
            return Err(io::Error::new(io::ErrorKind::InvalidData, TooLarge));
        }
    }

    // Read remaining bits.
    let mut buffer = [0u8; 16];
    reader.read_all_bits(&mut buffer, 0, msb as usize)?;
    let tail = T::from_le_bytes(buffer);

    let value = T::pow2(msb) + tail;

    Ok(value)
}

/// Decode unsigned value.
///
/// Reads the value encoded by `encode` function.
/// Returns an error if the encoded value is too large to fit in T.
pub fn decode<T, R>(reader: &mut ReadBits<R>) -> io::Result<T>
where
    T: Unsigned,
    R: io::Read,
{
    let msb = decode_gamma::<u32, R>(reader)? - 1;

    let mut buffer = [0u8; 16];
    reader.read_all_bits(&mut buffer, 0, msb as usize)?;

    let tail = T::from_le_bytes(buffer);

    if msb >= T::BITS {
        // If msb == BITS and tail is not zero, the value is larger than 2^BITS - 1, which is out of range for T.
        if msb > T::BITS || tail != T::ZERO {
            return Err(io::Error::new(io::ErrorKind::InvalidData, TooLarge));
        }

        return Ok(T::MAX);
    }

    Ok(T::pow2(msb) + tail - T::ONE)
}

/// Decode unsigned value.
///
/// Reads the value encoded by `encode` function.
/// Returns an error if the encoded value is too large to fit in T.
pub fn decode_non_zero<T, R>(reader: &mut ReadBits<R>) -> io::Result<T>
where
    T: Unsigned,
    R: io::Read,
{
    let msb = decode_gamma::<u32, R>(reader)? - 1;

    let mut buffer = [0u8; 16];
    reader.read_all_bits(&mut buffer, 0, msb as usize)?;

    let tail = T::from_le_bytes(buffer);

    if msb >= T::BITS {
        return Err(io::Error::new(io::ErrorKind::InvalidData, TooLarge));
    }

    Ok(T::pow2(msb) + tail)
}

/// Wrapper type to encode values using variable-length encoding.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vle<T>(pub T);

impl<T> Delta for Vle<T>
where
    T: Delta,
{
    #[inline]
    fn delta(self, base: Self) -> Vle<T> {
        Vle(self.0.delta(base.0))
    }

    #[inline]
    fn from_delta(base: Self, delta: Vle<T>) -> Self {
        Vle(T::from_delta(base.0, delta.0))
    }
}

impl<T> Encode for Vle<T>
where
    T: Unsigned,
{
    fn bit_len(&self) -> usize {
        encode_bit_len(self.0)
    }

    fn write(&self, write: &mut WriteBits<impl io::Write>) -> io::Result<()> {
        encode(self.0, write)
    }

    fn read(read: &mut ReadBits<impl io::Read>) -> io::Result<Self> {
        let value = decode(read)?;
        Ok(Self(value))
    }
}

#[cfg(test)]
mod tests {
    use crate::zigzaq::ZigZag;

    use super::*;

    #[test]
    fn zero_is_one_bit() {
        let mut buffer = Vec::new();
        let mut writer = WriteBits::new(&mut buffer);
        encode(0u32, &mut writer).unwrap();
        writer.finish().unwrap();
        assert_eq!(buffer, [0b1]);
    }

    #[test]
    fn roundtrip_unsigned() {
        let vals: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 255, 256, 1024, 1 << 40];
        let mut buffer = Vec::new();

        let mut writer = WriteBits::new(&mut buffer);
        for &v in &vals {
            encode(v, &mut writer).unwrap();
        }
        writer.finish().unwrap();

        let mut reader = ReadBits::new(&buffer[..]);
        let mut decoded = Vec::new();

        for _ in 0..vals.len() {
            decoded.push(decode::<u64, _>(&mut reader).unwrap());
        }

        assert_eq!(decoded, vals);
    }

    #[test]
    fn roundtrip_signed() {
        let vals: Vec<i32> = vec![0, -1, 1, -2, 2, -55, 55, -1000, 1000];
        let mut buffer = Vec::new();

        let mut writer = WriteBits::new(&mut buffer);
        for &v in &vals {
            encode(v.zigzag(), &mut writer).unwrap();
        }
        writer.finish().unwrap();

        let mut reader = ReadBits::new(&buffer[..]);
        let mut decoded = Vec::new();

        for _ in 0..vals.len() {
            decoded.push(i32::zagzig(decode(&mut reader).unwrap()));
        }

        assert_eq!(decoded, vals);
    }
}
