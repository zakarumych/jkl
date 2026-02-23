use std::{error::Error, io};

use crate::bits::{ReadBits, WriteBits};

// A trait for generic encoding of values that can be represented as a fixed-size array of bytes.
pub trait FixedCode: Sized {
    const SIZE: usize;
    type Array: Default + AsRef<[u8]> + AsMut<[u8]> + Send + Sync + 'static;
    type Error: Error + Send + Sync + 'static;
    fn encode(&self) -> Self::Array;
    fn decode(input: &Self::Array) -> Result<Self, Self::Error>;
}

impl<const N: usize> FixedCode for [u8; N]
where
    [u8; N]: Default,
{
    const SIZE: usize = N;
    type Array = [u8; N];
    type Error = std::convert::Infallible;

    #[inline]
    fn encode(&self) -> Self {
        *self
    }

    #[inline]
    fn decode(input: &Self) -> Result<Self, Self::Error> {
        Ok(*input)
    }
}

impl_fixedcode_le_bytes!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, f64);

pub trait Encode {
    fn write(&self, write: &mut WriteBits<impl io::Write>) -> io::Result<()>;
    fn read(read: &mut ReadBits<impl io::Read>) -> io::Result<Self>
    where
        Self: Sized;
}

impl<T> Encode for T
where
    T: FixedCode,
{
    #[inline]
    fn write(&self, write: &mut WriteBits<impl io::Write>) -> io::Result<()> {
        io::Write::write_all(write, self.encode().as_ref())
    }

    #[inline]
    fn read(read: &mut ReadBits<impl io::Read>) -> io::Result<T>
    where
        Self: Sized,
    {
        let mut buffer = T::Array::default();
        io::Read::read_exact(read, buffer.as_mut())?;
        Self::decode(&buffer).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}
