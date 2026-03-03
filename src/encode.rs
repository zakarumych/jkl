use std::{error::Error, io};

use crate::bits::{ReadBits, WriteBits};

/// A fixed-size byte buffer that can be used as the serialized representation of a [`FixedCode`] value.
///
/// Implementors provide a `zeroed` constructor for use as a read buffer during decoding.
pub trait ByteArray: AsRef<[u8]> + AsMut<[u8]> + Send + Sync + 'static {
    /// Returns a zeroed byte array.
    fn zeroed() -> Self;
}

impl<const N: usize> ByteArray for [u8; N] {
    #[inline]
    fn zeroed() -> Self {
        [0u8; N]
    }
}

/// A trait for generic encoding of values that can be represented as a fixed-size array of bytes.
pub trait FixedCode: Sized {
    const SIZE: usize;
    type Array: ByteArray;
    type Error: Error + Send + Sync + 'static;
    fn fix_encode(&self) -> Self::Array;
    fn fix_decode(input: &Self::Array) -> Result<Self, Self::Error>;

    /// Encodes `self` and writes the resulting bytes to `write`.
    #[inline]
    fn fix_write(&self, write: &mut impl io::Write) -> io::Result<()> {
        io::Write::write_all(write, self.fix_encode().as_ref())
    }

    /// Reads exactly [`SIZE`](Self::SIZE) bytes from `read` and decodes them.
    #[inline]
    fn fix_read(read: &mut impl io::Read) -> io::Result<Self> {
        let mut buffer = Self::Array::zeroed();
        io::Read::read_exact(read, buffer.as_mut())?;
        Self::fix_decode(&buffer).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

impl<const N: usize> FixedCode for [u8; N]
where
    [u8; N]: ByteArray,
{
    const SIZE: usize = N;
    type Array = [u8; N];
    type Error = std::convert::Infallible;

    #[inline]
    fn fix_encode(&self) -> Self {
        *self
    }

    #[inline]
    fn fix_decode(input: &Self) -> Result<Self, Self::Error> {
        Ok(*input)
    }
}

macro_rules! impl_fixedcode_le_bytes {
    ($($t:ty),* $(,)?) => {
        $(
            impl $crate::encode::FixedCode for $t {
                const SIZE: usize = std::mem::size_of::<Self>();
                type Array = [u8; Self::SIZE];
                type Error = std::convert::Infallible;

                fn fix_encode(&self) -> Self::Array {
                    self.to_le_bytes()
                }

                fn fix_decode(input: &Self::Array) -> Result<Self, Self::Error> {
                    Ok(Self::from_le_bytes(*input))
                }
            }
        )*
    };
}

impl_fixedcode_le_bytes!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, f64);

macro_rules! impl_fixedcode_tuple {
    () => {
        impl VarCode for () {
            fn var_bit_len(&self) -> usize {
                0
            }

            fn var_write(&self, _write: &mut WriteBits<impl io::Write>) -> io::Result<()> {
                Ok(())
            }

            fn var_read(_read: &mut ReadBits<impl io::Read>) -> io::Result<Self> {
                Ok(())
            }
        }
    };
    ($($a:ident)+) => {
        #[allow(non_snake_case)]
        impl<$($a),*> VarCode for ($($a,)*)
        where
            $($a: VarCode),*
        {
            fn var_bit_len(&self) -> usize {

                let ($($a,)*) = self;
                0 $(+ $a.var_bit_len())*
            }

            fn var_write(&self, write: &mut WriteBits<impl io::Write>) -> io::Result<()> {
                let ($($a,)*) = self;
                $(
                    $a.var_write(write)?;
                )*
                Ok(())
            }

            fn var_read(read: &mut ReadBits<impl io::Read>) -> io::Result<Self> {
                Ok((
                    $(
                        $a::var_read(read)?,
                    )*
                ))
            }
        }
    };
}

for_tuple!(impl_fixedcode_tuple);

/// A trait for variable-length, bit-level encoding and decoding.
///
/// Types implementing `VarCode` can serialize themselves into a variable number of bits
/// via [`WriteBits`] and deserialize from [`ReadBits`]. A blanket implementation is provided
/// for all [`FixedCode`] types, writing their fixed-size representation as whole bytes.
pub trait VarCode {
    fn var_bit_len(&self) -> usize;
    fn var_write(&self, write: &mut WriteBits<impl io::Write>) -> io::Result<()>;
    fn var_read(read: &mut ReadBits<impl io::Read>) -> io::Result<Self>
    where
        Self: Sized;
}

impl<T> VarCode for T
where
    T: FixedCode,
{
    fn var_bit_len(&self) -> usize {
        Self::SIZE * 8
    }

    #[inline]
    fn var_write(&self, write: &mut WriteBits<impl io::Write>) -> io::Result<()> {
        FixedCode::fix_write(self, write)
    }

    #[inline]
    fn var_read(read: &mut ReadBits<impl io::Read>) -> io::Result<T>
    where
        Self: Sized,
    {
        FixedCode::fix_read(read)
    }
}
