use std::io::{Read, Write};

pub trait LeBytes {
    fn write_to(&self, output: &mut impl Write) -> Result<(), std::io::Error>;
    fn read_from(input: &mut impl Read) -> Result<Self, std::io::Error>
    where
        Self: Sized;
}

macro_rules! impl_to_le_bytes {
    ($($ty:ty)*) => {
        $(
            impl LeBytes for $ty {
                fn write_to(&self, output: &mut impl Write) -> Result<(), std::io::Error> {
                    output.write_all(&<$ty>::to_le_bytes(*self))
                }

                fn read_from(input: &mut impl Read) -> Result<Self, std::io::Error> {
                    let mut bytes = [0; std::mem::size_of::<Self>()];
                    input.read_exact(&mut bytes)?;
                    Ok(<$ty>::from_le_bytes(bytes))
                }
            }
        )*
    };
}

impl_to_le_bytes!(u8 u16 u32 u64 u128);
