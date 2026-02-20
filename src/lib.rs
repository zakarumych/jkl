macro_rules! impl_fixedcode_struct {
    ($name:ident { $($field_name:ident: $field_ty:ty)* } | $error:ty) => {
        impl $crate::FixedCode for $name {
            const SIZE: usize = 0 $(+ <$field_ty as $crate::FixedCode>::SIZE)*;
            type Array = [u8; Self::SIZE];

            type Error = $error;

            fn encode(&self, output: &mut [u8; Self::SIZE]) {
                #![allow(unused_assignments)]

                let mut offset = 0;
                $(
                    self.$field_name.encode(output[offset..offset + <$field_ty as $crate::FixedCode>::SIZE].as_mut_array().unwrap());
                    offset += <$field_ty as $crate::FixedCode>::SIZE;
                )*
            }

            fn decode(input: &[u8; Self::SIZE]) -> Result<Self, $error> {
                #![allow(unused_assignments)]

                let mut offset = 0;
                Ok($name {
                    $(
                        $field_name: {
                            let value = <$field_ty as $crate::FixedCode>::decode(input[offset..offset + <$field_ty as $crate::FixedCode>::SIZE].as_array().unwrap())?;
                            offset += <$field_ty as $crate::FixedCode>::SIZE;
                            value
                        },
                    )*
                })
            }
        }
    };

    ($name:ident ( $($field_name:ident: $field_ty:ty)* ) | $error:ty) => {
        impl $crate::FixedCode for $name {
            const SIZE: usize = 0 $(+ <$field_ty as $crate::FixedCode>::SIZE)*;
            type Array = [u8; Self::SIZE];

            type Error = $error;

            fn encode(&self, output: &mut [u8; Self::SIZE]) {
                #![allow(unused_assignments)]

                let $name($($field_name,)*) = self;

                let mut offset = 0;
                $(
                    $field_name.encode(output[offset..offset + <$field_ty as $crate::FixedCode>::SIZE].as_mut_array().unwrap());
                    offset += <$field_ty as $crate::FixedCode>::SIZE;
                )*
            }

            fn decode(input: &[u8; Self::SIZE]) -> Result<Self, $error> {
                #![allow(unused_assignments)]

                let mut offset = 0;
                Ok($name(
                    $(
                        {
                            let $field_name = <$field_ty as $crate::FixedCode>::decode(input[offset..offset + <$field_ty as $crate::FixedCode>::SIZE].as_array().unwrap())?;
                            offset += <$field_ty as $crate::FixedCode>::SIZE;
                            $field_name
                        },
                    )*
                ))
            }
        }
    }
}

macro_rules! impl_fixedcode_le_bytes {
    ($($t:ty),* $(,)?) => {
        $(
            impl $crate::FixedCode for $t {
                const SIZE: usize = std::mem::size_of::<Self>();
                type Array = [u8; Self::SIZE];
                type Error = std::convert::Infallible;

                fn encode(&self, output: &mut Self::Array) {
                    *output = self.to_le_bytes();
                }

                fn decode(input: &Self::Array) -> Result<Self, Self::Error> {
                    Ok(Self::from_le_bytes(*input))
                }
            }
        )*
    };
}

macro_rules! impl_fixedcode_array {
    ($name:ident([$e:ty; $n:literal]) | $error:ty) => {
        impl $crate::FixedCode for $name {
            const SIZE: usize = <$e as $crate::FixedCode>::SIZE * $n;
            type Array = [u8; Self::SIZE];
            type Error = $error;

            fn encode(&self, output: &mut Self::Array) {
                let es = <$e as $crate::FixedCode>::SIZE;
                for (i, item) in self.0.iter().enumerate() {
                    item.encode(output[i * es..][..es].as_mut_array().unwrap());
                }
            }

            fn decode(input: &Self::Array) -> Result<Self, Self::Error> {
                let es = <$e as $crate::FixedCode>::SIZE;
                let mut result = [const { None }; $n];

                for (i, slot) in result.iter_mut().enumerate() {

                    match <$e as $crate::FixedCode>::decode(input[i * es..][..es].as_array().unwrap()) {
                        Ok(value) => {
                            *slot = Some(value)
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }

                Ok($name(result.map(|slot| slot.unwrap())))
            }
        }
    };
}

pub mod ans;
pub mod bc1;
pub mod bc2;
pub mod bc3;
pub mod bc4;
pub mod bc5;
pub mod bits;
pub mod cluster_fit;
pub mod filter;
pub mod jackal;
pub mod lz77;
pub mod lz78;
pub mod math;
pub mod max_rects;
pub mod rle;
pub mod vle;
pub mod z_curve;
pub mod zigzaq;

use std::error::Error;

pub use jackal::{DecodeError, DecompressError, Extent};

// A trait for generic encoding of values that can be represented as a fixed-size array of bytes.
pub trait FixedCode: Sized {
    const SIZE: usize;
    type Array: AsRef<[u8]> + AsMut<[u8]> + Default + Send + Sync + 'static;
    type Error: Error + Send + Sync + 'static;
    fn encode(&self, output: &mut Self::Array);
    fn decode(input: &Self::Array) -> Result<Self, Self::Error>;
}

impl<const N: usize> FixedCode for [u8; N]
where
    [u8; N]: Default,
{
    const SIZE: usize = N;
    type Array = [u8; N];
    type Error = std::convert::Infallible;

    fn encode(&self, output: &mut Self) {
        output.copy_from_slice(self);
    }

    fn decode(input: &Self) -> Result<Self, Self::Error> {
        Ok(*input)
    }
}

impl_fixedcode_le_bytes!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, f64);
