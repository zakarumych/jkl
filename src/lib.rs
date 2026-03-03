//! A compression and image processing toolkit.
//!
//! This crate provides entropy coders ([`ans`], [`rle`], [`vle`]),
//! dictionary compressors ([`lz77`], [`lz78`], [`lzw`]),
//! GPU-oriented block texture codecs ([`bc1`]–[`bc5`]),
//! bit-level I/O ([`bits`]), serialization traits ([`encode`]),
//! a 2D bin packer ([`max_rects`]), Z-order curve utilities ([`z_curve`]),
//! and the Jackal image format ([`jackal`]).

macro_rules! impl_fixedcode_struct {
    ($name:ident { $($field_name:ident: $field_ty:ty),* $(,)? } | $error:ty) => {
        impl $crate::encode::FixedCode for $name {
            const SIZE: usize = 0 $(+ <$field_ty as $crate::encode::FixedCode>::SIZE)*;
            type Array = [u8; Self::SIZE];
            type Error = $error;

            fn fix_encode(&self) -> [u8; Self::SIZE] {
                #![allow(unused_assignments)]

                let mut output = [0u8; Self::SIZE];
                let mut offset = 0;
                $(
                    output[offset..offset + <$field_ty as $crate::encode::FixedCode>::SIZE].copy_from_slice(&self.$field_name.fix_encode());
                    offset += <$field_ty as $crate::encode::FixedCode>::SIZE;
                )*

                output
            }

            fn fix_decode(input: &[u8; Self::SIZE]) -> Result<Self, $error> {
                #![allow(unused_assignments)]

                let mut offset = 0;
                Ok($name {
                    $(
                        $field_name: {
                            let value = <$field_ty as $crate::encode::FixedCode>::fix_decode(input[offset..offset + <$field_ty as $crate::encode::FixedCode>::SIZE].as_array().unwrap())?;
                            offset += <$field_ty as $crate::encode::FixedCode>::SIZE;
                            value
                        },
                    )*
                })
            }
        }
    };

    ($name:ident ( $($field_name:ident: $field_ty:ty),* $(,)? ) | $error:ty) => {
        impl $crate::encode::FixedCode for $name {
            const SIZE: usize = 0 $(+ <$field_ty as $crate::encode::FixedCode>::SIZE)*;
            type Array = [u8; Self::SIZE];
            type Error = $error;

            fn fix_encode(&self) -> [u8; Self::SIZE] {
                #![allow(unused_assignments)]

                let $name($($field_name,)*) = self;

                let mut output = [0u8; Self::SIZE];
                let mut offset = 0;
                $(
                    output[offset..offset + <$field_ty as $crate::encode::FixedCode>::SIZE].copy_from_slice(&$field_name.fix_encode());
                    offset += <$field_ty as $crate::encode::FixedCode>::SIZE;
                )*
                output
            }

            fn fix_decode(input: &[u8; Self::SIZE]) -> Result<Self, $error> {
                #![allow(unused_assignments)]

                let mut offset = 0;
                Ok($name(
                    $(
                        {
                            let $field_name = <$field_ty as $crate::encode::FixedCode>::fix_decode(input[offset..offset + <$field_ty as $crate::encode::FixedCode>::SIZE].as_array().unwrap())?;
                            offset += <$field_ty as $crate::encode::FixedCode>::SIZE;
                            $field_name
                        },
                    )*
                ))
            }
        }
    }
}

macro_rules! impl_fixedcode_array {
    ($name:ident([$e:ty; $n:literal]) | $error:ty) => {
        impl $crate::encode::FixedCode for $name {
            const SIZE: usize = <$e as $crate::encode::FixedCode>::SIZE * $n;
            type Array = [u8; Self::SIZE];
            type Error = $error;

            fn fix_encode(&self) -> Self::Array {
                let es = <$e as $crate::encode::FixedCode>::SIZE;
                let mut output = [0u8; Self::SIZE];
                for (i, item) in self.0.iter().enumerate() {
                    output[i * es..i * es + es].copy_from_slice(&item.fix_encode());
                }
                output
            }

            fn fix_decode(input: &Self::Array) -> Result<Self, Self::Error> {
                let es = <$e as $crate::encode::FixedCode>::SIZE;
                let mut result = [const { None }; $n];

                for (i, slot) in result.iter_mut().enumerate() {

                    match <$e as $crate::encode::FixedCode>::fix_decode(input[i * es..][..es].as_array().unwrap()) {
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

macro_rules! for_tuple {
    ($macro:ident) => {
        for_tuple!($macro for A B C D E F G H I J K L M N O P);
    };
    ($macro:ident for ) => {
        $macro!();
    };
    ($macro:ident for $head:ident $($tail:ident)*) => {
        for_tuple!($macro for $($tail)*);
        $macro!($head $($tail)*);
    };
}

macro_rules! impl_fixedcode_zero {
    ($name:ty) => {
        impl $crate::encode::FixedCode for $name {
            const SIZE: usize = 0;
            type Array = [u8; 0];
            type Error = std::convert::Infallible;

            fn fix_encode(&self) -> Self::Array {
                []
            }

            fn fix_decode(_: &Self::Array) -> Result<Self, Self::Error> {
                Ok(Self)
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
pub mod encode;
pub mod image;
pub mod jackal;
pub mod lz77;
pub mod lz78;
pub mod math;
pub mod max_rects;
// pub mod reference_map;
pub mod rle;
pub mod vle;
pub mod z_curve;
pub mod zigzaq;
