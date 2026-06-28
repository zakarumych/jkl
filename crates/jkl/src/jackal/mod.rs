//! Jackal file format.
//!
//! Jackal is a family of file formats for storing compressed data.
//! It is designed to be efficient during decompression and support on-GPU decompression,
//! for fastest load times and lowest memory usage.
//!

#[derive(Debug)]
struct InvalidMagic;

impl std::fmt::Display for InvalidMagic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid magic number")
    }
}

impl std::error::Error for InvalidMagic {}

macro_rules! define_magic {
    ($ident:ident => $literal:literal) => {
        #[derive(Clone, Copy, Debug)]
        struct $ident;

        impl $ident {
            const BYTES: [u8; 4] = *$literal;
        }

        impl crate::encode::FixedCode for $ident {
            const SIZE: usize = 4;
            type Array = [u8; 4];
            type Error = $crate::jackal::InvalidMagic;

            #[inline]
            fn fix_encode(&self) -> [u8; 4] {
                Self::BYTES
            }

            #[inline]
            fn fix_decode(bytes: &[u8; 4]) -> Result<Self, $crate::jackal::InvalidMagic> {
                if *bytes != Self::BYTES {
                    return Err($crate::jackal::InvalidMagic);
                }
                Ok($ident)
            }
        }
    };
}

pub mod atlas;
pub mod image;
