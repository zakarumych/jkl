use std::io::{Read, Write};

use crate::{bc1, jackal::DecompressError, math::Rgb565};

pub trait AnyBlock: Copy + 'static + Sized {
    const ASPECTS: usize;

    /// Compress one block aspect.
    fn compress<'a, const ASPECT: usize>(
        &self,
        encoder: &mut brotli::CompressorWriter<impl Write>,
    ) -> std::io::Result<()>;

    /// Decompress one block aspect.
    fn decompress<'a, const ASPECT: usize>(
        &mut self,
        decoder: &mut brotli::reader::Decompressor<impl Read>,
    ) -> Result<(), DecompressError>;
}

impl AnyBlock for bc1::Block {
    const ASPECTS: usize = 3;

    fn compress<'a, const ASPECT: usize>(
        &self,
        encoder: &mut brotli::CompressorWriter<impl Write>,
    ) -> std::io::Result<()> {
        match ASPECT {
            0 => {
                // Color0
                let bytes = self.color0.bits().to_le_bytes();
                encoder.write_all(&bytes)?;
            }
            1 => {
                // Color1
                let bytes = self.color1.bits().to_le_bytes();
                encoder.write_all(&bytes)?;
            }
            2 => {
                // Texels
                encoder.write_all(&self.texels)?;
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    fn decompress<'a, const ASPECT: usize>(
        &mut self,
        decoder: &mut brotli::reader::Decompressor<impl Read>,
    ) -> Result<(), DecompressError> {
        match ASPECT {
            0 => {
                // Color0
                let mut bytes = [0; 2];
                decoder.read_exact(&mut bytes)?;

                self.color0 = Rgb565::from_bits(u16::from_le_bytes(bytes));
            }
            1 => {
                // Color1
                let mut bytes = [0; 2];
                decoder.read_exact(&mut bytes)?;

                self.color1 = Rgb565::from_bits(u16::from_le_bytes(bytes));
            }
            2 => {
                // Texels
                decoder.read_exact(&mut self.texels)?;
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}
