use std::{hash::Hash, io};

use smallvec::SmallVec;

use crate::{
    bc1,
    bits::{write_bits_scope, ReadBits},
    encode::{FixedCode, VarCode},
    image::{Image2DMut, Image2DRef},
    math::{Rgb565, Rgb8U},
    vle::Vle,
};

use super::{compress::Compressor, DecodeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum Format {
    /// 8-bit single channel.
    R8,

    /// 8-bit two channels.
    RG8,

    /// 8-bit three channels.
    RGB8,

    /// 8-bit four channels.
    RGBA8,

    /// BC1 (DXT1) block compression format.
    BC1 = 256,

    /// BC2 (DXT3) block compression format.
    BC2,

    /// BC3 (DXT5) block compression format.
    BC3,

    /// BC4 block compression format.
    BC4,

    /// BC5 block compression format.
    BC5,

    /// BC6 block compression format.
    BC6,

    /// BC7 block compression format.
    BC7,
}

impl Format {
    /// Returns width of a block in pixels for the format.
    pub const fn block_width(&self) -> u16 {
        match self {
            Format::R8 | Format::RG8 | Format::RGB8 | Format::RGBA8 => 1,
            Format::BC1
            | Format::BC2
            | Format::BC3
            | Format::BC4
            | Format::BC5
            | Format::BC6
            | Format::BC7 => 4,
        }
    }

    /// Returns height of a block in pixels for the format.
    pub const fn block_height(&self) -> u16 {
        match self {
            Format::R8 | Format::RG8 | Format::RGB8 | Format::RGBA8 => 1,
            Format::BC1
            | Format::BC2
            | Format::BC3
            | Format::BC4
            | Format::BC5
            | Format::BC6
            | Format::BC7 => 4,
        }
    }
}

impl FixedCode for Format {
    const SIZE: usize = 2;
    type Array = [u8; 2];
    type Error = DecodeError;

    #[inline]
    fn fix_encode(&self) -> [u8; 2] {
        (*self as u16).to_le_bytes()
    }

    #[inline]
    fn fix_decode(bytes: &[u8; 2]) -> Result<Self, DecodeError> {
        let value = u16::from_le_bytes(*bytes);

        let format = match value {
            0 => Format::R8,
            1 => Format::RG8,
            2 => Format::RGB8,
            3 => Format::RGBA8,
            256 => Format::BC1,
            257 => Format::BC2,
            258 => Format::BC3,
            259 => Format::BC4,
            260 => Format::BC5,
            261 => Format::BC6,
            262 => Format::BC7,
            _ => return Err(DecodeError::InvalidFormat),
        };

        Ok(format)
    }
}

/// This trait is an interface for compression images.
pub trait Pixel: Copy + Eq + Hash + FixedCode + VarCode + 'static {
    type Context<C: Compressor>: VarCode;

    const FORMAT: Format;

    /// Compress the input images of data.
    ///
    /// `input` is an iterator over images, where each image is made of pixels of type `Self`.
    /// `compressor` is the compression algorithm to use.
    /// `write` is the output stream to write compressed data to.
    ///
    /// Function writes a context of type `Self::Context<C>` to the output stream,
    /// followed by offsets for each image tile in `input` and finally the compressed data for each tile
    /// at the corresponding offsets.
    fn compress_images<'a, C>(
        input: impl Iterator<Item = Image2DRef<'a, Self>> + Clone,
        compressor: C,
        write: impl io::Write + io::Seek,
    ) -> io::Result<()>
    where
        C: Compressor;

    #[inline]
    fn read_context<C>(read: impl io::Read) -> io::Result<Self::Context<C>>
    where
        C: Compressor,
    {
        let mut read_bits = ReadBits::new(read);
        Self::Context::<C>::var_read(&mut read_bits)
    }

    /// Decompress the input compressed data into output images.
    /// `compressor` is the compression algorithm to use.
    /// `context` is the decompression context produced during compression.
    /// `read` is the input stream to read compressed data from.
    /// `output` is an image buffer to write decompressed pixels to.
    fn decompress_image<'a, C>(
        compressor: C,
        context: &Self::Context<C>,
        read: impl io::Read,
        image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor;
}

impl Pixel for Rgb8U {
    type Context<C: Compressor> = C::Context<Vle<u32>>;

    const FORMAT: Format = Format::RGB8;

    fn compress_images<'a, C>(
        input: impl Iterator<Item = Image2DRef<'a, Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut tokens = Vec::new();

        let context = compressor.compress_symbols(
            input.map(|image| image.iter_pixels().map(|rgb| Vle(rgb.bits_interleaved()))),
            &mut tokens,
        )?;

        dbg!(tokens.len());

        let mut offsets = WriteOffsets::new(tokens.len(), &mut write)?;

        write_bits_scope(&mut write, |write_bits| context.var_write(write_bits))?;

        for token_group in &tokens {
            offsets.push_next(&mut write)?;

            write_bits_scope(&mut write, |write| {
                for token in token_group {
                    token.var_write(write)?;
                }
                Ok(())
            })?;
        }

        offsets.write(&mut write)?;
        Ok(())
    }

    fn decompress_image<'a, C>(
        compressor: C,
        context: &Self::Context<C>,
        read: impl io::Read,
        mut image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut read_bits = crate::bits::ReadBits::new(read);

        let mut symbols = compressor.decompress_tokens2(context, read_tokens(&mut read_bits));

        let height = image.height();
        for y in 0..height {
            let row = image.row_mut(y);
            for pixel in row {
                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                *pixel = Rgb8U::from_bits_interleaved(bits);
            }
        }

        Ok(())
    }
}

impl Pixel for bc1::Block {
    type Context<C: Compressor> = (
        C::Context<Vle<u16>>, // colors
        C::Context<u8>,       // texels
    );

    const FORMAT: Format = Format::BC1;

    fn compress_images<'a, C>(
        input: impl Iterator<Item = Image2DRef<'a, Self>> + Clone,
        compressor: C,
        mut write: impl io::Write + io::Seek,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let mut color_tokens = Vec::new();

        let color_cx = compressor.compress_symbols(
            input.clone().map(|image| {
                image.iter_pixels().flat_map(|b| {
                    [
                        Vle(b.color0.bits_interleaved()),
                        Vle(b.color1.bits_interleaved()),
                    ]
                })
            }),
            &mut color_tokens,
        )?;

        let mut texel_tokens = Vec::new();
        let texel_cx = compressor.compress_symbols(
            input.map(|image| image.iter_pixels().flat_map(|b| b.texels)),
            &mut texel_tokens,
        )?;

        assert_eq!(
            color_tokens.len(),
            texel_tokens.len(),
            "Tile count mismatch"
        );

        let mut offsets = WriteOffsets::new(color_tokens.len(), &mut write)?;

        write_bits_scope(&mut write, |write| {
            (color_cx, texel_cx).var_write(write)?;
            Ok(())
        })?;

        for idx in 0..color_tokens.len() {
            offsets.push_next(&mut write)?;

            let color_tile = &color_tokens[idx];
            let texel_tile = &texel_tokens[idx];

            write_bits_scope(&mut write, |write| {
                for token in color_tile {
                    token.var_write(write)?;
                }

                for token in texel_tile {
                    token.var_write(write)?;
                }

                Ok(())
            })?;
        }

        offsets.write(&mut write)?;

        Ok(())
    }

    fn decompress_image<'a, C>(
        compressor: C,
        context: &Self::Context<C>,
        read: impl io::Read,
        mut image: Image2DMut<'a, Self>,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        let (color_cx, texel_cx) = context;

        let mut read_bits = crate::bits::ReadBits::new(read);

        let mut symbols = compressor.decompress_tokens2(color_cx, read_tokens(&mut read_bits));

        let height = image.height();

        for y in 0..height {
            let row = image.row_mut(y);
            for pixel in row {
                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                pixel.color0 = Rgb565::from_bits_interleaved(bits);

                let Vle(bits) = symbols.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Not enough data to fill the image",
                    )
                })??;

                pixel.color1 = Rgb565::from_bits_interleaved(bits);
            }
        }

        drop(symbols);

        let mut symbols = compressor.decompress_tokens2(texel_cx, read_tokens(&mut read_bits));

        for y in 0..height {
            let row = image.row_mut(y);
            for pixel in row {
                for i in 0..4 {
                    let bits = symbols.next().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "Not enough data to fill the image",
                        )
                    })??;

                    pixel.texels[i] = bits;
                }
            }
        }

        Ok(())
    }
}

fn read_tokens<T>(read: &mut ReadBits<impl io::Read>) -> impl Iterator<Item = io::Result<T>> + '_
where
    T: VarCode,
{
    std::iter::from_fn(move || match T::var_read(read) {
        Ok(token) => Some(Ok(token)),
        Err(err) => Some(Err(err)),
    })
}

/// Array of tile offsets in the Jackal file,
/// provides population, reading and writing of tile offsets,
/// to ensure that implementation is consistent across the codebase.
pub(super) struct Offsets {
    array: SmallVec<[u64; 16]>,
}

impl Offsets {
    pub fn read(len: usize, mut read: impl io::Read) -> io::Result<Self> {
        let mut array = SmallVec::<[u64; 16]>::with_capacity(len);
        for _ in 0..len {
            let offset = u64::fix_read(&mut read)?;
            array.push(offset);
        }
        Ok(Offsets { array })
    }

    pub fn slice(&self) -> &[u64] {
        &self.array
    }

    pub fn bytes_size(&self) -> usize {
        self.array.len() * <u64 as FixedCode>::SIZE
    }
}

pub(super) struct WriteOffsets {
    offsets_start: u64,
    offsets: Offsets,
}

impl WriteOffsets {
    /// Creates a new Offsets object, reserves space for `len` offsets,
    /// and seeks the output stream to the end of the reserved space.
    pub fn new<W>(len: usize, write: &mut W) -> io::Result<Self>
    where
        W: io::Write + io::Seek,
    {
        let offsets_start = write.stream_position()?;
        let offsets_len = u64::try_from(<u64 as FixedCode>::SIZE * len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Too many tiles"))?; // 8 is the size of u64 in bytes
        let offsets_end = offsets_start + offsets_len;

        write.seek(io::SeekFrom::Start(offsets_end))?;

        Ok(WriteOffsets {
            offsets_start,
            offsets: Offsets {
                array: SmallVec::with_capacity(len),
            },
        })
    }

    pub fn push_next<W>(&mut self, write: &mut W) -> io::Result<()>
    where
        W: io::Seek,
    {
        let offset = write.stream_position()?;
        self.offsets.array.push(offset);
        Ok(())
    }

    /// Writes the offsets to the output stream at the reserved space.
    pub fn write<W>(&self, write: &mut W) -> io::Result<()>
    where
        W: io::Write + io::Seek,
    {
        write.seek(io::SeekFrom::Start(self.offsets_start))?;

        for offset in &self.offsets.array {
            offset.fix_write(write)?;
        }
        Ok(())
    }
}
