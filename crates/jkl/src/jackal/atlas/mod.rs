//! Atlas is an image collection stored in a single image.
//! Beside image atlas contains metadata describing
//! which rectangle of the atlas image corresponds to which image in the collection.
//!
//! Image atlases use JKLA format, that contains image portion similarly to JKLI.

use std::io;

use smallvec::SmallVec;

use crate::{
    bits::{ReadBits, WriteBits},
    encode::{FixedCode, VarCode},
    image::{
        Extent, ImageRef,
        format::Format,
        tiles::{Tile, TileSize},
    },
    jackal::image::{
        Compression, DecodeError, JackalImageReader, JackalTileReader, Offsets, Options, Pixel,
        tile_size, write_tiles,
    },
    math::Rect,
};

define_magic!(Magic => b"JKLA");

pub struct JackalAtlasHeader {
    magic: Magic,

    /// Number of rectangles in the atlas.
    rectangles: u32,

    /// Compression method used for the texel data.
    compression: Compression,

    /// Format of the blocks.
    format: Format,

    /// Extent of the image at mip-0.
    extent: Extent,

    /// Number of texture mip levels.
    levels: u16,

    /// Size of compression tiles.
    tile_size: TileSize,
}

impl JackalAtlasHeader {
    pub fn new(
        rectangles: u32,
        compression: Compression,
        format: Format,
        extent: Extent,
        levels: u16,
        tile_size: TileSize,
    ) -> JackalAtlasHeader {
        JackalAtlasHeader {
            magic: Magic,
            rectangles,
            compression,
            format,
            extent,
            levels,
            tile_size,
        }
    }
}

impl_fixedcode_struct! {
    JackalAtlasHeader {
        magic: Magic,
        rectangles: u32,
        compression: Compression,
        format: Format,
        extent: Extent,
        levels: u16,
        tile_size: TileSize,
    } | DecodeError
}

struct Rectangles {
    array: SmallVec<[(Rect<u32>, Box<str>); 32]>,
}

impl Rectangles {
    pub fn read(len: usize, read: &mut impl io::Read) -> io::Result<Self> {
        let mut array = SmallVec::with_capacity(len);
        for _ in 0..len {
            let x = u32::fix_read(read)?;
            let y = u32::fix_read(read)?;
            let w = u32::fix_read(read)?;
            let h = u32::fix_read(read)?;
            array.push((Rect { x, y, w, h }, Box::from("")));
        }

        let mut read = ReadBits::new(read);
        for (_, meta) in &mut array {
            *meta = String::var_read(&mut read)?.into_boxed_str();
        }

        Ok(Rectangles { array })
    }
}

fn write_rectangles(
    rectangles: &[(Rect<u32>, &str)],
    write: &mut impl io::Write,
) -> io::Result<()> {
    for (rect, _) in rectangles {
        rect.x.fix_write(write)?;
        rect.y.fix_write(write)?;
        rect.w.fix_write(write)?;
        rect.h.fix_write(write)?;
    }

    let mut write = WriteBits::new(write);
    for (_, meta) in rectangles {
        meta.var_write(&mut write)?;
    }

    Ok(())
}

/// Encode entire image to the IO stream.
///
/// Uses options to determine the compression method and tile size.
pub fn write_atlas<T>(
    input: ImageRef<T>,
    rectangles: &[(Rect<u32>, &str)],
    options: Options,
    mut write: impl io::Write + io::Seek,
) -> io::Result<()>
where
    T: Pixel,
{
    let rectangle_count = u32::try_from(rectangles.len()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "Too many rectangles for atlas")
    })?;

    let extent = input.extent();

    let tile_size = tile_size(options.tile_options, extent, T::FORMAT);

    let header = JackalAtlasHeader::new(
        rectangle_count,
        options.compression,
        T::FORMAT,
        extent,
        1,
        tile_size,
    );

    header.fix_write(&mut write)?;

    write_rectangles(rectangles, &mut write)?;

    write_tiles(input, options.compression, tile_size, write)
}

/// Convenience reader object for reading Jackal Images from a stream.
pub struct JackalAtlasReader<R> {
    inner: JackalImageReader<R>,
    rectangles: Rectangles,
}

impl<R> JackalAtlasReader<R> {
    /// Opens a JackalImageReader, reads the header and tile offsets from the stream.
    pub fn open(mut read: R) -> io::Result<Self>
    where
        R: io::Read + io::Seek,
    {
        let end = read.seek(io::SeekFrom::End(0))?;
        read.seek(io::SeekFrom::Start(0))?;

        let header = JackalAtlasHeader::fix_read(&mut read)?;
        let rectangles = Rectangles::read(header.rectangles as usize, &mut read)?;

        // Read tile offsets.
        let tiles_count = header.tile_size.tiles_count(header.extent);
        let offsets = Offsets::read(tiles_count, &mut read)?;

        Ok(JackalAtlasReader {
            inner: JackalImageReader::new(
                header.compression,
                header.format,
                header.extent,
                header.tile_size,
                offsets,
                read,
                end,
            ),
            rectangles,
        })
    }

    #[allow(unused)]
    #[inline]
    pub(crate) fn reader(&mut self) -> &mut R {
        self.inner.reader()
    }

    /// Returns rectangles of the atlas, where each rectangle corresponds to an image in the collection.
    pub fn rectangles(&self) -> impl Iterator<Item = (Rect<u32>, &str)> + '_ {
        self.rectangles
            .array
            .iter()
            .map(|(rect, name)| (*rect, &**name))
    }

    #[inline]
    pub fn format(&self) -> Format {
        self.inner.format()
    }

    #[inline]
    pub fn compression(&self) -> Compression {
        self.inner.compression()
    }

    #[inline]
    pub fn extent(&self) -> Extent {
        self.inner.extent()
    }

    #[inline]
    pub fn tiles(&self) -> usize {
        self.inner.tiles()
    }

    #[inline]
    pub fn tile_offsets(&self) -> &[u64] {
        self.inner.tile_offsets()
    }

    #[inline]
    pub fn tile_size(&self) -> TileSize {
        self.inner.tile_size()
    }

    #[inline]
    pub fn tile(&self, tile_index: usize) -> Tile {
        self.inner.tile(tile_index)
    }

    pub fn tile_reader<T>(&mut self) -> io::Result<JackalTileReader<'_, R, T>>
    where
        T: Pixel,
        R: io::Read + io::Seek,
    {
        self.inner.tile_reader()
    }

    /// Returns payload length of a tile.
    #[inline]
    pub fn tile_payload_len(&self, tile_index: usize) -> u64 {
        self.inner.tile_payload_len(tile_index)
    }

    /// Returns payload length of all tiles combined
    #[inline]
    pub fn tiles_payload_len(&self) -> u64 {
        self.inner.tiles_payload_len()
    }

    /// Copies compressed payload of the tile into destination buffer.
    #[inline]
    pub fn copy_tile_payload_into(&mut self, tile_index: usize, dst: &mut [u8]) -> io::Result<()>
    where
        R: io::Read + io::Seek,
    {
        self.inner.copy_tile_payload_into(tile_index, dst)
    }

    /// Returns a reader for the context portion of the image.
    #[inline]
    pub fn context_reader(&mut self) -> io::Result<impl io::Read + '_>
    where
        R: io::Read + io::Seek,
    {
        self.inner.context_reader()
    }
}
