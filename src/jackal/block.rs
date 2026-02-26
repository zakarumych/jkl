use std::io;

use crate::{
    image::ImageRef,
    jackal::{compress::Compressor, SuperBlockSize},
    math::Rgb8U,
};

/// This trait is an interface for compressing images composed of blocks of a specific type.
pub(super) trait AnyBlock: Sized + 'static {
    const ASPECTS: usize;

    fn compress_aspect<C, const ASPECT: usize>(
        image: ImageRef<'_, Self>,
        compressor: C,
    ) -> io::Result<()>
    where
        C: Compressor;
}

impl AnyBlock for Rgb8U {
    const ASPECTS: usize = 1;

    fn compress_aspect<C, const ASPECT: usize>(
        image: ImageRef<'_, Self>,
        compressor: C,
    ) -> io::Result<()>
    where
        C: Compressor,
    {
        todo!()
        // if image.width() < usize::from(superblock.width)
        //     || image.height() < usize::from(superblock.height)
        // {
        //     // Only 1 superblock can fit.

        //     let ctx = compressor.build_context(image.pixels().iter().map(|p| {
        //         let bits = p.bits_interleaved();
        //         let [a, b, c, _] = bits.to_le_bytes();
        //         [a, b, c]
        //     }));
        // }

        // todo!()
    }
}
