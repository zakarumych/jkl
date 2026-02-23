use std::io;

use crate::{
    bc1,
    encode::FixedCode,
    jackal::{header::Compression, visit_superblock, JackalHeader},
    math::Rgb8U,
};

pub(super) trait AnyBlock: Copy + 'static + Sized {
    const ASPECTS: usize;

    /// Compress specific block aspect.
    ///
    /// Writes compressed data into `writer`
    fn compress<const ASPECT: usize>(
        blocks: &[Self],
        header: &JackalHeader,
        write: impl io::Write + io::Seek,
    ) -> io::Result<()>;
}

impl AnyBlock for Rgb8U {
    const ASPECTS: usize = 3;

    fn compress<const ASPECT: usize>(
        blocks: &[Self],
        header: &JackalHeader,
        write: impl io::Write + io::Seek,
    ) -> io::Result<()> {
        // let superblocks_extent = header.superblocks_extent();
        // let mut superblocks = Vec::new();

        todo!();

        Ok(())
    }
}
