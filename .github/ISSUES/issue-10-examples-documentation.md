---
title: "Add examples and documentation"
labels: ["priority:medium", "type:documentation"]
assignees: []
---

## Description

The `examples/` directory only has one simple example (`max_rects.rs`). Comprehensive examples are needed to demonstrate all features and use cases, making the library accessible to new users and showcasing its capabilities.

## Current State

- **Examples**: 1 file (`examples/max_rects.rs`)
- **Documentation**: Minimal inline documentation
- **Tutorials**: None
- **Cookbook**: None

## Impact

Without good examples and documentation:
- Users don't know how to use the library
- Features go undiscovered
- Higher barrier to adoption
- More support questions

## Required Examples

### 1. Basic Compression Examples

#### `examples/compress_bc1.rs` - Basic BC1 Compression
```rust
//! Demonstrates basic BC1 texture compression.
//!
//! Usage: cargo run --example compress_bc1 -- input.png output.jkl

use jkl::{bc1, Extent, math::Rgb32F};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load image
    let img = image::open("input.png")?.into_rgb8();
    
    // Convert to BC1 blocks
    let blocks = encode_to_bc1(&img);
    
    // Compress with Jackal
    let mut output = Vec::new();
    jkl::jackal::compress_bc1_texture(
        Extent::D2 {
            width: (img.width() + 3) / 4,
            height: (img.height() + 3) / 4,
        },
        &blocks,
        std::io::Cursor::new(&mut output),
    )?;
    
    std::fs::write("output.jkl", output)?;
    println!("Compressed {} bytes", output.len());
    
    Ok(())
}

fn encode_to_bc1(img: &image::RgbImage) -> Vec<bc1::Block> {
    // Implementation details...
}
```

#### `examples/compress_bc3.rs` - BC3 with Alpha
Shows how to handle RGBA textures with BC3 format.

#### `examples/compress_all_formats.rs` - All BC Formats
Demonstrates encoding the same image to all supported formats and comparing sizes.

### 2. Jackal Compression Examples

#### `examples/compress_jackal.rs` - Jackal Compression
```rust
//! Demonstrates Jackal compression format.
//!
//! Shows compression ratio and explains format benefits.

use jkl::{bc1, jackal, Extent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open("texture.png")?.into_rgb8();
    let blocks = /* convert to blocks */;
    
    // Compress with Jackal
    let mut jackal_output = Vec::new();
    jackal::compress_bc1_texture(
        Extent::D2 { /* ... */ },
        &blocks,
        std::io::Cursor::new(&mut jackal_output),
    )?;
    
    // Compare with raw BC1
    let raw_bc1_size = blocks.len() * 8;
    let jackal_size = jackal_output.len();
    let ratio = raw_bc1_size as f32 / jackal_size as f32;
    
    println!("Raw BC1: {} bytes", raw_bc1_size);
    println!("Jackal:  {} bytes", jackal_size);
    println!("Ratio:   {:.2}x", ratio);
    
    Ok(())
}
```

#### `examples/decompress.rs` - Decompression
Shows how to decompress Jackal-compressed textures.

### 3. Texture Packing Examples

#### `examples/pack_textures.rs` - Texture Atlas Creation
```rust
//! Packs multiple textures into a single atlas.
//!
//! Usage: cargo run --example pack_textures -- texture1.png texture2.png

use jkl::max_rects::{MaximalRectangles, Heuristic};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load multiple images
    let images = vec![
        image::open("sprite1.png")?,
        image::open("sprite2.png")?,
        image::open("sprite3.png")?,
    ];
    
    // Create packer
    let mut packer = MaximalRectangles::new(2048, 2048);
    packer.with_heuristic(Heuristic::BestAreaFit);
    
    // Pack each image
    let mut atlas = image::RgbaImage::new(2048, 2048);
    let mut positions = Vec::new();
    
    for img in images {
        let pos = packer.insert(img.width(), img.height())
            .expect("Texture doesn't fit");
        
        // Copy image to atlas
        image::imageops::replace(&mut atlas, &img, pos.x as i64, pos.y as i64);
        positions.push(pos);
    }
    
    // Save atlas
    atlas.save("atlas.png")?;
    
    // Save metadata
    let metadata = serde_json::to_string_pretty(&positions)?;
    std::fs::write("atlas.json", metadata)?;
    
    Ok(())
}
```

### 4. Mipmap Examples

#### `examples/generate_mipmaps.rs` - Mipmap Generation
```rust
//! Generates and visualizes mipmaps.
//!
//! Shows each mipmap level side-by-side.

use jkl::mipmap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open("texture.png")?.into_rgba8();
    
    // Generate mipmaps
    let mipmaps = mipmap::generate_mipmaps(&img, mipmap::FilterType::Box, None);
    
    println!("Generated {} mip levels", mipmaps.len());
    
    // Save each level
    for (i, mip) in mipmaps.iter().enumerate() {
        mip.save(format!("mip_level_{}.png", i))?;
        println!("Level {}: {}×{}", i, mip.width(), mip.height());
    }
    
    Ok(())
}
```

### 5. Format Conversion Examples

#### `examples/convert_formats.rs` - Convert Between Formats
Shows how to convert textures from one BC format to another.

### 6. Batch Processing Examples

#### `examples/batch_process.rs` - Batch Compression
```rust
//! Compresses all PNG files in a directory.

use std::path::Path;
use walkdir::WalkDir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for entry in WalkDir::new("textures") {
        let entry = entry?;
        if entry.path().extension() == Some("png") {
            println!("Processing {:?}...", entry.path());
            compress_file(entry.path())?;
        }
    }
    Ok(())
}

fn compress_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Load, compress, save
    Ok(())
}
```

### 7. High-Level API Examples

#### `examples/custom_encoder.rs` - Using Encoder API
```rust
//! Demonstrates the high-level Encoder API.

use jkl::{Encoder, Format, Quality};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open("texture.png")?.into_rgba8();
    
    // Use builder pattern
    let encoded = Encoder::new()
        .format(Format::BC7)
        .jackal_compression(true)
        .compression_quality(Quality::High)
        .mipmaps(true)
        .encode(&img)?;
    
    println!("Encoded to {} bytes", encoded.data.len());
    std::fs::write("texture.jkl", encoded.data)?;
    
    Ok(())
}
```

### 8. Advanced Examples

#### `examples/quality_comparison.rs` - Quality vs. Speed
Compares different quality settings and measures encode time and visual quality.

#### `examples/parallel_encoding.rs` - Parallel Processing
Shows how to encode multiple textures in parallel using `rayon`.

## Documentation Requirements

### 1. API Documentation

Every public item needs documentation:

```rust
/// Encodes a 4×4 block of pixels to BC1 format.
///
/// BC1 (also known as DXT1) is a lossy compression format that encodes
/// RGB or 1-bit alpha data. Each 4×4 block is compressed to 8 bytes.
///
/// # Arguments
///
/// * `block` - A 4×4 array of RGB colors in 32-bit float format
///
/// # Returns
///
/// A compressed BC1 block (8 bytes)
///
/// # Examples
///
/// ```
/// use jkl::{bc1, math::Rgb32F};
///
/// let block = [[Rgb32F::BLACK; 4]; 4];
/// let compressed = bc1::Block::encode(block);
/// ```
///
/// # Performance
///
/// This function uses the cluster_fit algorithm which provides good quality
/// but is slower than the range_fit alternative. Expect ~5-10ms per block
/// on modern hardware.
pub fn encode(block: [[Rgb32F; 4]; 4]) -> Block {
    // ...
}
```

### 2. Module-Level Documentation

Each module needs overview documentation:

```rust
//! BC1 block compression format.
//!
//! BC1 (also known as DXT1 or S3TC) is a lossy texture compression format
//! widely supported by GPUs. It compresses 4×4 blocks of pixels to 8 bytes.
//!
//! # Format Details
//!
//! Each block contains:
//! - Two 16-bit RGB565 endpoint colors (4 bytes)
//! - 16 2-bit indices selecting interpolated colors (4 bytes)
//!
//! # Usage
//!
//! ```no_run
//! use jkl::bc1;
//! # use jkl::math::Rgb32F;
//!
//! let block = /* 4×4 pixels */;
//! let compressed = bc1::Block::encode(block);
//! let decompressed = compressed.decode();
//! ```

pub mod bc1;
```

### 3. Crate-Level Documentation

`lib.rs` needs comprehensive overview:

```rust
//! # JKL - Texture Compression Library
//!
//! JKL is a Rust library for compressing textures using BC (Block Compression)
//! formats and the custom Jackal compression format.
//!
//! ## Features
//!
//! - **BC1-BC5 Compression**: Industry-standard block compression formats
//! - **Jackal Format**: Custom hybrid compression for better ratios
//! - **Texture Packing**: Pack multiple textures into atlases
//! - **Mipmap Generation**: Automatic mipmap generation with various filters
//! - **High-Level API**: Easy-to-use encoder with sensible defaults
//!
//! ## Quick Start
//!
//! ```no_run
//! use jkl::{Encoder, Format};
//!
//! let image = image::open("texture.png")?.into_rgba8();
//! let encoded = Encoder::new()
//!     .format(Format::BC1)
//!     .encode(&image)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Examples
//!
//! See the `examples/` directory for more usage examples.

pub mod bc1;
// ...
```

### 4. Tutorial Documentation

Create `docs/` directory with tutorials:

#### `docs/getting-started.md` - Getting Started Guide
- Installation
- First compression
- Understanding output
- Common issues

#### `docs/formats.md` - Format Guide
- Explanation of each BC format
- When to use which format
- Quality vs. size tradeoffs
- Format compatibility

#### `docs/jackal-format.md` - Jackal Format Specification
- Technical details
- Benefits and use cases
- Comparison with other formats
- Integration guide

#### `docs/api-guide.md` - API Guide
- Low-level vs. high-level API
- Common patterns
- Error handling
- Best practices

## Acceptance Criteria

- [ ] At least 8 comprehensive examples exist
- [ ] Each example has detailed comments explaining what it does
- [ ] Examples cover common use cases:
  - [ ] Basic compression
  - [ ] Different formats
  - [ ] Jackal compression
  - [ ] Decompression
  - [ ] Texture packing
  - [ ] Mipmap generation
  - [ ] Batch processing
  - [ ] High-level API usage
- [ ] Examples can be run with `cargo run --example <name>`
- [ ] Examples include usage instructions in comments
- [ ] All examples compile and run successfully
- [ ] Examples use realistic input (included test images or generated data)
- [ ] API documentation exists for all public items
- [ ] Module-level documentation explains purpose and usage
- [ ] Crate-level documentation provides overview and quick start
- [ ] Examples are referenced in README
- [ ] Tutorial documentation covers getting started

## Additional Resources

### Example Test Images

Include a few test images in `examples/data/`:
- `test_gradient.png` - Gradient for quality testing
- `test_pattern.png` - Checkerboard or similar pattern
- `test_photo.png` - Photo-realistic image
- `test_sprite.png` - Sprite for packing demo

Generate these programmatically if needed:
```rust
// examples/generate_test_images.rs
fn generate_gradient() -> image::RgbaImage {
    let mut img = image::RgbaImage::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            img.put_pixel(x, y, image::Rgba([x as u8, y as u8, 128, 255]));
        }
    }
    img
}
```

### Interactive Examples (Optional)

Consider GUI examples using the existing GUI code:
- Visual quality comparison
- Interactive format selector
- Real-time compression preview

## Testing Examples

Ensure examples are tested:
```bash
# Test that all examples compile
cargo test --examples

# Run each example as a test
cargo run --example compress_bc1
cargo run --example pack_textures
# ...
```

## Documentation Publishing

Once library is ready for release:
1. Publish to crates.io
2. Documentation auto-published to docs.rs
3. Link from README
4. Consider additional hosting (GitHub Pages)

## Dependencies

**Depends on:**
- #2 (CLI implementation) - Examples should match CLI features
- #3 (README) - README should link to examples
- #8 (Encoder API) - Examples should showcase API

**Blocks:** None
