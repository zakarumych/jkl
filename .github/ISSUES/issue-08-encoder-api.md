---
title: "Implement encoder.rs module"
labels: ["priority:low", "type:feature", "component:api"]
assignees: []
---

## Description

The `encoder.rs` file currently only contains an empty `Encoder` struct (1 line). This should be a high-level encoding API that ties together all the compression functionality and provides a user-friendly interface for encoding textures.

## Details

- **File**: `src/encoder.rs` (currently 1 line: `pub struct Encoder {}`)
- **Current State**: Empty struct with no implementation
- **Purpose**: Provide high-level, user-friendly encoding API
- **Target Users**: Library users who want simple, powerful API

## Why This is Needed

Currently, users must:
1. Manually convert images to blocks
2. Understand BC format internals
3. Call low-level compression functions
4. Handle format-specific details

A high-level API should:
1. Handle format conversion automatically
2. Provide sensible defaults
3. Allow configuration when needed
4. Support all features (compression, packing, mipmaps)

## Proposed API Design

### Builder Pattern Approach

```rust
use jkl::{Encoder, Format};

// Simple usage with defaults
let output = Encoder::new()
    .format(Format::BC1)
    .encode(&image)?;

// Advanced usage with options
let output = Encoder::new()
    .format(Format::BC7)
    .jackal_compression(true)
    .compression_quality(Quality::High)
    .mipmaps(true)
    .mipmap_filter(FilterType::Lanczos3)
    .max_mip_levels(8)
    .encode(&image)?;

// Texture packing
let atlas = Encoder::new()
    .format(Format::BC1)
    .pack_textures(&images)?;
```

### Struct Definition

```rust
pub struct Encoder {
    format: Format,
    jackal_compression: bool,
    compression_quality: Quality,
    generate_mipmaps: bool,
    mipmap_filter: FilterType,
    max_mip_levels: Option<u32>,
    super_block_size: Option<SuperBlockSize>,
}

pub enum Quality {
    Fast,    // Fast compression, lower quality
    Normal,  // Balanced
    High,    // Slow compression, higher quality
}

pub struct EncodedTexture {
    pub data: Vec<u8>,
    pub extent: Extent,
    pub format: Format,
    pub mip_levels: u32,
}

impl Encoder {
    pub fn new() -> Self;
    pub fn format(self, format: Format) -> Self;
    pub fn jackal_compression(self, enable: bool) -> Self;
    pub fn compression_quality(self, quality: Quality) -> Self;
    pub fn mipmaps(self, generate: bool) -> Self;
    pub fn mipmap_filter(self, filter: FilterType) -> Self;
    pub fn max_mip_levels(self, levels: u32) -> Self;
    
    pub fn encode(&self, image: &image::RgbaImage) -> Result<EncodedTexture>;
    pub fn encode_to_writer(&self, image: &image::RgbaImage, writer: impl Write + Seek) -> Result<()>;
}
```

## Requirements

### Core Functionality

- [ ] Builder pattern for configuration
- [ ] Support all BC formats (BC1-BC5, BC6-BC7)
- [ ] Jackal compression on/off
- [ ] Quality settings affect compression parameters
- [ ] Mipmap generation integration
- [ ] Sensible defaults

### Input Handling

- [ ] Accept `image::RgbaImage`
- [ ] Accept `image::RgbImage`
- [ ] Handle HDR images for BC6 (future)
- [ ] Validate input dimensions
- [ ] Pad to block size if needed

### Output Options

- [ ] Encode to memory (`Vec<u8>`)
- [ ] Encode to file
- [ ] Encode to writer (`Write + Seek`)
- [ ] Return metadata (dimensions, format, etc.)

### Error Handling

- [ ] Clear error messages
- [ ] Specific error types
- [ ] Validation errors (invalid dimensions, format mismatch)
- [ ] I/O errors

### Documentation

- [ ] API documentation with examples
- [ ] Usage guide in README
- [ ] Doc examples that compile

## Acceptance Criteria

- [ ] `Encoder` struct has functional builder API
- [ ] Can encode to all supported formats
- [ ] Provides sensible defaults for all options
- [ ] Allows fine-grained control when needed
- [ ] Clear, comprehensive documentation
- [ ] Used by CLI implementation
- [ ] Unit tests cover all configuration options
- [ ] Doc examples compile and run
- [ ] Follows Rust API guidelines

## Technical Notes

### Builder Pattern Implementation

```rust
impl Encoder {
    pub fn new() -> Self {
        Encoder {
            format: Format::BC1,
            jackal_compression: true,
            compression_quality: Quality::Normal,
            generate_mipmaps: false,
            mipmap_filter: FilterType::Box,
            max_mip_levels: None,
            super_block_size: None,
        }
    }
    
    pub fn format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }
    
    // ... other builder methods
}
```

### Quality Settings

Map quality to compression parameters:
- **Fast**: Use simple encoders, larger super-blocks
- **Normal**: Balanced settings (default)
- **High**: Use cluster_fit, smaller super-blocks, higher Brotli level

### Image Conversion

```rust
fn rgba_to_blocks(&self, image: &image::RgbaImage) -> Vec<Block> {
    // Convert RGBA image to 4x4 blocks
    // Pad if dimensions not multiple of 4
    // Encode each block based on format
}
```

### Integration with Existing Code

The encoder should call existing low-level functions:
- `bc1::Block::encode()` for block encoding
- `jackal::compress_bc1_texture()` for Jackal compression
- `mipmap::generate_mipmaps()` for mipmap generation
- `TexturePacker::pack()` for texture packing

### Validation

```rust
fn validate_input(&self, image: &image::RgbaImage) -> Result<(), EncoderError> {
    // Check dimensions are reasonable
    // Check format is compatible with image type
    // Check mipmap settings are valid
}
```

### Advanced Features (Optional)

- **Parallel encoding** - Encode blocks in parallel
- **Progress callbacks** - Report encoding progress
- **Presets** - Named configuration presets (e.g., "web", "game", "high-quality")
- **Format auto-detection** - Choose format based on image content
- **Async API** - Async/await support for encoding

## Example Usage

### Basic Example

```rust
use jkl::{Encoder, Format};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = image::open("texture.png")?.into_rgba8();
    
    let encoded = Encoder::new()
        .format(Format::BC1)
        .encode(&image)?;
    
    std::fs::write("texture.jkl", encoded.data)?;
    
    Ok(())
}
```

### Advanced Example

```rust
use jkl::{Encoder, Format, Quality, FilterType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = image::open("texture.png")?.into_rgba8();
    
    let encoded = Encoder::new()
        .format(Format::BC7)
        .jackal_compression(true)
        .compression_quality(Quality::High)
        .mipmaps(true)
        .mipmap_filter(FilterType::Lanczos3)
        .max_mip_levels(10)
        .encode(&image)?;
    
    println!("Encoded {} bytes with {} mip levels", 
             encoded.data.len(), 
             encoded.mip_levels);
    
    std::fs::write("texture.jkl", encoded.data)?;
    
    Ok(())
}
```

## Testing Strategy

1. **Unit Tests**
   - Test each builder method
   - Test default values
   - Test with various combinations of options
   
2. **Integration Tests**
   - Encode images with different formats
   - Verify output correctness
   - Test roundtrip (encode → decode)

3. **Doc Tests**
   - Ensure all examples compile
   - Test common usage patterns

## Dependencies

**Depends on:**
- #4 (BC6/BC7 support) - Need all formats
- #5 (AnyBlock implementations) - Need Jackal for all formats
- #6 (Texture packing) - Need packing API
- #7 (Mipmap generation) - Need mipmap support

**Blocks:**
- #10 (Examples) - Examples should use this API
