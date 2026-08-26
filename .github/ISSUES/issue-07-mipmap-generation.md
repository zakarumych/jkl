---
title: "Add mipmap generation support"
labels: ["priority:medium", "type:feature", "component:compression"]
assignees: []
---

## Description

The Jackal format has a `MipLevels` field in the header, but it's currently hardcoded to 1. There's no functionality to generate mipmaps or compress textures with multiple mip levels. Mipmaps are essential for texture quality and performance in 3D graphics.

## Details

- **File**: `src/jackal/mod.rs` (line 64: hardcoded `levels: MipLevels(1)`)
- **Current State**: Single mip level only
- **Need**: Mipmap generation and multi-level compression
- **Impact**: Cannot create production-ready textures

## Background

**What are Mipmaps?**
- Pre-filtered versions of a texture at progressively lower resolutions
- Each level is typically half the resolution of the previous level
- Stored together with the base texture
- GPU selects appropriate level based on distance/screen size
- Improves both quality (reduces aliasing) and performance (better cache usage)

**Mipmap Chain Example:**
- Level 0: 1024×1024 (base)
- Level 1: 512×512
- Level 2: 256×256
- Level 3: 128×128
- ...
- Level 10: 1×1

## Requirements

### 1. Mipmap Generation

Create a module for mipmap generation (e.g., `src/mipmap.rs`):

```rust
pub enum FilterType {
    Box,      // Simple 2×2 box filter (fast)
    Triangle, // Bilinear filter
    Lanczos3, // High quality (slower)
}

pub fn generate_mipmaps(
    image: &image::RgbaImage,
    filter: FilterType,
    max_levels: Option<u32>,
) -> Vec<image::RgbaImage>;

pub fn calculate_mip_levels(width: u32, height: u32) -> u32 {
    // floor(log2(max(width, height))) + 1
}
```

### 2. Multi-Level Compression

Update Jackal compression to handle multiple mip levels:

```rust
pub fn compress_bc1_texture_with_mipmaps(
    extent: Extent,
    mip_blocks: Vec<Vec<bc1::Block>>, // Blocks for each mip level
    write: impl Write + Seek,
) -> std::io::Result<()>;
```

### 3. Multi-Level Decompression

Update Jackal decompression to extract specific mip levels:

```rust
pub fn decompress_bc1_texture_level(
    read: impl Read + Seek,
    level: u32,
) -> Result<(Extent, Vec<bc1::Block>), DecompressError>;
```

### 4. CLI Integration

```bash
# Generate mipmaps automatically
jkl compress input.png -o output.jkl --mipmaps

# Specify maximum mip levels
jkl compress input.png -o output.jkl --mipmaps --max-levels 8

# Specify filter type
jkl compress input.png -o output.jkl --mipmaps --filter lanczos3

# Extract specific mip level
jkl decompress input.jkl -o output.png --level 2
```

## Acceptance Criteria

- [ ] Can generate mipmaps from source image
- [ ] Supports multiple filter types (at least box and bilinear)
- [ ] Can compress textures with multiple mip levels
- [ ] Jackal format correctly stores multiple mip levels
- [ ] Can decompress specific mip levels
- [ ] CLI supports `--mipmaps` flag
- [ ] CLI supports `--max-levels` option
- [ ] CLI supports `--filter` option for filter type
- [ ] Documentation explains mipmap usage
- [ ] Tests verify multi-level compression/decompression
- [ ] Handles odd dimensions correctly (e.g., 513×512)

## Technical Notes

### Mipmap Generation Algorithms

**Box Filter (2×2 Average):**
- Simplest and fastest
- Average of 4 pixels
- Good enough for most cases
- Easy to implement:
  ```rust
  new_pixel = (p00 + p01 + p10 + p11) / 4
  ```

**Triangle/Bilinear Filter:**
- Slightly better quality
- Weighted average
- Standard in many tools

**Lanczos Filter:**
- Highest quality
- More expensive computationally
- Good for final production assets
- May use `image::imageops::resize()` with Lanczos3

### Handling Odd Dimensions

When dimensions are odd:
- **Option 1**: Round down (513 → 256)
- **Option 2**: Round up with edge replication
- **Standard**: Round down (most common)

Example: 513×512 → 256×256 → 128×128 → ...

### BC Format Considerations

Block compression formats work on 4×4 blocks:
- Mip levels must be at least 4×4 (or use edge replication)
- Smaller levels (2×2, 1×1) can be stored uncompressed
- Or stop at 4×4 as minimum level

### Memory Layout in Jackal Format

Store all mip levels sequentially:
```
[Header with levels=N]
[Super-blocks metadata for all levels]
[Level 0 data]
[Level 1 data]
[Level 2 data]
...
[Level N-1 data]
```

Need to update header format or track level offsets.

### Filter Implementation

Can use `image` crate's built-in filters:
```rust
use image::imageops::FilterType;

let next_level = image::imageops::resize(
    &current_level,
    new_width,
    new_height,
    FilterType::Lanczos3,
);
```

### Premultiplied Alpha

For proper filtering with alpha:
1. Premultiply RGB by alpha before filtering
2. Filter all channels
3. Un-premultiply after filtering
4. Prevents color bleeding from transparent areas

## Example Usage

```rust
use jkl::{mipmap, bc1, Extent};

// Generate mipmaps
let image = image::open("texture.png")?.into_rgba8();
let mipmaps = mipmap::generate_mipmaps(&image, mipmap::FilterType::Box, None);

// Compress each level
let mut all_blocks = Vec::new();
for mip in &mipmaps {
    let blocks = /* encode to BC1 */;
    all_blocks.push(blocks);
}

// Compress with Jackal
let mut output = Vec::new();
jkl::jackal::compress_bc1_texture_with_mipmaps(
    Extent::D2 { width: image.width(), height: image.height() },
    all_blocks,
    std::io::Cursor::new(&mut output),
)?;
```

## Testing Strategy

1. **Unit Tests**
   - Test mipmap generation for various sizes
   - Test POT sizes (256, 512, 1024)
   - Test NPOT sizes (513, 1000)
   - Test different filters
   - Verify mip chain length

2. **Roundtrip Tests**
   - Compress with mipmaps
   - Decompress each level
   - Verify correct dimensions
   - Verify data integrity

3. **Visual Tests** (optional)
   - Compare filter quality
   - Verify no obvious artifacts
   - Check color accuracy

## Dependencies

**Depends on:**
- #1 (Fix failing Jackal test) - Need working compression
- #5 (AnyBlock implementations) - Need compression for all formats

**Blocks:**
- #8 (Encoder API) - High-level API should support mipmaps
