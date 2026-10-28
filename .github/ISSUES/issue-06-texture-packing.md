---
title: "Implement texture packing functionality"
labels: ["priority:medium", "type:feature", "component:packing"]
assignees: []
---

## Description

The MaxRects algorithm for texture packing exists in `src/max_rects.rs` (287 lines) but is not integrated into the CLI or exposed as a public API for packing multiple textures into an atlas. This feature is essential for game developers and graphics applications.

## Details

- **File**: `src/max_rects.rs` (algorithm implemented)
- **Current State**: Low-level algorithm exists but not integrated
- **Need**: High-level API and CLI integration
- **Use Case**: Pack multiple sprites/textures into a single atlas for efficient GPU usage

## Current State

The `MaximalRectangles` struct provides the core packing algorithm with features:
- Multiple heuristics (BestAreaFit, BestShortSideFit, etc.)
- Rectangle rotation support
- Quantization support
- Efficient packing of rectangles

## Requirements

### 1. High-Level Packing API

Create a user-friendly API in a new module (e.g., `src/atlas.rs` or `src/packer.rs`):

```rust
pub struct TexturePacker {
    max_width: u32,
    max_height: u32,
    padding: u32,
    allow_rotation: bool,
}

pub struct PackedTexture {
    pub name: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub rotated: bool,
}

pub struct TextureAtlas {
    pub width: u32,
    pub height: u32,
    pub image: image::RgbaImage,
    pub textures: Vec<PackedTexture>,
}

impl TexturePacker {
    pub fn pack(&self, images: Vec<(String, image::RgbaImage)>) -> Result<TextureAtlas>;
}
```

### 2. CLI Integration

Add packing commands to CLI:

```bash
# Pack multiple textures into an atlas
jkl pack texture1.png texture2.png texture3.png -o atlas.png

# Generate metadata JSON file
jkl pack *.png -o atlas.png --metadata atlas.json

# Specify max atlas size
jkl pack *.png -o atlas.png --max-width 2048 --max-height 2048

# Add padding between textures
jkl pack *.png -o atlas.png --padding 2

# Allow rotation for better packing
jkl pack *.png -o atlas.png --allow-rotation
```

### 3. Metadata Output

Generate JSON metadata with texture positions:

```json
{
  "width": 2048,
  "height": 1024,
  "textures": [
    {
      "name": "sprite1.png",
      "x": 0,
      "y": 0,
      "width": 256,
      "height": 256,
      "rotated": false
    },
    {
      "name": "sprite2.png",
      "x": 256,
      "y": 0,
      "width": 128,
      "height": 512,
      "rotated": true
    }
  ]
}
```

### 4. Image Composition

- Load multiple input images
- Pack them using MaxRects algorithm
- Composite into single atlas image
- Handle transparency correctly
- Support padding between textures

## Acceptance Criteria

- [ ] Public API for packing textures exists
- [ ] Can pack multiple images into single atlas
- [ ] Outputs both atlas image and metadata
- [ ] Metadata format is JSON (standard format)
- [ ] CLI supports texture packing operations
- [ ] Supports PNG input/output
- [ ] Padding between textures configurable
- [ ] Rotation option available
- [ ] Documentation explains packing algorithm and usage
- [ ] Example demonstrates texture packing

## Technical Notes

### Integration Points

1. **MaxRects Usage**
   - Already has `MaximalRectangles::new(width, height)`
   - Call `insert(width, height)` for each texture
   - Returns `Option<Rect>` with position

2. **Image Handling**
   - Use `image` crate (already in workspace)
   - Load images with `image::open()`
   - Create atlas with `RgbaImage::new(width, height)`
   - Copy pixels with `copy_from()`

3. **Metadata Format**
   - Use `serde_json` for JSON output
   - Consider supporting multiple formats (JSON, XML, custom)
   - UV coordinates could be included (normalized 0-1)

### Error Handling

Consider cases:
- Texture too large for atlas
- No space left in atlas
- Invalid image files
- Duplicate texture names

### Algorithm Options

The MaxRects struct supports:
- Different heuristics for packing
- Rectangle rotation for better fit
- Quantization for alignment

Allow users to configure these options.

### Advanced Features (Optional)

- **Power-of-two atlas sizes** - Enforce or round to POT
- **Multi-page atlases** - Create multiple atlases if needed
- **Duplicate detection** - Reuse identical textures
- **Alpha trimming** - Trim transparent borders before packing
- **Premultiplied alpha** - Handle alpha blending correctly

## Example Usage

```rust
use jkl::TexturePacker;

let packer = TexturePacker::new()
    .max_size(2048, 2048)
    .padding(2)
    .allow_rotation(true);

let images = vec![
    ("sprite1.png", image::open("sprite1.png")?),
    ("sprite2.png", image::open("sprite2.png")?),
];

let atlas = packer.pack(images)?;

// Save atlas image
atlas.image.save("atlas.png")?;

// Save metadata
let json = serde_json::to_string_pretty(&atlas.textures)?;
std::fs::write("atlas.json", json)?;
```

## Testing Strategy

1. **Unit Tests**
   - Test packing various sized rectangles
   - Test with rotation enabled/disabled
   - Test padding behavior
   - Test edge cases (empty input, single image, etc.)

2. **Integration Tests**
   - Pack real images
   - Verify output image dimensions
   - Verify metadata accuracy
   - Extract textures from atlas and compare

3. **Performance Tests**
   - Measure packing time for large sets
   - Test with various heuristics
   - Profile memory usage

## Dependencies

**Depends on:**
- #2 (CLI implementation) - CLI should support packing

**Blocks:**
- #8 (Encoder API) - High-level API could include packing
