---
title: "Add AnyBlock implementation for BC2, BC3, BC4, BC5"
labels: ["priority:high", "type:feature", "component:compression"]
assignees: []
---

## Description

Currently, only BC1 format has an implementation of the `AnyBlock` trait in `src/jackal/block.rs`. This trait is required to compress textures with the Jackal format. BC2, BC3, BC4, and BC5 formats need their own implementations to support Jackal compression.

## Details

- **File**: `src/jackal/block.rs`
- **Current State**: Only `impl AnyBlock for bc1::Block` exists (lines 22-75)
- **Needed**: Implementations for BC2, BC3, BC4, BC5
- **Impact**: Without these, only BC1 can use Jackal compression

## AnyBlock Trait Reference

```rust
pub trait AnyBlock: Copy + 'static + Sized {
    const ASPECTS: usize;

    fn compress<'a, const ASPECT: usize>(&self, writer: impl Write) -> std::io::Result<()>;
    fn decompress<'a, const ASPECT: usize>(&mut self, reader: impl Read) -> Result<(), DecompressError>;
}
```

## Format Analysis

### BC1 (Current Implementation - Reference)
- **ASPECTS**: 3 (color0, color1, texels)
- **Aspect 0**: color0 (Rgb565, 2 bytes)
- **Aspect 1**: color1 (Rgb565, 2 bytes)
- **Aspect 2**: texels (4 bytes)

### BC2 (To Implement)
- **Structure**: alpha (8 bytes) + color0 + color1 + texels (8 bytes)
- **Proposed ASPECTS**: 4
  - Aspect 0: color0 (Rgb565, 2 bytes)
  - Aspect 1: color1 (Rgb565, 2 bytes)
  - Aspect 2: texels (4 bytes)
  - Aspect 3: alpha (8 bytes)

### BC3 (To Implement)
- **Structure**: bc4::Block (alpha, 8 bytes) + bc1::Block (8 bytes)
- **Proposed ASPECTS**: 5
  - Aspect 0: alpha.color0 (R8U, 1 byte)
  - Aspect 1: alpha.color1 (R8U, 1 byte)
  - Aspect 2: alpha.texels (6 bytes)
  - Aspect 3-4: Forward to BC1 aspects

### BC4 (To Implement)
- **Structure**: color0 + color1 + texels (6 bytes with 3-bit indices)
- **Proposed ASPECTS**: 3
  - Aspect 0: color0 (R8U, 1 byte)
  - Aspect 1: color1 (R8U, 1 byte)
  - Aspect 2: texels (6 bytes)

### BC5 (To Implement)
- **Structure**: Two BC4 blocks (red and green channels)
- **Proposed ASPECTS**: 6
  - Aspect 0-2: red channel (BC4 aspects)
  - Aspect 3-5: green channel (BC4 aspects)

## Requirements

For each format (BC2, BC3, BC4, BC5):
- [ ] Implement `AnyBlock` trait
- [ ] Define `ASPECTS` constant
- [ ] Implement `compress<ASPECT>` method for each aspect
- [ ] Implement `decompress<ASPECT>` method for each aspect
- [ ] Handle all edge cases (reading/writing correct number of bytes)

## Acceptance Criteria

- [ ] BC2 has complete AnyBlock implementation
- [ ] BC3 has complete AnyBlock implementation
- [ ] BC4 has complete AnyBlock implementation
- [ ] BC5 has complete AnyBlock implementation
- [ ] Each implementation correctly handles all aspects
- [ ] Unit tests verify compress/decompress roundtrip for each format
- [ ] No panics or undefined behavior
- [ ] Follows same style as BC1 implementation
- [ ] Code is well-documented

## Implementation Guide

### Step 1: Understand BC1 Implementation
Review the existing BC1 implementation to understand the pattern:
- How aspects are numbered
- How data is written in compress
- How data is read in decompress
- Error handling approach

### Step 2: Implement BC4 (Simplest)
BC4 is the simplest of the remaining formats (similar structure to BC1 but with 1 channel):
```rust
impl AnyBlock for bc4::Block {
    const ASPECTS: usize = 3;

    fn compress<'a, const ASPECT: usize>(&self, mut writer: impl Write) -> std::io::Result<()> {
        match ASPECT {
            0 => writer.write_all(&[self.color0.bits()])?,
            1 => writer.write_all(&[self.color1.bits()])?,
            2 => writer.write_all(&self.texels)?,
            _ => unreachable!(),
        }
        Ok(())
    }

    fn decompress<'a, const ASPECT: usize>(&mut self, mut reader: impl Read) -> Result<(), DecompressError> {
        match ASPECT {
            0 => {
                let mut byte = [0u8; 1];
                reader.read_exact(&mut byte)?;
                self.color0 = R8U::new(byte[0]);
            }
            1 => {
                let mut byte = [0u8; 1];
                reader.read_exact(&mut byte)?;
                self.color1 = R8U::new(byte[0]);
            }
            2 => reader.read_exact(&mut self.texels)?,
            _ => unreachable!(),
        }
        Ok(())
    }
}
```

### Step 3: Implement BC2
BC2 adds explicit alpha to BC1.

### Step 4: Implement BC5
BC5 is two BC4 blocks, so it can reuse the BC4 logic.

### Step 5: Implement BC3
BC3 combines BC4 (alpha) and BC1 (color).

## Testing Strategy

For each format:

```rust
#[test]
fn bc<N>_anyblock_roundtrip() {
    // Create a test block with known values
    let original = bc<N>::Block { /* ... */ };
    
    // Compress all aspects
    let mut compressed = Vec::new();
    for aspect in 0..bc<N>::Block::ASPECTS {
        original.compress::<aspect>(&mut compressed).unwrap();
    }
    
    // Decompress all aspects
    let mut decompressed = bc<N>::Block::default();
    let mut cursor = std::io::Cursor::new(&compressed);
    for aspect in 0..bc<N>::Block::ASPECTS {
        decompressed.decompress::<aspect>(&mut cursor).unwrap();
    }
    
    // Verify
    assert_eq!(original, decompressed);
}
```

## Dependencies

**Depends on:**
- #1 (Fix failing Jackal test) - Need to verify the pattern is correct

**Blocks:**
- #7 (Mipmap generation) - Needs working compression for all formats
- #8 (Encoder API) - High-level API needs all formats
