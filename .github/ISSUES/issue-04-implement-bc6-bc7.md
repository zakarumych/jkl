---
title: "Implement BC6 and BC7 format support"
labels: ["priority:medium", "type:feature", "component:compression"]
assignees: []
---

## Description

The Jackal header format defines BC6 and BC7 formats in the `Format` enum, but there are no corresponding encoder/decoder implementations for these formats. These are modern, high-quality compression formats that should be supported.

## Details

- **Files**: Need to create `src/bc6.rs` and `src/bc7.rs`
- **Current State**: BC6/BC7 defined in `Format` enum but not implemented
- **Similar Files**: `src/bc1.rs`, `src/bc4.rs` can serve as templates

## Format Specifications

### BC6H (High Dynamic Range)
- Block size: 16 bytes for 4x4 pixels
- Purpose: HDR color compression
- Channels: RGB (no alpha)
- Precision: Half-float (16-bit per channel)
- Use cases: HDR skyboxes, environment maps, lighting data

### BC7 (High Quality RGBA)
- Block size: 16 bytes for 4x4 pixels
- Purpose: Highest quality BC compression
- Channels: RGBA
- Modes: 8 different compression modes
- Use cases: High-quality albedo maps, normal maps

## Requirements

### For BC6 (`src/bc6.rs`)
- [ ] Define `Block` struct matching BC6 format
- [ ] Implement `encode()` function for 4x4 HDR blocks
- [ ] Implement `decode()` function
- [ ] Support both signed and unsigned modes
- [ ] Handle endpoint and index encoding
- [ ] Add constants for common values (BLACK, WHITE, etc.)
- [ ] Implement `bytes()` and `from_bytes()` methods

### For BC7 (`src/bc7.rs`)
- [ ] Define `Block` struct matching BC7 format
- [ ] Implement `encode()` function for 4x4 RGBA blocks
- [ ] Implement `decode()` function
- [ ] Support multiple partition modes (at least modes 0, 1, 6)
- [ ] Handle partition selection
- [ ] Implement endpoint and index encoding
- [ ] Add constants for common values
- [ ] Implement `bytes()` and `from_bytes()` methods

### Integration
- [ ] Export modules in `src/lib.rs`
- [ ] Add basic unit tests for both formats
- [ ] Add roundtrip tests
- [ ] Document format details in comments
- [ ] Consider AnyBlock trait implementation (for Issue #5)

## Acceptance Criteria

- [ ] `bc6.rs` module created with functional encode/decode
- [ ] `bc7.rs` module created with functional encode/decode
- [ ] Both modules exposed in `lib.rs`
- [ ] Basic unit tests pass for both formats
- [ ] Roundtrip tests verify encode→decode produces similar results
- [ ] Documentation comments explain format details
- [ ] No compilation warnings
- [ ] Follows same code style as existing BC modules

## Technical Notes

### Complexity Warning
BC6 and BC7 are complex formats. BC7 in particular has 8 different modes with various partition patterns.

### Implementation Strategies

#### Option 1: Full Implementation
- Implement all modes from scratch
- Follow the official BC specification
- Most control but most work
- Reference: DirectX BC6/BC7 documentation

#### Option 2: Simplified Implementation
- Implement only the most commonly used modes
- BC6: Unsigned mode only initially
- BC7: Modes 0, 1, and 6 only initially
- Faster to implement, good enough for most use cases

#### Option 3: Wrapper Around Existing Implementation
- Use existing Rust crates like `intel_tex_2` or similar
- Fastest to implement
- Less control over algorithm
- May have licensing considerations

### Recommended Approach
Start with Option 2 (simplified) for initial implementation, then expand modes as needed.

### Reference Implementations
- DirectX Texture Conversion Tool (texconv)
- Intel ISPC Texture Compressor
- AMD Compressonator
- Reference implementation in DirectXTex

### Block Structure References

**BC6H Block Structure:**
```
- Mode bits (2-5 bits)
- Compressed endpoints (75-79 bits)
- Partition index (0 or 5 bits)
- Indices (63 or 64 bits)
Total: 128 bits (16 bytes)
```

**BC7 Block Structure (varies by mode):**
```
- Mode bits (1-8 bits)
- Partition index (0-6 bits)
- Rotation bits (0-2 bits)
- Index selection bit (0-1 bits)
- Endpoints (variable)
- Indices (variable)
Total: 128 bits (16 bytes)
```

## Testing Strategy

1. **Unit Tests**
   - Test encode/decode with solid colors
   - Test encode/decode with gradients
   - Test encode/decode with checkerboard patterns

2. **Roundtrip Tests**
   - Encode then decode and measure error
   - Verify error is within acceptable bounds

3. **Reference Comparison** (optional)
   - Compare against known reference encoder
   - Validate output format correctness

## Dependencies

**Depends on:**
- #1 (Fix failing Jackal test) - Need to understand encoding pattern

**Blocks:**
- #8 (Encoder API) - High-level API needs all formats
