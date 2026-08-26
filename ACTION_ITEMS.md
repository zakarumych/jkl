# JKL Missing Functionality - Action Items

This document outlines the missing functionality identified in the JKL texture compression and packing tool, organized as action items with dependencies.

## Quick Reference: Dependency Graph

```
Issue 1 (Fix failing test)
  ├─> Issue 2 (CLI implementation)
  │     └─> Issue 3 (README documentation)
  ├─> Issue 4 (BC6/BC7 support)
  └─> Issue 5 (AnyBlock for BC2-BC5)

Issue 2 (CLI implementation)
  └─> Issue 6 (Texture packing API)

Issue 1, 5 (Compression working)
  └─> Issue 7 (Mipmap generation)

Issue 4, 5, 6, 7 (All components)
  └─> Issue 8 (High-level encoder API)

All implementation issues
  └─> Issue 9 (Comprehensive tests)

Issue 2, 3, 8 (Working APIs)
  └─> Issue 10 (Examples and docs)
```

---

## Issue 1: Fix failing Jackal roundtrip test

**Priority**: Critical  
**Status**: To Do  
**Dependencies**: None  
**Blocked By**: None  
**Blocks**: Issues 2, 4, 5

### Description
The `jackal::roundtrip` test is currently failing with an `InvalidData` error when attempting to decompress a BC1 texture that was just compressed.

### Details
- **File**: `src/jackal/mod.rs:619`
- **Error**: `Io(Custom { kind: InvalidData, error: "Invalid Data" })`
- **Test**: Basic roundtrip test compressing and decompressing a 2x1 block texture

### Acceptance Criteria
- [ ] The `jackal::roundtrip` test passes successfully
- [ ] Compression and decompression work correctly for BC1 format
- [ ] No regressions in other existing tests

### Technical Notes
The test creates a simple checkerboard pattern, encodes it with BC1, compresses with Jackal format, then decompresses and verifies the output matches. The failure occurs during decompression, suggesting an issue with the Brotli decompressor or data format.

---

## Issue 2: Implement functional CLI application

**Priority**: High  
**Status**: To Do  
**Dependencies**: Issue 1  
**Blocked By**: Issue 1 (Jackal compression must work)  
**Blocks**: Issues 3, 6

### Description
The CLI application (`cli/src/main.rs`) is currently just a placeholder that prints "Hello, world!". It needs to be implemented to provide texture compression, decompression, and processing functionality.

### Details
- **File**: `cli/src/main.rs` (currently 3 lines)
- **Current State**: Prints "Hello, world!" only

### Proposed Features
1. Compress textures to BC1-BC5 formats
2. Decompress Jackal compressed textures
3. Pack multiple textures into an atlas
4. Convert between texture formats
5. Generate mipmaps
6. Batch processing support

### Acceptance Criteria
- [ ] CLI can compress images to BC1 format with Jackal compression
- [ ] CLI can decompress Jackal-compressed textures
- [ ] CLI supports common image formats (PNG, JPEG, etc.) as input
- [ ] CLI provides meaningful error messages
- [ ] CLI has help text and usage examples
- [ ] Basic input/output operations work correctly

### Technical Notes
Consider using `clap` or similar for argument parsing. Need to add dependencies for image loading (already in workspace).

---

## Issue 3: Add README.md documentation

**Priority**: High  
**Status**: To Do  
**Dependencies**: Issue 2  
**Blocked By**: Issue 2 (CLI usage needs to be documented)  
**Blocks**: None

### Description
The repository currently has no README.md file, making it difficult for users to understand what the project does and how to use it.

### Details
- **File**: `README.md` (does not exist)
- **Impact**: Users cannot understand project purpose or usage

### Required Sections
1. Project overview and purpose
2. Features list
3. Installation instructions
4. CLI usage examples
5. Library usage examples
6. Supported formats
7. Build requirements
8. License information
9. Contributing guidelines (optional)

### Acceptance Criteria
- [ ] README.md exists in repository root
- [ ] Clearly explains what JKL is and what it does
- [ ] Provides installation instructions
- [ ] Includes CLI usage examples
- [ ] Includes library usage examples
- [ ] Lists supported texture compression formats
- [ ] Mentions the Jackal compression format

---

## Issue 4: Implement BC6 and BC7 format support

**Priority**: Medium  
**Status**: To Do  
**Dependencies**: Issue 1  
**Blocked By**: Issue 1 (Need to understand encoding pattern)  
**Blocks**: Issue 8

### Description
The Jackal header format defines BC6 and BC7 formats, but there are no corresponding encoder/decoder implementations for these formats.

### Details
- **Files**: Need to create `src/bc6.rs` and `src/bc7.rs`
- **Current State**: BC6/BC7 defined in `Format` enum but not implemented
- **Similar Files**: `src/bc1.rs`, `src/bc4.rs` can serve as templates

### Requirements
1. BC6 (HDR compression)
   - Encode/decode 4x4 blocks
   - Support HDR color values
   - Implement Block struct
   
2. BC7 (High-quality RGBA compression)
   - Encode/decode 4x4 blocks
   - Support multiple partition modes
   - Implement Block struct

### Acceptance Criteria
- [ ] `bc6.rs` module created with encode/decode functions
- [ ] `bc7.rs` module created with encode/decode functions
- [ ] Both modules exposed in `lib.rs`
- [ ] Basic unit tests for both formats
- [ ] Documentation comments explain format details

### Technical Notes
BC6 and BC7 are complex formats. BC7 has 8 different modes. Consider implementing a subset initially or using existing algorithms from reference implementations.

---

## Issue 5: Add AnyBlock implementation for BC2, BC3, BC4, BC5

**Priority**: High  
**Status**: To Do  
**Dependencies**: Issue 1  
**Blocked By**: Issue 1 (Need to verify the pattern is correct)  
**Blocks**: Issues 7, 8

### Description
Currently, only BC1 format has an implementation of the `AnyBlock` trait (in `src/jackal/block.rs`). This trait is needed to compress textures with the Jackal format. BC2, BC3, BC4, and BC5 formats need their own implementations.

### Details
- **File**: `src/jackal/block.rs`
- **Current State**: Only `impl AnyBlock for bc1::Block` exists
- **Needed**: Implementations for BC2, BC3, BC4, BC5

### Requirements
For each format (BC2, BC3, BC4, BC5):
1. Implement `AnyBlock` trait
2. Define `ASPECTS` constant (number of aspects to compress)
3. Implement `compress<ASPECT>` method
4. Implement `decompress<ASPECT>` method

### Acceptance Criteria
- [ ] BC2 has AnyBlock implementation
- [ ] BC3 has AnyBlock implementation
- [ ] BC4 has AnyBlock implementation
- [ ] BC5 has AnyBlock implementation
- [ ] Each implementation correctly handles all aspects of the format
- [ ] Unit tests verify compress/decompress roundtrip for each format

### Technical Notes
BC1 has 3 aspects (color0, color1, texels). Other formats have different structures:
- BC2: alpha (8 bytes) + BC1 color data
- BC3: BC4 alpha + BC1 color data
- BC4: single channel (color0, color1, texels with 3-bit indices)
- BC5: two BC4 blocks (red and green channels)

---

## Issue 6: Implement texture packing functionality

**Priority**: Medium  
**Status**: To Do  
**Dependencies**: Issue 2  
**Blocked By**: Issue 2 (CLI should support packing)  
**Blocks**: Issue 8

### Description
The MaxRects algorithm for texture packing exists (`src/max_rects.rs`) but is not integrated into the CLI or exposed as a public API for packing multiple textures into an atlas.

### Details
- **File**: `src/max_rects.rs` (287 lines, algorithm implemented)
- **Current State**: Low-level algorithm exists but not integrated
- **Need**: High-level API and CLI integration

### Requirements
1. Create a high-level packing API
2. Support packing multiple input images
3. Output packed atlas image
4. Output metadata (UV coordinates, positions)
5. Integrate with CLI

### Acceptance Criteria
- [ ] Public API for packing textures exists
- [ ] Can pack multiple images into single atlas
- [ ] Outputs both atlas image and metadata
- [ ] CLI supports texture packing operations
- [ ] Documentation explains packing algorithm and usage
- [ ] Example demonstrates texture packing

### Technical Notes
The existing `MaximalRectangles` struct provides the core algorithm. Need to add image handling, metadata output (JSON?), and CLI commands.

---

## Issue 7: Add mipmap generation support

**Priority**: Medium  
**Status**: To Do  
**Dependencies**: Issues 1, 5  
**Blocked By**: Issues 1, 5 (Need working compression for all formats)  
**Blocks**: Issue 8

### Description
The Jackal format has a `MipLevels` field in the header, but it's hardcoded to 1. There's no functionality to generate mipmaps or compress textures with multiple mip levels.

### Details
- **File**: `src/jackal/mod.rs` (hardcoded `levels: MipLevels(1)`)
- **Current State**: Single mip level only
- **Need**: Mipmap generation and multi-level compression

### Requirements
1. Generate mipmaps from source image
2. Compress each mip level
3. Store multiple levels in Jackal format
4. Decompress specific mip levels
5. CLI flag for mipmap generation

### Acceptance Criteria
- [ ] Can generate mipmaps from source image
- [ ] Can compress textures with multiple mip levels
- [ ] Jackal format correctly stores and retrieves mip levels
- [ ] CLI supports `--mipmaps` or similar flag
- [ ] Documentation explains mipmap usage
- [ ] Tests verify multi-level compression/decompression

### Technical Notes
Mipmap generation typically uses box filtering or higher-quality filters. Each level is half the resolution of the previous level. Need to handle odd dimensions properly.

---

## Issue 8: Implement encoder.rs module

**Priority**: Low  
**Status**: To Do  
**Dependencies**: Issues 4, 5, 6, 7  
**Blocked By**: Issues 4, 5, 6, 7 (Needs other components)  
**Blocks**: Issue 10

### Description
The `encoder.rs` file currently only contains an empty `Encoder` struct. This should be a high-level encoding API that ties together all the compression functionality.

### Details
- **File**: `src/encoder.rs` (currently 1 line)
- **Current State**: Empty struct
- **Purpose**: Provide user-friendly encoding API

### Requirements
1. High-level API for encoding textures
2. Support all BC formats
3. Support Jackal compression
4. Support texture packing
5. Support mipmap generation
6. Builder pattern or configuration struct

### Acceptance Criteria
- [ ] `Encoder` struct has functional API
- [ ] Can encode to all supported formats
- [ ] Provides sensible defaults
- [ ] Allows fine-grained control when needed
- [ ] Documentation with examples
- [ ] Used by CLI implementation

### Technical Notes
Consider a builder pattern like:
```rust
Encoder::new()
    .format(Format::BC1)
    .jackal_compression(true)
    .mipmaps(true)
    .encode(image)?
```

---

## Issue 9: Add comprehensive test coverage

**Priority**: Medium  
**Status**: To Do  
**Dependencies**: All implementation issues (1, 2, 4, 5, 6, 7, 8)  
**Blocked By**: All implementation issues  
**Blocks**: None

### Description
The project currently has only 11 unit tests, with 1 failing. Comprehensive test coverage is needed for all compression formats, algorithms, and functionality.

### Current Test Status
- Total tests: 11
- Passing: 10
- Failing: 1 (jackal::roundtrip)

### Required Test Coverage
1. **Compression Formats**
   - BC1, BC2, BC3, BC4, BC5 encode/decode
   - BC6, BC7 encode/decode (once implemented)
   - Edge cases (solid colors, gradients, patterns)

2. **Jackal Compression**
   - Roundtrip for all BC formats
   - Different texture sizes
   - Super-block handling

3. **Algorithms**
   - LZ77, LZ78, LZW, RLE, ANS
   - Edge cases and error conditions

4. **Texture Packing**
   - Various rectangle sizes
   - Rotation and quantization
   - Different heuristics

5. **Mipmap Generation**
   - Various sizes and formats
   - Odd dimensions

### Acceptance Criteria
- [ ] Test coverage > 80%
- [ ] All compression formats have roundtrip tests
- [ ] All public APIs have tests
- [ ] Edge cases are tested
- [ ] Error conditions are tested
- [ ] Tests are well-documented

---

## Issue 10: Add examples and documentation

**Priority**: Medium  
**Status**: To Do  
**Dependencies**: Issues 2, 3, 8  
**Blocked By**: Issues 2, 3, 8 (Need working APIs)  
**Blocks**: None

### Description
The `examples/` directory only has one simple example (`max_rects.rs`). Comprehensive examples are needed to demonstrate all features and use cases.

### Details
- **Directory**: `examples/`
- **Current State**: 1 example (max_rects.rs)
- **Need**: Examples for all major features

### Required Examples
1. `compress_bc1.rs` - Basic BC1 compression
2. `compress_jackal.rs` - Jackal compression format
3. `decompress.rs` - Decompressing textures
4. `pack_textures.rs` - Texture atlas packing
5. `generate_mipmaps.rs` - Mipmap generation
6. `convert_formats.rs` - Converting between formats
7. `batch_processing.rs` - Processing multiple files
8. `custom_encoder.rs` - Using the Encoder API

### Acceptance Criteria
- [ ] At least 8 comprehensive examples exist
- [ ] Each example has detailed comments
- [ ] Examples cover common use cases
- [ ] Examples can be run with `cargo run --example <name>`
- [ ] Examples are referenced in README
- [ ] API documentation has doc examples

### Technical Notes
Examples should be simple, focused, and well-commented. Each should demonstrate one primary feature or use case.

---

## Summary of Dependencies

### Critical Path (No Dependencies)
1. **Issue 1**: Fix failing Jackal roundtrip test

### Second Tier (Depends on Issue 1)
2. **Issue 2**: Implement functional CLI application
3. **Issue 4**: Implement BC6 and BC7 format support
4. **Issue 5**: Add AnyBlock implementation for BC2-BC5

### Third Tier
5. **Issue 3**: Add README.md documentation (depends on Issue 2)
6. **Issue 6**: Implement texture packing functionality (depends on Issue 2)
7. **Issue 7**: Add mipmap generation support (depends on Issues 1, 5)

### Fourth Tier
8. **Issue 8**: Implement encoder.rs module (depends on Issues 4, 5, 6, 7)
9. **Issue 9**: Add comprehensive test coverage (depends on all implementations)
10. **Issue 10**: Add examples and documentation (depends on Issues 2, 3, 8)

---

## Recommended Implementation Order

1. **Start with Issue 1** - Fix the failing test (critical foundation)
2. **Then Issue 5** - Implement AnyBlock for BC2-BC5 (enables Jackal for more formats)
3. **Then Issue 2** - Implement CLI (provides user-facing functionality)
4. **Then Issue 3** - Add README (makes project accessible)
5. **Then Issue 6** - Implement texture packing (important feature)
6. **Then Issue 7** - Add mipmap support (important feature)
7. **Then Issue 4** - Implement BC6/BC7 (additional formats)
8. **Then Issue 8** - Implement encoder API (high-level convenience)
9. **Finally Issues 9 & 10** - Add tests and examples (polish and documentation)

---

## Notes for GitHub Issue Creation

Each section above represents a separate GitHub issue. When creating issues:

1. Use the section title as the issue title
2. Include the Description, Details, Requirements/Proposed Features
3. Add the Acceptance Criteria as a task list
4. Include Technical Notes
5. Add labels based on Priority (critical, high, medium, low)
6. Add appropriate labels (bug, enhancement, documentation, testing)
7. Link dependencies in the issue description using GitHub's issue references
8. Consider using GitHub Projects or Milestones to track progress

### Suggested Labels
- `priority:critical` - Issue 1
- `priority:high` - Issues 2, 3, 5
- `priority:medium` - Issues 4, 6, 7, 9, 10
- `priority:low` - Issue 8
- `type:bug` - Issue 1
- `type:feature` - Issues 2, 4, 5, 6, 7, 8
- `type:documentation` - Issues 3, 10
- `type:testing` - Issue 9
- `component:cli` - Issue 2
- `component:compression` - Issues 1, 4, 5, 7
- `component:packing` - Issue 6
- `component:api` - Issue 8

---

## Document Version

**Created**: 2026-01-01  
**Last Updated**: 2026-01-01  
**Status**: Initial version
