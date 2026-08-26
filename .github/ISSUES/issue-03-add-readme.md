---
title: "Add README.md documentation"
labels: ["priority:high", "type:documentation"]
assignees: []
---

## Description

The repository currently has no README.md file, making it difficult for users to understand what the project does and how to use it. A comprehensive README is essential for project visibility and adoption.

## Details

- **File**: `README.md` (does not exist)
- **Impact**: Users cannot understand project purpose or usage
- **Target**: Professional, comprehensive README

## Required Sections

### 1. Project Header
- Project name and tagline
- Badges (build status, crates.io version, license)
- Brief one-sentence description

### 2. Overview
- What is JKL?
- What problem does it solve?
- Key features and benefits

### 3. Features
- Supported texture compression formats (BC1-BC5, BC6/BC7 future)
- Jackal compression format
- Texture packing with MaxRects
- Mipmap generation
- CLI and library interfaces

### 4. Installation

```bash
# From crates.io (once published)
cargo install jkl-cli

# From source
git clone https://github.com/zakarumych/jkl
cd jkl
cargo build --release
```

### 5. Quick Start / Usage Examples

#### CLI Usage
```bash
# Compress a texture
jkl compress input.png -o output.jkl --format bc1

# Decompress a texture
jkl decompress input.jkl -o output.png
```

#### Library Usage
```rust
use jkl::{bc1, Extent};

// Compress a texture
let blocks = /* ... */;
let mut output = Vec::new();
jkl::jackal::compress_bc1_texture(
    Extent::D2 { width: 256, height: 256 },
    &blocks,
    std::io::Cursor::new(&mut output)
)?;
```

### 6. Supported Formats
- BC1 (DXT1) - RGB/1-bit Alpha
- BC2 (DXT3) - RGBA Explicit Alpha
- BC3 (DXT5) - RGBA Interpolated Alpha
- BC4 - Single Channel
- BC5 - Dual Channel
- BC6 (future) - HDR
- BC7 (future) - High Quality RGBA

### 7. Jackal Format
- Brief explanation of hybrid compression
- Benefits (GPU-friendly, parallel decompression)
- Use cases

### 8. Building from Source
- Requirements (Rust version)
- Build commands
- Running tests
- Building examples

### 9. Documentation
- Link to docs.rs (once published)
- Link to examples
- Link to wiki/extended docs (if exists)

### 10. Contributing
- How to report bugs
- How to submit PRs
- Code style guidelines
- Testing requirements

### 11. License
- MIT OR Apache-2.0
- Copyright information

### 12. Acknowledgments
- Credits for algorithms used
- Related projects

## Acceptance Criteria

- [ ] README.md exists in repository root
- [ ] Clearly explains what JKL is and what it does
- [ ] Provides installation instructions
- [ ] Includes CLI usage examples
- [ ] Includes library usage examples
- [ ] Lists supported texture compression formats
- [ ] Explains the Jackal compression format
- [ ] Has proper markdown formatting
- [ ] Images/diagrams included (optional but recommended)
- [ ] All links work correctly
- [ ] Follows standard README best practices

## Visual Elements (Optional but Recommended)

- Logo or project icon
- Comparison images (original vs. compressed)
- Compression ratio charts
- Dependency graph diagram
- Architecture diagram

## Dependencies

**Depends on:**
- #2 (CLI implementation) - Need working CLI to document usage

**Blocks:** None
