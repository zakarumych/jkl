---
title: "Implement functional CLI application"
labels: ["priority:high", "type:feature", "component:cli"]
assignees: []
---

## Description

The CLI application (`cli/src/main.rs`) is currently just a placeholder that prints "Hello, world!". It needs to be implemented to provide texture compression, decompression, and processing functionality.

## Details

- **File**: `cli/src/main.rs` (currently 3 lines)
- **Current State**: Prints "Hello, world!" only
- **Target**: Full-featured CLI for texture processing

## Proposed Features

### Core Functionality
1. **Compress textures** to BC1-BC5 formats
2. **Decompress** Jackal compressed textures
3. **Pack multiple textures** into an atlas
4. **Convert** between texture formats
5. **Generate mipmaps**
6. **Batch processing** support

### Command Structure (Proposed)

```bash
# Compress a texture
jkl compress input.png -o output.jkl --format bc1

# Decompress a texture
jkl decompress input.jkl -o output.png

# Pack multiple textures
jkl pack texture1.png texture2.png -o atlas.jkl

# Convert formats
jkl convert input.bc1 -o output.bc3

# Generate mipmaps
jkl compress input.png -o output.jkl --mipmaps
```

## Acceptance Criteria

- [ ] CLI can compress images to BC1 format with Jackal compression
- [ ] CLI can decompress Jackal-compressed textures
- [ ] CLI supports common image formats (PNG, JPEG, etc.) as input
- [ ] CLI provides meaningful error messages
- [ ] CLI has help text and usage examples (`--help` flag)
- [ ] Basic input/output operations work correctly
- [ ] Exit codes indicate success/failure appropriately

## Technical Notes

### Recommended Dependencies
- `clap` (v4) for argument parsing with derive macros
- `anyhow` for error handling
- `image` crate (already in workspace) for loading/saving images

### CLI Design Considerations
- Follow Unix conventions for flags and arguments
- Support both short (`-o`) and long (`--output`) flags
- Allow piping input/output where appropriate
- Verbose mode for debugging (`-v`, `--verbose`)
- Quiet mode for scripting (`-q`, `--quiet`)

### Example Implementation Structure
```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "jkl")]
#[command(about = "Texture compression and packing tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Compress { /* ... */ },
    Decompress { /* ... */ },
    Pack { /* ... */ },
}
```

## Dependencies

**Depends on:**
- #1 (Fix failing Jackal test) - Must have working compression

**Blocks:**
- #3 (README documentation) - Need CLI to document
- #6 (Texture packing) - CLI should expose this feature
