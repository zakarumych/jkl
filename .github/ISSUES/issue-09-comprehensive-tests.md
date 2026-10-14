---
title: "Add comprehensive test coverage"
labels: ["priority:medium", "type:testing"]
assignees: []
---

## Description

The project currently has only 11 unit tests, with 1 failing. Comprehensive test coverage is needed for all compression formats, algorithms, and functionality to ensure reliability and prevent regressions.

## Current Test Status

```
Running unittests src/lib.rs
running 11 tests
- bits::test_reader ... ok
- bits::test_writer ... ok
- bits::test_test ... ok
- lz78::test_u16 ... ok
- math::test_round_down ... ok
- math::test_round_up ... ok
- ans::test_u16 ... ok
- z_curve::test_even_odd_split_squash ... ok
- z_curve::test_rect_z_order ... ok
- lz77::test_u16 ... ok
- jackal::roundtrip ... FAILED
```

**Total**: 11 tests  
**Passing**: 10  
**Failing**: 1  
**Coverage**: Very low (estimated < 20%)

## Goals

- **Target Coverage**: > 80% line coverage
- **Test All Public APIs**: Every public function/method should have tests
- **Test Edge Cases**: Boundary conditions, error cases
- **Test Integration**: Test components working together
- **Fix Existing Failures**: jackal::roundtrip must pass

## Required Test Coverage

### 1. Compression Formats (BC1-BC5, BC6, BC7)

Each format needs:
- [ ] **Encode/decode roundtrip tests** - Verify encode→decode preserves data
- [ ] **Solid color tests** - Black, white, red, etc.
- [ ] **Gradient tests** - Smooth color transitions
- [ ] **Pattern tests** - Checkerboard, stripes
- [ ] **Edge case tests** - Transparent, semi-transparent
- [ ] **Bytes conversion tests** - `bytes()` and `from_bytes()`

```rust
#[test]
fn bc1_roundtrip_solid_colors() {
    let colors = [Rgb32F::BLACK, Rgb32F::WHITE, Rgb32F::RED];
    for color in colors {
        let block = [[color; 4]; 4];
        let encoded = bc1::Block::encode(block);
        let decoded = encoded.decode();
        // Assert colors are close enough
    }
}
```

### 2. Jackal Compression

- [ ] **Fix existing roundtrip test** (Issue #1)
- [ ] **Roundtrip for all BC formats** (once AnyBlock implemented)
- [ ] **Different texture sizes** - 4×4, 64×64, 256×256, etc.
- [ ] **Non-square textures** - 256×128, 512×256
- [ ] **Super-block handling** - Various super-block sizes
- [ ] **Mip level support** - Multiple mip levels (once implemented)
- [ ] **Header read/write** - Verify header serialization

```rust
#[test]
fn jackal_roundtrip_various_sizes() {
    let sizes = [(4, 4), (64, 64), (256, 128)];
    for (w, h) in sizes {
        // Create test data
        // Compress
        // Decompress
        // Verify
    }
}
```

### 3. Compression Algorithms

#### LZ77, LZ78, LZW
- [ ] **Compress/decompress roundtrip**
- [ ] **Empty input**
- [ ] **Single token**
- [ ] **Repetitive data**
- [ ] **Random data**
- [ ] **Maximum dictionary size**

```rust
#[test]
fn lz77_roundtrip_repetitive() {
    let data = vec![1u8; 1000];
    let compressed = lz77::compress(&data);
    let decompressed = lz77::decompress(&compressed);
    assert_eq!(data, decompressed);
}
```

#### RLE (Run-Length Encoding)
- [ ] **Long runs**
- [ ] **No runs (all different)**
- [ ] **Mixed runs and singles**

#### ANS (Asymmetric Numeral Systems)
- [ ] **Various data distributions**
- [ ] **Edge cases (single symbol, all symbols)**

### 4. Bit Operations

- [ ] **WriteBits/ReadBits roundtrip**
- [ ] **Various bit lengths** (1-32 bits)
- [ ] **Alignment tests**
- [ ] **Flush/finish behavior**

Existing tests cover some of this, but need expansion.

### 5. Math Operations

- [ ] **Color space conversions** (RGB↔YIQ, etc.)
- [ ] **Vector operations**
- [ ] **Rounding operations** (partially covered)
- [ ] **Perceptual distance calculations**

### 6. Texture Packing (MaxRects)

- [ ] **Pack various sized rectangles**
- [ ] **Fill efficiency** (how well space is used)
- [ ] **Rotation enabled/disabled**
- [ ] **Different heuristics** (BestAreaFit, BestShortSideFit, etc.)
- [ ] **Quantization**
- [ ] **Empty bin**
- [ ] **Rectangle too large**

```rust
#[test]
fn maxrects_pack_efficiency() {
    let mut packer = MaximalRectangles::new(1024, 1024);
    let rects = vec![(256, 256), (512, 512), (128, 128)];
    
    for (w, h) in rects {
        let pos = packer.insert(w, h);
        assert!(pos.is_some(), "Should fit in 1024×1024");
    }
}
```

### 7. Mipmap Generation (once implemented)

- [ ] **Generate correct number of levels**
- [ ] **Each level has correct dimensions**
- [ ] **Power-of-two dimensions**
- [ ] **Non-power-of-two dimensions**
- [ ] **Different filters** (Box, Triangle, Lanczos)
- [ ] **Minimum level size** (stop at 4×4 or 1×1)

### 8. High-Level Encoder API (once implemented)

- [ ] **Builder pattern** - All builder methods
- [ ] **Default values** - Encoder with defaults works
- [ ] **All format support** - Can encode to each format
- [ ] **Error cases** - Invalid input, unsupported format
- [ ] **Integration** - Full encode pipeline works

### 9. CLI (once implemented)

- [ ] **Command parsing** - Arguments parsed correctly
- [ ] **File I/O** - Can read/write files
- [ ] **Error handling** - Graceful error messages
- [ ] **Exit codes** - Correct success/failure codes

Use integration tests for CLI:
```rust
#[test]
fn cli_compress_decompress() {
    let output = Command::new("target/debug/jkl")
        .arg("compress")
        .arg("test.png")
        .arg("-o")
        .arg("test.jkl")
        .output()
        .expect("Failed to run CLI");
    
    assert!(output.status.success());
}
```

## Test Organization

### Unit Tests
Place near the code they test (in same file):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature() {
        // ...
    }
}
```

### Integration Tests
Place in `tests/` directory:
```
tests/
├── bc_formats.rs       # Test all BC formats
├── jackal.rs           # Jackal compression tests
├── packing.rs          # Texture packing tests
└── cli.rs              # CLI integration tests
```

### Test Utilities
Create helper functions for common test operations:
```rust
// tests/common/mod.rs
pub fn create_test_image(width: u32, height: u32) -> RgbaImage {
    // Create test pattern
}

pub fn assert_colors_close(a: Rgb32F, b: Rgb32F, tolerance: f32) {
    // Compare with tolerance
}
```

## Property-Based Testing (Optional)

Consider using `proptest` or `quickcheck` for:
- Testing with random inputs
- Finding edge cases automatically
- Verifying properties (e.g., "encode→decode always succeeds")

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn lz77_roundtrip_any_data(data: Vec<u8>) {
        let compressed = lz77::compress(&data);
        let decompressed = lz77::decompress(&compressed);
        prop_assert_eq!(data, decompressed);
    }
}
```

## Performance Benchmarks (Optional)

Consider adding benchmarks with `criterion`:
```rust
// benches/compression.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_bc1_encode(c: &mut Criterion) {
    let block = /* test block */;
    c.bench_function("bc1_encode", |b| {
        b.iter(|| bc1::Block::encode(black_box(block)))
    });
}
```

## Acceptance Criteria

- [ ] Test coverage > 80% (measure with `cargo tarpaulin` or similar)
- [ ] All compression formats have comprehensive tests
- [ ] All public APIs have tests
- [ ] Edge cases are tested (empty input, maximum sizes, etc.)
- [ ] Error conditions are tested
- [ ] Tests are well-documented with comments
- [ ] All tests pass consistently
- [ ] No flaky tests (tests that randomly fail)
- [ ] Tests run in reasonable time (< 1 minute for unit tests)
- [ ] CI runs all tests automatically

## Testing Tools

### Coverage Measurement
```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run coverage
cargo tarpaulin --out Html --output-dir coverage/
```

### Test Execution
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture

# Run ignored tests
cargo test -- --ignored

# Run with multiple threads
cargo test -- --test-threads=4
```

## Documentation Tests

Ensure all doc examples compile and run:
```rust
/// Encodes a 4×4 block to BC1 format.
///
/// # Example
///
/// ```
/// use jkl::bc1;
/// let block = [[Rgb32F::BLACK; 4]; 4];
/// let encoded = bc1::Block::encode(block);
/// ```
pub fn encode(block: [[Rgb32F; 4]; 4]) -> Block {
    // ...
}
```

## Continuous Integration

Add CI configuration (`.github/workflows/test.yml`):
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run tests
        run: cargo test --all-features
      - name: Check coverage
        run: cargo tarpaulin --out Xml
```

## Dependencies

**Depends on:**
- All implementation issues (#1-#8) - Need code to test

**Blocks:** None (testing doesn't block other work, but should be done alongside)

## Testing Strategy by Priority

### Phase 1: Critical (Do First)
1. Fix jackal::roundtrip test (#1)
2. Add BC1-BC5 format tests
3. Add basic integration tests

### Phase 2: Important (Do Next)
1. Add Jackal compression tests for all formats
2. Add algorithm tests (LZ77, LZ78, LZW, RLE, ANS)
3. Add texture packing tests

### Phase 3: Complete (Do Last)
1. Add mipmap generation tests
2. Add encoder API tests
3. Add CLI integration tests
4. Measure and improve coverage to > 80%
