---
title: "Fix failing Jackal roundtrip test"
labels: ["priority:critical", "type:bug", "component:compression"]
assignees: []
---

## Description

The `jackal::roundtrip` test is currently failing with an `InvalidData` error when attempting to decompress a BC1 texture that was just compressed. This is a critical issue that blocks other functionality development.

## Details

- **File**: `src/jackal/mod.rs:619`
- **Error**: `Io(Custom { kind: InvalidData, error: "Invalid Data" })`
- **Test**: Basic roundtrip test compressing and decompressing a 2x1 block texture

## Current Behavior

```
test jackal::roundtrip ... FAILED

---- jackal::roundtrip stdout ----
thread 'jackal::roundtrip' panicked at src/jackal/mod.rs:619:88:
called `Result::unwrap()` on an `Err` value: Io(Custom { kind: InvalidData, error: "Invalid Data" })
```

## Expected Behavior

The test should:
1. Create a simple checkerboard pattern
2. Encode it with BC1
3. Compress with Jackal format
4. Decompress and verify output matches input

## Acceptance Criteria

- [ ] The `jackal::roundtrip` test passes successfully
- [ ] Compression and decompression work correctly for BC1 format
- [ ] No regressions in other existing tests
- [ ] Root cause is identified and documented

## Technical Notes

The test creates a simple checkerboard pattern, encodes it with BC1, compresses with Jackal format, then decompresses and verifies the output matches. The failure occurs during decompression, suggesting an issue with the Brotli decompressor or data format mismatch between compress/decompress paths.

Potential areas to investigate:
- Brotli compression parameters
- Data alignment or padding issues
- Stream position/seeking issues
- Header or block metadata

## Dependencies

**Blocks:**
- #2 (CLI implementation)
- #4 (BC6/BC7 support)
- #5 (AnyBlock implementations)

**Blocked by:** None
