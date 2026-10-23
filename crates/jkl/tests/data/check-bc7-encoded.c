/* Check JKL-emitted 80-byte records against independent bcdec.
 * Usage: check-bc7-encoded target/bc7-encoded.bin */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define BCDEC_IMPLEMENTATION
#include "bcdec.h"
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    FILE *in = fopen(argv[1], "rb"); if (!in) return 2;
    uint64_t block[2]; unsigned char expected[64], actual[64]; unsigned count = 0;
    size_t bytes;
    while ((bytes = fread(block,1,16,in)) == 16) {
        if (fread(expected,1,64,in) != 64) return 3;
        bcdec_bc7(block, actual, 16);
        if (memcmp(actual, expected, 64)) { fprintf(stderr, "mismatch block %u\n", count); return 4; }
        ++count;
    }
    if (bytes != 0 || ferror(in) || fclose(in)) return 5;
    printf("Independent decoder agrees on %u JKL blocks.\n", count);
    return count ? 0 : 6;
}
