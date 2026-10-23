/* Independent fixtures: compile with iOrange/bcdec bcdec.h on include path.
 * Usage: generate-bc7-reference OUTPUT.bin */
#include <stdint.h>
#include <stdio.h>
#define BCDEC_IMPLEMENTATION
#include "bcdec.h"

static uint32_t state = 0xb7c72026u;
static uint32_t random_u32(void) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
static void bits(unsigned char *b, unsigned start, unsigned count, unsigned value) {
    for (unsigned i = 0; i < count; ++i)
        b[(start+i)/8] = (b[(start+i)/8] & ~(1u << ((start+i)%8))) |
            (((value >> i) & 1u) << ((start+i)%8));
}
static int emit(FILE *out, uint64_t *storage) {
    unsigned char rgba[64];
    bcdec_bc7(storage, rgba, 16);
    return fwrite(storage, 1, 16, out) == 16 && fwrite(rgba, 1, 64, out) == 64;
}
int main(int argc, char **argv) {
    static const unsigned partitions[] = {16,64,64,64,1,1,1,64};
    static const unsigned pbit_count[] = {6,2,0,4,0,0,2,4};
    static const unsigned pbit_start[] = {77,80,0,94,0,0,63,94};
    if (argc != 2) return 1;
    FILE *out = fopen(argv[1], "wb");
    if (!out) return 2;
    for (unsigned mode = 0; mode < 8; ++mode) {
        unsigned variants = mode == 4 ? 8 : mode == 5 ? 4 : 1;
        for (unsigned part = 0; part < partitions[mode]; ++part)
            for (unsigned variant = 0; variant < variants; ++variant)
                for (unsigned pattern = 0; pattern < 8; ++pattern) {
                    uint64_t storage[2]; unsigned char *b = (unsigned char *)storage;
                    for (unsigned i = 0; i < 16; ++i)
                        b[i] = pattern == 0 ? 0 : pattern == 1 ? 255 :
                            pattern == 2 ? 0xaa : pattern == 3 ? 0x55 : random_u32() & 255;
                    bits(b, 0, mode + 1, 1u << mode);
                    if (partitions[mode] > 1) bits(b, mode+1, mode == 0 ? 4 : 6, part);
                    if (mode == 4 || mode == 5) bits(b, mode+1, 2, variant & 3);
                    if (mode == 4) bits(b, mode+3, 1, variant >> 2);
                    if (!emit(out, storage)) return 3;
                }
        for (unsigned pbits = 0; pbit_count[mode] && pbits < (1u << pbit_count[mode]); ++pbits) {
            uint64_t storage[2]; unsigned char *b = (unsigned char *)storage;
            for (unsigned i = 0; i < 16; ++i) b[i] = random_u32() & 255;
            bits(b, 0, mode+1, 1u << mode);
            bits(b, pbit_start[mode], pbit_count[mode], pbits);
            if (!emit(out, storage)) return 3;
        }
        for (unsigned n = 0; n < 256; ++n) {
            uint64_t storage[2]; unsigned char *b = (unsigned char *)storage;
            for (unsigned i = 0; i < 16; ++i) b[i] = random_u32() & 255;
            bits(b, 0, mode+1, 1u << mode);
            if (!emit(out, storage)) return 3;
        }
    }
    for (unsigned n = 0; n < 16; ++n) {
        uint64_t storage[2]; unsigned char *b = (unsigned char *)storage;
        for (unsigned i = 0; i < 16; ++i) b[i] = random_u32() & 255;
        b[0] = 0;
        if (!emit(out, storage)) return 3;
    }
    return fclose(out) != 0;
}
