/* Reference fixture generator. Compile with iOrange/bcdec's bcdec.h on the
 * include path, then redirect stdout to bc6-reference.bin. */
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#define BCDEC_IMPLEMENTATION
#include "bcdec.h"

int main(void) {
    static const unsigned modes[] = {0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15};
    uint32_t random = 0x6bc6f16u;
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    for (unsigned mode_index = 0; mode_index < 14; ++mode_index) {
        for (unsigned pattern = 0; pattern < 34; ++pattern) {
            /* uint64_t guarantees alignment for the reference decoder. */
            uint64_t storage[2];
            unsigned char *block = (unsigned char *)storage;
            unsigned short decoded[48];
            unsigned mode = modes[mode_index];
            for (unsigned i = 0; i < 16; ++i) {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                block[i] = pattern == 32 ? 0 : pattern == 33 ? 255 : random & 255;
            }
            block[0] = (block[0] & (mode < 2 ? 252 : 224)) | mode;
            if (mode_index < 10) {
                block[9] = (block[9] & 31) | ((pattern & 7) << 5);
                block[10] = (block[10] & 252) | ((pattern >> 3) & 3);
            }
            fwrite(block, 1, 16, stdout);
            for (unsigned signed_mode = 0; signed_mode < 2; ++signed_mode) {
                bcdec_bc6h_half(block, decoded, 12, signed_mode);
                for (unsigned i = 0; i < 48; ++i) {
                    putchar(decoded[i] & 255);
                    putchar(decoded[i] >> 8);
                }
            }
        }
    }
    return ferror(stdout) != 0;
}
