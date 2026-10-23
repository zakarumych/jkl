// Independent reference corpus. Compile with bc7enc.cpp and bcdec.h.
// Usage: generate-bc7-quality OUTPUT.bin
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include "bc7enc.h"
#define BCDEC_IMPLEMENTATION
#include "bcdec.h"

static uint32_t state = 0xbc72026u;
static uint32_t rng() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static uint8_t clamp(int x) { return (uint8_t)std::max(0, std::min(255, x)); }

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    FILE *out = fopen(argv[1], "wb"); if (!out) return 2;
    bc7enc_compress_block_init();
    bc7enc_compress_block_params p;
    bc7enc_compress_block_params_init(&p);
    bc7enc_compress_block_params_init_linear_weights(&p);
    p.m_uber_level = 4;
    p.m_max_partitions = 64;
    p.m_mode17_partition_estimation_filterbank = false;
    for (unsigned category = 0; category < 8; ++category)
        for (unsigned ordinal = 0; ordinal < 8; ++ordinal) {
            uint8_t source[64], decoded[64]; uint64_t block[2];
            int endpoints[3][4];
            for (auto &endpoint : endpoints) for (auto &c : endpoint) c = rng() & 255;
            for (unsigned i = 0; i < 16; ++i) {
                unsigned x = i%4, y = i/4;
                unsigned group = category == 2 ? x/2 : ((x/2) + (y/2)) % 3;
                int correlated = int(rng() % 65) - 32;
                for (unsigned c = 0; c < 4; ++c) {
                    int a = endpoints[0][c], b = endpoints[1][c], v;
                    switch (category) {
                    case 0: v = a; break;
                    case 1: v = (a*int(15-i)+b*int(i)+7)/15; break;
                    case 2: case 3: v = endpoints[group][c] + int(rng()%9)-4; break;
                    case 4: v = c == 3 ? int(rng()&255) : (a*int(15-i)+b*int(i)+7)/15; break;
                    case 5: v = a + correlated + int(rng()%9)-4; break;
                    case 6: v = rng() & 255; break;
                    default: v = 0; break;
                    }
                    source[i*4+c] = clamp(v);
                }
                if (category == 2 || category == 3 || (category == 1 && ordinal < 4)) source[i*4+3] = 255;
            }
            if (category == 7) {
                for (auto &word : block) word = uint64_t(rng()) | (uint64_t(rng()) << 32);
                unsigned char *bytes = (unsigned char *)block;
                bytes[0] = (bytes[0] & ~((1u << (ordinal+1))-1u)) | (1u << ordinal);
                bcdec_bc7(block, source, 16);
            }
            bc7enc_compress_block(block, source, &p);
            bcdec_bc7(block, decoded, 16);
            uint8_t header[] = {(uint8_t)category, (uint8_t)ordinal, 0, 0};
            if (fwrite(header,1,4,out)!=4 || fwrite(source,1,64,out)!=64 ||
                fwrite(block,1,16,out)!=16 || fwrite(decoded,1,64,out)!=64) return 3;
        }
    return fclose(out) != 0;
}
