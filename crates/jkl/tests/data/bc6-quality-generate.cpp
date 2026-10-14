// Independent BC6H quality corpus generator. See bc6-quality.md for build inputs.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#include "BC.h"

using namespace DirectX;
using namespace DirectX::PackedVector;

static uint32_t random_state = 0xbc6a110fu;
static uint32_t random32() {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}
static int random_code(bool signed_mode) {
    int v = int(random32() % 31744);
    return signed_mode && (random32() & 1) ? -v : v;
}
static uint16_t half_code(int v) {
    v = std::clamp(v, -31743, 31743);
    return uint16_t(v < 0 ? 0x8000 | -v : v);
}
static void little16(FILE* out, uint16_t v) {
    std::fputc(v & 255, out);
    std::fputc(v >> 8, out);
}
static void emit(FILE* out, bool signed_mode, int category, int ordinal,
                 const std::array<uint16_t, 48>& input, const uint8_t* exact = nullptr) {
    XMVECTOR pixels[16], decoded[16];
    uint8_t reference[16];
    for (int i = 0; i < 16; ++i)
        pixels[i] = XMVectorSet(XMConvertHalfToFloat(input[i*3]),
            XMConvertHalfToFloat(input[i*3+1]), XMConvertHalfToFloat(input[i*3+2]), 1);
    if (signed_mode) {
        D3DXEncodeBC6HS(reference, pixels, 0);
        D3DXDecodeBC6HS(decoded, reference);
    } else {
        D3DXEncodeBC6HU(reference, pixels, 0);
        D3DXDecodeBC6HU(decoded, reference);
    }
    std::fputc(signed_mode, out);
    std::fputc(category, out);
    little16(out, uint16_t(ordinal));
    for (auto v : input) little16(out, v);
    std::fwrite(reference, 1, 16, out);
    for (auto v : decoded) {
        little16(out, XMConvertFloatToHalf(XMVectorGetX(v)));
        little16(out, XMConvertFloatToHalf(XMVectorGetY(v)));
        little16(out, XMConvertFloatToHalf(XMVectorGetZ(v)));
    }
    uint8_t zero[16] = {};
    std::fwrite(exact ? exact : zero, 1, 16, out);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: bc6-quality-generate bc6-reference.bin bc6-quality.bin\n");
        return 1;
    }
    FILE* out = std::fopen(argv[2], "wb");
    if (!out) return 2;
    for (int signed_mode = 0; signed_mode < 2; ++signed_mode) {
        for (int category = 0; category < 8; ++category) {
            for (int sample = 0; sample < 8; ++sample) {
                int a[3], b[3], d[3];
                for (int c = 0; c < 3; ++c) {
                    a[c] = random_code(signed_mode);
                    b[c] = random_code(signed_mode);
                    d[c] = random_code(signed_mode);
                }
                std::array<uint16_t, 48> input;
                for (int i = 0; i < 16; ++i) {
                    int x = i % 4, y = i / 4;
                    int t = int(random32() % 16);
                    for (int c = 0; c < 3; ++c) {
                        int v;
                        switch (category) {
                        case 0: // Smooth gradients in logarithmic HDR code space.
                            v = (a[c] * (15-i) + b[c] * i) / 15; break;
                        case 1: // Two spatial color groups, each with local variation.
                            v = (x < 2 ? a[c] : b[c]) + (y-1)*37; break;
                        case 2: // Correlated noise, shared interpolation factor.
                            v = (a[c]*(15-t) + b[c]*t)/15 + int(random32()%257)-128; break;
                        case 3: // Uncorrelated full-range HDR.
                            v = random_code(signed_mode); break;
                        case 4: // Cross-channel three-cluster adversary.
                            v = (i%3 == 0 ? a[c] : i%3 == 1 ? b[c] : d[c]); break;
                        case 5: // Near-zero subnormals and sign crossings.
                            v = int(random32()%1024) * (signed_mode && (i+c)%2 ? -1 : 1); break;
                        case 6: // High precision, narrow bright range.
                            v = (signed_mode && c%2 ? -1 : 1) * (29000 + sample*300 + int(random32()%129)); break;
                        default: // Smooth gradients in linear-light HDR.
                            float av = XMConvertHalfToFloat(half_code(a[c]));
                            float bv = XMConvertHalfToFloat(half_code(b[c]));
                            v = 0;
                            input[i*3+c] = XMConvertFloatToHalf(av + (bv-av)*(i/15.0f));
                            continue;
                        }
                        input[i*3+c] = half_code(v);
                    }
                }
                emit(out, signed_mode, category, sample, input);
            }
        }
    }
    FILE* original = std::fopen(argv[1], "rb");
    if (!original) return 3;
    for (int mode = 0; mode < 14; ++mode) {
        for (int pattern : {0, 17}) {
            uint8_t source[208];
            std::fseek(original, (mode*34+pattern)*208, SEEK_SET);
            if (std::fread(source, 1, sizeof(source), original) != sizeof(source)) return 4;
            for (int signed_mode = 0; signed_mode < 2; ++signed_mode) {
                std::array<uint16_t, 48> input;
                for (int i = 0; i < 48; ++i) {
                    int offset = 16 + signed_mode*96 + i*2;
                    input[i] = uint16_t(source[offset] | (source[offset+1]<<8));
                }
                emit(out, signed_mode, 8, mode*2 + (pattern == 17), input, source);
            }
        }
    }
    bool failed = std::ferror(out);
    std::fclose(original);
    failed |= std::fclose(out) != 0;
    return failed ? 5 : 0;
}
