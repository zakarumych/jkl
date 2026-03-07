#ifndef JKL_RLE_DECODER_H
#define JKL_RLE_DECODER_H

#include <assert.h>
#include <stdint.h>

#include "error.h"
#include "bit_reader.h"

typedef struct JklRleToken {
    uint32_t value;
    uint64_t count;
} JklRleToken;

int jkl_read_rle_token(JklBitReader *reader, JklRleToken *out_token);

typedef struct JklRleDecoder {
    uint64_t run_remaining;
    uint32_t run_value;
} JklRleDecoder;

void jkl_rle_decoder_init(JklRleDecoder *decoder);

int jkl_rle_decode(
    JklRleDecoder *decoder,
    uint32_t *out_symbol);

int jkl_rle_feed_token(
    JklRleDecoder *decoder,
    JklRleToken token);

static inline JklRleToken jkl_rle_token_default(void) {
    JklRleToken out_token;
    out_token.value = 0;
    out_token.count = 1;
    return out_token;
}

static inline JklRleToken jkl_rle_token_delta(
    JklRleToken me,
    JklRleToken base) {
    JklRleToken out_delta;
    assert(me.value >= base.value);
    out_delta.value = me.value - base.value;
    out_delta.count = me.count;
    return out_delta;
}

static inline JklRleToken jkl_rle_token_from_delta(
    JklRleToken base,
    JklRleToken delta) {
    JklRleToken out_value;
    assert(delta.value <= UINT32_MAX - base.value);
    out_value.value = base.value + delta.value;
    out_value.count = delta.count;
    assert(out_value.count != 0);
    return out_value;
}

#endif
