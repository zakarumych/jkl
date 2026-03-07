#ifndef JKL_LZ77_DECODER_H
#define JKL_LZ77_DECODER_H

#include <assert.h>
#include <stdint.h>

#include "error.h"
#include "bits.h"

#define JKLI_LZ77_WINDOW_SIZE 1024u

typedef enum JklLz77TokenKind
{
    JKL_LZ77_TOKEN_LITERAL = 0,
    JKL_LZ77_TOKEN_REFERENCE = 1
} JklLz77TokenKind;

typedef struct JklLz77Token
{
    JklLz77TokenKind kind;
    union
    {
        uint32_t literal;
        struct
        {
            uint32_t length;
            uint32_t distance;
        } reference;
    } v;
} JklLz77Token;

int jkl_read_lz77_token(JklBitReader *reader, JklLz77Token *out_token);

static inline JklLz77Token jkl_lz77_token_delta(
    JklLz77Token me,
    JklLz77Token base)
{
    uint64_t v;
    JklLz77Token out_delta;

    if (me.kind == JKL_LZ77_TOKEN_LITERAL && base.kind == JKL_LZ77_TOKEN_LITERAL)
    {
        assert(me.v.literal >= base.v.literal);
        out_delta.kind = JKL_LZ77_TOKEN_LITERAL;
        out_delta.v.literal = me.v.literal - base.v.literal;
        return out_delta;
    }

    if (me.kind == JKL_LZ77_TOKEN_REFERENCE && base.kind == JKL_LZ77_TOKEN_REFERENCE)
    {
        out_delta.kind = JKL_LZ77_TOKEN_REFERENCE;

        if (me.v.reference.length == base.v.reference.length)
        {
            assert(me.v.reference.distance >= base.v.reference.distance);
            out_delta.v.reference.length = 2;
            out_delta.v.reference.distance = me.v.reference.distance - base.v.reference.distance;
            return out_delta;
        }

        assert(me.v.reference.length >= base.v.reference.length);
        v = (uint64_t)(me.v.reference.length - base.v.reference.length) + 2ULL;
        assert(v <= UINT32_MAX);
        out_delta.v.reference.length = (uint32_t)v;
        out_delta.v.reference.distance = me.v.reference.distance;
        return out_delta;
    }

    return me;
}

static inline JklLz77Token jkl_lz77_token_from_delta(
    JklLz77Token base,
    JklLz77Token delta)
{
    uint64_t v;
    JklLz77Token out_value;

    if (base.kind == JKL_LZ77_TOKEN_LITERAL && delta.kind == JKL_LZ77_TOKEN_LITERAL)
    {
        assert(delta.v.literal <= UINT32_MAX - base.v.literal);
        out_value.kind = JKL_LZ77_TOKEN_LITERAL;
        out_value.v.literal = base.v.literal + delta.v.literal;
        return out_value;
    }

    if (base.kind == JKL_LZ77_TOKEN_LITERAL && delta.kind == JKL_LZ77_TOKEN_REFERENCE)
    {
        return delta;
    }

    if (base.kind == JKL_LZ77_TOKEN_REFERENCE && delta.kind == JKL_LZ77_TOKEN_LITERAL)
    {
        return delta;
    }

    out_value.kind = JKL_LZ77_TOKEN_REFERENCE;

    if (delta.v.reference.length == 2)
    {
        v = (uint64_t)base.v.reference.distance + (uint64_t)delta.v.reference.distance;
        assert(v <= UINT32_MAX);
        out_value.v.reference.length = base.v.reference.length;
        out_value.v.reference.distance = (uint32_t)v;
        return out_value;
    }

    assert(delta.v.reference.length >= 2);
    v = (uint64_t)base.v.reference.length + (uint64_t)(delta.v.reference.length - 2);
    assert(v <= UINT32_MAX);
    out_value.v.reference.length = (uint32_t)v;
    out_value.v.reference.distance = delta.v.reference.distance;
    return out_value;
}

typedef struct JklLz77Decoder
{
    uint32_t window[JKLI_LZ77_WINDOW_SIZE];
    uint32_t head;
    uint32_t pending_literal;
    uint32_t entry_distance;
    uint32_t entry_length;
} JklLz77Decoder;

void jkl_lz77_decoder_init(JklLz77Decoder *decoder);

/* Produces one symbol from decoder state, or JKL_ERR_NEED_TOKEN if state is empty. */
int jkl_lz77_decode(
    JklLz77Decoder *decoder,
    uint32_t *out_symbol);

/* Feeds exactly one token into decoder state. Symbol is produced on next decode call. */
int jkl_lz77_feed_token(
    JklLz77Decoder *decoder,
    JklLz77Token token);

static inline JklLz77Token jkl_lz77_token_default(void)
{
    JklLz77Token out_token;
    out_token.kind = JKL_LZ77_TOKEN_LITERAL;
    out_token.v.literal = 0;
    return out_token;
}

#endif
