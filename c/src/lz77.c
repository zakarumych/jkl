#include "jkl/lz77.h"

#include <string.h>

#include "jkl/elias.h"

static uint32_t jkl_lz77_window_get(const JklLz77Decoder *decoder, uint32_t index)
{
    uint32_t idx = (decoder->head + JKLI_LZ77_WINDOW_SIZE - 1U - index) % JKLI_LZ77_WINDOW_SIZE;
    return decoder->window[idx];
}

static void jkl_lz77_window_push(JklLz77Decoder *decoder, uint32_t value)
{
    decoder->window[decoder->head] = value;
    decoder->head = (decoder->head + 1U) % JKLI_LZ77_WINDOW_SIZE;
}

void jkl_lz77_decoder_init(JklLz77Decoder *decoder)
{
    memset(decoder->window, 0, sizeof(decoder->window));
    decoder->head = 0;
    decoder->pending_literal = 0;
    decoder->entry_distance = 0;
    decoder->entry_length = 0;
}

int jkl_lz77_decode(
    JklLz77Decoder *decoder,
    uint32_t *out_symbol)
{
    if (decoder->entry_distance == JKLI_LZ77_WINDOW_SIZE)
    {
        decoder->entry_distance = 0;
        *out_symbol = decoder->pending_literal;
        jkl_lz77_window_push(decoder, decoder->pending_literal);
        return JKL_OK;
    }
    assert(decoder->entry_distance < JKLI_LZ77_WINDOW_SIZE);

    if (decoder->entry_length > 0)
    {
        decoder->entry_length -= 1;
        uint32_t literal = jkl_lz77_window_get(decoder, decoder->entry_distance);
        *out_symbol = literal;
        jkl_lz77_window_push(decoder, literal);
        return JKL_OK;
    }

    return JKL_ERR_NEED_TOKEN;
}

int jkl_lz77_feed_token(
    JklLz77Decoder *decoder,
    JklLz77Token token)
{
    assert(decoder->entry_length == 0);
    assert(decoder->entry_distance < JKLI_LZ77_WINDOW_SIZE);

    if (token.kind == JKL_LZ77_TOKEN_LITERAL)
    {
        decoder->pending_literal = token.v.literal;
        decoder->entry_distance = JKLI_LZ77_WINDOW_SIZE;
        return JKL_OK;
    }

    if (token.v.reference.distance >= JKLI_LZ77_WINDOW_SIZE)
    {
        return JKL_ERR_INVALID_DATA;
    }

    if (token.v.reference.length == 0)
    {
        return JKL_ERR_INVALID_DATA;
    }

    decoder->entry_distance = token.v.reference.distance;
    decoder->entry_length = token.v.reference.length;

    return JKL_OK;
}

int jkl_read_lz77_token(JklBitReader *reader, JklLz77Token *out_token)
{
    uint32_t length;
    uint32_t distance;

    JKL_RETURN_IF_ERROR(jkl_elias_delta_decode_nonzero(reader, &length));

    if (length == 1)
    {
        out_token->kind = JKL_LZ77_TOKEN_LITERAL;
        return jkl_elias_delta_decode(reader, &out_token->v.literal);
    }

    if (length > UINT32_MAX)
    {
        return JKL_ERR_TOO_LARGE;
    }

    JKL_RETURN_IF_ERROR(jkl_elias_delta_decode(reader, &distance));
    if (distance >= UINT32_MAX)
    {
        return JKL_ERR_TOO_LARGE;
    }

    out_token->kind = JKL_LZ77_TOKEN_REFERENCE;
    out_token->v.reference.length = length;
    out_token->v.reference.distance = distance;
    return JKL_OK;
}
