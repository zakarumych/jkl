#include "rle.h"

#include "elias.h"

int jkl_read_rle_token(JklBitReader *reader, JklRleToken *out_token)
{
    uint64_t count;

    JKL_RETURN_IF_ERROR(jkl_elias_delta_decode(reader, &out_token->value));
    JKL_RETURN_IF_ERROR(jkl_elias_delta_decode_nonzero(reader, &count));
    out_token->count = count;
    return JKL_OK;
}

void jkl_rle_decoder_init(JklRleDecoder *decoder)
{
    decoder->run_remaining = 0;
    decoder->run_value = 0;
}

int jkl_rle_decode(JklRleDecoder *decoder, uint32_t *out_symbol)
{
    if (decoder->run_remaining > 0)
    {
        decoder->run_remaining -= 1;
        *out_symbol = decoder->run_value;
        return JKL_OK;
    }

    return JKL_ERR_NEED_TOKEN;
}

int jkl_rle_feed_token(JklRleDecoder *decoder, JklRleToken token)
{
    if (decoder->run_remaining > 0)
    {
        return JKL_ERR_INVALID_DATA;
    }

    if (token.count == 0)
    {
        return JKL_ERR_INVALID_DATA;
    }

    decoder->run_remaining = token.count;
    decoder->run_value = token.value;

    return JKL_OK;
}
