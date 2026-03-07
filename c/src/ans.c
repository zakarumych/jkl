#include "ans.h"

#include <stdlib.h>

#include "bit_reader.h"
#include "elias.h"

static int jkl_checked_add_u64(uint64_t a, uint64_t b, uint64_t *out_value)
{
    if (a > UINT64_MAX - b)
    {
        return JKL_ERR_TOO_LARGE;
    }

    *out_value = a + b;
    return JKL_OK;
}

#define JKL_DEFINE_FIND_INDEX(name, CtxType)           \
    static size_t name(const CtxType *ctx, uint64_t c) \
    {                                                  \
        size_t lo = 0;                                 \
        size_t hi = ctx->len;                          \
                                                       \
        while (lo < hi)                                \
        {                                              \
            size_t mid = lo + (hi - lo) / 2;           \
            if (ctx->entries[mid].cumul == c)          \
            {                                          \
                return mid;                            \
            }                                          \
            if (ctx->entries[mid].cumul < c)           \
            {                                          \
                lo = mid + 1;                          \
            }                                          \
            else                                       \
            {                                          \
                hi = mid;                              \
            }                                          \
        }                                              \
                                                       \
        return lo ? lo - 1 : 0;                        \
    }

JKL_DEFINE_FIND_INDEX(jkl_ans_find_index_32, JklAnsContext32)
JKL_DEFINE_FIND_INDEX(jkl_ans_find_index_lz77, JklAnsContextLz77)
JKL_DEFINE_FIND_INDEX(jkl_ans_find_index_rle, JklAnsContextRle)

void jkl_ans_decoder_init_32(JklAnsDecoder32 *decoder, const JklAnsContext32 *ctx)
{
    decoder->ctx = ctx;
    decoder->state = 0;
}

void jkl_ans_decoder_init_lz77(JklAnsDecoderLz77 *decoder, const JklAnsContextLz77 *ctx)
{
    decoder->ctx = ctx;
    decoder->state = 0;
}

void jkl_ans_decoder_init_rle(JklAnsDecoderRle *decoder, const JklAnsContextRle *ctx)
{
    decoder->ctx = ctx;
    decoder->state = 0;
}

int jkl_ans_feed_token_32(JklAnsDecoder32 *decoder, uint32_t token)
{
    if (decoder->state >= 0x80000000ULL)
    {
        return JKL_ERR_INVALID_DATA;
    }

    decoder->state = (decoder->state << 32) | token;
    return JKL_OK;
}

int jkl_ans_feed_token_lz77(JklAnsDecoderLz77 *decoder, uint32_t token)
{
    if (decoder->state >= 0x80000000ULL)
    {
        return JKL_ERR_INVALID_DATA;
    }

    decoder->state = (decoder->state << 32) | token;
    return JKL_OK;
}

int jkl_ans_feed_token_rle(JklAnsDecoderRle *decoder, uint32_t token)
{
    if (decoder->state >= 0x80000000ULL)
    {
        return JKL_ERR_INVALID_DATA;
    }

    decoder->state = (decoder->state << 32) | token;
    return JKL_OK;
}

int jkl_ans_decode_32(JklAnsDecoder32 *decoder, uint32_t *out_symbol)
{
    uint64_t c;
    size_t index;
    JklAnsEntry32 *entry;

    if (decoder->ctx->len == 0 || decoder->ctx->total == 0)
    {
        return JKL_ERR_INVALID_DATA;
    }

    if (decoder->state < 0x80000000ULL)
    {
        return JKL_ERR_NEED_TOKEN;
    }

    c = decoder->state % decoder->ctx->total;
    index = jkl_ans_find_index_32(decoder->ctx, c);
    if (index >= decoder->ctx->len)
    {
        return JKL_ERR_INVALID_DATA;
    }

    entry = &decoder->ctx->entries[index];
    if (entry->freq == 0 || c < entry->cumul)
    {
        return JKL_ERR_INVALID_DATA;
    }

    *out_symbol = entry->symbol;
    decoder->state =
        (decoder->state / decoder->ctx->total) * entry->freq +
        c - entry->cumul;

    return JKL_OK;
}

int jkl_ans_decode_lz77(JklAnsDecoderLz77 *decoder, JklLz77Token *out_symbol)
{
    uint64_t c;
    size_t index;
    JklAnsEntryLz77 *entry;

    if (decoder->ctx->len == 0 || decoder->ctx->total == 0)
    {
        return JKL_ERR_INVALID_DATA;
    }

    if (decoder->state < 0x80000000ULL)
    {
        return JKL_ERR_NEED_TOKEN;
    }

    c = decoder->state % decoder->ctx->total;
    index = jkl_ans_find_index_lz77(decoder->ctx, c);
    if (index >= decoder->ctx->len)
    {
        return JKL_ERR_INVALID_DATA;
    }

    entry = &decoder->ctx->entries[index];
    if (entry->freq == 0 || c < entry->cumul)
    {
        return JKL_ERR_INVALID_DATA;
    }

    *out_symbol = entry->symbol;
    decoder->state =
        (decoder->state / decoder->ctx->total) * entry->freq +
        c - entry->cumul;

    return JKL_OK;
}

int jkl_ans_decode_rle(JklAnsDecoderRle *decoder, JklRleToken *out_symbol)
{
    uint64_t c;
    size_t index;
    JklAnsEntryRle *entry;

    if (decoder->ctx->len == 0 || decoder->ctx->total == 0)
    {
        return JKL_ERR_INVALID_DATA;
    }

    if (decoder->state < 0x80000000ULL)
    {
        return JKL_ERR_NEED_TOKEN;
    }

    c = decoder->state % decoder->ctx->total;
    index = jkl_ans_find_index_rle(decoder->ctx, c);
    if (index >= decoder->ctx->len)
    {
        return JKL_ERR_INVALID_DATA;
    }

    entry = &decoder->ctx->entries[index];
    if (entry->freq == 0 || c < entry->cumul)
    {
        return JKL_ERR_INVALID_DATA;
    }

    *out_symbol = entry->symbol;
    decoder->state =
        (decoder->state / decoder->ctx->total) * entry->freq +
        c - entry->cumul;

    return JKL_OK;
}

int jkl_ans_context_read_32(FILE *fp, JklAnsContext32 *out_ctx)
{
    JklBitReader br;
    uint64_t len_u64;
    size_t len;
    size_t i;
    uint64_t cumul = 0;
    uint32_t last = 0;

    out_ctx->entries = NULL;
    out_ctx->len = 0;
    out_ctx->total = 0;

    JKL_RETURN_IF_ERROR(jkl_bit_reader_init_file(fp, &br));
    JKL_RETURN_IF_ERROR(jkl_elias_delta_decode(&br, &len_u64));

    if (len_u64 > SIZE_MAX)
    {
        return JKL_ERR_TOO_LARGE;
    }
    len = (size_t)len_u64;

    if (len > 0)
    {
        out_ctx->entries = (JklAnsEntry32 *)malloc(len * sizeof(JklAnsEntry32));
        if (out_ctx->entries == NULL)
        {
            return JKL_ERR_OOM;
        }
    }

    out_ctx->len = len;

    for (i = 0; i < len; ++i)
    {
        uint64_t count;
        uint32_t delta;
        uint32_t symbol;

        JKL_RETURN_IF_ERROR(jkl_elias_delta_decode(&br, &count));
        JKL_RETURN_IF_ERROR(jkl_elias_delta_decode(&br, &delta));

        if (delta > UINT32_MAX - last)
        {
            jkl_ans_context_free_32(out_ctx);
            return JKL_ERR_TOO_LARGE;
        }
        symbol = last + delta;

        out_ctx->entries[i].cumul = cumul;
        out_ctx->entries[i].freq = count;
        out_ctx->entries[i].symbol = symbol;

        last = symbol;
        if (jkl_checked_add_u64(cumul, count, &cumul) != JKL_OK)
        {
            jkl_ans_context_free_32(out_ctx);
            return JKL_ERR_TOO_LARGE;
        }
    }

    if (len == 1)
    {
        out_ctx->entries[0].freq = 1;
        cumul = 2;
    }

    if (cumul >= 0x80000000ULL)
    {
        jkl_ans_context_free_32(out_ctx);
        return JKL_ERR_INVALID_DATA;
    }

    out_ctx->total = cumul;
    return JKL_OK;
}

int jkl_ans_context_read_lz77(FILE *fp, JklAnsContextLz77 *out_ctx)
{
    JklBitReader br;
    uint64_t len_u64;
    size_t len;
    size_t i;
    uint64_t cumul = 0;
    JklLz77Token last;

    out_ctx->entries = NULL;
    out_ctx->len = 0;
    out_ctx->total = 0;

    last = jkl_lz77_token_default();
    JKL_RETURN_IF_ERROR(jkl_bit_reader_init_file(fp, &br));
    JKL_RETURN_IF_ERROR(jkl_elias_delta_decode(&br, &len_u64));

    if (len_u64 > SIZE_MAX)
    {
        return JKL_ERR_TOO_LARGE;
    }
    len = (size_t)len_u64;

    if (len > 0)
    {
        out_ctx->entries = (JklAnsEntryLz77 *)malloc(len * sizeof(JklAnsEntryLz77));
        if (out_ctx->entries == NULL)
        {
            return JKL_ERR_OOM;
        }
    }

    out_ctx->len = len;

    for (i = 0; i < len; ++i)
    {
        uint64_t count;
        JklLz77Token delta;
        JklLz77Token symbol;

        JKL_RETURN_IF_ERROR(jkl_elias_delta_decode(&br, &count));
        JKL_RETURN_IF_ERROR(jkl_read_lz77_token(&br, &delta));
        symbol = jkl_lz77_token_from_delta(last, delta);

        out_ctx->entries[i].cumul = cumul;
        out_ctx->entries[i].freq = count;
        out_ctx->entries[i].symbol = symbol;

        last = symbol;
        if (jkl_checked_add_u64(cumul, count, &cumul) != JKL_OK)
        {
            jkl_ans_context_free_lz77(out_ctx);
            return JKL_ERR_TOO_LARGE;
        }
    }

    if (len == 1)
    {
        out_ctx->entries[0].freq = 1;
        cumul = 2;
    }

    if (cumul >= 0x80000000ULL)
    {
        jkl_ans_context_free_lz77(out_ctx);
        return JKL_ERR_INVALID_DATA;
    }

    out_ctx->total = cumul;
    return JKL_OK;
}

int jkl_ans_context_read_rle(FILE *fp, JklAnsContextRle *out_ctx)
{
    JklBitReader br;
    uint64_t len_u64;
    size_t len;
    size_t i;
    uint64_t cumul = 0;
    JklRleToken last;

    out_ctx->entries = NULL;
    out_ctx->len = 0;
    out_ctx->total = 0;

    last = jkl_rle_token_default();
    JKL_RETURN_IF_ERROR(jkl_bit_reader_init_file(fp, &br));
    JKL_RETURN_IF_ERROR(jkl_elias_delta_decode(&br, &len_u64));

    if (len_u64 > SIZE_MAX)
    {
        return JKL_ERR_TOO_LARGE;
    }
    len = (size_t)len_u64;

    if (len > 0)
    {
        out_ctx->entries = (JklAnsEntryRle *)malloc(len * sizeof(JklAnsEntryRle));
        if (out_ctx->entries == NULL)
        {
            return JKL_ERR_OOM;
        }
    }

    out_ctx->len = len;

    for (i = 0; i < len; ++i)
    {
        uint64_t count;
        JklRleToken delta;
        JklRleToken symbol;

        JKL_RETURN_IF_ERROR(jkl_elias_delta_decode(&br, &count));
        JKL_RETURN_IF_ERROR(jkl_read_rle_token(&br, &delta));
        symbol = jkl_rle_token_from_delta(last, delta);

        out_ctx->entries[i].cumul = cumul;
        out_ctx->entries[i].freq = count;
        out_ctx->entries[i].symbol = symbol;

        last = symbol;
        if (jkl_checked_add_u64(cumul, count, &cumul) != JKL_OK)
        {
            jkl_ans_context_free_rle(out_ctx);
            return JKL_ERR_TOO_LARGE;
        }
    }

    if (len == 1)
    {
        out_ctx->entries[0].freq = 1;
        cumul = 2;
    }

    if (cumul >= 0x80000000ULL)
    {
        jkl_ans_context_free_rle(out_ctx);
        return JKL_ERR_INVALID_DATA;
    }

    out_ctx->total = cumul;
    return JKL_OK;
}

void jkl_ans_context_free_32(JklAnsContext32 *ctx)
{
    free(ctx->entries);
    ctx->entries = NULL;
    ctx->len = 0;
    ctx->total = 0;
}

void jkl_ans_context_free_lz77(JklAnsContextLz77 *ctx)
{
    free(ctx->entries);
    ctx->entries = NULL;
    ctx->len = 0;
    ctx->total = 0;
}

void jkl_ans_context_free_rle(JklAnsContextRle *ctx)
{
    free(ctx->entries);
    ctx->entries = NULL;
    ctx->len = 0;
    ctx->total = 0;
}
