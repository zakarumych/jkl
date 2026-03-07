#ifndef JKL_ANS_H
#define JKL_ANS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "error.h"
#include "lz77.h"
#include "rle.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct JklAnsEntry32 {
    uint64_t cumul;
    uint64_t freq;
    uint32_t symbol;
} JklAnsEntry32;

typedef struct JklAnsEntryLz77 {
    uint64_t cumul;
    uint64_t freq;
    JklLz77Token symbol;
} JklAnsEntryLz77;

typedef struct JklAnsEntryRle {
    uint64_t cumul;
    uint64_t freq;
    JklRleToken symbol;
} JklAnsEntryRle;

typedef struct JklAnsContext32 {
    JklAnsEntry32 *entries;
    size_t len;
    uint64_t total;
} JklAnsContext32;

typedef struct JklAnsContextLz77 {
    JklAnsEntryLz77 *entries;
    size_t len;
    uint64_t total;
} JklAnsContextLz77;

typedef struct JklAnsContextRle {
    JklAnsEntryRle *entries;
    size_t len;
    uint64_t total;
} JklAnsContextRle;

typedef struct JklAnsDecoder32 {
    const JklAnsContext32 *ctx;
    uint64_t state;
} JklAnsDecoder32;

typedef struct JklAnsDecoderLz77 {
    const JklAnsContextLz77 *ctx;
    uint64_t state;
} JklAnsDecoderLz77;

typedef struct JklAnsDecoderRle {
    const JklAnsContextRle *ctx;
    uint64_t state;
} JklAnsDecoderRle;

void jkl_ans_decoder_init_32(JklAnsDecoder32 *decoder, const JklAnsContext32 *ctx);
void jkl_ans_decoder_init_lz77(JklAnsDecoderLz77 *decoder, const JklAnsContextLz77 *ctx);
void jkl_ans_decoder_init_rle(JklAnsDecoderRle *decoder, const JklAnsContextRle *ctx);

int jkl_ans_decode_32(JklAnsDecoder32 *decoder, uint32_t *out_symbol);
int jkl_ans_decode_lz77(JklAnsDecoderLz77 *decoder, JklLz77Token *out_symbol);
int jkl_ans_decode_rle(JklAnsDecoderRle *decoder, JklRleToken *out_symbol);

int jkl_ans_feed_token_32(JklAnsDecoder32 *decoder, uint32_t token);
int jkl_ans_feed_token_lz77(JklAnsDecoderLz77 *decoder, uint32_t token);
int jkl_ans_feed_token_rle(JklAnsDecoderRle *decoder, uint32_t token);

int jkl_ans_context_read_32(FILE *fp, JklAnsContext32 *out_ctx);
int jkl_ans_context_read_lz77(FILE *fp, JklAnsContextLz77 *out_ctx);
int jkl_ans_context_read_rle(FILE *fp, JklAnsContextRle *out_ctx);

void jkl_ans_context_free_32(JklAnsContext32 *ctx);
void jkl_ans_context_free_lz77(JklAnsContextLz77 *ctx);
void jkl_ans_context_free_rle(JklAnsContextRle *ctx);

#ifdef __cplusplus
} /* extern "C" */
#else
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define jkl_ans_decoder_init(decoder, ctx) _Generic((decoder),                \
    JklAnsDecoder32 *: jkl_ans_decoder_init_32,                               \
    JklAnsDecoderLz77 *: jkl_ans_decoder_init_lz77,                           \
    JklAnsDecoderRle *: jkl_ans_decoder_init_rle)(decoder, ctx)

#define jkl_ans_decode(decoder, out_symbol) _Generic((decoder),               \
    JklAnsDecoder32 *: jkl_ans_decode_32,                                      \
    JklAnsDecoderLz77 *: jkl_ans_decode_lz77,                                  \
    JklAnsDecoderRle *: jkl_ans_decode_rle)(decoder, out_symbol)

#define jkl_ans_feed_token(decoder, token) _Generic((decoder),                \
    JklAnsDecoder32 *: jkl_ans_feed_token_32,                                  \
    JklAnsDecoderLz77 *: jkl_ans_feed_token_lz77,                              \
    JklAnsDecoderRle *: jkl_ans_feed_token_rle)(decoder, token)

#define jkl_ans_context_read(fp, out_ctx) _Generic((out_ctx),                 \
    JklAnsContext32 *: jkl_ans_context_read_32,                                \
    JklAnsContextLz77 *: jkl_ans_context_read_lz77,                            \
    JklAnsContextRle *: jkl_ans_context_read_rle)(fp, out_ctx)

#define jkl_ans_context_free(ctx) _Generic((ctx),                             \
    JklAnsContext32 *: jkl_ans_context_free_32,                                \
    JklAnsContextLz77 *: jkl_ans_context_free_lz77,                            \
    JklAnsContextRle *: jkl_ans_context_free_rle)(ctx)
#endif
#endif

#endif
