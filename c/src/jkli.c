#include "jkli.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "ans.h"
#include "bit_reader.h"
#include "elias.h"
#include "lz77.h"
#include "rle.h"

#if !defined(_WIN32)
#include <sys/types.h>
#endif

#define JKLI_MAGIC 0x494C4B4Au

typedef union JkliContext
{
    JklAnsContext32 ans;
    JklAnsContextLz77 lz77_ans;
    JklAnsContextRle rle_ans;
} JkliContext;

struct JkliFile
{
    FILE *fp;
    int owns_fp;

    JkliCompression compression;
    Format format;
    Extent extent;
    TileSize tile_size;

    uint64_t *offsets;

    JkliContext context;
};

static int jkl_read_u8(FILE *fp, uint8_t *out_value)
{
    int c;

    c = fgetc(fp);
    if (c == EOF)
    {
        if (ferror(fp) != 0)
        {
            return JKL_ERR_IO;
        }
        return JKL_ERR_EOF;
    }

    *out_value = (uint8_t)c;
    return JKL_OK;
}

static int jkl_read_exact(FILE *fp, uint8_t *buffer, size_t size)
{
    if (size == 0)
    {
        return JKL_OK;
    }

    if (fread(buffer, 1, size, fp) != size)
    {
        if (ferror(fp) != 0)
        {
            return JKL_ERR_IO;
        }
        return JKL_ERR_EOF;
    }

    return JKL_OK;
}

static int jkl_read_u16_le(FILE *fp, uint16_t *out_value)
{
    uint8_t b[2];
    JKL_RETURN_IF_ERROR(jkl_read_exact(fp, b, 2));
    *out_value = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
    return JKL_OK;
}

static int jkl_read_u32_le(FILE *fp, uint32_t *out_value)
{
    uint8_t b[4];
    JKL_RETURN_IF_ERROR(jkl_read_exact(fp, b, 4));
    *out_value = (uint32_t)b[0] |
                 ((uint32_t)b[1] << 8) |
                 ((uint32_t)b[2] << 16) |
                 ((uint32_t)b[3] << 24);
    return JKL_OK;
}

static int jkl_read_u64_le(FILE *fp, uint64_t *out_value)
{
    uint8_t b[8];
    JKL_RETURN_IF_ERROR(jkl_read_exact(fp, b, 8));
    *out_value = (uint64_t)b[0] |
                 ((uint64_t)b[1] << 8) |
                 ((uint64_t)b[2] << 16) |
                 ((uint64_t)b[3] << 24) |
                 ((uint64_t)b[4] << 32) |
                 ((uint64_t)b[5] << 40) |
                 ((uint64_t)b[6] << 48) |
                 ((uint64_t)b[7] << 56);
    return JKL_OK;
}

static uint32_t jkl_compact8_3(uint32_t x)
{
    x = x & 0x249249u;
    x = (x | (x >> 2)) & 0x0C30C3u;
    x = (x | (x >> 4)) & 0x00F00Fu;
    x = (x | (x >> 8)) & 0x00000FFFu;
    return x & 0xFFu;
}

static void jkl_rgb_from_bits_interleaved(uint32_t bits, uint8_t out_rgb[3])
{
    uint8_t r = (uint8_t)jkl_compact8_3(bits);
    uint8_t b = (uint8_t)jkl_compact8_3(bits >> 1);
    uint8_t g = (uint8_t)jkl_compact8_3(bits >> 2);

    out_rgb[0] = r;
    out_rgb[1] = g;
    out_rgb[2] = b;
}

static void jkl_free_context(JkliFile *file)
{
    switch (file->compression)
    {
    case JKLI_COMPRESSION_ANS:
        jkl_ans_context_free_32(&file->context.ans);
        break;
    case JKLI_COMPRESSION_LZ77_ANS:
        jkl_ans_context_free_lz77(&file->context.lz77_ans);
        break;
    case JKLI_COMPRESSION_RLE_ANS:
        jkl_ans_context_free_rle(&file->context.rle_ans);
        break;
    case JKLI_COMPRESSION_NONE:
    case JKLI_COMPRESSION_LZ77:
    default:
        break;
    }
}

static int jkl_read_extent_from_header(Extent *extent, const uint8_t raw_extent[25])
{
    uint8_t d;
    uint64_t w;
    uint64_t h;
    uint64_t z;

    d = raw_extent[0];
    w = (uint64_t)raw_extent[1] |
        ((uint64_t)raw_extent[2] << 8) |
        ((uint64_t)raw_extent[3] << 16) |
        ((uint64_t)raw_extent[4] << 24) |
        ((uint64_t)raw_extent[5] << 32) |
        ((uint64_t)raw_extent[6] << 40) |
        ((uint64_t)raw_extent[7] << 48) |
        ((uint64_t)raw_extent[8] << 56);

    h = (uint64_t)raw_extent[9] |
        ((uint64_t)raw_extent[10] << 8) |
        ((uint64_t)raw_extent[11] << 16) |
        ((uint64_t)raw_extent[12] << 24) |
        ((uint64_t)raw_extent[13] << 32) |
        ((uint64_t)raw_extent[14] << 40) |
        ((uint64_t)raw_extent[15] << 48) |
        ((uint64_t)raw_extent[16] << 56);

    z = (uint64_t)raw_extent[17] |
        ((uint64_t)raw_extent[18] << 8) |
        ((uint64_t)raw_extent[19] << 16) |
        ((uint64_t)raw_extent[20] << 24) |
        ((uint64_t)raw_extent[21] << 32) |
        ((uint64_t)raw_extent[22] << 40) |
        ((uint64_t)raw_extent[23] << 48) |
        ((uint64_t)raw_extent[24] << 56);

    switch (d)
    {
    case 0:
        if (h != 1 || z != 1)
        {
            return JKL_ERR_INVALID_DIMENSIONS;
        }
        extent->dimensions = JKL_IMAGE_DIMENSIONS_D1;
        if (w > SIZE_MAX)
        {
            return JKL_ERR_TOO_LARGE;
        }
        extent->width = (size_t)w;
        extent->height = 1;
        extent->depth = 1;
        extent->layers = 1;
        break;
    case 1:
        if (z != 1)
        {
            return JKL_ERR_INVALID_DIMENSIONS;
        }
        extent->dimensions = JKL_IMAGE_DIMENSIONS_D2;
        if (w > SIZE_MAX || h > SIZE_MAX)
        {
            return JKL_ERR_TOO_LARGE;
        }
        extent->width = (size_t)w;
        extent->height = (size_t)h;
        extent->depth = 1;
        extent->layers = 1;
        break;
    case 2:
        extent->dimensions = JKL_IMAGE_DIMENSIONS_D3;
        if (w > SIZE_MAX || h > SIZE_MAX || z > SIZE_MAX)
        {
            return JKL_ERR_TOO_LARGE;
        }
        extent->width = (size_t)w;
        extent->height = (size_t)h;
        extent->depth = (size_t)z;
        extent->layers = 1;
        break;
    case 3:
        if (h != 1)
        {
            return JKL_ERR_INVALID_DIMENSIONS;
        }
        extent->dimensions = JKL_IMAGE_DIMENSIONS_D1_ARRAY;
        if (w > SIZE_MAX || z > SIZE_MAX)
        {
            return JKL_ERR_TOO_LARGE;
        }
        extent->width = (size_t)w;
        extent->height = 1;
        extent->depth = 1;
        extent->layers = (size_t)z;
        break;
    case 4:
        extent->dimensions = JKL_IMAGE_DIMENSIONS_D2_ARRAY;
        if (w > SIZE_MAX || h > SIZE_MAX || z > SIZE_MAX)
        {
            return JKL_ERR_TOO_LARGE;
        }
        extent->width = (size_t)w;
        extent->height = (size_t)h;
        extent->depth = 1;
        extent->layers = (size_t)z;
        break;
    default:
        return JKL_ERR_INVALID_DIMENSIONS;
    }

    return JKL_OK;
}

size_t jkli_tile_count(const JkliFile *file)
{
    return jkl_tiles_count(file->tile_size, file->extent);
}

static int jkl_seek_u64(FILE *fp, uint64_t pos)
{
#if defined(_WIN32)
    if (_fseeki64(fp, (long long)pos, SEEK_SET) != 0)
    {
        return JKL_ERR_IO;
    }
#else
    if (fseeko(fp, (off_t)pos, SEEK_SET) != 0)
    {
        return JKL_ERR_IO;
    }
#endif
    return JKL_OK;
}

static int jkl_read_all_offsets(FILE *fp, size_t tile_count, uint64_t *offsets)
{
    size_t i;

    for (i = 0; i < tile_count; ++i)
    {
        JKL_RETURN_IF_ERROR(jkl_read_u64_le(fp, &offsets[i]));
    }

    return JKL_OK;
}

static size_t jkl_context_size(JkliCompression compression)
{
    switch (compression)
    {
    case JKLI_COMPRESSION_ANS:
        return sizeof(JklAnsContext32);
    case JKLI_COMPRESSION_LZ77_ANS:
        return sizeof(JklAnsContextLz77);
    case JKLI_COMPRESSION_RLE_ANS:
        return sizeof(JklAnsContextRle);
    case JKLI_COMPRESSION_NONE:
    case JKLI_COMPRESSION_LZ77:
    default:
        return 0;
    }
}

int jkli_open_file(FILE *file, int take_ownership, JkliFile **out_file)
{
    JkliFile *f;
    uint32_t magic;
    uint8_t compression_u8;
    uint16_t format_u16;
    uint8_t raw_extent[25];
    uint16_t levels;
    uint16_t tile_w;
    uint16_t tile_h;
    size_t tile_count;
    size_t offsets_size;
    int err;

    *out_file = NULL;

    f = (JkliFile *)calloc(1, sizeof(JkliFile));
    if (f == NULL)
    {
        return JKL_ERR_OOM;
    }

    f->fp = file;
    f->owns_fp = take_ownership ? 1 : 0;

    err = jkl_read_u32_le(file, &magic);
    if (err != JKL_OK)
    {
        jkli_close(f);
        return err;
    }

    if (magic != JKLI_MAGIC)
    {
        jkli_close(f);
        return JKL_ERR_INVALID_MAGIC;
    }

    err = jkl_read_u8(file, &compression_u8);
    if (err != JKL_OK)
    {
        jkli_close(f);
        return err;
    }

    switch (compression_u8)
    {
    case 0:
        f->compression = JKLI_COMPRESSION_NONE;
        break;
    case 1:
        f->compression = JKLI_COMPRESSION_LZ77;
        break;
    case 2:
        f->compression = JKLI_COMPRESSION_ANS;
        break;
    case 3:
        f->compression = JKLI_COMPRESSION_LZ77_ANS;
        break;
    case 4:
        f->compression = JKLI_COMPRESSION_RLE_ANS;
        break;
    default:
        jkli_close(f);
        return JKL_ERR_INVALID_COMPRESSION;
    }

    err = jkl_read_u16_le(file, &format_u16);
    if (err != JKL_OK)
    {
        jkli_close(f);
        return err;
    }

    switch (format_u16)
    {
    case 0:
        f->format = JKL_FORMAT_R8;
        break;
    case 1:
        f->format = JKL_FORMAT_RG8;
        break;
    case 2:
        f->format = JKL_FORMAT_RGB8;
        break;
    case 3:
        f->format = JKL_FORMAT_RGBA8;
        break;
    case 256:
        f->format = JKL_FORMAT_BC1;
        break;
    case 257:
        f->format = JKL_FORMAT_BC2;
        break;
    case 258:
        f->format = JKL_FORMAT_BC3;
        break;
    case 259:
        f->format = JKL_FORMAT_BC4;
        break;
    case 260:
        f->format = JKL_FORMAT_BC5;
        break;
    case 261:
        f->format = JKL_FORMAT_BC6;
        break;
    case 262:
        f->format = JKL_FORMAT_BC7;
        break;
    default:
        jkli_close(f);
        return JKL_ERR_INVALID_FORMAT;
    }

    err = jkl_read_exact(file, raw_extent, sizeof(raw_extent));
    if (err != JKL_OK)
    {
        jkli_close(f);
        return err;
    }

    err = jkl_read_extent_from_header(&f->extent, raw_extent);
    if (err != JKL_OK)
    {
        jkli_close(f);
        return err;
    }

    err = jkl_read_u16_le(file, &levels);
    if (err != JKL_OK)
    {
        jkli_close(f);
        return err;
    }

    if (levels == 0)
    {
        jkli_close(f);
        return JKL_ERR_MIP_ZERO;
    }

    err = jkl_read_u16_le(file, &tile_w);
    if (err != JKL_OK)
    {
        jkli_close(f);
        return err;
    }

    err = jkl_read_u16_le(file, &tile_h);
    if (err != JKL_OK)
    {
        jkli_close(f);
        return err;
    }

    f->tile_size.width = tile_w;
    f->tile_size.height = tile_h;

    tile_count = jkli_tile_count(f);

    offsets_size = tile_count * sizeof(uint64_t);
    f->offsets = malloc(offsets_size);
    if (f->offsets == NULL && offsets_size != 0)
    {
        jkli_close(f);
        return JKL_ERR_OOM;
    }

    err = jkl_read_all_offsets(file, tile_count, f->offsets);
    if (err != JKL_OK)
    {
        jkli_close(f);
        return err;
    }

    switch (f->compression)
    {
    case JKLI_COMPRESSION_NONE:
    case JKLI_COMPRESSION_LZ77:
        /* LZ77 context is unit/empty in JKLI. */
        break;
    case JKLI_COMPRESSION_ANS:
        err = jkl_ans_context_read_32(file, &f->context.ans);
        if (err != JKL_OK)
        {
            jkli_close(f);
            return err;
        }
        break;
    case JKLI_COMPRESSION_LZ77_ANS:
        /* Tuple context starts with empty LZ77 context; only ANS payload is read. */
        err = jkl_ans_context_read_lz77(file, &f->context.lz77_ans);
        if (err != JKL_OK)
        {
            jkli_close(f);
            return err;
        }
        break;
    case JKLI_COMPRESSION_RLE_ANS:
        /* Tuple context starts with empty RLE context; only ANS payload is read. */
        err = jkl_ans_context_read_rle(file, &f->context.rle_ans);
        if (err != JKL_OK)
        {
            jkli_close(f);
            return err;
        }
        break;
    default:
        jkli_close(f);
        return JKL_ERR_INVALID_COMPRESSION;
    }

    *out_file = f;
    return JKL_OK;
}

int jkli_open(const char *path, JkliFile **out_file)
{
    FILE *fp;

    if (fopen_s(&fp, path, "rb") != 0)
    {
        return JKL_ERR_IO;
    }

    return jkli_open_file(fp, 1, out_file);
}

int jkli_close(JkliFile *file)
{
    if (file == NULL)
    {
        return JKL_OK;
    }

    jkl_free_context(file);
    free(file->offsets);
    file->offsets = NULL;

    if (file->owns_fp && file->fp != NULL)
    {
        fclose(file->fp);
    }

    free(file);
    return JKL_OK;
}

Format jkli_format(const JkliFile *file)
{
    return file->format;
}

Extent jkli_extent(const JkliFile *file)
{
    return file->extent;
}

TileSize jkli_tile_size(const JkliFile *file)
{
    return file->tile_size;
}

Tile jkli_tile_at(const JkliFile *file, size_t tile_index)
{
    assert(tile_index < jkli_tile_count(file));
    return jkl_tile_at(file->tile_size, file->extent, tile_index);
}

static int jkl_decode_tile_none_rgb8(
    FILE *fp,
    Image2D output)
{
    size_t y;
    size_t row_bytes;
    uint64_t decoded = 0;

    row_bytes = output.width * 3;
    for (y = 0; y < output.height; ++y)
    {
        uint8_t *row = jkl_image2d_row_at(&output, y);
        JKL_RETURN_IF_ERROR(jkl_read_exact(fp, row, row_bytes));
        decoded += output.width;
    }

    return JKL_OK;
}

static int jkl_decode_tile_lz77_rgb8(
    FILE *fp,
    Image2D output)
{
    JklBitReader br;
    JklLz77Decoder lz;
    size_t i;

    JKL_RETURN_IF_ERROR(jkl_bit_reader_init_file(fp, &br));
    jkl_lz77_decoder_init(&lz);

    size_t pixels = output.width * output.height;

    for (i = 0; i < pixels; ++i)
    {
        uint32_t symbol;
        JklLz77Token token;
        int err = jkl_lz77_decode(&lz, &symbol);
        if (err == JKL_ERR_NEED_TOKEN)
        {
            JKL_RETURN_IF_ERROR(jkl_read_lz77_token(&br, &token));
            JKL_RETURN_IF_ERROR(jkl_lz77_feed_token(&lz, token));
            JKL_RETURN_IF_ERROR(jkl_lz77_decode(&lz, &symbol));
        }
        else
        {
            JKL_RETURN_IF_ERROR(err);
        }

        {
            size_t x = (size_t)(i % output.width);
            size_t y = (size_t)(i / output.width);
            uint8_t *dst = jkl_image2d_at(&output, x, y);
            jkl_rgb_from_bits_interleaved(symbol, dst);
        }
    }

    return JKL_OK;
}

static int jkl_decode_tile_ans_rgb8(
    FILE *fp,
    const JklAnsContext32 *ctx,
    Image2D output)
{
    JklBitReader br;
    JklAnsDecoder32 decoder;
    size_t i;

    JKL_RETURN_IF_ERROR(jkl_bit_reader_init_file(fp, &br));
    jkl_ans_decoder_init_32(&decoder, ctx);

    size_t pixels = output.width * output.height;

    for (i = 0; i < pixels; ++i)
    {
        uint32_t symbol;
        for (;;)
        {
            int err = jkl_ans_decode_32(&decoder, &symbol);
            if (err == JKL_OK)
            {
                break;
            }

            if (err != JKL_ERR_NEED_TOKEN)
            {
                return err;
            }

            {
                uint64_t token;
                JKL_RETURN_IF_ERROR(jkl_bit_read_bits(&br, 32U, &token));
                JKL_RETURN_IF_ERROR(jkl_ans_feed_token_32(&decoder, (uint32_t)token));
            }
        }

        {
            size_t x = (size_t)(i % output.width);
            size_t y = (size_t)(i / output.width);
            uint8_t *dst = jkl_image2d_at(&output, x, y);
            jkl_rgb_from_bits_interleaved(symbol, dst);
        }
    }

    return JKL_OK;
}

static int jkl_decode_tile_lz77_ans_rgb8(
    FILE *fp,
    const JklAnsContextLz77 *ctx,
    Image2D output)
{
    JklBitReader br;
    JklAnsDecoderLz77 ans;
    JklLz77Decoder lz;
    size_t i;

    JKL_RETURN_IF_ERROR(jkl_bit_reader_init_file(fp, &br));

    jkl_ans_decoder_init_lz77(&ans, ctx);
    jkl_lz77_decoder_init(&lz);

    size_t pixels = output.width * output.height;

    for (i = 0; i < pixels; ++i)
    {
        uint32_t symbol;
        int err = jkl_lz77_decode(&lz, &symbol);
        if (err == JKL_ERR_NEED_TOKEN)
        {
            JklLz77Token lz77_token;

            for (;;)
            {
                int ans_err = jkl_ans_decode_lz77(&ans, &lz77_token);
                if (ans_err == JKL_OK)
                {
                    break;
                }

                if (ans_err != JKL_ERR_NEED_TOKEN)
                {
                    return ans_err;
                }

                {
                    uint64_t ans_token;
                    JKL_RETURN_IF_ERROR(jkl_bit_read_bits(&br, 32U, &ans_token));
                    JKL_RETURN_IF_ERROR(jkl_ans_feed_token_lz77(&ans, (uint32_t)ans_token));
                }
            }

            JKL_RETURN_IF_ERROR(jkl_lz77_feed_token(&lz, lz77_token));
            JKL_RETURN_IF_ERROR(jkl_lz77_decode(&lz, &symbol));
        }
        else
        {
            JKL_RETURN_IF_ERROR(err);
        }

        {
            size_t x = (size_t)(i % output.width);
            size_t y = (size_t)(i / output.width);
            uint8_t *dst = jkl_image2d_at(&output, x, y);
            jkl_rgb_from_bits_interleaved(symbol, dst);
        }
    }

    return JKL_OK;
}

static int jkl_decode_tile_rle_ans_rgb8(
    FILE *fp,
    const JklAnsContextRle *ctx,
    Image2D output)
{
    JklBitReader br;
    JklAnsDecoderRle ans;
    JklRleDecoder rle;
    size_t i;

    JKL_RETURN_IF_ERROR(jkl_bit_reader_init_file(fp, &br));

    jkl_ans_decoder_init_rle(&ans, ctx);
    jkl_rle_decoder_init(&rle);

    size_t pixels = output.width * output.height;

    for (i = 0; i < pixels; ++i)
    {
        uint32_t symbol;
        int err = jkl_rle_decode(&rle, &symbol);
        if (err == JKL_ERR_NEED_TOKEN)
        {
            JklRleToken rle_token;

            for (;;)
            {
                int ans_err = jkl_ans_decode_rle(&ans, &rle_token);
                if (ans_err == JKL_OK)
                {
                    break;
                }

                if (ans_err != JKL_ERR_NEED_TOKEN)
                {
                    return ans_err;
                }

                {
                    uint64_t ans_token;
                    JKL_RETURN_IF_ERROR(jkl_bit_read_bits(&br, 32U, &ans_token));
                    JKL_RETURN_IF_ERROR(jkl_ans_feed_token_rle(&ans, (uint32_t)ans_token));
                }
            }

            JKL_RETURN_IF_ERROR(jkl_rle_feed_token(&rle, rle_token));
            JKL_RETURN_IF_ERROR(jkl_rle_decode(&rle, &symbol));
        }
        else
        {
            JKL_RETURN_IF_ERROR(err);
        }

        {
            size_t x = (size_t)(i % output.width);
            size_t y = (size_t)(i / output.width);
            uint8_t *dst = jkl_image2d_at(&output, x, y);
            jkl_rgb_from_bits_interleaved(symbol, dst);
        }
    }

    return JKL_OK;
}

int jkli_decode_tile(
    JkliFile *file,
    size_t tile_index,
    Image2D output)
{
    Tile tile;

    assert(output.format == file->format);
    assert(tile_index < jkli_tile_count(file));

    tile = jkli_tile_at(file, tile_index);

    assert(output.width == tile.rect.w);
    assert(output.height == tile.rect.h);

    JKL_RETURN_IF_ERROR(jkl_seek_u64(file->fp, file->offsets[tile_index]));

    switch (file->compression)
    {
    case JKLI_COMPRESSION_NONE:
        return jkl_decode_tile_none_rgb8(
            file->fp,
            output);

    case JKLI_COMPRESSION_LZ77:
        return jkl_decode_tile_lz77_rgb8(
            file->fp,
            output);

    case JKLI_COMPRESSION_ANS:
        return jkl_decode_tile_ans_rgb8(
            file->fp,
            &file->context.ans,
            output);

    case JKLI_COMPRESSION_LZ77_ANS:
        return jkl_decode_tile_lz77_ans_rgb8(
            file->fp,
            &file->context.lz77_ans,
            output);

    case JKLI_COMPRESSION_RLE_ANS:
        return jkl_decode_tile_rle_ans_rgb8(
            file->fp,
            &file->context.rle_ans,
            output);

    default:
        return JKL_ERR_INVALID_COMPRESSION;
    }
}
