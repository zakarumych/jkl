#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"
#include "jkli.h"

static void print_usage(const char *exe)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s decode <input.jkli> [output.bmp]\n"
            "\n"
            "Only BMP output is supported.\n",
            exe);
}

static int ascii_tolower(int c)
{
    if (c >= 'A' && c <= 'Z')
    {
        return c - 'A' + 'a';
    }
    return c;
}

static int ascii_strcasecmp(const char *a, const char *b)
{
    while (*a != '\0' && *b != '\0')
    {
        int ca = ascii_tolower((unsigned char)*a);
        int cb = ascii_tolower((unsigned char)*b);
        if (ca != cb)
        {
            return ca - cb;
        }
        ++a;
        ++b;
    }
    return ascii_tolower((unsigned char)*a) - ascii_tolower((unsigned char)*b);
}

static int has_bmp_extension(const char *path)
{
    const char *dot = strrchr(path, '.');
    if (dot == NULL)
    {
        return 0;
    }
    return ascii_strcasecmp(dot, ".bmp") == 0;
}

static char *default_bmp_path(const char *input)
{
    size_t len = strlen(input);
    size_t i = len;
    char *out;

    while (i > 0)
    {
        char c = input[i - 1];
        if (c == '/' || c == '\\')
        {
            break;
        }
        if (c == '.')
        {
            len = i - 1;
            break;
        }
        --i;
    }

    out = (char *)malloc(len + 5);
    if (out == NULL)
    {
        return NULL;
    }

    memcpy(out, input, len);
    memcpy(out + len, ".bmp", 5);
    return out;
}

static void write_u16_le(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t)(value & 0xFFu);
    dst[1] = (uint8_t)((value >> 8) & 0xFFu);
}

static void write_u32_le(uint8_t *dst, uint32_t value)
{
    dst[0] = (uint8_t)(value & 0xFFu);
    dst[1] = (uint8_t)((value >> 8) & 0xFFu);
    dst[2] = (uint8_t)((value >> 16) & 0xFFu);
    dst[3] = (uint8_t)((value >> 24) & 0xFFu);
}

static int write_bmp24(const char *path, size_t width, size_t height, const uint8_t *rgb)
{
    FILE *fp = NULL;
    uint8_t header[54] = {0};
    size_t row_bytes = width * 3;
    size_t row_stride = (row_bytes + 3u) & ~3u;
    size_t pixel_bytes = row_stride * height;
    uint32_t file_size;
    int y;

    if (width > 0x7FFFFFFFu || height > 0x7FFFFFFFu)
    {
        return JKL_ERR_TOO_LARGE;
    }
    if (row_bytes / 3 != width || row_stride < row_bytes || pixel_bytes / row_stride != height)
    {
        return JKL_ERR_TOO_LARGE;
    }
    if (pixel_bytes > (size_t)(UINT32_MAX - 54u))
    {
        return JKL_ERR_TOO_LARGE;
    }

    file_size = (uint32_t)(54u + pixel_bytes);

    if (fopen_s(&fp, path, "wb") != 0)
    {
        return JKL_ERR_IO;
    }

    header[0] = 'B';
    header[1] = 'M';
    write_u32_le(header + 2, file_size);
    write_u32_le(header + 10, 54u);
    write_u32_le(header + 14, 40u);
    write_u32_le(header + 18, (uint32_t)width);
    write_u32_le(header + 22, (uint32_t)height);
    write_u16_le(header + 26, 1u);
    write_u16_le(header + 28, 24u);
    write_u32_le(header + 34, (uint32_t)pixel_bytes);

    if (fwrite(header, 1, sizeof(header), fp) != sizeof(header))
    {
        fclose(fp);
        return JKL_ERR_IO;
    }

    for (y = (int)height - 1; y >= 0; --y)
    {
        size_t x;
        uint8_t pad[3] = {0, 0, 0};
        const uint8_t *row = rgb + (size_t)y * row_bytes;

        for (x = 0; x < width; ++x)
        {
            uint8_t bgr[3];
            const uint8_t *px = row + x * 3;
            bgr[0] = px[2];
            bgr[1] = px[1];
            bgr[2] = px[0];
            if (fwrite(bgr, 1, 3, fp) != 3)
            {
                fclose(fp);
                return JKL_ERR_IO;
            }
        }

        if (row_stride > row_bytes)
        {
            size_t pad_len = row_stride - row_bytes;
            if (fwrite(pad, 1, pad_len, fp) != pad_len)
            {
                fclose(fp);
                return JKL_ERR_IO;
            }
        }
    }

    if (fclose(fp) != 0)
    {
        return JKL_ERR_IO;
    }

    return JKL_OK;
}

static int decode_jkli_to_bmp(const char *input_path, const char *output_path)
{
    JkliFile *file = NULL;
    Extent extent;
    TileSize tile_size;
    size_t tile_count;
    uint8_t *pixels = NULL;
    Image2D full_image;
    size_t total_pixels;
    size_t i;
    int err;

    err = jkli_open(input_path, &file);
    if (err != JKL_OK)
    {
        return err;
    }

    extent = jkli_extent(file);

    if (extent.dimensions != JKL_IMAGE_DIMENSIONS_D2)
    {
        jkli_close(file);
        return JKL_ERR_UNSUPPORTED_FORMAT;
    }
    if (jkli_format(file) != JKL_FORMAT_RGB8)
    {
        jkli_close(file);
        return JKL_ERR_UNSUPPORTED_FORMAT;
    }

    if (extent.width == 0 || extent.height == 0)
    {
        jkli_close(file);
        return JKL_ERR_INVALID_EXTENT;
    }

    total_pixels = extent.width * extent.height;
    if (total_pixels / extent.width != extent.height || total_pixels > (SIZE_MAX / 3u))
    {
        jkli_close(file);
        return JKL_ERR_TOO_LARGE;
    }

    pixels = (uint8_t *)malloc(total_pixels * 3u);
    if (pixels == NULL)
    {
        jkli_close(file);
        return JKL_ERR_OOM;
    }

    full_image = jkl_image2d_init(extent.width, extent.height, JKL_FORMAT_RGB8, pixels);

    tile_size = jkli_tile_size(file);
    tile_count = jkl_tiles_count(tile_size, extent);

    for (i = 0; i < tile_count; ++i)
    {
        Tile tile = jkli_tile_at(file, i);
        Image2D view = jkl_image2d_get_rect(&full_image, tile.rect);
        err = jkli_decode_tile(file, i, view);
        if (err != JKL_OK)
        {
            free(pixels);
            jkli_close(file);
            return err;
        }
    }

    err = write_bmp24(output_path, extent.width, extent.height, pixels);
    free(pixels);
    jkli_close(file);
    return err;
}

int main(int argc, char **argv)
{
    const char *exe = (argc > 0 && argv[0] != NULL) ? argv[0] : "cli";
    const char *input = NULL;
    const char *output = NULL;
    char *owned_output = NULL;
    int err;

    if (argc < 3 || argc > 4 || strcmp(argv[1], "decode") != 0)
    {
        print_usage(exe);
        return 2;
    }

    input = argv[2];
    if (argc == 4)
    {
        output = argv[3];
    }
    else
    {
        owned_output = default_bmp_path(input);
        if (owned_output == NULL)
        {
            fprintf(stderr, "Out of memory\n");
            return 1;
        }
        output = owned_output;
    }

    if (!has_bmp_extension(output))
    {
        fprintf(stderr, "Output must have .bmp extension\n");
        free(owned_output);
        return 2;
    }

    err = decode_jkli_to_bmp(input, output);
    if (err != JKL_OK)
    {
        fprintf(stderr, "Decode failed (%d)\n", err);
        free(owned_output);
        return 1;
    }

    free(owned_output);
    return 0;
}
