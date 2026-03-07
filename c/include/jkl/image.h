#ifndef JKL_IMAGE_H
#define JKL_IMAGE_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum Format {
    JKL_FORMAT_R8 = 0,
    JKL_FORMAT_RG8 = 1,
    JKL_FORMAT_RGB8 = 2,
    JKL_FORMAT_RGBA8 = 3,
    JKL_FORMAT_BC1 = 256,
    JKL_FORMAT_BC2 = 257,
    JKL_FORMAT_BC3 = 258,
    JKL_FORMAT_BC4 = 259,
    JKL_FORMAT_BC5 = 260,
    JKL_FORMAT_BC6 = 261,
    JKL_FORMAT_BC7 = 262
} Format;

/* Returns bytes per block for the format. */
static inline size_t jkl_format_block_size_bytes(Format format) {
    switch (format) {
        case JKL_FORMAT_R8:
            return 1;
        case JKL_FORMAT_RG8:
            return 2;
        case JKL_FORMAT_RGB8:
            return 3;
        case JKL_FORMAT_RGBA8:
            return 4;
        case JKL_FORMAT_BC1:
            return 8;
        case JKL_FORMAT_BC2:
        case JKL_FORMAT_BC3:
        case JKL_FORMAT_BC5:
        case JKL_FORMAT_BC6:
        case JKL_FORMAT_BC7:
            return 16;
        case JKL_FORMAT_BC4:
            return 8;
        default:
            return 0;
    }
}

/* Returns bytes per block and asserts that format is known. */
static inline size_t jkl_format_block_size_bytes_known(Format format) {
    size_t block_size = jkl_format_block_size_bytes(format);
    assert(block_size != 0);
    return block_size;
}

typedef enum JklImageDimensions {
    JKL_IMAGE_DIMENSIONS_D1 = 0,
    JKL_IMAGE_DIMENSIONS_D2 = 1,
    JKL_IMAGE_DIMENSIONS_D3 = 2,
    JKL_IMAGE_DIMENSIONS_D1_ARRAY = 3,
    JKL_IMAGE_DIMENSIONS_D2_ARRAY = 4
} JklImageDimensions;

typedef struct Rect {
    size_t x;
    size_t y;
    size_t w;
    size_t h;
} Rect;

typedef struct Extent {
    JklImageDimensions dimensions;
    size_t width;
    size_t height;
    size_t depth;
    size_t layers;
} Extent;

typedef struct TileSize {
    uint16_t width;
    uint16_t height;
} TileSize;

typedef struct Tile {
    size_t plane;
    Rect rect;
} Tile;

typedef struct Image2D {
    /* Width in blocks. */
    size_t width;
    /* Height in blocks. */
    size_t height;
    /* Row stride in blocks. */
    size_t stride;
    uint8_t *pixels;
    Format format;
} Image2D;

typedef struct Image {
    JklImageDimensions dimensions;
    /* Extent in blocks: [width, height, depth_or_layers]. */
    size_t extent[3];
    /* Strides in blocks: [row_stride, plane_or_layer_stride]. */
    size_t stride[2];
    uint8_t *pixels;
    Format format;
} Image;

static inline size_t jkl_image2d_width(const Image2D *image) {
    return image->width;
}

static inline size_t jkl_image2d_height(const Image2D *image) {
    return image->height;
}

static inline size_t jkl_image2d_stride(const Image2D *image) {
    return image->stride;
}

static inline JklImageDimensions jkl_image_dimensions(const Image *image) {
    return image->dimensions;
}

static inline size_t jkl_image_width(const Image *image) {
    return image->extent[0];
}

static inline size_t jkl_image_height(const Image *image) {
    return image->extent[1];
}

static inline size_t jkl_image_depth_or_layers(const Image *image) {
    return image->extent[2];
}

Image2D jkl_image2d_init(
    size_t width,
    size_t height,
    Format format,
    uint8_t *pixels);

Image2D jkl_image2d_init_with_stride(
    size_t width,
    size_t height,
    size_t stride,
    Format format,
    uint8_t *pixels);

Image2D jkl_image2d_from_row(
    size_t width,
    Format format,
    uint8_t *pixels);

Image2D jkl_image2d_get_rect(const Image2D *image, Rect rect);

size_t jkl_tiles_count(TileSize tile_size, Extent extent);

Tile jkl_tile_at(TileSize tile_size, Extent extent, size_t index);

size_t jkl_image2d_row_index(const Image2D *image, size_t y);

uint8_t *jkl_image2d_row_at(const Image2D *image, size_t y);

void jkl_image2d_row_get(const Image2D *image, size_t y, uint8_t *out_row_data);

void jkl_image2d_row_set(const Image2D *image, size_t y, const uint8_t *row_data);

size_t jkl_image2d_index(const Image2D *image, size_t x, size_t y);

uint8_t *jkl_image2d_at(const Image2D *image, size_t x, size_t y);

void jkl_image2d_get(const Image2D *image, size_t x, size_t y, uint8_t *out_block_data);

void jkl_image2d_set(const Image2D *image, size_t x, size_t y, const uint8_t *block_data);

Image jkl_image_init(
    JklImageDimensions dimensions,
    const size_t extent[3],
    Format format,
    uint8_t *pixels);

Image jkl_image_init_with_stride(
    JklImageDimensions dimensions,
    const size_t extent[3],
    const size_t stride[2],
    Format format,
    uint8_t *pixels);

Image jkl_image_new_1d(
    size_t width,
    Format format,
    uint8_t *pixels);

Image jkl_image_new_2d(
    size_t width,
    size_t height,
    Format format,
    uint8_t *pixels);

Image jkl_image_new_3d(
    size_t width,
    size_t height,
    size_t depth,
    Format format,
    uint8_t *pixels);

Image jkl_image_new_1d_array(
    size_t width,
    size_t layers,
    Format format,
    uint8_t *pixels);

Image jkl_image_new_2d_array(
    size_t width,
    size_t height,
    size_t layers,
    Format format,
    uint8_t *pixels);

Image jkl_image_with_stride_2d(
    size_t width,
    size_t height,
    size_t row_stride,
    Format format,
    uint8_t *pixels);

Image jkl_image_with_stride_3d(
    size_t width,
    size_t height,
    size_t depth,
    size_t row_stride,
    size_t plane_stride,
    Format format,
    uint8_t *pixels);

Image jkl_image_with_stride_1d_array(
    size_t width,
    size_t layers,
    size_t row_stride,
    Format format,
    uint8_t *pixels);

Image jkl_image_with_stride_2d_array(
    size_t width,
    size_t height,
    size_t layers,
    size_t row_stride,
    size_t layer_stride,
    Format format,
    uint8_t *pixels);

Image2D jkl_image_plane(const Image *image, size_t plane_index);

size_t jkl_image_row_index(const Image *image, size_t y, size_t plane_index);

uint8_t *jkl_image_row_at(const Image *image, size_t y, size_t plane_index);

void jkl_image_row_get(const Image *image, size_t y, size_t plane_index, uint8_t *out_row_data);

void jkl_image_row_set(const Image *image, size_t y, size_t plane_index, const uint8_t *row_data);

size_t jkl_image_index(const Image *image, size_t x, size_t y, size_t z);

uint8_t *jkl_image_at(const Image *image, size_t x, size_t y, size_t z);

void jkl_image_get(const Image *image, size_t x, size_t y, size_t z, uint8_t *out_block_data);

void jkl_image_set(const Image *image, size_t x, size_t y, size_t z, const uint8_t *block_data);

#ifdef __cplusplus
}
#endif

#endif
