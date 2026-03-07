#include "image.h"

#include <assert.h>
#include <string.h>

static size_t jkl_div_ceil_size(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static void jkl_extent_to_raw(Extent extent, size_t out_raw[3]) {
    out_raw[0] = extent.width;
    out_raw[1] = extent.height;
    out_raw[2] = extent.depth * extent.layers;
}

static void jkl_tiles_from_extent(TileSize tile_size, Extent extent, size_t out_tiles[3]) {
    size_t raw[3];

    assert(tile_size.width != 0 && tile_size.height != 0);

    jkl_extent_to_raw(extent, raw);
    out_tiles[0] = jkl_div_ceil_size(raw[0], (size_t)tile_size.width);
    out_tiles[1] = jkl_div_ceil_size(raw[1], (size_t)tile_size.height);
    out_tiles[2] = raw[2];
}

Image2D jkl_image2d_init(
    size_t width,
    size_t height,
    Format format,
    uint8_t *pixels) {
    return jkl_image2d_init_with_stride(width, height, width, format, pixels);
}

Image2D jkl_image2d_init_with_stride(
    size_t width,
    size_t height,
    size_t stride,
    Format format,
    uint8_t *pixels) {
    Image2D out_image;

    (void)jkl_format_block_size_bytes_known(format);
    assert(stride >= width);

    out_image.width = width;
    out_image.height = height;
    out_image.stride = stride;
    out_image.pixels = pixels;
    out_image.format = format;
    return out_image;
}

Image2D jkl_image2d_from_row(
    size_t width,
    Format format,
    uint8_t *pixels) {
    return jkl_image2d_init(width, 1, format, pixels);
}

Image2D jkl_image2d_get_rect(const Image2D *image, Rect rect) {
    size_t byte_index;
    Image2D out_rect;

    assert(rect.x <= image->width && rect.y <= image->height);
    assert(rect.w <= image->width - rect.x && rect.h <= image->height - rect.y);

    if (rect.w == 0 || rect.h == 0) {
        byte_index = 0;
    } else {
        byte_index = jkl_image2d_index(image, rect.x, rect.y);
    }

    out_rect.width = rect.w;
    out_rect.height = rect.h;
    out_rect.stride = image->stride;
    out_rect.pixels = image->pixels + byte_index;
    out_rect.format = image->format;
    return out_rect;
}

size_t jkl_tiles_count(TileSize tile_size, Extent extent) {
    size_t tiles[3];
    jkl_tiles_from_extent(tile_size, extent, tiles);
    return tiles[0] * tiles[1] * tiles[2];
}

Tile jkl_tile_at(TileSize tile_size, Extent extent, size_t index) {
    size_t tiles[3];
    size_t tiles_per_plane;
    size_t plane;
    size_t tile_index;
    size_t tile_x;
    size_t tile_y;
    size_t x;
    size_t y;
    size_t w;
    size_t h;
    Tile out_tile;

    jkl_tiles_from_extent(tile_size, extent, tiles);
    tiles_per_plane = tiles[0] * tiles[1];
    assert(tiles_per_plane != 0 && tiles[2] != 0);

    plane = index / tiles_per_plane;
    assert(plane < tiles[2]);

    tile_index = index % tiles_per_plane;
    tile_y = tile_index / tiles[0];
    tile_x = tile_index % tiles[0];

    x = tile_x * (size_t)tile_size.width;
    y = tile_y * (size_t)tile_size.height;

    w = (size_t)tile_size.width;
    h = (size_t)tile_size.height;
    if (x + w > extent.width) {
        w = extent.width - x;
    }
    if (y + h > extent.height) {
        h = extent.height - y;
    }

    out_tile.plane = plane;
    out_tile.rect.x = x;
    out_tile.rect.y = y;
    out_tile.rect.w = w;
    out_tile.rect.h = h;
    return out_tile;
}

size_t jkl_image2d_row_index(const Image2D *image, size_t y) {
    return jkl_image2d_index(image, 0, y);
}

uint8_t *jkl_image2d_row_at(const Image2D *image, size_t y) {
    return jkl_image2d_at(image, 0, y);
}

void jkl_image2d_row_get(const Image2D *image, size_t y, uint8_t *out_row_data) {
    uint8_t *row;
    size_t block_size;
    size_t row_len;

    row = jkl_image2d_row_at(image, y);
    block_size = jkl_format_block_size_bytes_known(image->format);
    row_len = image->width * block_size;
    memcpy(out_row_data, row, row_len);
}

void jkl_image2d_row_set(const Image2D *image, size_t y, const uint8_t *row_data) {
    uint8_t *row;
    size_t block_size;
    size_t row_len;

    row = jkl_image2d_row_at(image, y);
    block_size = jkl_format_block_size_bytes_known(image->format);
    row_len = image->width * block_size;
    memcpy(row, row_data, row_len);
}

size_t jkl_image2d_index(const Image2D *image, size_t x, size_t y) {
    size_t block_size;
    size_t block_index;

    assert(x < image->width && y < image->height);
    block_size = jkl_format_block_size_bytes_known(image->format);

    block_index = y * image->stride + x;
    return block_index * block_size;
}

uint8_t *jkl_image2d_at(const Image2D *image, size_t x, size_t y) {
    size_t idx = jkl_image2d_index(image, x, y);
    return &image->pixels[idx];
}

void jkl_image2d_get(const Image2D *image, size_t x, size_t y, uint8_t *out_block_data) {
    uint8_t *block;
    size_t block_size;

    block = jkl_image2d_at(image, x, y);
    block_size = jkl_format_block_size_bytes_known(image->format);
    memcpy(out_block_data, block, block_size);
}

void jkl_image2d_set(const Image2D *image, size_t x, size_t y, const uint8_t *block_data) {
    uint8_t *block;
    size_t block_size;

    block = jkl_image2d_at(image, x, y);
    block_size = jkl_format_block_size_bytes_known(image->format);
    memcpy(block, block_data, block_size);
}

static void jkl_validate_extent_for_dimensions(
    JklImageDimensions dimensions,
    const size_t extent[3]) {
    switch (dimensions) {
        case JKL_IMAGE_DIMENSIONS_D1:
            assert(extent[1] == 1 && extent[2] == 1);
            break;
        case JKL_IMAGE_DIMENSIONS_D2:
            assert(extent[2] == 1);
            break;
        case JKL_IMAGE_DIMENSIONS_D3:
            break;
        case JKL_IMAGE_DIMENSIONS_D1_ARRAY:
            assert(extent[2] == 1);
            break;
        case JKL_IMAGE_DIMENSIONS_D2_ARRAY:
            break;
        default:
            assert(0);
            break;
    }
}

Image jkl_image_init(
    JklImageDimensions dimensions,
    const size_t extent[3],
    Format format,
    uint8_t *pixels) {
    size_t stride[2];

    stride[0] = extent[0];
    stride[1] = extent[0] * extent[1];

    return jkl_image_init_with_stride(dimensions, extent, stride, format, pixels);
}

Image jkl_image_init_with_stride(
    JklImageDimensions dimensions,
    const size_t extent[3],
    const size_t stride[2],
    Format format,
    uint8_t *pixels) {
    size_t min_plane_stride;
    Image out_image;

    (void)jkl_format_block_size_bytes_known(format);
    jkl_validate_extent_for_dimensions(dimensions, extent);

    assert(stride[0] >= extent[0]);

    min_plane_stride = stride[0] * extent[1];
    if (dimensions == JKL_IMAGE_DIMENSIONS_D3 || dimensions == JKL_IMAGE_DIMENSIONS_D2_ARRAY) {
        assert(stride[1] >= min_plane_stride);
    }

    out_image.dimensions = dimensions;
    out_image.extent[0] = extent[0];
    out_image.extent[1] = extent[1];
    out_image.extent[2] = extent[2];
    out_image.stride[0] = stride[0];
    out_image.stride[1] = stride[1];
    out_image.pixels = pixels;
    out_image.format = format;

    return out_image;
}

Image jkl_image_new_1d(size_t width, Format format, uint8_t *pixels) {
    const size_t extent[3] = { width, 1, 1 };
    return jkl_image_init(JKL_IMAGE_DIMENSIONS_D1, extent, format, pixels);
}

Image jkl_image_new_2d(
    size_t width,
    size_t height,
    Format format,
    uint8_t *pixels) {
    const size_t extent[3] = { width, height, 1 };
    return jkl_image_init(JKL_IMAGE_DIMENSIONS_D2, extent, format, pixels);
}

Image jkl_image_new_3d(
    size_t width,
    size_t height,
    size_t depth,
    Format format,
    uint8_t *pixels) {
    const size_t extent[3] = { width, height, depth };
    return jkl_image_init(JKL_IMAGE_DIMENSIONS_D3, extent, format, pixels);
}

Image jkl_image_new_1d_array(
    size_t width,
    size_t layers,
    Format format,
    uint8_t *pixels) {
    const size_t extent[3] = { width, layers, 1 };
    return jkl_image_init(JKL_IMAGE_DIMENSIONS_D1_ARRAY, extent, format, pixels);
}

Image jkl_image_new_2d_array(
    size_t width,
    size_t height,
    size_t layers,
    Format format,
    uint8_t *pixels) {
    const size_t extent[3] = { width, height, layers };
    return jkl_image_init(JKL_IMAGE_DIMENSIONS_D2_ARRAY, extent, format, pixels);
}

Image jkl_image_with_stride_2d(
    size_t width,
    size_t height,
    size_t row_stride,
    Format format,
    uint8_t *pixels) {
    const size_t extent[3] = { width, height, 1 };
    size_t plane_stride;
    size_t stride[2];

    plane_stride = row_stride * height;
    stride[0] = row_stride;
    stride[1] = plane_stride;
    return jkl_image_init_with_stride(JKL_IMAGE_DIMENSIONS_D2, extent, stride, format, pixels);
}

Image jkl_image_with_stride_3d(
    size_t width,
    size_t height,
    size_t depth,
    size_t row_stride,
    size_t plane_stride,
    Format format,
    uint8_t *pixels) {
    const size_t extent[3] = { width, height, depth };
    const size_t stride[2] = { row_stride, plane_stride };
    return jkl_image_init_with_stride(JKL_IMAGE_DIMENSIONS_D3, extent, stride, format, pixels);
}

Image jkl_image_with_stride_1d_array(
    size_t width,
    size_t layers,
    size_t row_stride,
    Format format,
    uint8_t *pixels) {
    const size_t extent[3] = { width, layers, 1 };
    size_t plane_stride;
    size_t stride[2];

    plane_stride = row_stride * layers;
    stride[0] = row_stride;
    stride[1] = plane_stride;
    return jkl_image_init_with_stride(JKL_IMAGE_DIMENSIONS_D1_ARRAY, extent, stride, format, pixels);
}

Image jkl_image_with_stride_2d_array(
    size_t width,
    size_t height,
    size_t layers,
    size_t row_stride,
    size_t layer_stride,
    Format format,
    uint8_t *pixels) {
    const size_t extent[3] = { width, height, layers };
    const size_t stride[2] = { row_stride, layer_stride };
    return jkl_image_init_with_stride(JKL_IMAGE_DIMENSIONS_D2_ARRAY, extent, stride, format, pixels);
}

Image2D jkl_image_plane(const Image *image, size_t plane_index) {
    size_t block_size;
    size_t plane_blocks;
    size_t plane_byte_offset;
    size_t plane_width;
    size_t plane_height;
    Image2D out_plane;

    assert(plane_index < image->extent[2]);
    block_size = jkl_format_block_size_bytes_known(image->format);

    plane_width = image->extent[0];
    plane_height = image->extent[1];
    plane_blocks = plane_index * image->stride[1];
    plane_byte_offset = plane_blocks * block_size;

    out_plane.width = plane_width;
    out_plane.height = plane_height;
    out_plane.stride = image->stride[0];
    out_plane.pixels = image->pixels + plane_byte_offset;
    out_plane.format = image->format;

    return out_plane;
}

size_t jkl_image_row_index(const Image *image, size_t y, size_t plane_index) {
    return jkl_image_index(image, 0, y, plane_index);
}

uint8_t *jkl_image_row_at(const Image *image, size_t y, size_t plane_index) {
    return jkl_image_at(image, 0, y, plane_index);
}

void jkl_image_row_get(const Image *image, size_t y, size_t plane_index, uint8_t *out_row_data) {
    uint8_t *row;
    size_t block_size;
    size_t row_len;

    row = jkl_image_row_at(image, y, plane_index);
    block_size = jkl_format_block_size_bytes_known(image->format);
    row_len = image->extent[0] * block_size;
    memcpy(out_row_data, row, row_len);
}

void jkl_image_row_set(const Image *image, size_t y, size_t plane_index, const uint8_t *row_data) {
    uint8_t *row;
    size_t block_size;
    size_t row_len;

    row = jkl_image_row_at(image, y, plane_index);
    block_size = jkl_format_block_size_bytes_known(image->format);
    row_len = image->extent[0] * block_size;
    memcpy(row, row_data, row_len);
}

size_t jkl_image_index(const Image *image, size_t x, size_t y, size_t z) {
    size_t block_size;
    size_t block_index = 0;

    assert(x < image->extent[0] && y < image->extent[1] && z < image->extent[2]);
    block_size = jkl_format_block_size_bytes_known(image->format);

    switch (image->dimensions) {
        case JKL_IMAGE_DIMENSIONS_D1:
            block_index = x;
            break;
        case JKL_IMAGE_DIMENSIONS_D2:
        case JKL_IMAGE_DIMENSIONS_D1_ARRAY:
            block_index = y * image->stride[0] + x;
            break;
        case JKL_IMAGE_DIMENSIONS_D3:
        case JKL_IMAGE_DIMENSIONS_D2_ARRAY: {
            size_t layer_index;
            size_t row_index;
            layer_index = z * image->stride[1];
            row_index = y * image->stride[0];
            block_index = layer_index + row_index + x;
            break;
        }
        default:
            assert(0);
            break;
    }

    return block_index * block_size;
}

uint8_t *jkl_image_at(const Image *image, size_t x, size_t y, size_t z) {
    size_t idx = jkl_image_index(image, x, y, z);
    return &image->pixels[idx];
}

void jkl_image_get(const Image *image, size_t x, size_t y, size_t z, uint8_t *out_block_data) {
    uint8_t *block;
    size_t block_size;

    block = jkl_image_at(image, x, y, z);
    block_size = jkl_format_block_size_bytes_known(image->format);
    memcpy(out_block_data, block, block_size);
}

void jkl_image_set(const Image *image, size_t x, size_t y, size_t z, const uint8_t *block_data) {
    uint8_t *block;
    size_t block_size;

    block = jkl_image_at(image, x, y, z);
    block_size = jkl_format_block_size_bytes_known(image->format);
    memcpy(block, block_data, block_size);
}

