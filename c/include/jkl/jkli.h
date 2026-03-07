#ifndef JKLI_H
#define JKLI_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "error.h"
#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum JkliCompression {
    JKLI_COMPRESSION_NONE = 0,
    JKLI_COMPRESSION_LZ77 = 1,
    JKLI_COMPRESSION_ANS = 2,
    JKLI_COMPRESSION_LZ77_ANS = 3,
    JKLI_COMPRESSION_RLE_ANS = 4
} JkliCompression;

typedef struct JkliFile JkliFile;

int jkli_open(const char *path, JkliFile **out_file);
int jkli_open_file(FILE *file, int take_ownership, JkliFile **out_file);
int jkli_close(JkliFile *file);

Format jkli_format(const JkliFile *file);
Extent jkli_extent(const JkliFile *file);
TileSize jkli_tile_size(const JkliFile *file);
size_t jkli_tile_count(const JkliFile *file);
Tile jkli_tile_at(const JkliFile *file, size_t tile_index);

/* Decodes one tile into output image blocks. Destination must match tile geometry and format. */
int jkli_decode_tile(
    JkliFile *file,
    size_t tile_index,
    Image2D output);

#ifdef __cplusplus
}
#endif

#endif
