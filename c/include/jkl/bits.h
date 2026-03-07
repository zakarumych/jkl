#ifndef JKL_BIT_READER_H
#define JKL_BIT_READER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "error.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum JklBitReaderSource
    {
        JKL_BIT_SOURCE_FILE = 0,
        JKL_BIT_SOURCE_MEMORY = 1
    } JklBitReaderSource;

    typedef struct JklBitReader
    {
        JklBitReaderSource source;
        uint64_t buffer;
        uint32_t buffer_len;
        union
        {
            FILE *file;
            struct
            {
                const uint8_t *data;
                size_t size;
                size_t pos;
            } mem;
        } src;
    } JklBitReader;

    int jkl_bit_reader_init_file(FILE *file, JklBitReader *out_reader);
    int jkl_bit_reader_init_memory(const uint8_t *data, size_t size, JklBitReader *out_reader);

    int jkl_bit_read_bit(JklBitReader *reader, int *out_bit);
    int jkl_bit_read_bits(JklBitReader *reader, uint32_t bit_count, uint64_t *out_value);
    int jkl_bit_discard_bits(JklBitReader *reader, uint32_t bit_count);

#ifdef __cplusplus
}
#endif

#endif
