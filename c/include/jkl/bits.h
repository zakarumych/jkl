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
    typedef struct JklBitReader
    {
        /* Buffered bits */
        uint64_t buffer;

        /* Number of bits currently in buffer */
        uint8_t buffer_len;

        enum
        {
            JKL_BIT_READER_FILE,
            JKL_BIT_READER_MEMORY
        } kind;

        union
        {
            FILE *fp;
            struct
            {
                const uint8_t *data;
                size_t size;
            } memory;
        } impl;
    } JklBitReader;

    int jkl_bit_reader_init_file(FILE *file, JklBitReader *out_reader);
    int jkl_bit_reader_init_memory(const uint8_t *data, size_t size, JklBitReader *out_reader);

    int jkl_bit_read_bit(JklBitReader *reader, int *out_bit);
    int jkl_bit_read_until_set_bit(JklBitReader *reader, uint64_t *pos);
    int jkl_bit_read_bits(JklBitReader *reader, uint32_t bit_count, uint64_t *out_value);
#ifdef __cplusplus
}
#endif

#endif
