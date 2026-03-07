#include "bit_reader.h"

#include <assert.h>

static int jkl_bit_reader_refill_bits(JklBitReader *reader, uint32_t need_bits)
{
    uint32_t missing;
    uint32_t max_bytes;
    uint32_t want_bytes;

    if (need_bits <= reader->buffer_len)
    {
        return JKL_OK;
    }

    if (reader->buffer_len >= 64U)
    {
        return JKL_OK;
    }

    missing = need_bits - reader->buffer_len;
    max_bytes = (64U - reader->buffer_len) / 8U;
    want_bytes = (missing + 7U) / 8U;

    if (want_bytes > max_bytes)
    {
        want_bytes = max_bytes;
    }

    if (want_bytes == 0U)
    {
        return JKL_OK;
    }

    if (reader->source == JKL_BIT_SOURCE_FILE)
    {
        uint8_t temp[8];
        size_t got = fread(temp, 1U, (size_t)want_bytes, reader->src.file);
        size_t i;

        if (got == 0U)
        {
            if (ferror(reader->src.file) != 0)
            {
                return JKL_ERR_IO;
            }
            return JKL_ERR_EOF;
        }

        for (i = 0; i < got; ++i)
        {
            reader->buffer |= ((uint64_t)temp[i]) << reader->buffer_len;
            reader->buffer_len += 8U;
        }

        return JKL_OK;
    }

    {
        uint32_t available = 0;
        uint32_t to_take;
        uint32_t i;

        if (reader->src.mem.pos < reader->src.mem.size)
        {
            size_t rem = reader->src.mem.size - reader->src.mem.pos;
            if (rem > 0xFFFFFFFFu)
            {
                available = 0xFFFFFFFFu;
            }
            else
            {
                available = (uint32_t)rem;
            }
        }

        if (available == 0U)
        {
            return JKL_ERR_EOF;
        }

        to_take = want_bytes;
        if (to_take > available)
        {
            to_take = available;
        }

        for (i = 0; i < to_take; ++i)
        {
            uint8_t b = reader->src.mem.data[reader->src.mem.pos++];
            reader->buffer |= ((uint64_t)b) << reader->buffer_len;
            reader->buffer_len += 8U;
        }
    }

    if (reader->buffer_len == 0U)
    {
        return JKL_ERR_EOF;
    }

    return JKL_OK;
}

int jkl_bit_reader_init_file(FILE *file, JklBitReader *out_reader)
{
    out_reader->source = JKL_BIT_SOURCE_FILE;
    out_reader->buffer = 0;
    out_reader->buffer_len = 0;
    out_reader->src.file = file;
    return JKL_OK;
}

int jkl_bit_reader_init_memory(const uint8_t *data, size_t size, JklBitReader *out_reader)
{
    assert(data != NULL || size == 0);
    assert(out_reader != NULL);

    out_reader->source = JKL_BIT_SOURCE_MEMORY;
    out_reader->buffer = 0;
    out_reader->buffer_len = 0;
    out_reader->src.mem.data = data;
    out_reader->src.mem.size = size;
    out_reader->src.mem.pos = 0;
    return JKL_OK;
}

int jkl_bit_read_bit(JklBitReader *reader, int *out_bit)
{
    JKL_RETURN_IF_ERROR(jkl_bit_reader_refill_bits(reader, 1U));
    assert(reader->buffer_len > 0U);

    *out_bit = (int)(reader->buffer & 1ULL);
    reader->buffer >>= 1;
    reader->buffer_len -= 1U;

    return JKL_OK;
}

int jkl_bit_read_bits(JklBitReader *reader, uint32_t bit_count, uint64_t *out_value)
{
    uint64_t value = 0;
    uint32_t written = 0;

    assert(bit_count <= 64U);

    while (written < bit_count)
    {
        uint32_t remain = bit_count - written;
        uint32_t take;
        uint64_t chunk;

        if (reader->buffer_len < remain)
        {
            JKL_RETURN_IF_ERROR(jkl_bit_reader_refill_bits(reader, remain));
        }

        take = reader->buffer_len;
        if (take > remain)
        {
            take = remain;
        }

        if (take == 64U)
        {
            chunk = reader->buffer;
        }
        else
        {
            chunk = reader->buffer & ((1ULL << take) - 1ULL);
        }

        value |= chunk << written;

        reader->buffer >>= take;
        reader->buffer_len -= take;
        written += take;
    }

    *out_value = value;
    return JKL_OK;
}

int jkl_bit_discard_bits(JklBitReader *reader, uint32_t bit_count)
{
    uint32_t discarded = 0;

    assert(bit_count <= 64U);

    while (discarded < bit_count)
    {
        uint32_t remain = bit_count - discarded;
        uint32_t take;

        if (reader->buffer_len < remain)
        {
            JKL_RETURN_IF_ERROR(jkl_bit_reader_refill_bits(reader, remain));
        }

        take = reader->buffer_len;
        if (take > remain)
        {
            take = remain;
        }

        reader->buffer >>= take;
        reader->buffer_len -= take;
        discarded += take;
    }

    return JKL_OK;
}