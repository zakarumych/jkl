#include "jkl/bits.h"

#include <assert.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

static uint8_t jkl_ctz64_nonzero(uint64_t value)
{
    assert(value != 0);

#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward64(&index, value);
    return (uint8_t)index;
#elif defined(__GNUC__) || defined(__clang__)
    return (uint8_t)__builtin_ctzll(value);
#else
    uint8_t c = 0;
    while ((value & 1ULL) == 0)
    {
        value >>= 1;
        c += 1;
    }
    return c;
#endif
}

static int jkl_bit_reader_refill_bits(JklBitReader *reader)
{
    FILE *fp = reader->impl.fp;

    assert(reader->buffer_len == 0);
    reader->buffer = 0;

    if (reader->kind == JKL_BIT_READER_FILE)
    {
        uint8_t temp[8];
        size_t got = fread(temp, 1U, sizeof(reader->buffer), fp);
        size_t i;

        if (got == 0U)
        {
            if (ferror(fp) != 0)
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
    }
    else
    {
        assert(reader->kind == JKL_BIT_READER_MEMORY);

        const uint8_t *data = reader->impl.memory.data;
        size_t size = reader->impl.memory.size;

        if (size == 0)
        {
            return JKL_ERR_EOF;
        }

        while (reader->buffer_len < 64U && size > 0)
        {
            reader->buffer |= ((uint64_t)(*data)) << reader->buffer_len;
            reader->buffer_len += 8U;
            data += 1;
            size -= 1;
        }

        reader->impl.memory.data = data;
        reader->impl.memory.size = size;
    }

    return JKL_OK;
}

int jkl_bit_reader_init_file(FILE *file, JklBitReader *out_reader)
{
    assert(file != NULL);
    assert(out_reader != NULL);

    out_reader->kind = JKL_BIT_READER_FILE;
    out_reader->impl.fp = file;
    out_reader->buffer = 0;
    out_reader->buffer_len = 0;
    return JKL_OK;
}

int jkl_bit_reader_init_memory(const uint8_t *data, size_t size, JklBitReader *out_reader)
{
    assert(data != NULL || size == 0);
    assert(out_reader != NULL);

    out_reader->kind = JKL_BIT_READER_MEMORY;
    out_reader->impl.memory.data = data;
    out_reader->impl.memory.size = size;
    out_reader->buffer = 0;
    out_reader->buffer_len = 0;
    return JKL_OK;
}

int jkl_bit_read_bit(JklBitReader *reader, int *out_bit)
{
    if (reader->buffer_len == 0)
    {
        JKL_RETURN_IF_ERROR(jkl_bit_reader_refill_bits(reader));
    }

    assert(reader->buffer_len > 0U);

    *out_bit = (int)(reader->buffer & 1ULL);
    reader->buffer >>= 1;
    reader->buffer_len -= 1U;

    return JKL_OK;
}

int jkl_bit_read_until_set_bit(JklBitReader *reader, uint64_t *pos)
{
    uint64_t bit_pos = 0;

    while (reader->buffer == 0)
    {
        bit_pos += reader->buffer_len;
        reader->buffer_len = 0;
        JKL_RETURN_IF_ERROR(jkl_bit_reader_refill_bits(reader));
    }

    uint8_t trailing_zeros = jkl_ctz64_nonzero(reader->buffer);
    uint8_t consumed = (uint8_t)(trailing_zeros + 1u);
    reader->buffer >>= consumed;
    reader->buffer_len = (uint8_t)(reader->buffer_len - consumed);
    *pos = bit_pos + trailing_zeros;
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
        uint8_t take;
        uint64_t chunk;

        if (reader->buffer_len == 0)
        {
            JKL_RETURN_IF_ERROR(jkl_bit_reader_refill_bits(reader));
        }

        take = reader->buffer_len;
        if (take > remain)
        {
            take = (uint8_t)remain;
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
        reader->buffer_len = (uint8_t)(reader->buffer_len - take);
        written += take;
    }

    *out_value = value;
    return JKL_OK;
}
